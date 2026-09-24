"""Cell-level orchestrator for the cash caching pipeline.

Owns the 7-phase pipeline that ``%cash_on``'s ``run_cell`` and
``run_cell_async`` hooks run every cell through:

    1. Cell ID & notebook path resolution (the caller's: ``cell_id``)
    2. Badge & timing initialisation
    3. Module change detection
    4. Upstream dependency resolution
    5. AST parse
    6. Pre-execution notification assembly
    7. Statement-by-statement execution

Both hooks delegate to :meth:`CellExecutor.execute_cell` (or its async
twin, which shares every phase but statement execution): there is exactly
one cell-execution code path.

**Anti-god-class rule (load-bearing):**

- ``CellExecutor`` does not call IPython's ``display()`` or
  ``publish_display_data()`` directly.  The badge and the error display are
  drawn by the :class:`BadgePresenter` it is given, the same one
  ``CashMagics`` draws the final badge with.
- ``CellExecutor`` does not restore variables.  Variable-granular cache
  work is :class:`Restorer`'s job.  The executor calls
  ``restorer.restore_variable(var_name)`` during upstream resolution; it
  never reaches into the backend itself.

**`original_run_cell` parameter**:

The hook supplies its captured ``_original_run_cell`` so error paths
that arise mid-pipeline (SyntaxError from upstream simulation,
``RuntimeError`` / :class:`AmbiguousCellError`, generic exception
fallback) can be surfaced through IPython's normal execution machinery
and the kernel reply status stays as "error".  The async hook passes
``None``: those exceptions propagate to it, and it re-raises them through
the original ``run_cell_async`` itself.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import sys
import time
import uuid
from collections.abc import Awaitable, Callable, Generator, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from ...analysis.annotations import audited_lines, get_statement_annotations
from ...analysis.code_analyzer import CodeAnalyzer, splitlines_like_the_parser, statement_code
from ...backends._writes import discarded_writes
from ...diagnostics import warn_diagnostic
from ...exceptions import (
    AmbiguousCellError,
    CashCacheIneffectiveWarning,
    ForwardReferenceError,
    UpstreamStateError,
)
from ...remote_source import measured_validation as _measured_validation
from ...tracking.file_dep_snapshot import begin_file_state_epoch, end_file_state_epoch
from ...tracking.randomness import get_drawing_rng_modules, rng_lineage_fingerprint
from .._protocols import ShellProtocol
from ..cache_status import CacheStatus
from ..consumables import consumable_state, is_consumable_unrestorable
from ..control_structures import contains_top_level_await, is_control_structure
from ..statement import ProcessResult
from ..statement.capture import replay_outputs
from ..tracking_state import TrackingState

if TYPE_CHECKING:
    from ..control_structures import ControlStructureProcessor
    from ..module_invalidator import ModuleInvalidator
    from ..restore import Restorer
    from ..statement import StatementProcessor
    from ..upstream import UpstreamChecker
    from ._types import TimingBreakdown
    from .badges import BadgePresenter

import logging

logger = logging.getLogger(__name__)


class EarlyReturn:
    """Sentinel wrapper for early-exit values that flow back up to the
    hook proxy unchanged.  Carries an IPython ``run_cell`` result."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value


class PipelineSyntaxError:
    """Sentinel returned by :meth:`CellExecutor.execute_cell` when the cell's
    own AST fails to parse.  Caller decides how to react."""

    __slots__ = ()


class PipelineCompleted:
    """Successful pipeline run: carries everything the finaliser needs."""

    __slots__ = (
        "all_metrics",
        "buffered_outputs",
        "badge_display_id",
        "hook_start",
        "timing_breakdown",
        "badge_render_time",
    )

    def __init__(
        self,
        all_metrics: list,
        buffered_outputs: list,
        badge_display_id: str,
        hook_start: float,
        timing_breakdown: "TimingBreakdown",
        badge_render_time: float,
    ) -> None:
        self.all_metrics = all_metrics
        self.buffered_outputs = buffered_outputs
        self.badge_display_id = badge_display_id
        self.hook_start = hook_start
        self.timing_breakdown = timing_breakdown
        self.badge_render_time = badge_render_time


@dataclass
class _CellRun:
    """A cell on its way through the pipeline, once phases 1-6 are done."""

    raw_cell: str
    tree: ast.Module
    #: Every row the badge will show, the upstream rows first.
    all_metrics: list[ProcessResult]
    badge_display_id: str
    hook_start: float
    timing_breakdown: TimingBreakdown
    #: The TTL the cell's statements are stored with.
    ttl: int | None = None


@dataclass(frozen=True)
class _Step:
    """The one part of a cell the sync and async paths run differently.

    A statement for the statement processor (*kwargs* are its arguments), or,
    with *await_unit*, a control structure whose body awaits, run as one
    awaited unit (*kwargs* then go to ``process_await_unit``).
    """

    kwargs: dict[str, Any]
    await_unit: ast.stmt | None = None


_StepResult = TypeVar("_StepResult")


def _drive(steps: Generator[_Step, Any, _StepResult], run: Callable[[_Step], Any]) -> _StepResult:
    """Run *steps* to completion, handing each yielded step to *run*.

    An exception *run* raises is raised back inside *steps*, at the ``yield``
    that asked for the step, so the loop's own handlers see it exactly as if
    the step had run in place.
    """
    try:
        step = next(steps)
        while True:
            try:
                reply = run(step)
            except BaseException as exc:  # noqa: BLE001 - re-raised inside the loop, where it happened
                step = steps.throw(exc)
            else:
                step = steps.send(reply)
    except StopIteration as done:
        return done.value


async def _drive_async(
    steps: Generator[_Step, Any, _StepResult], run: Callable[[_Step], Awaitable[Any]]
) -> _StepResult:
    """:func:`_drive`, awaiting each step."""
    try:
        step = next(steps)
        while True:
            try:
                reply = await run(step)
            except BaseException as exc:  # noqa: BLE001 - re-raised inside the loop, where it happened
                step = steps.throw(exc)
            else:
                step = steps.send(reply)
    except StopIteration as done:
        return done.value


def _pyplot_open_fignums() -> set[int]:
    """Open matplotlib figure numbers, or an empty set if pyplot isn't loaded.

    Only inspects an already-imported ``matplotlib.pyplot`` — never imports it,
    so it stays a no-op for notebooks that don't plot.
    """
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return set()
    try:
        return set(plt.get_fignums())
    except Exception:  # noqa: BLE001 - a broken backend must not break execution
        return set()


def _close_pyplot_figures(nums: set[int]) -> None:
    """Close the given matplotlib figures, removing them from pyplot's registry.

    Used after upstream re-execution: a figure that reconstruction OPENED (to
    rebuild a ``fig``/``ax`` a downstream cell needs) would otherwise be flushed
    by the inline backend's post-execute hook into the DOWNSTREAM cell's output —
    a stray plot. A normally-run cell closes its figures on flush anyway, so
    closing the reconstructed ones matches that end state. The Figure/Axes
    objects stay valid (``fig.savefig`` / ``ax.*`` still work) for the cell that
    asked for them.
    """
    if not nums:
        return
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return
    for num in nums:
        try:
            plt.close(num)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass


def staleness_notification(tracker) -> dict | None:
    """Badge row for a notebook file cash has PROVEN is out of date.

    Returns None unless there is proof. This is deliberately quiet: a warning
    that appears when cash is merely unsure is a warning users learn to skip,
    and this one needs to be believed the once it matters.

    ASCII only: `code` is written into the saved .ipynb and may be read back
    by a different process (nbconvert, a log scraper, an agent) on a console
    whose codepage cash cannot know. See `cash.notebook.staleness._to_ascii`,
    which sanitises `hint()` for the same reason before it ever gets here.

    Message order is load-bearing, not stylistic. `%cash_badge print` renders
    `code` through `renderers.text._row_line`, which hard-truncates a row's
    first line at `theme.HEADER_MAX_LEN` (80 chars) -- there is no tooltip or
    drawer in that mode to hold the rest, unlike HTML. The fact of staleness
    and the remedy ("Save and re-run") are what make the row actionable, so
    they go FIRST, comfortably inside the cap; the save time and the cell
    hint are supporting evidence, appended after, and may be silently cut off
    in print mode. Keep the essential clause short enough that it plus a
    small margin stays under 80 chars even after the RNG suffix a future
    change might add.
    """
    if not tracker.is_stale():
        return None
    saved = tracker.saved_at()
    when = time.strftime("%H:%M:%S", time.localtime(saved)) if saved else "an earlier time"
    hint = tracker.hint()
    where = f" '{hint}' differs from the saved copy." if hint else ""
    return {
        "status": CacheStatus.WARNING,
        "code": (
            f"[!] Notebook file is stale -- Save (Ctrl+S) and re-run to be sure. "
            f"Upstream check used the copy saved at {when}.{where} "
            f"Other cells may have changed too."
        ),
        "is_upstream": True,
        "total_time": 0.0,
        "execution_time": 0.0,
        "outputs": [],
    }


def discarded_writes_notification(seen_before: int) -> tuple[dict | None, int]:
    """Badge row said when a cache write failed and was thrown away.

    Returns ``(row_or_None, new_total)`` so the caller can carry the watermark
    to the next cell.

    A discarded write is the one failure the rest of the badge cannot express.
    It is not a miss -- a miss is a row that says EXECUTED and tells you so. It
    is a hit that never got the chance to exist: the entry is absent, the work
    recomputes every run, and every counter on the badge looks healthy. Windows
    spent an unknown period doing exactly this on every run (fixed in 0.4.1),
    and the only report was a logger warning at kernel shutdown, which in a
    notebook means never.

    Loud on every occurrence: this reports work being lost right now, and a
    second occurrence is a second lost result rather than a repeat of the same
    news.

    ASCII only and short, for the reasons `staleness_notification` gives -- the
    print renderer caps a row at 80 characters.
    """
    try:
        total = len(discarded_writes())
    except Exception:  # noqa: BLE001 - a diagnostic must never break a cell
        return None, seen_before
    if total <= seen_before:
        return None, total

    new = total - seen_before
    plural = "s" if new != 1 else ""
    return {
        "status": CacheStatus.WARNING,
        "code": (f"[!] {new} cache write{plural} failed -- not cached, will recompute. See %cash_stats."),
        "is_upstream": False,
        "total_time": 0.0,
        "execution_time": 0.0,
        "outputs": [],
    }, total


def _statement_source(raw_cell: str, node: ast.stmt) -> str | None:
    """The statement's original text, as the badge shows it.

    Also what a cache-miss statement compiles from (``exec_source``) in place
    of the ``ast.unparse`` form, which strips comments; the cache key is
    always the unparsed form. Continuation lines are dedented by the node's
    own ``col_offset``, since ``get_source_segment`` leaves them at their
    absolute indentation.

    A top-level ``def``/``class`` returns ``None``: running it only binds the
    name, and its full body would make the badge very tall. Lifting that
    would also change what EXECUTES, since ``_exec_source_for_node`` returns
    this text whenever it is set, and so bypass that function's waiver gate.
    A ``match`` is not excluded: it runs as one unit, so its text is the code
    that ran.

    A single-line statement is sliced out directly: ``get_source_segment``
    re-splits the whole cell on every call (O(statements x cell length)).
    Offsets are UTF-8 byte offsets, as there, and the split follows the
    parser's line breaks (``splitlines_like_the_parser``), not
    ``str.splitlines``'s, or the line index drifts from ``node.lineno``.

    Returns ``None`` when the segment cannot be recovered; the caller falls
    back to the unparsed form. Never raises.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return None
    try:
        end_lineno = node.end_lineno
        if end_lineno is not None and end_lineno == node.lineno and node.end_col_offset is not None:
            lines = splitlines_like_the_parser(raw_cell)
            segment = lines[node.lineno - 1].encode()[node.col_offset : node.end_col_offset].decode()
        else:
            segment = ast.get_source_segment(raw_cell, node)
    except (IndexError, ValueError, TypeError, AttributeError):  # display only
        return None
    if not segment:
        return None

    indent = getattr(node, "col_offset", 0) or 0
    if indent <= 0:
        return segment

    head, *rest = segment.split("\n")
    return "\n".join([head] + [line[indent:] if line[:indent].isspace() else line for line in rest])


def _exec_source_for_node(
    raw_cell: str,
    node: ast.stmt,
    stmt_display: str | None,
) -> str | None:
    """The text to EXECUTE for *node*, where it differs from ``stmt_display``.

    ``stmt_display`` is the text the badge shows. It withholds a top-level
    ``def``/``class`` body (``_statement_source``), and the unparsed fallback
    has no comments, so a ``# @cash:assume-safe`` waiver inside a function
    defined in a cell would be invisible to ``inspect.getsource`` and so to
    the purity analyzer. For such a function this recovers the original text,
    decorators and a trailing comment on its last line included.

    Only when the body carries a real waiver, as decided by
    ``annotations.audited_lines`` (the analyzer's own test, so the two
    cannot disagree). Every other ``def``/``class`` must compile from the
    unparsed text on every path: the upstream re-execution path always
    compiles that form, and two texts for one function give it two identity
    hashes, which re-keys every cached call to it. A waived function
    redefined through that path still hashes differently once per session
    (one lost call-cache hit; the values stay correct).

    The recovered text is checked with ``compile()`` (with top-level
    ``await`` allowed, as the async path compiles), since the line-based
    reconstruction can go wrong -- a PEP 614 decorator expression starting
    below its ``@`` line, for one.

    Returns ``stmt_display`` when it is set, else the recovered text, else
    ``None`` (the executor then runs the unparsed code). Never raises.
    """
    if stmt_display is not None:
        return stmt_display
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return None
    # A pre-filter only: `body` is part of `raw_cell`, so no waiver can be
    # found in it without this text somewhere in the cell.
    if "@cash:" not in raw_cell:
        return None
    try:
        body = ast.get_source_segment(raw_cell, node)
        if not body:
            return None
        lines = splitlines_like_the_parser(raw_cell)
        # The segment starts at the `def`/`class` line, not the decorators.
        decorators = node.decorator_list
        if decorators:
            prefix = "".join(lines[decorators[0].lineno - 1 : node.lineno - 1])
            body = prefix + body
        end_lineno = getattr(node, "end_lineno", None)
        end_col = getattr(node, "end_col_offset", None)
        # ...and ends at `end_col_offset`, before a comment on the last line.
        if end_lineno is not None and end_col is not None and 0 < end_lineno <= len(lines):
            rest = lines[end_lineno - 1].encode()[end_col:].decode()
            trailing = rest.split("\n", 1)[0].rstrip("\r")
            if trailing.strip() == "" or trailing.lstrip().startswith("#"):
                body = body + trailing
        # The function-scope flag is derived from the marked lines, so no
        # marked line means no waiver of either kind.
        waived_lines, _ = audited_lines(body)
        if not waived_lines:
            return None
        # Top-level `await` allowed, as `CodeRunner.run_async` compiles: an
        # `@await get_deco()` decorator is legal there. On the sync path such
        # text fails to compile either way, so the flag decides nothing.
        compile(body, "<cash-recovery-check>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        return body
    except Exception:  # noqa: BLE001 - execution must never break over this
        return None


def _builtin_trap(shell: Any):
    """The shell's builtin trap, which IPython enters around every cell it runs.

    It puts ``get_ipython`` (and ``display``) into ``builtins`` for the cell's
    duration. cash runs a cell's statements itself, outside IPython's run, so
    they ran without it: pandas imported in a cached cell asked ``get_ipython()``,
    got NameError, decided it was in a terminal and set ``display.max_columns``
    to 0 instead of 20 -- tables printed differently with cash on.
    Anything else that detects a notebook that way was fooled too. The
    trap nests, so IPython's own run inside it is unaffected.
    """
    trap = getattr(shell, "builtin_trap", None)
    if trap is None or not hasattr(trap, "__enter__"):
        return contextlib.nullcontext()
    return trap


def _set_written_later(executor: Any, names: frozenset[str]) -> None:
    """Tell the statement processor which names the rest of the cell writes."""
    processor = getattr(executor, "_statement_processor", None)
    if processor is not None:
        processor.set_written_later_in_cell(names)


def _written_later_in_cell(body: list[ast.stmt]) -> list[frozenset[str]]:
    """For each top-level statement, the names a LATER statement of the cell
    writes -- rebinds or changes in place.

    A value every one of whose names is written again before the cell ends is
    an intermediate: the cell leaves a later version, and the end-of-cell pass
    writes that one to disk when restoring beats rebuilding
    (``TieredBackend.persist_from_memory``). Writing each intermediate to disk
    as it was made cost a cleaning cell 3.8 s of pickling on a cold run,
    for ~500 MB versions of ``sales`` that nothing restores.
    """
    outputs: list[set[str]] = []
    for node in body:
        try:
            _inputs, outs = CodeAnalyzer.analyze_code_block(ast.unparse(node))
        except Exception:  # noqa: BLE001 - unknown writes defer nothing
            outs = set()
        outputs.append(set(outs))
    later: list[frozenset[str]] = [frozenset()] * len(body)
    acc: set[str] = set()
    for i in range(len(body) - 1, -1, -1):
        later[i] = frozenset(acc)
        acc |= outputs[i]
    return later


def _jumpable_runs(body: list[ast.stmt], raw_cell: str, touches_rng) -> dict[int, int]:
    """``{start: end}`` of the runs of plain assignments a restore can jump in.

    A run is consecutive top-level assignments that rebuild a name more than
    once (``sales = ...``, ``sales["t"] = ...``, ...), with no ``# @cash:``
    directive, no random draw, and no statement reading a name the run writes
    before the run has written it -- so where the run starts from is what the
    cell had before it, and every version inside it is the run's own. See
    ``UpstreamChecker.plan_cell_run``.
    """

    def plain(node) -> bool:
        if isinstance(node, ast.AnnAssign) and node.value is None:
            return False
        if not isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            return False
        if any(isinstance(n, (ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)) for n in ast.walk(node)):
            return False
        if "@cash:" in raw_cell and get_statement_annotations(raw_cell, node).has_directives():
            return False
        return not touches_rng(ast.unparse(node))

    runs: dict[int, int] = {}
    i, n = 0, len(body)
    while i < n:
        if not plain(body[i]):
            i += 1
            continue
        j = i
        while j < n and plain(body[j]):
            j += 1
        reads_writes = []
        for node in body[i:j]:
            try:
                inputs, outputs = CodeAnalyzer.analyze_code_block(ast.unparse(node))
            except Exception:  # noqa: BLE001 - an unanalysable statement ends the run
                reads_writes = []
                break
            reads_writes.append((set(inputs), set(outputs)))
        if len(reads_writes) >= 2:
            written = [w for _, w in reads_writes]
            everything = set().union(*written)
            rebuilt = any(sum(1 for w in written if name in w) > 1 for name in everything)
            so_far: set[str] = set()
            ordered = True
            for reads, writes in reads_writes:
                if reads & (everything - so_far):
                    ordered = False
                    break
                so_far |= writes
            if rebuilt and ordered and _writes_only_into_its_own_objects(body[i:j]):
                runs[i] = j
        i = j
    return runs


#: Methods whose result may be the object they are called on, or share its data.
_VIEW_METHODS = frozenset(
    {
        "view",
        "reshape",
        "ravel",
        "squeeze",
        "transpose",
        "swapaxes",
        "pipe",
        "asarray",
        "asanyarray",
        "ascontiguousarray",
        "__getitem__",
        "get",
    }
)


def _makes_a_new_object(value: ast.expr) -> bool:
    """Whether *value* evaluates to an object no other name holds.

    Conservative: a name, an attribute, a slice (``arr[1:]`` is a view of
    ``arr``) or a column (``df['a']``) may be shared, as may a call known to
    return its receiver or a view. A mask or a list of columns selects a copy;
    arithmetic, literals and other calls make a new object.
    """
    if isinstance(value, ast.Call):
        func = value.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        return name not in _VIEW_METHODS
    if isinstance(
        value,
        (
            ast.BinOp,
            ast.UnaryOp,
            ast.Compare,
            ast.BoolOp,
            ast.List,
            ast.Dict,
            ast.Set,
            ast.ListComp,
            ast.DictComp,
            ast.SetComp,
            ast.JoinedStr,
        ),
    ):
        return True
    if isinstance(value, ast.Subscript):
        if isinstance(value.value, (ast.Tuple, ast.List)) and isinstance(value.slice, ast.Constant):
            items = value.value.elts
            index = value.slice.value
            return isinstance(index, int) and -len(items) <= index < len(items) and _makes_a_new_object(items[index])
        return isinstance(value.slice, (ast.Compare, ast.BoolOp, ast.UnaryOp, ast.List))
    return False


def _writes_only_into_its_own_objects(nodes: list[ast.stmt]) -> bool:
    """Whether every in-place write of the run lands in an object the run made.

    ``y = x; y[0] += 5`` changes ``x``, and ``v = arr[1:]; v += 1`` changes
    ``arr``: skipping or restoring those statements loses the change to the
    object outside the run. So a name the run writes into -- ``name[...] =``,
    ``name.attr =``, ``name += ...`` -- must have been bound in the run, before
    the write, by an expression that makes a new object.
    """
    fresh: set[str] = set()
    for node in nodes:
        targets = [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else list(node.targets)
        for target in targets:
            if isinstance(node, ast.AugAssign) and isinstance(target, ast.Name):
                if target.id not in fresh:
                    return False
                continue
            base = target
            while isinstance(base, (ast.Subscript, ast.Attribute)):
                base = base.value
            if base is not target:
                if not isinstance(base, ast.Name) or base.id not in fresh:
                    return False
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            names = {t.id for t in targets if isinstance(t, ast.Name)}
            if len(targets) == 1 and names and _makes_a_new_object(node.value):
                fresh |= names
            else:
                # A rebinding by anything else (``a = b = x``, ``y, z = pair``)
                # may share; a write into ``name[...]`` keeps the name's object.
                fresh -= {
                    n.id
                    for t in targets
                    if not isinstance(t, (ast.Subscript, ast.Attribute))
                    for n in ast.walk(t)
                    if isinstance(n, ast.Name)
                }
    return True


class CellExecutor:
    """Run a single notebook cell through the cached-execution pipeline.

    Single public entry: :meth:`execute_cell` (and its async twin
    :meth:`execute_cell_async`) — there is no separate code path.
    """

    def __init__(
        self,
        shell: ShellProtocol,
        cash_instance: Any,
        badges: "BadgePresenter",
        tracking_state: "TrackingState",
        statement_processor: "StatementProcessor",
        upstream_checker: "UpstreamChecker",
        restorer: "Restorer",
        module_invalidator: "ModuleInvalidator",
        control_structure_processor: "ControlStructureProcessor",
    ) -> None:
        self.shell = shell
        self._cash_instance = cash_instance
        self._badges = badges
        self.tracking_state = tracking_state
        self._statement_processor = statement_processor
        self._upstream_checker = upstream_checker
        self._restorer = restorer
        self._module_invalidator = module_invalidator
        self._control_structure_processor = control_structure_processor

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute_cell(
        self,
        raw_cell: str,
        args: tuple = (),
        kwargs: dict | None = None,
        original_run_cell: Callable[..., Any] | None = None,
        *,
        ttl: int | None = None,
        cell_id: str | None = None,
    ) -> PipelineCompleted | PipelineSyntaxError | EarlyReturn:
        """Run *raw_cell* through the 7-phase cached-execution pipeline.

        Returns one of:
        - :class:`PipelineCompleted` — caller invokes the finaliser
        - :class:`PipelineSyntaxError` — the cell's own AST failed to parse
        - :class:`EarlyReturn` — propagate the wrapped value (hook only)
        """
        with self._cell_scope():
            cell = self._prepare_cell(raw_cell, args, kwargs or {}, original_run_cell, ttl, cell_id)
            if not isinstance(cell, _CellRun):
                return cell
            with self._statements_scope(cell):
                result = self._execute_cell_statements(cell)
            return self._complete_cell(cell, result)

    async def execute_cell_async(
        self,
        raw_cell: str,
        args: tuple = (),
        kwargs: dict | None = None,
        original_run_cell: Callable[..., Any] | None = None,
        *,
        ttl: int | None = None,
        cell_id: str | None = None,
    ) -> PipelineCompleted | PipelineSyntaxError | EarlyReturn:
        """:meth:`execute_cell` for a cell with a top-level ``await``.

        Every phase is the sync pipeline's own; only the statements are run
        through :meth:`_execute_cell_statements_async`, so a top-level
        ``await`` runs on IPython's live loop.
        """
        with self._cell_scope():
            cell = self._prepare_cell(raw_cell, args, kwargs or {}, original_run_cell, ttl, cell_id)
            if not isinstance(cell, _CellRun):
                return cell
            with self._statements_scope(cell):
                result = await self._execute_cell_statements_async(cell)
            return self._complete_cell(cell, result)

    @contextlib.contextmanager
    def _cell_scope(self) -> Iterator[None]:
        """What holds for the whole cell run: one file-state epoch (each file
        is hashed at most once per cell), one batch of backend notices (one
        CACHE-NOT-WORTH-BYTES per cell, not per statement), and IPython's
        builtin trap."""
        begin_file_state_epoch()
        backend = self._cell_warning_backend()
        if backend is not None:
            backend.hold_notices()
        try:
            with _builtin_trap(self.shell):
                yield
        finally:
            end_file_state_epoch()
            if backend is not None:
                backend.release_notices()

    def _cell_warning_backend(self):
        """The backend that batches this cell's warnings, if it does."""
        try:
            cash = self._statement_processor.get_cash_instance()
            return cash.backend if cash is not None else None
        except Exception:  # noqa: BLE001 - batching is cosmetic; never block a cell
            return None

    def _prepare_cell(
        self,
        raw_cell: str,
        args: tuple,
        kwargs: dict,
        original_run_cell: Callable[..., Any] | None,
        ttl: int | None,
        cell_id: str | None,
    ) -> _CellRun | PipelineSyntaxError | EarlyReturn:
        """Phases 2-6: everything before the cell's statements run.

        Phase 1, the cell id and the notebook path, is the caller's: *cell_id*
        is what it resolved, and *ttl* the TTL the cell's entries are stored
        with.
        """
        # 2. Badge & timing init
        badge_display_id = str(uuid.uuid4())
        timing_breakdown = self._init_cell_timing_and_badge(badge_display_id)
        hook_start = time.time()
        logger.debug("[TIMING_PROXY] Start cached_run_cell")

        # 3. Module change detection (must precede upstream check)
        pre_upstream_metrics = self._detect_module_changes(raw_cell)

        # 4. Upstream resolution
        upstream_result = self._resolve_upstream_state(
            raw_cell,
            pre_upstream_metrics,
            badge_display_id,
            timing_breakdown,
            args,
            kwargs,
            original_run_cell,
            ttl=ttl,
            cell_id=cell_id,
        )
        if isinstance(upstream_result, EarlyReturn):
            return upstream_result
        upstream_metrics, _restore_time, _execution_time = upstream_result

        # 5. AST parse (tolerate a top-level ``await``; a bare
        # ast.parse rejects module-level await and would silently skip the cell)
        try:
            tree = CodeAnalyzer.parse_cell(raw_cell)
        except SyntaxError:
            self._badges.close(badge_display_id)
            return PipelineSyntaxError()

        # 6. Pre-execution notifications
        all_metrics = self._build_pre_execution_notifications(
            raw_cell,
            pre_upstream_metrics,
            upstream_metrics,
        )
        return _CellRun(raw_cell, tree, all_metrics, badge_display_id, hook_start, timing_breakdown, ttl)

    @contextlib.contextmanager
    def _statements_scope(self, cell: _CellRun) -> Iterator[None]:
        """Phase 7's frame around the statements: a fresh RNG observation and
        statement log for the cell, remote freshness checks measured into the
        breakdown, and -- once every statement ran -- the persisting of what
        the cell left that would be costly to rebuild after a restart."""
        logger.debug("[TIMING_PROXY] Start executing statements")
        self._statement_processor.begin_cell_rng_observation()
        self._statement_processor.begin_cell_statement_log()
        # Remote freshness checks (a cached function reading s3:// and friends)
        # are network round trips that land on the HIT path, where the badge
        # reports a saving and nothing reports what establishing it cost.
        with _measured_validation(sink=cell.timing_breakdown):
            yield
        t_persist = time.time()
        self._statement_processor.end_cell_persistence()
        cell.timing_breakdown["persist_final"] = time.time() - t_persist

    def _complete_cell(
        self,
        cell: _CellRun,
        result: EarlyReturn | tuple[list[ProcessResult], list, float],
    ) -> PipelineCompleted | EarlyReturn:
        """What the finaliser needs, once the cell's statements have run."""
        if isinstance(result, EarlyReturn):
            return result
        all_metrics, buffered_result_outputs, badge_render_time = result
        cell.timing_breakdown["badge_progress"] = badge_render_time
        self._record_executed_cell_hash(cell.raw_cell)
        return PipelineCompleted(
            all_metrics=all_metrics,
            buffered_outputs=buffered_result_outputs,
            badge_display_id=cell.badge_display_id,
            hook_start=cell.hook_start,
            timing_breakdown=cell.timing_breakdown,
            badge_render_time=badge_render_time,
        )

    def _record_executed_cell_hash(self, raw_cell: str) -> None:
        """Remember that this exact cell source ran, so the upstream checker can
        tell an edited-but-not-rerun seed() cell from one that actually ran.
        Also snapshot the RNG state around a cell that
        TOUCHED the global RNG so a downstream draw can be restored to its
        position-correct state, and record which
        modules it changed — which catches draws inside called functions that
        static analysis cannot see.

        The RNG half is HARVESTED from the statement-level observer rather than
        re-measured here. There used to be two observers snapshotting the same
        streams to answer the same question at different granularities; the
        statement one is finer and can reconstruct this one (union of what its
        statements changed), so it is now the single source of truth.

        That also narrows the recorded window to the user's statements. The old
        cell-wide diff spanned cash's OWN machinery, so a cell containing only
        ``import random`` was recorded as having changed ``numpy.random``
        because cash imported numpy while handling it. Those incidental entries
        used to be load-bearing — they were the only thing giving a first
        drawing cell something to rewind to — until the per-cell snapshot recorded each
        cell's own start position instead."""
        try:
            state = self._statement_processor.tracking_state
            digest = hashlib.sha256(raw_cell.encode("utf-8")).hexdigest()
            state.executed_cell_source_hashes.add(digest)
            changed, pre, post = self._statement_processor.cell_rng_observation()
            if changed and post is not None:
                state.rng_post_states[digest] = post
                state.observed_rng_cells[digest] = changed
                if pre is not None:
                    # Where this cell's randomness STARTED, plus the seeds in
                    # force for it. Re-executing a draw reproduces its value only
                    # by rewinding to this. The fingerprint is what
                    # makes it safe to prefer over the upstream-anchor scan: it
                    # expires the position when the seed behind it changes,
                    # using the same lineage check that invalidates any other
                    # value.
                    drawing = set(get_drawing_rng_modules(raw_cell)) | changed
                    state.rng_pre_states[digest] = (
                        pre,
                        rng_lineage_fingerprint(state.variable_lineage, drawing),
                    )
        except (AttributeError, TypeError):  # pragma: no cover - defensive
            pass

    # ------------------------------------------------------------------
    # Phase 2: badge & timing init
    # ------------------------------------------------------------------

    def _init_cell_timing_and_badge(self, badge_display_id: str) -> "TimingBreakdown":
        """Set up timing tracking and render the initial 'RUNNING' badge."""
        timing_breakdown: "TimingBreakdown" = {}
        t_badge_init = time.time()
        self._badges.start_cell(badge_display_id)
        timing_breakdown["badge_init"] = time.time() - t_badge_init
        return timing_breakdown

    # ------------------------------------------------------------------
    # Phase 3: module change detection
    # ------------------------------------------------------------------

    def _detect_module_changes(self, raw_cell: str) -> list[ProcessResult]:
        """Check for changed tracked modules, reload them, and invalidate lineage.

        Returns a list of notification metrics (MODULE_RELOADED entries) for
        the badge display.
        """
        ft = self._statement_processor.function_tracker
        notifications: list[ProcessResult] = []

        # Auto-track local module imports found in this cell
        try:
            newly_tracked = ft.auto_track_local_imports(raw_cell)
            if newly_tracked:
                logger.debug("[AUTO_TRACK] Auto-tracking local modules: %s", ", ".join(sorted(newly_tracked)))
        except (ImportError, AttributeError, OSError, TypeError) as exc:
            logger.debug("Failed to auto-track local imports: %s", exc)

        # Check tracked modules for source file changes and reload if needed
        try:
            changed_modules, per_module_changed_symbols = ft.check_and_reload_changed_modules(
                self.shell.user_ns,
            )
            if changed_modules:
                self._module_invalidator.invalidate(
                    changed_modules,
                    self._statement_processor,
                    per_module_changed_symbols,
                )

                mod_names = ", ".join(sorted(changed_modules.keys()))
                notification: ProcessResult = {
                    "status": CacheStatus.MODULE_RELOADED,
                    # No glyph: this text reaches `%cash_badge print`, whose readers are
                    # often cp1252 consoles. The label says it already.
                    "code": f"Module{'s' if len(changed_modules) > 1 else ''} reloaded: {mod_names}",
                    "is_upstream": True,
                    "total_time": 0.0,
                    "execution_time": 0.0,
                    "saved_time": 0.0,
                    "error": None,
                    "restored_vars": [],
                    "uncacheable_reasons": [],
                    "outputs": [],
                    "changed_modules": dict(changed_modules.items()),
                }
                notifications.append(notification)
                for mod, path in changed_modules.items():
                    syms = per_module_changed_symbols.get(mod)
                    sym_info = f"changed symbols: {syms}" if syms is not None else "full invalidation"
                    logger.debug("[AUTO_TRACK] Reloaded changed module '%s' (%s) (%s)", mod, path, sym_info)
        except (ImportError, AttributeError, OSError, TypeError, ValueError) as exc:
            logger.debug("Failed to check/reload changed modules: %s", exc)

        return notifications

    # ------------------------------------------------------------------
    # Phase 4: upstream resolution
    # ------------------------------------------------------------------

    def _check_and_reexecute_upstream_cells(
        self,
        cell_code: str,
        required_inputs: set,
        progress_callback: Callable[..., None] | None = None,
        *,
        ttl: int | None = None,
        cell_id: str | None = None,
    ) -> tuple[list[ProcessResult], float, float]:
        """Delegate to ``UpstreamChecker``.

        Returns a list of metrics for any executed or restored upstream
        statements, plus the total restore and execution times.
        """
        return self._upstream_checker.check_and_reexecute(
            cell_code,
            required_inputs,
            self._statement_processor.process_statement,
            ttl,
            cell_id=cell_id,
            progress_callback=progress_callback,
            control_structure_callback=self._control_structure_processor.process,
        )

    def _record_consumable_bases(self, inputs: set[str]) -> None:
        """Record the cell-entry drain position of every consumable input.

        Only consumable, unrestorable objects (generator / queue / file handle)
        get an entry; everything else is left out so the dict stays small and
        the simulator's lookup is a plain miss. Stale names are dropped so a
        rebound variable cannot be compared against an unrelated predecessor's
        token.
        """
        state = self._statement_processor.tracking_state
        bases = state.consumable_bases
        user_ns = self.shell.user_ns
        for var_name in inputs:
            value = user_ns.get(var_name)
            if value is None:
                bases.pop(var_name, None)
                continue
            try:
                if not is_consumable_unrestorable(value):
                    bases.pop(var_name, None)
                    continue
                token = consumable_state(value)
            except (TypeError, ValueError, AttributeError, RecursionError):
                bases.pop(var_name, None)
                continue
            if token is None:
                bases.pop(var_name, None)
            else:
                bases[var_name] = token

    def _ensure_state_for_inputs(
        self,
        cell_code: str,
        progress_callback: Callable[..., None] | None = None,
        *,
        ttl: int | None = None,
        cell_id: str | None = None,
    ) -> tuple[list[ProcessResult], float, float]:
        """Ensure all required inputs are available in ``user_ns``.

        First attempts a fast-path restore via :class:`Restorer`; then
        falls through to upstream re-execution via
        :meth:`_check_and_reexecute_upstream_cells`.
        """
        # Reconstructing an upstream PLOT cell (to rebuild a fig/ax a downstream
        # cell needs) opens a matplotlib figure. The inline backend's
        # post-execute hook would then flush that figure into THIS (downstream)
        # cell's output — a stray plot. Close any figure reconstruction opens so
        # the downstream cell only shows its own output (a normally-run cell
        # closes its figures on flush anyway).
        figs_before = _pyplot_open_fignums()
        try:
            inputs, outputs = CodeAnalyzer.analyze_code_block(cell_code)
            logger.debug("[ENSURE_STATE_DEBUG] Cell %.50r: inputs %s, outputs %s", cell_code, inputs, outputs)

            total_restore_time = 0.0
            upstream_metrics: list[ProcessResult] = []

            for var_name in inputs:
                if var_name not in self.shell.user_ns:
                    start_restore = time.time()
                    try:
                        metrics = self._restorer.restore_variable(var_name)
                        total_restore_time += time.time() - start_restore
                        if metrics:
                            upstream_metrics.extend(metrics)
                    except NameError:
                        # Could not find a source — proceed; upstream re-execution may provide it.
                        logger.debug(
                            "[STATE] Could not restore '%s' from cache. Hoping for upstream re-execution.", var_name
                        )

            reexec_metrics, upstream_restore_time, total_execution_time = self._check_and_reexecute_upstream_cells(
                cell_code,
                inputs,
                progress_callback=progress_callback,
                ttl=ttl,
                cell_id=cell_id,
            )
            total_restore_time += upstream_restore_time
            upstream_metrics.extend(reexec_metrics)

            # Snapshot how far each consumable input has been drained, now that
            # upstream resolution has settled the namespace and before the cell
            # body draws from it. This is the cell-ENTRY baseline the simulator
            # compares against on the next run of this cell: equal means the
            # producer handed us the same state as last time (run_all -> no-op),
            # different means we are looking at our own previous run's leftovers
            # (isolated re-run -> re-execute the producer). Must run AFTER
            # re-execution, or an isolated re-run would record the drained state
            # and destroy the signal for the run after it.
            self._record_consumable_bases(inputs)

        except (RuntimeError, SyntaxError, AmbiguousCellError, ForwardReferenceError):
            raise
        except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.debug("[STATE] Error in state restoration logic: %s", e)
            raise
        finally:
            _close_pyplot_figures(_pyplot_open_fignums() - figs_before)

        return upstream_metrics, total_restore_time, total_execution_time

    def _resolve_upstream_state(
        self,
        raw_cell: str,
        pre_upstream_metrics: list[ProcessResult],
        badge_display_id: str,
        timing_breakdown: "TimingBreakdown",
        args: tuple,
        kwargs: dict,
        original_run_cell: Callable[..., Any] | None,
        *,
        ttl: int | None = None,
        cell_id: str | None = None,
    ) -> tuple[list[ProcessResult], float, float] | EarlyReturn:
        """Run upstream dependency checking and state restoration.

        On error: if *original_run_cell* is provided (hook path), fall back
        through IPython so the user sees the error in the cell.  When None
        (the async hook), re-raise so the caller sees a normal Python
        exception.
        """
        t_ensure = time.time()

        def _upstream_progress_cb(
            upstream_metrics_so_far: list,
            current_stmt_code: str,
            current_step: int | None = None,
            total_steps: int | None = None,
        ) -> None:
            combined = pre_upstream_metrics + upstream_metrics_so_far
            upstream_label = f"↑ {current_stmt_code}" if current_stmt_code else current_stmt_code
            self._badges.maybe_progress(
                combined,
                display_id=badge_display_id,
                step=current_step if current_step is not None else len(combined),
                total=total_steps or 0,
                code=upstream_label,
            )

        caught: Exception | None = None
        try:
            upstream_metrics, total_restore_time, total_execution_time = self._ensure_state_for_inputs(
                raw_cell,
                progress_callback=_upstream_progress_cb,
                ttl=ttl,
                cell_id=cell_id,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001 - broad fallback for upstream simulation failures
            # Do NOT dispatch original_run_cell (or render the badge) from inside
            # this suite: it is a LIVE except block, so sys.exc_info() is set to
            # this internal exception.  Any exception IPython raises while
            # surfacing the user's error would then be implicitly chained onto it
            # via __context__, leaking cash's own frames plus a spurious "During
            # handling of the above exception, another exception occurred" banner
            # into the user's traceback.  Capture here and dispatch
            # AFTER the block exits, when sys.exc_info() is clear.
            caught = e

        if caught is not None:
            return self._handle_upstream_resolution_failure(
                caught,
                raw_cell,
                badge_display_id,
                args,
                kwargs,
                original_run_cell,
            )

        timing_breakdown["upstream_check_raw"] = time.time() - t_ensure
        timing_breakdown["total_restore_time"] = total_restore_time
        timing_breakdown["total_execution_time"] = total_execution_time
        timing_breakdown["upstream_check"] = (time.time() - t_ensure) - total_restore_time - total_execution_time

        if logger.isEnabledFor(logging.DEBUG):
            ensure = time.time() - t_ensure
            logger.debug(
                "[TIMING_PROXY] Ensure state: %.2fms (restore %.2fms, execution %.2fms, overhead %.2fms)",
                ensure * 1000,
                total_restore_time * 1000,
                total_execution_time * 1000,
                (ensure - total_restore_time - total_execution_time) * 1000,
            )

        return upstream_metrics, total_restore_time, total_execution_time

    def _handle_upstream_resolution_failure(
        self,
        caught: Exception,
        raw_cell: str,
        badge_display_id: str,
        args: tuple,
        kwargs: dict,
        original_run_cell: Callable[..., Any] | None,
    ) -> EarlyReturn:
        """Surface an upstream-resolution failure to the user with a clean traceback.

        Deliberately called AFTER :meth:`_resolve_upstream_state`'s try/except
        has fully exited, so ``sys.exc_info()`` is already clear.  That timing is
        load-bearing: dispatching ``original_run_cell`` from *inside*
        the live ``except`` block made Python implicitly chain the fresh (or
        IPython-raised) exception onto cash's internal one via ``__context__``,
        and IPython's ultratb then rendered cash's own frames
        (``analysis.py``/``virtual_lineage.py``/``cell_executor.py``/
        ``checker.py``) plus a spurious "During handling of the above exception,
        another exception occurred" banner — making a plain user typo look like
        cash crashed.  Running the dispatch here keeps the traceback as short and
        clean as cash-off.

        Behaviour is otherwise identical to the old in-``except`` dispatch:

        - ``original_run_cell is None`` (the async hook): a SyntaxError
          becomes a quiet "log + return"; anything else re-raises so the
          caller sees the real error.
        - SyntaxError (hook path): re-run the raw cell through IPython so the
          user sees the parse error attributed to their cell.
        - RuntimeError / AmbiguousCellError / UpstreamStateError /
          ForwardReferenceError: synthesise a
          fresh raise inside the user's cell (the "fail the cell
          loudly" path) so IPython attributes the traceback to the cell.
        - anything else: log and fall back to normal execution.
        """
        if original_run_cell is None:
            # No run_cell to fall back on: a SyntaxError from upstream sim is
            # surfaced as a normal "log + return" (matches the executor's own
            # AST-parse SyntaxError path).  Any other exception propagates so
            # the caller sees the real error.
            if isinstance(caught, SyntaxError):
                self._badges.close(badge_display_id)
                return EarlyReturn(None)
            raise caught
        if isinstance(caught, SyntaxError):
            self._badges.close(badge_display_id)
            return EarlyReturn(original_run_cell(raw_cell, *args, **kwargs))
        if isinstance(caught, (RuntimeError, AmbiguousCellError, UpstreamStateError, ForwardReferenceError)):
            # Re-raise inside the user's cell so IPython renders the traceback
            # as if the cell itself raised.  Import the exception class
            # explicitly because the user's namespace may not have it.  The
            # trailing ``from None`` suppresses any ambient context so the
            # synthesised raise carries only the message, never a chain back
            # into cash's internals.
            cls = type(caught)
            # repr(), not a triple-quoted literal. Python quotes names in its
            # own messages -- "No such file or directory: 'side.txt'" -- and a
            # message ending in a quote closed the literal early, so the user's
            # cell died with `SyntaxError: unterminated string literal` from
            # code cash wrote, with the real failure nowhere in sight. repr()
            # also handles the newlines this message routinely carries.
            error_code = f"from {cls.__module__} import {cls.__name__}; raise {cls.__name__}({str(caught)!r}) from None"
            self._badges.close(badge_display_id)
            return EarlyReturn(original_run_cell(error_code, *args, **kwargs))
        # An internal failure, and the cell is about to run UNCACHED. This used
        # to be logger.error only -- invisible in a notebook, where nobody is
        # watching the kernel log -- so the sole trace was an empty badge, which
        # itself then read as "EXECUTED 0.00s". A user hitting this saw a cell
        # produce nothing and had no way to learn why. Warn where they are.
        # With the traceback: this message asks the user to report the failure,
        # and "ModuleNotFoundError: No module named 'openpyxl'" on its own says
        # nothing about where in cash it came from.
        logger.error("Cash auto-caching failed: %s. Falling back to normal execution.", caught, exc_info=caught)
        try:
            warn_diagnostic(
                CashCacheIneffectiveWarning,
                "NOTEBOOK-BAILOUT",
                what=(
                    f"cash hit an internal error and stepped aside: "
                    f"{type(caught).__name__}: {caught}. This cell ran normally "
                    f"but was NOT cached, and neither were its results."
                ),
                fix=(
                    "Nothing in your code caused this and re-running is safe -- "
                    "the cell's result is correct, just uncached. Please report "
                    "it with the message above."
                ),
            )
        except Exception:  # noqa: BLE001 - a diagnostic must never break a cell
            pass
        self._badges.close(badge_display_id, status="BYPASSED")
        return EarlyReturn(original_run_cell(raw_cell, *args, **kwargs))

    # ------------------------------------------------------------------
    # Phase 6: pre-execution notifications
    # ------------------------------------------------------------------

    def _make_function_change_metrics(self) -> list[ProcessResult]:
        """Return notification metrics for any user-defined functions that changed source."""
        ft = self._statement_processor.function_tracker
        try:
            changed_funcs = ft.detect_changed_functions(self.shell.user_ns)
            if not changed_funcs:
                return []
            func_names = ", ".join(sorted(changed_funcs))
            logger.debug("[FUNCTION_CHANGE] Detected changed functions: %s", func_names)
            return [
                {
                    "status": CacheStatus.FUNCTION_CHANGED,
                    "code": f"Function{'s' if len(changed_funcs) > 1 else ''} changed: {func_names}",
                    "is_upstream": True,
                    "execution_time": 0.0,
                    "total_time": 0.0,
                    "saved_time": 0.0,
                    "error": None,
                    "restored_vars": [],
                    "uncacheable_reasons": [],
                    "outputs": [],
                    "changed_functions": sorted(changed_funcs),
                }
            ]
        except (AttributeError, TypeError, OSError) as exc:
            logger.debug("Failed to check function changes: %s", exc)
            return []

    def _make_opaque_warning_metrics(self, raw_cell: str) -> list[ProcessResult]:
        """Return WARNING metrics for opaque call patterns detected in raw_cell."""
        ft = self._statement_processor.function_tracker
        try:
            opaque_warnings = ft.detect_opaque_call_patterns(raw_cell, self.shell.user_ns)
            if not opaque_warnings:
                return []
            for w in opaque_warnings:
                logger.debug("[OPAQUE_CALL] %s", w)
            return [
                {
                    "status": CacheStatus.WARNING,
                    "code": f"⚠️ {msg}",
                    "is_upstream": True,
                    "execution_time": 0.0,
                    "total_time": 0.0,
                    "saved_time": 0.0,
                    "error": None,
                    "restored_vars": [],
                    "uncacheable_reasons": [],
                    "outputs": [],
                }
                for msg in opaque_warnings
            ]
        except (AttributeError, TypeError, SyntaxError, ValueError) as exc:
            logger.debug("Failed to detect opaque call patterns: %s", exc)
            return []

    def _make_staleness_metrics(self) -> list[ProcessResult]:
        """Return the WARNING notification for a proven-stale notebook file.

        Only proof is reported. A once-per-session "cash cannot see unsaved
        edits here" row used to join it whenever cash read the saved file; it
        fired on every fresh kernel of every headless run, where nothing can be
        unsaved, and no user could act on it. Where edits CAN be
        unsaved -- JupyterLab with the extension, VS Code, Colab -- cash reads
        the live cells. Guarded
        like `_make_function_change_metrics` / `_make_opaque_warning_metrics`
        above. Nothing in `StalenessTracker`'s current implementation raises,
        but this is a diagnostic nicety layered on top of upstream resolution
        (which must already have succeeded to reach this point) -- its failure
        must never be able to take down a user's cell execution over what is,
        at worst, a missed warning. Broader than the siblings' exception tuples
        on purpose: unlike theirs, there is no specific failure mode to name
        here, so the guarantee has to be unconditional.
        """
        try:
            tracker = self._upstream_checker.staleness
            notifications = []
            stale = staleness_notification(tracker)
            if stale is not None:
                notifications.append(stale)
            return notifications
        except Exception as exc:  # noqa: BLE001 - a diagnostic must never break execution
            logger.debug("Failed to check notebook staleness: %s", exc)
            return []

    def _build_pre_execution_notifications(
        self,
        raw_cell: str,
        pre_upstream_metrics: list[ProcessResult],
        upstream_metrics: list[ProcessResult],
    ) -> list[ProcessResult]:
        """Assemble the initial metrics list from module, upstream, and function-change notifications."""
        all_metrics: list[ProcessResult] = []
        if pre_upstream_metrics:
            all_metrics.extend(pre_upstream_metrics)
        if upstream_metrics:
            all_metrics.extend(upstream_metrics)
        all_metrics.extend(self._make_function_change_metrics())
        all_metrics.extend(self._make_opaque_warning_metrics(raw_cell))
        # Deliberately NOT built in `_detect_module_changes` alongside
        # MODULE_RELOADED: that phase runs BEFORE upstream resolution
        # (`_resolve_upstream_state` / `check_and_reexecute`), which is where
        # `staleness.observe()` is called for THIS cell (checker.py's
        # `_find_current_cell_index`). Reading the tracker there would still
        # see the PREVIOUS cell's verdict, so the run that actually proves
        # staleness would show the warning one cell late. This function runs
        # after upstream resolution has completed for the current cell, so the
        # tracker is current.
        all_metrics.extend(self._make_staleness_metrics())
        return all_metrics

    # ------------------------------------------------------------------
    # Phase 7: statement execution
    # ------------------------------------------------------------------

    @staticmethod
    def _flush_rich_outputs(
        rich_outputs: list,
        is_last_statement: bool,
        buffered_result_outputs: list,
    ) -> list:
        """Publish or buffer rich outputs depending on statement position.

        Returns the (possibly updated) buffer — callers should reassign the
        returned value, as the last-statement path replaces the buffer.
        """
        if is_last_statement:
            return rich_outputs
        replay_outputs(rich=rich_outputs)
        return buffered_result_outputs

    def _handle_regular_stmt_metrics(
        self,
        metrics: ProcessResult | None,
        is_last_statement: bool,
        all_metrics: list[ProcessResult],
        buffered_result_outputs: list,
    ) -> list:
        """Consume a single non-control statement's metrics; return updated buffer.

        Shared tail for the sync and async statement paths — the only thing
        that differs upstream is how ``metrics`` was produced (sync
        ``process_statement`` vs awaited ``process_statement_async``).
        """
        if not metrics:
            return buffered_result_outputs

        all_metrics.append(metrics)
        replay_outputs(metrics.get("stdout", ""), metrics.get("stderr", ""))
        if metrics.get("status") == CacheStatus.ERROR and metrics.get("error"):
            raise metrics["error"]

        return self._flush_rich_outputs(
            metrics.get("rich_outputs", []),
            is_last_statement,
            buffered_result_outputs,
        )

    def _collect_ctrl_outputs(
        self,
        ctrl_result: Any,
        is_last_statement: bool,
        all_metrics: list[ProcessResult],
        buffered_result_outputs: list,
    ) -> list:
        """Flush outputs from all metrics in a control structure result."""
        for metrics in ctrl_result.metrics:
            if not metrics:
                continue
            all_metrics.append(metrics)
            if not metrics.get("_output_flushed"):
                replay_outputs(metrics.get("stdout", ""), metrics.get("stderr", ""))
            buffered_result_outputs = self._flush_rich_outputs(
                metrics.get("rich_outputs", []),
                is_last_statement,
                buffered_result_outputs,
            )
        return buffered_result_outputs

    def _finalize_error_badge(self, e: BaseException, cell: _CellRun, node: ast.stmt) -> None:
        """Show a clean error display + render the final DONE badge.

        Does **not** re-raise — that is the pipeline caller's job.
        """
        # A statement that just raised may have an armed progress timer
        # (it hadn't finished, so nothing cancelled it yet) -- stop it before
        # rendering the DONE badge below so a late fire can't overwrite it.
        self._badges.cancel_progress()
        self._badges.show_error(e, cell.raw_cell, node)
        self._badges.finish(
            cell.all_metrics,
            cell.badge_display_id,
            time.time() - cell.hook_start,
            cell.timing_breakdown,
        )

    def _execute_cell_statements(self, cell: _CellRun) -> tuple[list[ProcessResult], list, float]:
        """Run the cell's statements (see :meth:`_cell_steps`).

        Returns ``(all_metrics, buffered_result_outputs, badge_render_time)``
        on success, or raises if a statement raised an error (after first
        rendering the error badge via :meth:`_finalize_error_badge`).
        """
        return _drive(self._cell_steps(cell, awaitable=False), self._run_step)

    async def _execute_cell_statements_async(self, cell: _CellRun) -> tuple[list[ProcessResult], list, float]:
        """:meth:`_execute_cell_statements`, awaiting each statement, so a
        top-level ``await`` runs on IPython's live loop."""
        return await _drive_async(self._cell_steps(cell, awaitable=True), self._run_step_async)

    def _run_step(self, step: _Step) -> Any:
        return self._statement_processor.process_statement(**step.kwargs)

    async def _run_step_async(self, step: _Step) -> Any:
        if step.await_unit is not None:
            return await self._control_structure_processor.process_await_unit(step.await_unit, **step.kwargs)
        return await self._statement_processor.process_statement_async(**step.kwargs)

    def _cell_steps(
        self, cell: _CellRun, *, awaitable: bool
    ) -> Generator[_Step, Any, tuple[list[ProcessResult], list, float]]:
        """Iterate over the cell's statements, executing or caching each one.

        Yields a :class:`_Step` for the work the sync and async paths run
        differently, and is sent back its result: each regular statement, and
        -- when *awaitable* -- a control structure whose body awaits.
        """
        raw_cell, tree, all_metrics = cell.raw_cell, cell.tree, cell.all_metrics
        buffered_result_outputs: list = []
        badge_render_time = 0.0

        upstream_step_count = len(
            [m for m in all_metrics if m.get("is_upstream", False) and m.get("status") is not CacheStatus.SKIPPED]
        )
        total_steps_unified = upstream_step_count + len(tree.body)
        stmt_occurrence_counts: dict[str, int] = {}
        written_later = _written_later_in_cell(tree.body)
        checker = self._upstream_checker
        try:
            jump_runs = _jumpable_runs(tree.body, raw_cell, checker.cell_touches_rng) if checker is not None else {}
        except Exception:  # noqa: BLE001 - no jump is the ordinary run
            jump_runs = {}
        #: Statements a restore of a later version made unnecessary.
        planned: dict[int, ProcessResult] = {}

        for i, node in enumerate(tree.body):
            if i in jump_runs:
                plan = checker.plan_cell_run(tree.body[i : jump_runs[i]], raw_cell, dict(stmt_occurrence_counts))
                planned = {i + k: m for k, m in (plan or {}).items()}
            texts = self._statement_texts(raw_cell, node)
            if texts is None:
                continue
            stmt_code, stmt_display, stmt_exec_source = texts

            occ = stmt_occurrence_counts.get(stmt_code, 0)
            stmt_occurrence_counts[stmt_code] = occ + 1
            if i in planned:
                all_metrics.append(planned.pop(i))
                continue
            annotation = get_statement_annotations(raw_cell, node)
            is_last = i == len(tree.body) - 1
            unified_step = upstream_step_count + i + 1

            t_badge_pre = time.time()
            self._badges.arm_progress(
                all_metrics,
                display_id=cell.badge_display_id,
                step=unified_step,
                total=total_steps_unified,
                code=stmt_code,
            )
            badge_render_time += time.time() - t_badge_pre

            try:
                try:
                    if is_control_structure(node):
                        buffered_result_outputs = yield from self._control_structure_steps(
                            cell,
                            i,
                            stmt_code,
                            is_last=is_last,
                            buffered=buffered_result_outputs,
                            awaitable=awaitable,
                        )
                    else:
                        buffered_result_outputs = yield from self._statement_steps(
                            cell,
                            stmt_code,
                            annotation=annotation,
                            display_code=stmt_display,
                            exec_source=stmt_exec_source,
                            occurrence_index=occ,
                            is_last=is_last,
                            written_later=written_later[i],
                            buffered=buffered_result_outputs,
                        )

                    self._badges.cancel_progress()
                    t_badge = time.time()
                    # `unified_step`, NOT `unified_step + 1`: this fires when a
                    # statement has FINISHED, and the next one has not started.
                    # The number means "the furthest statement cash has reached",
                    # which is what `arm_progress` publishes too.
                    self._badges.maybe_progress(
                        all_metrics,
                        display_id=cell.badge_display_id,
                        step=unified_step,
                        total=total_steps_unified,
                        code=None,
                    )
                    badge_render_time += time.time() - t_badge

                except Exception as e:  # intentionally broad: catches user code exceptions
                    self._finalize_error_badge(e, cell, node)
                    raise
            finally:
                # Cancel on EVERY exit from this statement, not just the two
                # paths above. A BaseException that isn't an Exception --
                # KeyboardInterrupt, or asyncio.CancelledError from an interrupted
                # await -- skips the `except` above entirely. Left armed, that
                # timer fires later, on whatever cell is running by then.
                # Safe to call unconditionally: a no-op once already cancelled.
                self._badges.cancel_progress()

        return (all_metrics, buffered_result_outputs, badge_render_time)

    def _statement_steps(
        self,
        cell: _CellRun,
        stmt_code: str,
        *,
        annotation: Any,
        display_code: str | None,
        exec_source: str | None,
        occurrence_index: int,
        is_last: bool,
        written_later: frozenset[str],
        buffered: list,
    ) -> Generator[_Step, Any, list]:
        """Run one regular top-level statement; returns the buffered result
        outputs with its own added.

        ``display_code`` and ``exec_source`` differ only for a top-level
        ``def``/``class``: the badge withholds the body, and the original text
        is executed only under an ``# @cash:assume-safe`` waiver
        (``_exec_source_for_node``).
        """
        _set_written_later(self, written_later)
        try:
            metrics = yield _Step(
                {
                    "code": stmt_code,
                    "ttl": cell.ttl,
                    "silent": True,
                    "annotation": annotation,
                    "display_code": display_code,
                    "exec_source": exec_source,
                    "occurrence_index": occurrence_index,
                    # IPython echoes only the CELL's last expression;
                    # cash executes each statement as its own unit.
                    "is_last": is_last,
                }
            )
            return self._handle_regular_stmt_metrics(metrics, is_last, cell.all_metrics, buffered)
        finally:
            _set_written_later(self, frozenset())

    def _statement_texts(self, raw_cell: str, node: ast.stmt) -> tuple[str, str | None, str | None] | None:
        """``(code, display_code, exec_source)`` for the top-level *node*, or
        None when it cannot be unparsed.

        ``code`` is the keyed and executed text; the badge shows the user's
        own layout (see `_statement_source`).
        """
        # With an expression's trailing ``;`` (``df.head();`` shows no
        # repr): the suppression rides through the cache key AND the
        # execution path (``CodeRunner`` skips the display), so a cached
        # re-run doesn't emit a phantom repr. See ``statement_code``.
        try:
            stmt_code = statement_code(node, raw_cell)
        except (ValueError, TypeError):
            return None
        stmt_display = _statement_source(raw_cell, node)
        stmt_exec_source = _exec_source_for_node(raw_cell, node, stmt_display)
        return stmt_code, stmt_display, stmt_exec_source

    def _control_structure_steps(
        self,
        cell: _CellRun,
        i: int,
        stmt_code: str,
        *,
        is_last: bool,
        buffered: list,
        awaitable: bool,
    ) -> Generator[_Step, Any, list]:
        """Run the control structure at ``cell.tree.body[i]``; returns the
        buffered result outputs with its own added.

        ``raw_cell`` (not the node's annotation) is what goes down: the
        structure's statements each resolve their OWN directive against the
        original source, and ``ast.unparse`` has already dropped the comments
        by the time they are dispatched. The node's annotation is its
        WHOLE-range merge, which cannot tell a directive on the loop from one
        on a single body statement -- passing it would disable caching for
        every sibling in the body.
        """
        node = cell.tree.body[i]
        raw_cell = cell.raw_cell
        # Logged as ONE statement, as the upstream simulation traces it, for
        # the figure histories a writer records.
        control_log = self._statement_processor.begin_control_log(stmt_code)
        try:
            if awaitable and contains_top_level_await(node):
                # The sync ControlStructureProcessor cannot compile a body that
                # awaits (``'await' outside function``), so the whole structure
                # runs as one awaited unit.
                logger.debug("[CONTROL] Await inside control body, running as awaited single unit")
                ctrl_result = yield _Step(
                    {"ttl": cell.ttl, "silent": True, "raw_cell": raw_cell},
                    await_unit=node,
                )
            else:
                logger.debug("[CONTROL] Detected control structure, delegating to ControlStructureProcessor")
                ctrl_result = self._control_structure_processor.process(
                    node,
                    ttl=cell.ttl,
                    silent=True,
                    raw_cell=raw_cell,
                    prev_node=cell.tree.body[i - 1] if i > 0 else None,
                )
        finally:
            self._statement_processor.end_control_log(control_log)
        buffered = self._collect_ctrl_outputs(ctrl_result, is_last, cell.all_metrics, buffered)
        logger.debug(
            "[CONTROL] Completed: %s iterations, %s cached, %s computed",
            ctrl_result.total_iterations,
            ctrl_result.cached_iterations,
            ctrl_result.computed_iterations,
        )
        if not ctrl_result.success:
            raise ctrl_result.error or RuntimeError("Unknown error in control structure execution")
        return buffered
