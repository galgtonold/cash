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
- ``CellExecutor`` does not restore variables.  What the cell reads is
  brought up to date by :class:`UpstreamResolution`; the executor never
  reaches into the backend itself.

**Stepping aside**:

When the upstream check fails (a SyntaxError in a cell above,
``RuntimeError`` / :class:`AmbiguousCellError`, an internal error), the
executor returns :class:`RunInstead` rather than running anything itself.
The sync and the async hook each hand its source to their own original
``run_cell`` / ``run_cell_async``, so a cell with a top-level ``await``
falls back exactly as any other cell does, and the kernel reply status
stays "error" where the cell failed.
"""

from __future__ import annotations

import ast
import contextlib
import io
import uuid
from collections.abc import Awaitable, Callable, Generator, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from IPython.core.inputtransformer2 import leading_indent

from ..._clock import perf_counter as _perf_counter
from ...analysis.annotations import get_statement_annotations
from ...analysis.cell_runs import jumpable_runs, written_later_in_cell
from ...analysis.code_analyzer import CodeAnalyzer
from ...diagnostics import warn_diagnostic
from ...exceptions import CashWarning
from ...remote_source import measured_validation as _measured_validation
from ...source_norm import exact_source_digest
from ...tracking.file_dep_snapshot import begin_file_state_epoch, end_file_state_epoch
from ...tracking.randomness import get_drawing_rng_modules, rng_lineage_fingerprint
from .._protocols import ShellProtocol
from ..cache_status import CacheStatus
from ..control_structures import contains_top_level_await, is_control_structure
from ..statement import ProcessResult
from ..statement.capture import replay_outputs
from ..tracking_state import TrackingState
from ._types import PipelineCompleted, PipelineSyntaxError, RunInstead
from .ipython_cell import CellMagic, IPythonCell, ipython_cell
from .notifications import (
    function_change_rows,
    module_load_failed_row,
    module_reloaded_row,
    opaque_call_rows,
    stale_notebook_rows,
)
from .statement_source import statement_texts
from .upstream_phase import UpstreamResolution

if TYPE_CHECKING:
    from ..control_structures import ControlStructureProcessor
    from ..module_invalidator import ModuleInvalidator
    from ..statement import StatementProcessor
    from ..upstream import UpstreamChecker
    from ._types import TimingBreakdown
    from .badges import BadgePresenter

import logging

logger = logging.getLogger(__name__)


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
    #: The cell as IPython's transform writes it, when it holds IPython syntax.
    ipython: IPythonCell | None = None


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


def _runs_no_python(raw_cell: str) -> bool:
    """Whether *raw_cell* is only cash line magics (``%cash_off``) and ``!`` shell
    commands (blank and comment lines aside), so it runs none of the user's
    modules."""
    lines = [line.strip() for line in raw_cell.splitlines()]
    code = [line for line in lines if line and not line.startswith("#")]
    return bool(code) and all(line.startswith("!") or _is_cash_line_magic(line) for line in code)


def _is_cash_line_magic(line: str) -> bool:
    return line.startswith("%") and line[1:].startswith("cash")


#: The name a ``%%time``/``%%prun`` body runs under, for the magic to call.
_BODY_HOOK = "__cash_cell_body__"


class _BodyNotRun(Exception):
    """A cell magic raised before it ran the body it was handed (a usage
    error): IPython runs the cell instead and reports it."""


def _set_written_later(executor: Any, names: frozenset[str]) -> None:
    """Tell the statement processor which names the rest of the cell writes."""
    processor = getattr(executor, "_statement_processor", None)
    if processor is not None:
        processor.set_written_later_in_cell(names)


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
        module_invalidator: "ModuleInvalidator",
        control_structure_processor: "ControlStructureProcessor",
    ) -> None:
        self.shell = shell
        self._cash_instance = cash_instance
        self._badges = badges
        self.tracking_state = tracking_state
        self._statement_processor = statement_processor
        self._upstream_checker = upstream_checker
        self._module_invalidator = module_invalidator
        self._control_structure_processor = control_structure_processor
        self._upstream = UpstreamResolution(
            shell,
            badges,
            statement_processor,
            upstream_checker,
            control_structure_processor,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute_cell(
        self,
        raw_cell: str,
        *,
        ttl: int | None = None,
        cell_id: str | None = None,
    ) -> PipelineCompleted | PipelineSyntaxError | RunInstead:
        """Run *raw_cell* through the 7-phase cached-execution pipeline.

        Returns one of:
        - :class:`PipelineCompleted` — caller invokes the finaliser
        - :class:`PipelineSyntaxError` — the cell's own AST failed to parse
        - :class:`RunInstead` — the caller runs its source through IPython
        """
        with self._cell_scope():
            cell = self._prepare_cell(raw_cell, ttl, cell_id)
            if not isinstance(cell, _CellRun):
                return cell
            try:
                with self._statements_scope(cell):
                    result = self._execute_cell_statements(cell)
            except _BodyNotRun:
                self._badges.close(cell.badge_display_id)
                return PipelineSyntaxError()
            return self._complete_cell(cell, result)

    async def execute_cell_async(
        self,
        raw_cell: str,
        *,
        ttl: int | None = None,
        cell_id: str | None = None,
    ) -> PipelineCompleted | PipelineSyntaxError | RunInstead:
        """:meth:`execute_cell` for a cell with a top-level ``await``.

        Every phase is the sync pipeline's own; only the statements are run
        through :meth:`_execute_cell_statements_async`, so a top-level
        ``await`` runs on IPython's live loop.
        """
        with self._cell_scope():
            cell = self._prepare_cell(raw_cell, ttl, cell_id)
            if not isinstance(cell, _CellRun):
                return cell
            try:
                with self._statements_scope(cell):
                    result = await self._execute_cell_statements_async(cell)
            except _BodyNotRun:
                self._badges.close(cell.badge_display_id)
                return PipelineSyntaxError()
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
        ttl: int | None,
        cell_id: str | None,
    ) -> _CellRun | PipelineSyntaxError | RunInstead:
        """Phases 2-6: everything before the cell's statements run.

        Phase 1, the cell id and the notebook path, is the caller's: *cell_id*
        is what it resolved, and *ttl* the TTL the cell's entries are stored
        with.
        """
        # IPython runs a cell that starts with a space or a tab with that
        # indentation taken off every line; a parser given the raw text
        # refuses it, and the cell would run outside the pipeline, changing
        # variables nothing tracks.
        raw_cell = "".join(leading_indent(raw_cell.splitlines(keepends=True)))

        # 2. Badge & timing init
        badge_display_id = str(uuid.uuid4())
        timing_breakdown = self._init_cell_timing_and_badge(badge_display_id)
        hook_start = _perf_counter()
        logger.debug("[TIMING_PROXY] Start cached_run_cell")

        # 3. Module change detection (must precede upstream check)
        pre_upstream_metrics = self._detect_module_changes(raw_cell)
        self._raise_failed_reload(raw_cell, badge_display_id, pre_upstream_metrics, hook_start, timing_breakdown)

        # 4. Upstream resolution
        upstream_result = self._upstream.resolve(
            raw_cell,
            pre_upstream_metrics,
            badge_display_id,
            timing_breakdown,
            ttl=ttl,
            cell_id=cell_id,
        )
        if isinstance(upstream_result, RunInstead):
            return upstream_result
        upstream_metrics, _restore_time, _execution_time = upstream_result

        # 5. AST parse (tolerate a top-level ``await``; a bare
        # ast.parse rejects module-level await and would silently skip the cell)
        ipython = None
        try:
            tree = CodeAnalyzer.parse_cell(raw_cell)
        except SyntaxError:
            ipython = ipython_cell(raw_cell, self.shell.transform_cell)
            if ipython is None:
                self._badges.close(badge_display_id)
                return PipelineSyntaxError()
            tree = ipython.tree

        # 6. Pre-execution notifications
        all_metrics = self._build_pre_execution_notifications(
            raw_cell,
            pre_upstream_metrics,
            upstream_metrics,
        )
        return _CellRun(raw_cell, tree, all_metrics, badge_display_id, hook_start, timing_breakdown, ttl, ipython)

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
        t_persist = _perf_counter()
        self._statement_processor.end_cell_persistence()
        cell.timing_breakdown["persist_final"] = _perf_counter() - t_persist

    def _complete_cell(
        self,
        cell: _CellRun,
        result: tuple[list[ProcessResult], list, float],
    ) -> PipelineCompleted:
        """What the finaliser needs, once the cell's statements have run."""
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
            digest = exact_source_digest(raw_cell)
            state.executed_cell_source_hashes.add(digest)
            state.failed_cells.pop(digest, None)
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
        t_badge_init = _perf_counter()
        self._badges.start_cell(badge_display_id)
        timing_breakdown["badge_init"] = _perf_counter() - t_badge_init
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
            newly_tracked = ft.auto_track_local_imports(raw_cell, self.shell.user_ns)
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
                self._replay_module_state(changed_modules)

                notifications.append(module_reloaded_row(changed_modules))
                for mod, path in changed_modules.items():
                    syms = per_module_changed_symbols.get(mod)
                    sym_info = f"changed symbols: {syms}" if syms is not None else "full invalidation"
                    logger.debug("[AUTO_TRACK] Reloaded changed module '%s' (%s) (%s)", mod, path, sym_info)
        except (ImportError, AttributeError, OSError, TypeError, ValueError) as exc:
            logger.debug("Failed to check/reload changed modules: %s", exc)

        return notifications

    def _replay_module_state(self, changed_modules: Mapping[str, Any]) -> None:
        """Run again the statements that set state on each reloaded module.

        A reload runs the module's top level again: ``mylib.K = 7``, a
        ``mylib.REGISTRY["a"] = ...`` or a ``mylib.set_k(7)`` a cell made is
        gone, and the cells that made them are not run again -- the notebook
        computed on the file's defaults, which neither a top-to-bottom run
        nor the kernel before the edit had. They run here, in the order they
        last ran, uncached and with their output dropped; a statement that
        raises is reported (``NOTEBOOK-RELOAD-STATE``) and the rest still run.
        """
        writers = self._statement_processor.tracking_state.module_state_writers
        done: set[str] = set()
        for module in changed_modules:
            for code in list(writers.get(module, ())):
                if code in done:
                    continue
                done.add(code)
                try:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        exec(compile(code, "<cash: module state>", "exec", dont_inherit=True), self.shell.user_ns)  # noqa: S102 - replays the user's own statement
                except Exception as exc:  # noqa: BLE001 - arbitrary user code
                    writers[module].remove(code)
                    first_line = code.strip().splitlines()[0] if code.strip() else code
                    warn_diagnostic(
                        CashWarning,
                        "NOTEBOOK-RELOAD-STATE",
                        f"reloading the edited module {module!r} dropped what `{first_line}` set on it, "
                        f"and running it again raised {type(exc).__name__}: {exc}",
                        "run the cell that sets it again.",
                    )

    def _raise_failed_reload(
        self,
        raw_cell: str,
        badge_display_id: str,
        module_rows: list[ProcessResult],
        hook_start: float,
        timing_breakdown: "TimingBreakdown",
    ) -> None:
        """Raise what reloading an edited module raised, before the cell runs.

        A module whose file no longer loads (a ``SyntaxError``, a top level
        that raises) keeps its old code in the kernel. Running the cell
        would run that old code as if the edit had been picked up, so the
        cell fails instead, with the error a fresh import of the file gives,
        and keeps failing until the module loads. A cell of only cash
        magics and ``!`` shell commands still runs, so cash can be turned off.
        """
        errors = self._statement_processor.function_tracker.reload_errors()
        if not errors or _runs_no_python(raw_cell):
            return
        self._badges.finish(
            [*module_rows, module_load_failed_row(errors)],
            badge_display_id,
            _perf_counter() - hook_start,
            timing_breakdown,
        )
        for mod_name, exc in errors.items():
            self._badges.show_module_load_error(mod_name, exc)
        raise next(iter(errors.values()))

    # ------------------------------------------------------------------
    # Phase 6: pre-execution notifications
    # ------------------------------------------------------------------

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
        function_tracker = self._statement_processor.function_tracker
        all_metrics.extend(function_change_rows(function_tracker, self.shell.user_ns))
        all_metrics.extend(opaque_call_rows(function_tracker, raw_cell, self.shell.user_ns))
        # Deliberately NOT built in `_detect_module_changes` alongside
        # MODULE_RELOADED: that phase runs BEFORE upstream resolution
        # (`_resolve_upstream_state` / `check_and_reexecute`), which is where
        # `staleness.observe()` is called for THIS cell (checker.py's
        # `_find_current_cell_index`). Reading the tracker there would still
        # see the PREVIOUS cell's verdict, so the run that actually proves
        # staleness would show the warning one cell late. This function runs
        # after upstream resolution has completed for the current cell, so the
        # tracker is current.
        all_metrics.extend(stale_notebook_rows(self._upstream_checker))
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
        e._cash_shown = True  # type: ignore[attr-defined] - read by CashMagics._raising_quietly
        self._badges.finish(
            cell.all_metrics,
            cell.badge_display_id,
            _perf_counter() - cell.hook_start,
            cell.timing_breakdown,
        )

    def _execute_cell_statements(self, cell: _CellRun) -> tuple[list[ProcessResult], list, float]:
        """Run the cell's statements (see :meth:`_cell_steps`).

        Returns ``(all_metrics, buffered_result_outputs, badge_render_time)``
        on success, or raises if a statement raised an error (after first
        rendering the error badge via :meth:`_finalize_error_badge`).
        """

        def run() -> tuple[list[ProcessResult], list, float]:
            return _drive(self._cell_steps(cell, awaitable=False), self._run_step)

        magic = cell.ipython.magic if cell.ipython is not None else None
        return run() if magic is None else self._under_cell_magic(magic, run)

    async def _execute_cell_statements_async(self, cell: _CellRun) -> tuple[list[ProcessResult], list, float]:
        """:meth:`_execute_cell_statements`, awaiting each statement, so a
        top-level ``await`` runs on IPython's live loop. A ``%%time`` body
        runs inside the magic, which does not await: synchronously."""
        if cell.ipython is not None and cell.ipython.magic is not None:
            return self._execute_cell_statements(cell)
        return await _drive_async(self._cell_steps(cell, awaitable=True), self._run_step_async)

    def _under_cell_magic(
        self, magic: CellMagic, run: Callable[[], tuple[list[ProcessResult], list, float]]
    ) -> tuple[list[ProcessResult], list, float]:
        """*run* the cell's statements as the body of ``%%time``/``%%prun``.

        The magic is handed a body that calls *run*, so it times or profiles
        the run cash makes, prints what it prints, and the body runs once.
        An error from the body is raised as the cell's, as ``%%time`` raises
        it; under ``--no-raise-error`` too, since cash has shown it as the
        cell's error already.
        """
        outcome: dict[str, Any] = {}

        def body() -> None:
            outcome["started"] = True
            try:
                outcome["result"] = run()
            except BaseException as exc:
                outcome["error"] = exc
                raise

        ns = self.shell.user_ns
        ns[_BODY_HOOK] = body
        try:
            self.shell.run_cell_magic(magic.name, magic.line, f"{_BODY_HOOK}()")
        except BaseException:
            if "started" not in outcome:
                raise _BodyNotRun() from None
            raise
        finally:
            ns.pop(_BODY_HOOK, None)
        if "error" in outcome:
            raise outcome["error"]
        return outcome["result"]

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
        written_later = written_later_in_cell(tree.body)
        jump_runs = self._jump_runs(tree.body, raw_cell)
        #: Statements a restore of a later version made unnecessary.
        planned: dict[int, ProcessResult] = {}

        for i, node in enumerate(tree.body):
            if i in jump_runs:
                plan = self._upstream_checker.plan_cell_run(
                    tree.body[i : jump_runs[i]], raw_cell, dict(stmt_occurrence_counts)
                )
                planned = {i + k: m for k, m in (plan or {}).items()}
            texts = statement_texts(cell.ipython.source if cell.ipython is not None else raw_cell, node)
            if texts is None:
                continue
            stmt_code, stmt_display, stmt_exec_source = texts
            if cell.ipython is not None:
                stmt_display = cell.ipython.display_text(raw_cell, node) or stmt_display

            occ = stmt_occurrence_counts.get(stmt_code, 0)
            stmt_occurrence_counts[stmt_code] = occ + 1
            if i in planned:
                all_metrics.append(planned.pop(i))
                continue
            annotation = get_statement_annotations(raw_cell, node)
            is_last = i == len(tree.body) - 1
            unified_step = upstream_step_count + i + 1

            t_badge_pre = _perf_counter()
            self._badges.arm_progress(
                all_metrics,
                display_id=cell.badge_display_id,
                step=unified_step,
                total=total_steps_unified,
                code=stmt_code,
            )
            badge_render_time += _perf_counter() - t_badge_pre

            if is_control_structure(node):
                steps = self._control_structure_steps(
                    cell,
                    i,
                    stmt_code,
                    is_last=is_last,
                    buffered=buffered_result_outputs,
                    awaitable=awaitable,
                )
            else:
                steps = self._statement_steps(
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
            try:
                buffered_result_outputs, render_time = yield from self._badged_steps(
                    cell, node, steps, step=unified_step, total=total_steps_unified
                )
            except BaseException:
                # What follows never ran; the upstream check must not credit it.
                self.tracking_state.failed_cells[exact_source_digest(raw_cell)] = i
                raise
            badge_render_time += render_time

        return (all_metrics, buffered_result_outputs, badge_render_time)

    def _jump_runs(self, body: list[ast.stmt], raw_cell: str) -> dict[int, int]:
        """The runs of the cell a restore may jump (see ``jumpable_runs``);
        none when there is no upstream checker or the analysis fails."""
        checker = self._upstream_checker
        try:
            return jumpable_runs(body, raw_cell, checker.cell_touches_rng) if checker is not None else {}
        except Exception:  # noqa: BLE001 - no jump is the ordinary run
            return {}

    def _badged_steps(
        self,
        cell: _CellRun,
        node: ast.stmt,
        steps: Generator[_Step, Any, list],
        *,
        step: int,
        total: int,
    ) -> Generator[_Step, Any, tuple[list, float]]:
        """Run one top-level statement's *steps* under the badge.

        Returns the buffered result outputs *steps* returned and the time
        spent drawing the badge's progress. A statement that raises gets the
        error display and the final badge before the error propagates.
        """
        try:
            try:
                buffered = yield from steps

                self._badges.cancel_progress()
                t_badge = _perf_counter()
                # `step`, NOT `step + 1`: this fires when a statement has
                # FINISHED, and the next one has not started. The number means
                # "the furthest statement cash has reached", which is what
                # `arm_progress` publishes too.
                self._badges.maybe_progress(
                    cell.all_metrics,
                    display_id=cell.badge_display_id,
                    step=step,
                    total=total,
                    code=None,
                )
                return buffered, _perf_counter() - t_badge

            except (Exception, KeyboardInterrupt, SystemExit) as e:  # intentionally broad: the cell's own error
                # An interrupt and a ``sys.exit()`` end the cell as an error
                # does in IPython: shown, and the cell's reply says error.
                self._finalize_error_badge(e, cell, node)
                raise
        finally:
            # Cancel on EVERY exit from this statement, not just the two
            # paths above. A BaseException the `except` does not take --
            # asyncio.CancelledError from an interrupted await, a
            # GeneratorExit -- skips it entirely. Left armed, that
            # timer fires later, on whatever cell is running by then.
            # Safe to call unconditionally: a no-op once already cancelled.
            self._badges.cancel_progress()

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
        (``exec_source_for_node``).
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
