"""Cell-level orchestrator for the cash caching pipeline.

Owns the 7-phase pipeline shared by ``%cash_on`` (the ``pre_run_cell`` hook
proxy) and ``%%cash`` (the cell magic):

    1. Cell ID & notebook path resolution
    2. Badge & timing initialisation
    3. Module change detection
    4. Upstream dependency resolution
    5. AST parse
    6. Pre-execution notification assembly
    7. Statement-by-statement execution

Both magic entry points delegate to :meth:`CellExecutor.execute_cell`.  This
is what makes the drift bug structurally impossible to reintroduce: there
is exactly one cell-execution code path.

**Anti-god-class rule (load-bearing):**

- ``CellExecutor`` does not call IPython's ``display()`` or
  ``publish_display_data()`` directly.  Display side effects live in the
  IPython adapter (``CashMagics``).  Today the executor invokes the
  adapter's badge methods through a back-reference (``self._magics``);
  the long-term plan (see ``.github/planning/ARCHITECTURE_DEEPENING.md``
  §6) is to replace that scaffold with a typed ``ProgressEvent`` callback.
- ``CellExecutor`` does not restore variables.  Variable-granular cache
  work is :class:`Restorer`'s job.  The executor calls
  ``restorer.restore_variable(var_name)`` during upstream resolution; it
  never reaches into the backend itself.

**`original_run_cell` parameter**:

The hook supplies its captured ``_original_run_cell`` so error paths
that arise mid-pipeline (SyntaxError from upstream simulation,
``RuntimeError`` / :class:`AmbiguousCellError`, generic exception
fallback) can be surfaced through IPython's normal execution machinery
and the kernel reply status stays as "error".  The ``%%cash`` magic
passes ``None`` so those exceptions propagate naturally to IPython's
magic-error path instead.
"""

from __future__ import annotations

import ast
import hashlib
import sys
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from IPython.display import display, publish_display_data

from ...exceptions import AmbiguousCellError, UpstreamStateError
from ...remote_source import measured_validation as _measured_validation
from .._protocols import ShellProtocol
from ..analysis import CodeAnalyzer
from ..annotations import get_statement_annotations
from ..cache_status import CacheStatus
from ..consumables import consumable_state, is_consumable_unrestorable
from ..control_structures import contains_top_level_await, is_control_structure
from ..randomness import get_drawing_rng_modules, rng_lineage_fingerprint
from ..statement import ProcessResult

if TYPE_CHECKING:
    from ..lineage_store import TrackingState
    from ._types import TimingBreakdown
    from .magics import CashMagics
    from ..module_invalidator import ModuleInvalidator
    from ..restore import Restorer
    from ..statement import StatementProcessor
    from ..upstream import UpstreamChecker
    from ..control_structures import ControlStructureProcessor

import logging

logger = logging.getLogger(__name__)


class _EarlyReturn:
    """Sentinel wrapper for early-exit values that flow back up to the
    hook proxy unchanged.  Carries an IPython ``run_cell`` result."""
    __slots__ = ('value',)

    def __init__(self, value: Any) -> None:
        self.value = value


class _PipelineSyntaxError:
    """Sentinel returned by :meth:`CellExecutor.execute_cell` when the cell's
    own AST fails to parse.  Caller decides how to react."""
    __slots__ = ()


class _PipelineCompleted:
    """Successful pipeline run: carries everything the finaliser needs."""
    __slots__ = (
        'all_metrics', 'buffered_outputs', 'badge_display_id',
        'hook_start', 'timing_breakdown', 'badge_render_time',
    )

    def __init__(
        self,
        all_metrics: list,
        buffered_outputs: list,
        badge_display_id: str,
        hook_start: float,
        timing_breakdown: 'TimingBreakdown',
        badge_render_time: float,
    ) -> None:
        self.all_metrics = all_metrics
        self.buffered_outputs = buffered_outputs
        self.badge_display_id = badge_display_id
        self.hook_start = hook_start
        self.timing_breakdown = timing_breakdown
        self.badge_render_time = badge_render_time


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
        'status': 'WARNING',
        'code': (f"[!] Notebook file is stale -- Save (Ctrl+S) and re-run to be sure. "
                 f"Upstream check used the copy saved at {when}.{where} "
                 f"Other cells may have changed too."),
        'is_upstream': True,
        'total_time': 0.0,
        'execution_time': 0.0,
        'outputs': [],
    }


def unverifiable_notification(tracker) -> dict | None:
    """Badge row said ONCE when cash cannot see unsaved edits at all.

    Distinct from `staleness_notification`, which reports proven staleness. This
    reports a missing capability: cash is reading the saved file, so an edit you
    have not saved is invisible to it and it cannot promise the check was
    current. That is permanent for the session, hence once.

    ASCII only, and short enough to survive print mode's 80-char row cap -- see
    `staleness_notification` for why both matter.
    """
    if not tracker.take_unverifiable_announcement():
        return None
    return {
        'status': 'WARNING',
        'code': ("[!] cash cannot see unsaved edits here -- Save before running "
                 "to be sure. It is reading the saved notebook file."),
        'is_upstream': True,
        'total_time': 0.0,
        'execution_time': 0.0,
        'outputs': [],
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

    Loud on every occurrence rather than once per session, unlike
    `unverifiable_notification`: that one reports a permanent property of the
    environment, where this reports work being lost right now, and a second
    occurrence is a second lost result rather than a repeat of the same news.

    ASCII only and short, for the reasons `staleness_notification` gives -- the
    print renderer caps a row at 80 characters.
    """
    try:
        from cash.backends._base import discarded_writes
        total = len(discarded_writes())
    except Exception:      # noqa: BLE001 - a diagnostic must never break a cell
        return None, seen_before
    if total <= seen_before:
        return None, total

    new = total - seen_before
    plural = "s" if new != 1 else ""
    return {
        'status': 'WARNING',
        'code': (f"[!] {new} cache write{plural} failed -- not cached, will "
                 f"recompute. See %cash_stats."),
        'is_upstream': False,
        'total_time': 0.0,
        'execution_time': 0.0,
        'outputs': [],
    }, total


# Characters ``str.splitlines()`` treats as line breaks that the CPython
# parser does not -- the parser (and therefore ``node.lineno`` /
# ``node.end_lineno``) recognizes only "\r\n", "\r" and "\n". Built with
# ``chr()`` rather than escape literals so the exact code points stay
# unambiguous on the page: vertical tab, form feed, FILE/GROUP/RECORD
# SEPARATOR (U+001C-U+001E), NEL (U+0085), LINE SEPARATOR (U+2028) and
# PARAGRAPH SEPARATOR (U+2029). See ``_splitlines_like_the_parser`` below.
_PARSER_INCOMPATIBLE_LINEBREAKS = "".join(
    chr(c) for c in (0x0B, 0x0C, 0x1C, 0x1D, 0x1E, 0x85, 0x2028, 0x2029)
)
_LINEBREAK_MASK = str.maketrans(
    _PARSER_INCOMPATIBLE_LINEBREAKS, " " * len(_PARSER_INCOMPATIBLE_LINEBREAKS)
)


def _splitlines_like_the_parser(raw_cell: str) -> list[str]:
    """Split ``raw_cell`` into lines using the parser's line-ending rules,
    not ``str.splitlines()``'s.

    ``str.splitlines(keepends=True)`` breaks on more characters than the
    CPython tokenizer does -- see ``_PARSER_INCOMPATIBLE_LINEBREAKS`` above
    -- so indexing its result by ``node.lineno`` desyncs the moment any of
    those appear anywhere earlier in the cell (observed: a vertical tab
    inside one string literal silently truncated that statement's display;
    a form feed at the end of one line corrupted the display of the NEXT
    statement). Fixed here by masking those characters to a plain space --
    one-for-one, so every position keeps its original index -- before
    calling ``str.splitlines()``, then slicing the boundaries it finds back
    out of the ORIGINAL (unmasked) ``raw_cell``. The returned lines contain
    the real original characters verbatim; only the *decision of where a
    line ends* used the masked copy.

    Both ``str.translate`` and ``str.splitlines`` are single C-level passes
    over the whole string, so this stays cheap enough for the fast path
    below to keep its measured win over always calling
    ``ast.get_source_segment`` (see that docstring for the numbers).
    """
    masked = raw_cell.translate(_LINEBREAK_MASK)
    lines = []
    pos = 0
    for masked_line in masked.splitlines(keepends=True):
        length = len(masked_line)
        lines.append(raw_cell[pos:pos + length])
        pos += length
    return lines


def _statement_source(raw_cell: str, node: ast.stmt) -> str | None:
    """The statement's ORIGINAL text.

    Originally written for display only, and still what the badge shows --
    but its result is also reused, unchanged, as what a cache-miss statement
    COMPILES from (``exec_source`` in ``processor.py``'s
    ``_execute_statement``), instead of the ``ast.unparse`` form, whenever it
    is not ``None``. ``ast.unparse`` normalizes a statement onto one logical
    line and strips comments -- both fine for the CACHE KEY, which stays the
    unparsed form always (never this), but wrong for compiling a function
    DEFINED in the cell: a per-line ``# @cash:assume-safe`` in its body would
    be invisible to ``inspect.getsource`` (and so to the purity analyzer) if
    compiled from text with no comments in it. See ``_execute_statement`` for
    the cache-key/exec-source boundary in full, and ``_exec_source_for_node``
    below for why a top-level ``def``/``class`` -- excluded here -- still
    gets a source to execute despite that.

    ``get_source_segment`` returns a nested statement with its first line flush
    and every continuation line at its ABSOLUTE file indentation, which reads as
    ragged in a row. ``textwrap.dedent`` cannot fix that -- the first line
    shares no common prefix -- so continuation lines are dedented by the node's
    own ``col_offset``. For a top-level statement that offset is 0 and this is a
    no-op.

    A top-level ``def``/``class`` is deliberately excluded, returning ``None``
    same as an unrecoverable segment. Executing one only BINDS the name -- the
    body never runs -- so the body is not "the code that ran" the way it is
    for every other captured statement, and showing it in full would make the
    badge very tall in any notebook that defines functions. The caller's
    existing fallback (the unparsed form, clipped to one line with a
    "... +N lines" hint) is the right treatment for these, not a compromise.
    This exclusion is a DISPLAY decision only -- ``_exec_source_for_node``
    recovers a def/class's source separately, for execution, without
    widening what this function hands the badge.

    ``ast.Match`` is deliberately NOT in the exclusion below, even though a
    ``match`` statement is just as multi-line as a ``def``/``class``. The
    line is drawn on BINDING vs. EXECUTING, not on "is it multi-line": a
    ``match`` genuinely executes its matched branch (unlike a def/class body,
    which only runs when called), and since ``match`` is not one of
    ``is_control_structure()``'s node types (For/While/If/With/Try), it has
    no per-branch rows of its own -- the runtime caches and executes it as
    ONE unit, so its full source IS "the code that ran", same as any other
    captured statement. Do not "fix" the def/class-vs-match inconsistency by
    adding ``Match`` here -- that would re-collapse a match statement's body
    to a first-line summary for something that actually ran in full.

    Returns ``None`` when the segment cannot be recovered, or is withheld as
    above; the caller falls back to the unparsed form. Never raises: a badge
    must not be able to break a cell.

    **Fast path for a single-line statement** (``node.end_lineno ==
    node.lineno`` -- the overwhelmingly common case). ``get_source_segment``
    re-splits the WHOLE cell into lines on every call, and on 3.10-3.12 that
    split (``ast._splitlines_no_ff``) is a pure-Python, character-by-character
    loop with no early exit -- it scans to the end of the cell even for a
    one-line statement on line 1. 3.13+ replaced it with a regex-based split
    that accepts ``maxlines`` and stops at ``end_lineno``, which is most of
    why this was never noticed there. Measured (best-of-N) calling this
    function once per node over a 200-statement ~10 KB cell: on 3.11.15,
    113ms with the old always-``get_source_segment`` code vs 2ms with this
    fast path; on 3.13.12, 14ms vs 1ms. Calling this function once per
    top-level statement (see the two call sites in this module) made the
    difference O(statements x cell length) instead of O(cell length).

    The replacement does exactly what ``get_source_segment`` does for a
    single-line node -- split the cell into lines (the C-level
    ``str.splitlines``, not the manual loop above -- see
    ``_splitlines_like_the_parser``) and slice the one at ``node.lineno``
    -- INCLUDING its byte-offset semantics: ``col_offset``/``end_col_offset``
    are UTF-8 byte offsets, not character indices, so the line is encoded,
    sliced, and decoded, same as upstream. Any node shape this fast path
    can't handle (missing/`None` location attributes) falls through to the
    original call below, so behaviour for every other case -- multi-line
    segments, the dedent, the ``None`` fallback -- is untouched.

    **The splitter must use the parser's line-ending rules, not
    ``str.splitlines()``'s own.** ``str.splitlines()`` treats several
    characters as line breaks that the CPython tokenizer does not (see
    ``_PARSER_INCOMPATIBLE_LINEBREAKS``), so a naive
    ``raw_cell.splitlines(keepends=True)`` desyncs its line index from
    ``node.lineno`` the moment any of those appear anywhere earlier in the
    cell -- silently, with no exception to trigger the ``None`` fallback.
    ``_splitlines_like_the_parser`` fixes this while keeping the same
    single-pass-over-the-whole-cell shape (so the perf win above holds);
    see its own docstring for how.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return None
    try:
        end_lineno = node.end_lineno
        if (
            end_lineno is not None
            and end_lineno == node.lineno
            and node.end_col_offset is not None
        ):
            lines = _splitlines_like_the_parser(raw_cell)
            segment = (
                lines[node.lineno - 1]
                .encode()[node.col_offset:node.end_col_offset]
                .decode()
            )
        else:
            segment = ast.get_source_segment(raw_cell, node)
    except Exception:  # noqa: BLE001 - display only, never break the cell
        return None
    if not segment:
        return None

    indent = getattr(node, "col_offset", 0) or 0
    if indent <= 0:
        return segment

    head, *rest = segment.split("\n")
    return "\n".join(
        [head] + [line[indent:] if line[:indent].isspace() else line
                  for line in rest]
    )


def _exec_source_for_node(
    raw_cell: str, node: ast.stmt, stmt_display: str | None,
) -> str | None:
    """The text to EXECUTE for *node*, diverging from ``stmt_display`` (the
    text to DISPLAY) only for a top-level ``def``/``class`` that carries a
    ``# @cash:`` directive somewhere in its body.

    ``_statement_source`` withholds a ``def``/``class`` body on purpose --
    showing one in full would make the badge very tall (see its docstring),
    so ``stmt_display`` is ``None`` there and the badge falls back to a
    "+N lines" summary (``_row_code_html``). That withholding is a DISPLAY
    decision; it must not also decide what gets compiled. ``ast.unparse``
    strips comments, so a function DEFINED in a cell used to be compiled with
    no comments in it, and a per-line ``# @cash:assume-safe`` inside its body
    was invisible to ``inspect.getsource`` (and so to the purity analyzer) --
    exactly the def/class case ``stmt_display`` withholds. Recovering it here
    for EXECUTION closes that gap without widening what the badge shows.

    Reuses ``stmt_display`` whenever it is not ``None`` -- one extraction per
    statement, same as before this function existed. Only a top-level
    ``def``/``class`` (rare relative to ordinary statements, and always
    multi-line, so never the single-line case ``_statement_source``'s fast
    path exists for) pays a second, direct ``ast.get_source_segment`` call.
    No dedent is needed for it: every node this is called on is top-level
    (``col_offset == 0``), same precondition ``_statement_source``'s own
    dedent step relies on.

    **Decorators are prepended manually.** ``ast.get_source_segment`` anchors
    to ``node.lineno``, which -- since Python 3.8 -- is the ``def``/``class``
    KEYWORD line, not the decorator, even though ``@c.cache`` is
    unambiguously part of "the statement that ran". Measured: without this,
    the segment recovered for ``@c.cache\\ndef audited(n): ...`` was just
    ``def audited(n): ...`` with NO decorator -- so the executed function was
    never actually wrapped in ``@cash.cache`` at all, silently disabling
    caching (and purity checking) for every decorated function defined in a
    cell. ``node.decorator_list`` gives the exact AST-derived boundary: a
    decorator is always alone on its own line(s) by Python's grammar, so the
    lines from the first decorator's own start through the line before
    ``node.lineno`` are exactly the decorator block, verbatim, regardless of
    multiple decorators, multi-line decorator arguments, or blank lines
    between them. Uses ``_splitlines_like_the_parser`` (not
    ``str.splitlines()``) for the same reason every other line-indexed read
    in this module does -- see that function's docstring.

    **Gated on ``"@cash:" in body`` -- not unconditional, and this is
    load-bearing, not an optimisation.** A ``def``/``class`` this recovers
    can be RE-COMPILED a second time later, by an entirely different path:
    the upstream checker/restorer re-executes an earlier statement (e.g. to
    replay a self-modifying global back to its pre-cell state -- see
    ``test_a_same_session_rerun_neither_freezes_nor_accumulates``), and that
    path has never threaded ``display_code``/``exec_source`` (same as a
    control body or a loop-split iteration -- it always compiles the
    unparsed form). Before this function existed, EVERY path compiled a
    function from the same canonical ``ast.unparse`` text, so a function's
    identity hash (``FunctionTracker.get_function_source_hash``, which feeds
    the CAS-243 call-cache key -- "editing the callee re-keys the call")
    was stable regardless of which path (re)created it. Recovering the
    original text unconditionally broke that: the FIRST definition (via the
    normal split loop) got the original text, but a LATER same-session
    redefinition via the upstream/restorer path still fell back to the
    unparsed form -- two textually different, behaviourally identical
    representations of the same, unedited function. ``source_identity_digest``
    normalises comments away but NOT other harmless textual variance (a raw
    string literal vs. its ``ast.unparse``-escaped equivalent, measured with
    a Windows path baked into the source hashed differently either way), so
    the two representations hashed differently, the call-cache key for a
    call to that function moved, and its body executed an extra time on the
    very next same-session re-run (measured: ``compute_f``'s tick file
    incremented once where 0 was expected). Gating on the ``@cash:`` marker
    -- the same substring check ``_drop_audited`` itself uses as a fast path
    -- means a function with no directive in it (the overwhelming majority)
    is completely unaffected by this function, on EITHER path, exactly as
    before it existed: no behaviour change, no risk of the above. A function
    that DOES carry a directive can still hit the same cross-path
    inconsistency if it is later redefined via the upstream/restorer path in
    the same session -- that residual case is not fixed here (fixing it
    would mean threading original-source recovery into upstream
    restoration too, well beyond what this function is scoped to do) --
    but it no longer taxes every def/class in every notebook to leave it
    open for the rare annotated one.

    **A trailing comment on the node's own LAST line is recovered too.**
    ``ast.get_source_segment`` trims its last line at ``end_col_offset``, so
    ``return x  # @cash:assume-safe`` as literally the final line of a
    function loses the comment -- a directive on any INTERIOR line survives
    untouched (only the first/last lines of a multi-line segment are
    column-trimmed), so this only bites the specific case of the waiver
    sitting on the function's very last physical line. Recovered the same
    way ``_expr_has_trailing_semicolon`` reads past a node's end elsewhere in
    this module: byte-offset slice the end line past ``end_col_offset``, and
    append it ONLY when what remains, stripped, is empty or starts with
    ``#`` -- never anything else, which would mean a semicolon-separated
    SIBLING statement on the same physical line (``def f(): return 1; y =
    2``), whose text belongs to a different node entirely and must not be
    swallowed here.

    Returns ``None`` -- falling back to the unparsed ``code`` at the
    executor, unchanged from before this function existed -- for every other
    reason ``stmt_display`` came back empty (a control body, a loop-split
    iteration, a rewritten statement, or a genuinely unrecoverable segment),
    and for a def/class with no ``@cash:`` directive in it. Never raises:
    this must never be able to break a cell.
    """
    if stmt_display is not None:
        return stmt_display
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return None
    try:
        body = ast.get_source_segment(raw_cell, node)
        if not body:
            return None
        lines = _splitlines_like_the_parser(raw_cell)
        decorators = node.decorator_list
        if decorators:
            prefix = "".join(lines[decorators[0].lineno - 1: node.lineno - 1])
            body = prefix + body
        end_lineno = getattr(node, "end_lineno", None)
        end_col = getattr(node, "end_col_offset", None)
        if (
            end_lineno is not None and end_col is not None
            and 0 < end_lineno <= len(lines)
        ):
            rest = lines[end_lineno - 1].encode()[end_col:].decode()
            trailing = rest.split("\n", 1)[0].rstrip("\r")
            if trailing.strip() == "" or trailing.lstrip().startswith("#"):
                body = body + trailing
        return body if "@cash:" in body else None
    except Exception:  # noqa: BLE001 - execution must never break over this
        return None


class CellExecutor:
    """Run a single notebook cell through the cached-execution pipeline.

    Single public entry: :meth:`execute_cell`.  Both ``%cash_on`` and
    ``%%cash`` route through it — there is no separate code path.
    """

    def __init__(
        self,
        shell: ShellProtocol,
        cash_instance: Any,
        magics: 'CashMagics',
        tracking_state: 'TrackingState',
        statement_processor: 'StatementProcessor',
        upstream_checker: 'UpstreamChecker',
        restorer: 'Restorer',
        module_invalidator: 'ModuleInvalidator',
        control_structure_processor: 'ControlStructureProcessor',
        debug: bool = False,
    ) -> None:
        self.shell = shell
        self._cash_instance = cash_instance
        self._magics = magics  # back-ref for badge rendering — scaffold for typed ProgressEvent callback
        self._tracking_state = tracking_state
        self._statement_processor = statement_processor
        self._upstream_checker = upstream_checker
        self._restorer = restorer
        self._module_invalidator = module_invalidator
        self._control_structure_processor = control_structure_processor
        self._debug = debug

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute_cell(
        self,
        raw_cell: str,
        args: tuple = (),
        kwargs: dict | None = None,
        original_run_cell: Callable[..., Any] | None = None,
    ) -> _PipelineCompleted | _PipelineSyntaxError | _EarlyReturn:
        """Run *raw_cell* through the 7-phase cached-execution pipeline.

        Returns one of:
        - :class:`_PipelineCompleted` — caller invokes the finaliser
        - :class:`_PipelineSyntaxError` — the cell's own AST failed to parse
        - :class:`_EarlyReturn` — propagate the wrapped value (hook only)
        """
        kwargs = kwargs or {}

        # 1. Cell ID & notebook path
        self._extract_cell_id_and_notebook_path()

        # 2. Badge & timing init
        badge_display_id = str(uuid.uuid4())
        timing_breakdown = self._init_cell_timing_and_badge(badge_display_id)
        hook_start = time.time()
        if self._debug:
            print(f"[TIMING_PROXY] Start cached_run_cell: {datetime.now().strftime('%H:%M:%S.%f')}")

        # 3. Module change detection (must precede upstream check)
        pre_upstream_metrics = self._detect_module_changes(raw_cell)

        # 4. Upstream resolution
        upstream_result = self._resolve_upstream_state(
            raw_cell, pre_upstream_metrics, badge_display_id,
            timing_breakdown, args, kwargs, original_run_cell,
        )
        if isinstance(upstream_result, _EarlyReturn):
            return upstream_result
        upstream_metrics, _restore_time, _execution_time = upstream_result

        # 5. AST parse (tolerate a top-level ``await``; a bare
        # ast.parse rejects module-level await and would silently skip the cell)
        try:
            tree = CodeAnalyzer._parse_cell(raw_cell)
        except SyntaxError:
            self._magics._cancel_progress_badge()
            self._magics._render_interactive_badge([], display_id=badge_display_id, status="DONE")
            return _PipelineSyntaxError()

        # 6. Pre-execution notifications
        all_metrics = self._build_pre_execution_notifications(
            raw_cell, pre_upstream_metrics, upstream_metrics,
        )

        if self._debug:
            print("[TIMING_PROXY] Start executing statements...")

        # 7. Statement execution. The per-statement RNG observer does the
        # measuring; this only opens a fresh accumulation for the cell.
        self._statement_processor.begin_cell_rng_observation()
        # Remote freshness checks (a cached function reading s3:// and friends)
        # are network round trips that land on the HIT path, where the badge
        # reports a saving and nothing reports what establishing it cost. The
        # measurement writes itself into the breakdown on exit, so it survives
        # the early return below.
        with _measured_validation(sink=timing_breakdown):
            result = self._execute_cell_statements(
                raw_cell, tree, all_metrics, badge_display_id,
                hook_start, timing_breakdown,
            )
        if isinstance(result, _EarlyReturn):
            return result

        all_metrics, buffered_result_outputs, badge_render_time = result
        timing_breakdown['badge_progress'] = badge_render_time
        self._record_executed_cell_hash(raw_cell)

        return _PipelineCompleted(
            all_metrics=all_metrics,
            buffered_outputs=buffered_result_outputs,
            badge_display_id=badge_display_id,
            hook_start=hook_start,
            timing_breakdown=timing_breakdown,
            badge_render_time=badge_render_time,
        )

    def _record_executed_cell_hash(self, raw_cell: str) -> None:
        """Remember that this exact cell source ran, so the upstream checker can
        tell an edited-but-not-rerun seed() cell from one that actually ran
        (ADR-017). Also snapshot the RNG state around a cell that
        TOUCHED the global RNG so a downstream draw can be restored to its
        position-correct state (ADR-018), and record which
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
            state = self._statement_processor._tracking_state
            digest = hashlib.sha256(raw_cell.encode('utf-8')).hexdigest()
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

    async def execute_cell_async(
        self,
        raw_cell: str,
        args: tuple = (),
        kwargs: dict | None = None,
        original_run_cell: Callable[..., Any] | None = None,
    ) -> _PipelineCompleted | _PipelineSyntaxError | _EarlyReturn:
        """Async twin of :meth:`execute_cell` for top-level-await cells.

        Runs the identical 7-phase pipeline — cell id, badge/timing init, module
        change detection, upstream resolution, AST parse, pre-execution
        notifications — then awaits :meth:`_execute_cell_statements_async` so a
        top-level ``await`` executes on IPython's live loop.  Because every phase
        but statement execution is shared verbatim, the async path can never
        drift from the sync path in upstream reconstruction, cacheability, or
        badge accounting.
        """
        kwargs = kwargs or {}

        # 1. Cell ID & notebook path
        self._extract_cell_id_and_notebook_path()

        # 2. Badge & timing init
        badge_display_id = str(uuid.uuid4())
        timing_breakdown = self._init_cell_timing_and_badge(badge_display_id)
        hook_start = time.time()
        if self._debug:
            print(f"[TIMING_PROXY] Start cached_run_cell (async): {datetime.now().strftime('%H:%M:%S.%f')}")

        # 3. Module change detection (must precede upstream check)
        pre_upstream_metrics = self._detect_module_changes(raw_cell)

        # 4. Upstream resolution
        upstream_result = self._resolve_upstream_state(
            raw_cell, pre_upstream_metrics, badge_display_id,
            timing_breakdown, args, kwargs, original_run_cell,
        )
        if isinstance(upstream_result, _EarlyReturn):
            return upstream_result
        upstream_metrics, _restore_time, _execution_time = upstream_result

        # 5. AST parse (tolerate a top-level ``await``; a bare
        # ast.parse rejects module-level await and would silently skip the cell)
        try:
            tree = CodeAnalyzer._parse_cell(raw_cell)
        except SyntaxError:
            self._magics._cancel_progress_badge()
            self._magics._render_interactive_badge([], display_id=badge_display_id, status="DONE")
            return _PipelineSyntaxError()

        # 6. Pre-execution notifications
        all_metrics = self._build_pre_execution_notifications(
            raw_cell, pre_upstream_metrics, upstream_metrics,
        )

        if self._debug:
            print("[TIMING_PROXY] Start executing statements (async)...")

        # 7. Statement execution (awaited). Same single observer as the sync path.
        self._statement_processor.begin_cell_rng_observation()
        # Remote freshness checks, measured exactly as in the sync path.
        with _measured_validation(sink=timing_breakdown):
            result = await self._execute_cell_statements_async(
                raw_cell, tree, all_metrics, badge_display_id,
                hook_start, timing_breakdown,
            )
        if isinstance(result, _EarlyReturn):
            return result

        all_metrics, buffered_result_outputs, badge_render_time = result
        timing_breakdown['badge_progress'] = badge_render_time
        self._record_executed_cell_hash(raw_cell)

        return _PipelineCompleted(
            all_metrics=all_metrics,
            buffered_outputs=buffered_result_outputs,
            badge_display_id=badge_display_id,
            hook_start=hook_start,
            timing_breakdown=timing_breakdown,
            badge_render_time=badge_render_time,
        )

    # ------------------------------------------------------------------
    # Phase 1: cell ID & notebook path
    # ------------------------------------------------------------------

    def _extract_cell_id_and_notebook_path(self) -> None:
        """Resolve cell_id and notebook path from IPython kernel metadata.

        Must run BEFORE the upstream check so the notebook path is available
        for reading upstream cells.  Stores the cell id on the adapter via
        the back-reference.
        """
        try:
            cell_id = self._magics._cell_id_from_parent_metadata(self.shell)
            self._magics._current_cell_id = cell_id
            self._magics._maybe_seed_notebook_path(cell_id)

            # Debug-level logging (not a raw print): silent unless %cash_debug
            # is on. The "No cell_id" branch otherwise fires on every cell in
            # the default proxy path for environments that don't supply one.
            if cell_id:
                logger.debug("[PROXY_CELL_ID] Captured cell_id early: %s", cell_id)
            else:
                logger.debug("[PROXY_CELL_ID] No cell_id in parent metadata")
        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            logger.debug("[PROXY_CELL_ID] Could not capture cell_id early: %s", e)

    # ------------------------------------------------------------------
    # Phase 2: badge & timing init
    # ------------------------------------------------------------------

    def _init_cell_timing_and_badge(self, badge_display_id: str) -> 'TimingBreakdown':
        """Set up timing tracking and render the initial 'RUNNING' badge."""
        timing_breakdown: 'TimingBreakdown' = {}
        cell_start = time.time()

        self._magics._badge_cell_start_time = cell_start
        self._magics._last_badge_render_time = 0.0

        t_badge_init = time.time()
        if self._magics._badge_mode == 'html':
            self._magics._render_interactive_badge(
                [], display_id=badge_display_id,
                status="RUNNING", update_existing=False,
            )
        timing_breakdown['badge_init'] = time.time() - t_badge_init
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
            if newly_tracked and self._debug:
                print(f"[AUTO_TRACK] Auto-tracking local modules: {', '.join(sorted(newly_tracked))}")
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

                mod_names = ', '.join(sorted(changed_modules.keys()))
                notification: ProcessResult = {
                    'status': 'MODULE_RELOADED',
                    'code': f"🔄 Module{'s' if len(changed_modules) > 1 else ''} reloaded: {mod_names}",
                    'is_upstream': True,
                    'total_time': 0.0,
                    'execution_time': 0.0,
                    'outputs': [],
                    'changed_modules': dict(changed_modules.items()),
                }
                notifications.append(notification)
                if self._debug:
                    for mod, path in changed_modules.items():
                        syms = per_module_changed_symbols.get(mod)
                        sym_info = f" (changed symbols: {syms})" if syms is not None else " (full invalidation)"
                        print(f"[AUTO_TRACK] Reloaded changed module '{mod}' ({path}){sym_info}")
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
    ) -> tuple[list[ProcessResult], float, float]:
        """Delegate to ``UpstreamChecker``.

        Returns a list of metrics for any executed or restored upstream
        statements, plus the total restore and execution times.
        """
        return self._upstream_checker.check_and_reexecute(
            cell_code,
            required_inputs,
            self._statement_processor.process_statement,
            self._magics._global_ttl,
            cell_id=self._magics._current_cell_id,
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
        state = self._statement_processor._tracking_state
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
            if self._debug:
                print(f"[ENSURE_STATE_DEBUG] Cell code: {cell_code[:50]}...")

            inputs, outputs = CodeAnalyzer.analyze_code_block(cell_code)

            if self._debug:
                print(f"[ENSURE_STATE_DEBUG] Analyzed inputs: {inputs}")
                print(f"[ENSURE_STATE_DEBUG] Analyzed outputs: {outputs}")
                print(f"[ENSURE_STATE_DEBUG] Current user_ns keys (first 10): {list(self.shell.user_ns.keys())[:10]}")

            total_restore_time = 0.0
            upstream_metrics: list[ProcessResult] = []

            for var_name in inputs:
                if var_name not in self.shell.user_ns:
                    start_restore = time.time()
                    try:
                        metrics = self._restorer.restore_variable(var_name)
                        total_restore_time += (time.time() - start_restore)
                        if metrics:
                            upstream_metrics.extend(metrics)
                    except NameError:
                        # Could not find a source — proceed; upstream re-execution may provide it.
                        if self._debug:
                            print(f"[STATE] Could not restore '{var_name}' from cache. Hoping for upstream re-execution.")

            reexec_metrics, upstream_restore_time, total_execution_time = self._check_and_reexecute_upstream_cells(
                cell_code, inputs, progress_callback=progress_callback,
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

        except (RuntimeError, SyntaxError, AmbiguousCellError):
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
        timing_breakdown: 'TimingBreakdown',
        args: tuple,
        kwargs: dict,
        original_run_cell: Callable[..., Any] | None,
    ) -> tuple[list[ProcessResult], float, float] | _EarlyReturn:
        """Run upstream dependency checking and state restoration.

        On error: if *original_run_cell* is provided (hook path), fall back
        through IPython so the user sees the error in the cell.  When None
        (magic path), re-raise so the magic's caller sees a normal Python
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
            self._magics._maybe_progress_badge(
                combined, display_id=badge_display_id,
                step=current_step if current_step is not None else len(combined),
                total=total_steps or 0,
                code=upstream_label,
            )

        caught: Exception | None = None
        try:
            upstream_metrics, total_restore_time, total_execution_time = self._ensure_state_for_inputs(
                raw_cell, progress_callback=_upstream_progress_cb,
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
                caught, raw_cell, badge_display_id, args, kwargs, original_run_cell,
            )

        timing_breakdown['upstream_check_raw'] = time.time() - t_ensure
        timing_breakdown['total_restore_time'] = total_restore_time
        timing_breakdown['total_execution_time'] = total_execution_time
        timing_breakdown['upstream_check'] = (
            (time.time() - t_ensure) - total_restore_time - total_execution_time
        )

        if self._debug:
            print(f"[TIMING_PROXY] Ensure state: {(time.time() - t_ensure)*1000:.2f}ms")
            print(f"[TIMING_PROXY] Total restore time: {total_restore_time*1000:.2f}ms")
            print(f"[TIMING_PROXY] Total execution time: {total_execution_time*1000:.2f}ms")
            print(f"[TIMING_PROXY] Pure overhead (excl. restore+exec): {((time.time() - t_ensure) - total_restore_time - total_execution_time)*1000:.2f}ms")

        return upstream_metrics, total_restore_time, total_execution_time

    def _handle_upstream_resolution_failure(
        self,
        caught: Exception,
        raw_cell: str,
        badge_display_id: str,
        args: tuple,
        kwargs: dict,
        original_run_cell: Callable[..., Any] | None,
    ) -> _EarlyReturn:
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

        - ``original_run_cell is None`` (``%%cash`` magic path): a SyntaxError
          becomes a quiet "log + return"; anything else re-raises so the magic's
          caller sees the real error.
        - SyntaxError (hook path): re-run the raw cell through IPython so the
          user sees the parse error attributed to their cell.
        - RuntimeError / AmbiguousCellError / UpstreamStateError: synthesise a
          fresh raise inside the user's cell (the "fail the cell
          loudly" path) so IPython attributes the traceback to the cell.
        - anything else: log and fall back to normal execution.
        """
        if original_run_cell is None:
            # Magic path: SyntaxError from upstream sim is best surfaced as
            # a normal "log + return" (matches the executor's own AST-parse
            # SyntaxError path).  Any other exception propagates so the
            # magic's caller sees the real error.
            if isinstance(caught, SyntaxError):
                self._magics._cancel_progress_badge()
                self._magics._render_interactive_badge([], display_id=badge_display_id, status="DONE")
                return _EarlyReturn(None)
            raise caught
        if isinstance(caught, SyntaxError):
            self._magics._cancel_progress_badge()
            self._magics._render_interactive_badge([], display_id=badge_display_id, status="DONE")
            return _EarlyReturn(original_run_cell(raw_cell, *args, **kwargs))
        if isinstance(caught, (RuntimeError, AmbiguousCellError, UpstreamStateError)):
            # Re-raise inside the user's cell so IPython renders the traceback
            # as if the cell itself raised.  Import the exception class
            # explicitly because the user's namespace may not have it.  The
            # trailing ``from None`` suppresses any ambient context so the
            # synthesised raise carries only the message, never a chain back
            # into cash's internals.
            cls = type(caught)
            error_code = (
                f"from {cls.__module__} import {cls.__name__}; "
                f"raise {cls.__name__}('''{str(caught)}''') from None"
            )
            self._magics._cancel_progress_badge()
            self._magics._render_interactive_badge([], display_id=badge_display_id, status="DONE")
            return _EarlyReturn(original_run_cell(error_code, *args, **kwargs))
        logger.error("Cash auto-caching failed: %s. Falling back to normal execution.", caught)
        self._magics._cancel_progress_badge()
        self._magics._render_interactive_badge([], display_id=badge_display_id, status="DONE")
        return _EarlyReturn(original_run_cell(raw_cell, *args, **kwargs))

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
            func_names = ', '.join(sorted(changed_funcs))
            if self._debug:
                print(f"[FUNCTION_CHANGE] Detected changed functions: {func_names}")
            return [{
                'status': 'FUNCTION_CHANGED',
                'code': f"🔄 Function{'s' if len(changed_funcs) > 1 else ''} changed: {func_names}",
                'is_upstream': True,
                'execution_time': 0.0,
                'total_time': 0.0,
                'saved_time': 0.0,
                'error': None,
                'restored_vars': [],
                'uncacheable_reasons': [],
                'outputs': [],
                'changed_functions': sorted(changed_funcs),
            }]
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
            if self._debug:
                for w in opaque_warnings:
                    print(f"[OPAQUE_CALL] {w}")
            return [{
                'status': 'WARNING',
                'code': f"⚠️ {msg}",
                'is_upstream': True,
                'execution_time': 0.0,
                'total_time': 0.0,
                'saved_time': 0.0,
                'error': None,
                'restored_vars': [],
                'uncacheable_reasons': [],
                'outputs': [],
            } for msg in opaque_warnings]
        except (AttributeError, TypeError, SyntaxError, ValueError) as exc:
            logger.debug("Failed to detect opaque call patterns: %s", exc)
            return []

    def _make_staleness_metrics(self) -> list[ProcessResult]:
        """Return WARNING notifications from the staleness tracker's two checks.

        Two independent rows share this guard: `staleness_notification` (proven
        stale -- loud every occurrence) and `unverifiable_notification`
        (freshness cannot be verified at all -- loud once per session). Guarded
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
            unverifiable = unverifiable_notification(tracker)
            if unverifiable is not None:
                notifications.append(unverifiable)
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
        for output in rich_outputs:
            if isinstance(output, dict) and 'data' in output:
                publish_display_data(data=output['data'], metadata=output.get('metadata', {}))
            else:
                display(output)
        return buffered_result_outputs

    @staticmethod
    def _expr_has_trailing_semicolon(raw_cell: str, node: ast.stmt) -> bool:
        """True if expression statement *node* is followed by a ``;`` in the raw
        source (IPython display suppression). ``ast.unparse`` discards it, so we
        recover it from the original cell text.

        Both coordinates must be read the way the PARSER wrote them, or this
        silently answers ``False`` and echoes a repr the user suppressed:

        * the line index needs the parser's line-break rules, not
          ``str.splitlines()``'s wider set -- see
          ``_splitlines_like_the_parser``. One vertical tab or form feed
          anywhere earlier in the cell shifts every later index.
        * ``end_col_offset`` is a UTF-8 *byte* offset, not a character index,
          so the line is sliced as bytes. Reading it as characters slid the
          slice past the ``;`` whenever anything non-ASCII sat earlier on the
          same line (``df[df.city == "Zürich"];``).

        Lines here keep their endings, so the remainder of the cell appends
        verbatim rather than being rebuilt with ``"\\n".join``.
        """
        if not isinstance(node, ast.Expr):
            return False
        end_line = getattr(node, "end_lineno", None)
        end_col = getattr(node, "end_col_offset", None)
        if end_line is None or end_col is None:
            return False
        lines = _splitlines_like_the_parser(raw_cell)
        if end_line > len(lines):
            return False
        rest = lines[end_line - 1].encode()[end_col:].decode()
        rest += "".join(lines[end_line:])
        return rest.lstrip().startswith(";")

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
        if metrics.get('stdout'):
            print(metrics['stdout'], end='')
        if metrics.get('stderr'):
            print(metrics['stderr'], end='', file=sys.stderr)
        if metrics.get('status') == CacheStatus.ERROR and metrics.get('error'):
            raise metrics['error']

        return self._flush_rich_outputs(
            metrics.get('rich_outputs', []), is_last_statement, buffered_result_outputs,
        )

    def _process_regular_stmt(
        self,
        stmt_code: str,
        annotation: Any,
        occurrence_index: int,
        is_last_statement: bool,
        all_metrics: list[ProcessResult],
        buffered_result_outputs: list,
        display_code: str | None = None,
        exec_source: str | None = None,
    ) -> list:
        """Process one non-control statement with caching; return updated buffer.

        ``display_code`` and ``exec_source`` are usually the SAME text (the
        loop passes ``stmt_display`` for both -- see ``_statement_source``),
        but diverge for a top-level ``def``/``class``: the badge withholds
        the body there (``display_code`` stays ``None``), while execution
        still needs it (``exec_source`` recovers it directly) -- see the
        caller in ``_execute_cell_statements``.
        """
        metrics = self._statement_processor.process_statement(
            stmt_code, self._magics._global_ttl, silent=True,
            annotation=annotation,
            display_code=display_code,
            exec_source=exec_source,
            occurrence_index=occurrence_index,
            # IPython echoes only the CELL's last expression; cash executes each
            # statement as its own unit, so it must be told which one that is
            #.
            is_last=is_last_statement,
        )
        return self._handle_regular_stmt_metrics(
            metrics, is_last_statement, all_metrics, buffered_result_outputs,
        )

    async def _process_regular_stmt_async(
        self,
        stmt_code: str,
        annotation: Any,
        occurrence_index: int,
        is_last_statement: bool,
        all_metrics: list[ProcessResult],
        buffered_result_outputs: list,
        display_code: str | None = None,
        exec_source: str | None = None,
    ) -> list:
        """Async twin of :meth:`_process_regular_stmt`.

        Routes every regular statement through ``process_statement_async`` so a
        top-level ``await`` is executed on IPython's live loop.  A statement
        with no top-level await compiles without ``CO_COROUTINE`` and runs
        through the same synchronous exec/eval inside the async executor, so its
        behaviour (and cache key) is identical to the sync path.
        """
        metrics = await self._statement_processor.process_statement_async(
            stmt_code, self._magics._global_ttl, silent=True,
            annotation=annotation,
            display_code=display_code,
            exec_source=exec_source,
            occurrence_index=occurrence_index,
            is_last=is_last_statement,  # as in the sync path
        )
        return self._handle_regular_stmt_metrics(
            metrics, is_last_statement, all_metrics, buffered_result_outputs,
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
            if not metrics.get('_output_flushed'):
                if metrics.get('stdout'):
                    print(metrics['stdout'], end='')
                if metrics.get('stderr'):
                    print(metrics['stderr'], end='', file=sys.stderr)
            buffered_result_outputs = self._flush_rich_outputs(
                metrics.get('rich_outputs', []), is_last_statement, buffered_result_outputs,
            )
        return buffered_result_outputs

    def _finalize_error_badge(
        self,
        e: BaseException,
        raw_cell: str,
        node: ast.stmt,
        all_metrics: list[ProcessResult],
        badge_display_id: str,
        hook_start: float,
        timing_breakdown: 'TimingBreakdown',
    ) -> None:
        """Show a clean error display + render the final DONE badge.

        Does **not** re-raise — that is the pipeline caller's job.
        """
        # A statement that just raised may have an armed progress timer
        # (it hadn't finished, so nothing cancelled it yet) -- stop it before
        # rendering the DONE badge below so a late fire can't overwrite it.
        self._magics._cancel_progress_badge()
        self._magics._show_clean_error(e, raw_cell, node)
        hook_total = time.time() - hook_start
        if self._magics._badge_mode == 'html':
            self._magics._render_interactive_badge(
                all_metrics, display_id=badge_display_id,
                cell_total_time=hook_total, timing_breakdown=timing_breakdown,
                status="DONE",
            )
        elif self._magics._badge_mode == 'print':
            self._magics._print_text_badge(all_metrics, cell_total_time=hook_total)

    def _execute_cell_statements(
        self,
        raw_cell: str,
        tree: ast.Module,
        all_metrics: list[ProcessResult],
        badge_display_id: str,
        hook_start: float,
        timing_breakdown: 'TimingBreakdown',
    ) -> _EarlyReturn | tuple[list[ProcessResult], list, float]:
        """Iterate over AST statements, executing or caching each one.

        Returns ``(all_metrics, buffered_result_outputs, badge_render_time)``
        on success, or raises if a statement raised an error (after first
        rendering the error badge via :meth:`_finalize_error_badge`).
        """
        buffered_result_outputs: list = []
        badge_render_time = 0.0

        upstream_step_count = len([
            m for m in all_metrics
            if m.get('is_upstream', False) and m.get('status') != 'SKIPPED'
        ])
        total_steps_unified = upstream_step_count + len(tree.body)
        stmt_occurrence_counts: dict[str, int] = {}

        for i, node in enumerate(tree.body):
            try:
                stmt_code = ast.unparse(node)
            except (ValueError, TypeError):
                continue

            # The badge shows the user's own layout; `stmt_code` above stays the
            # keyed and executed text. See `_statement_source`.
            stmt_display = _statement_source(raw_cell, node)
            stmt_exec_source = _exec_source_for_node(raw_cell, node, stmt_display)

            # ``ast.unparse`` drops a trailing ``;``, losing IPython's display
            # suppression (``df.head();`` shows no repr). Re-attach it so the
            # suppression rides through the cache key AND the execution path
            # (``_execute_statement`` skips the display), so a cached re-run
            # doesn't emit a phantom repr.
            if self._expr_has_trailing_semicolon(raw_cell, node):
                stmt_code = stmt_code + ";"

            occ = stmt_occurrence_counts.get(stmt_code, 0)
            stmt_occurrence_counts[stmt_code] = occ + 1
            annotation = get_statement_annotations(raw_cell, node)
            is_last = (i == len(tree.body) - 1)
            unified_step = upstream_step_count + i + 1

            t_badge_pre = time.time()
            self._magics._arm_progress_badge(
                all_metrics, display_id=badge_display_id, step=unified_step,
                total=total_steps_unified, code=stmt_code,
            )
            badge_render_time += time.time() - t_badge_pre

            try:
                try:
                    if is_control_structure(node):
                        if self._debug:
                            print("[CONTROL] Detected control structure, delegating to ControlStructureProcessor")
                        # ``raw_cell`` (not the annotation) is what goes down: the
                        # structure's statements each resolve their OWN directive
                        # against the original source, and ``ast.unparse`` has already
                        # dropped the comments by the time they are dispatched. The
                        # ``annotation`` computed above is the node's WHOLE-range
                        # merge, which cannot tell a directive on the loop from one on
                        # a single body statement — passing it would disable caching
                        # for every sibling in the body.
                        ctrl_result = self._control_structure_processor.process(
                            node, ttl=self._magics._global_ttl, silent=True,
                            raw_cell=raw_cell,
                            prev_node=tree.body[i - 1] if i > 0 else None,
                        )
                        buffered_result_outputs = self._collect_ctrl_outputs(
                            ctrl_result, is_last, all_metrics, buffered_result_outputs,
                        )
                        if self._debug:
                            print(f"[CONTROL] Completed: {ctrl_result.total_iterations} iterations, "
                                  f"{ctrl_result.cached_iterations} cached, {ctrl_result.computed_iterations} computed")
                        if not ctrl_result.success:
                            raise ctrl_result.error or RuntimeError("Unknown error in control structure execution")
                    else:
                        buffered_result_outputs = self._process_regular_stmt(
                            stmt_code, annotation, occ, is_last, all_metrics,
                            buffered_result_outputs, display_code=stmt_display,
                            exec_source=stmt_exec_source,
                        )

                    self._magics._cancel_progress_badge()
                    t_badge = time.time()
                    self._magics._maybe_progress_badge(
                        all_metrics, display_id=badge_display_id,
                        step=unified_step + 1, total=total_steps_unified, code=None,
                    )
                    badge_render_time += time.time() - t_badge

                except Exception as e:  # noqa: BLE001 - intentionally broad: catches user code exceptions
                    self._finalize_error_badge(
                        e, raw_cell, node, all_metrics, badge_display_id,
                        hook_start, timing_breakdown,
                    )
                    raise
            finally:
                # Cancel on EVERY exit from this statement, not just the two
                # paths above (the explicit cancel on success, and the one
                # inside _finalize_error_badge for a caught Exception). A
                # BaseException that isn't an Exception -- KeyboardInterrupt,
                # or asyncio.CancelledError from an interrupted await -- skips
                # the `except` above entirely, and neither
                # _execute_cell_inner nor its async twin cancels either. Left
                # armed, that timer fires up to _BADGE_MIN_RENDER_INTERVAL
                # after the interrupt, on whatever cell is running by then.
                # Safe to call unconditionally: a no-op once already cancelled.
                self._magics._cancel_progress_badge()

        return (all_metrics, buffered_result_outputs, badge_render_time)

    async def _execute_cell_statements_async(
        self,
        raw_cell: str,
        tree: ast.Module,
        all_metrics: list[ProcessResult],
        badge_display_id: str,
        hook_start: float,
        timing_breakdown: 'TimingBreakdown',
    ) -> _EarlyReturn | tuple[list[ProcessResult], list, float]:
        """Async twin of :meth:`_execute_cell_statements` for top-level-await cells.

        Identical badge / occurrence / trailing-semicolon / control-structure /
        error-badge plumbing as the sync loop — the ONLY difference is that a
        regular (non-control) statement is executed via the awaited
        :meth:`_process_regular_stmt_async`, so a top-level ``await`` runs on
        IPython's live loop.  Control structures still go through the sync
        ``ControlStructureProcessor`` (they own their own body/mutation lineage
        and never contain top-level await).
        """
        buffered_result_outputs: list = []
        badge_render_time = 0.0

        upstream_step_count = len([
            m for m in all_metrics
            if m.get('is_upstream', False) and m.get('status') != 'SKIPPED'
        ])
        total_steps_unified = upstream_step_count + len(tree.body)
        stmt_occurrence_counts: dict[str, int] = {}

        for i, node in enumerate(tree.body):
            try:
                stmt_code = ast.unparse(node)
            except (ValueError, TypeError):
                continue

            # The badge shows the user's own layout; `stmt_code` above stays the
            # keyed and executed text. See `_statement_source`.
            stmt_display = _statement_source(raw_cell, node)
            stmt_exec_source = _exec_source_for_node(raw_cell, node, stmt_display)

            if self._expr_has_trailing_semicolon(raw_cell, node):
                stmt_code = stmt_code + ";"

            occ = stmt_occurrence_counts.get(stmt_code, 0)
            stmt_occurrence_counts[stmt_code] = occ + 1
            annotation = get_statement_annotations(raw_cell, node)
            is_last = (i == len(tree.body) - 1)
            unified_step = upstream_step_count + i + 1

            t_badge_pre = time.time()
            self._magics._arm_progress_badge(
                all_metrics, display_id=badge_display_id, step=unified_step,
                total=total_steps_unified, code=stmt_code,
            )
            badge_render_time += time.time() - t_badge_pre

            try:
                try:
                    if is_control_structure(node):
                        if contains_top_level_await(node):
                            # A control body that awaits (``for x in xs: r = await
                            # fetch(x)``) cannot be compiled by the sync
                            # ControlStructureProcessor — its unflagged compile()
                            # raises ``SyntaxError: 'await' outside function``
                            #. Run the whole structure as one awaited unit
                            # through the PyCF_ALLOW_TOP_LEVEL_AWAIT-capable path.
                            if self._debug:
                                print("[CONTROL] Await inside control body, running as awaited single unit")
                            ctrl_result = await self._control_structure_processor.process_await_unit(
                                node, ttl=self._magics._global_ttl, silent=True,
                                raw_cell=raw_cell,
                            )
                        else:
                            if self._debug:
                                print("[CONTROL] Detected control structure, delegating to ControlStructureProcessor")
                            # ``raw_cell`` (not the annotation) is what goes down: the
                            # structure's statements each resolve their OWN directive
                            # against the original source, and ``ast.unparse`` has already
                            # dropped the comments by the time they are dispatched. The
                            # ``annotation`` computed above is the node's WHOLE-range
                            # merge, which cannot tell a directive on the loop from one on
                            # a single body statement — passing it would disable caching
                            # for every sibling in the body.
                            ctrl_result = self._control_structure_processor.process(
                                node, ttl=self._magics._global_ttl, silent=True,
                                raw_cell=raw_cell,
                                prev_node=tree.body[i - 1] if i > 0 else None,
                            )
                        buffered_result_outputs = self._collect_ctrl_outputs(
                            ctrl_result, is_last, all_metrics, buffered_result_outputs,
                        )
                        if self._debug:
                            print(f"[CONTROL] Completed: {ctrl_result.total_iterations} iterations, "
                                  f"{ctrl_result.cached_iterations} cached, {ctrl_result.computed_iterations} computed")
                        if not ctrl_result.success:
                            raise ctrl_result.error or RuntimeError("Unknown error in control structure execution")
                    else:
                        buffered_result_outputs = await self._process_regular_stmt_async(
                            stmt_code, annotation, occ, is_last, all_metrics,
                            buffered_result_outputs, display_code=stmt_display,
                            exec_source=stmt_exec_source,
                        )

                    self._magics._cancel_progress_badge()
                    t_badge = time.time()
                    self._magics._maybe_progress_badge(
                        all_metrics, display_id=badge_display_id,
                        step=unified_step + 1, total=total_steps_unified, code=None,
                    )
                    badge_render_time += time.time() - t_badge

                except Exception as e:  # noqa: BLE001 - intentionally broad: catches user code exceptions
                    self._finalize_error_badge(
                        e, raw_cell, node, all_metrics, badge_display_id,
                        hook_start, timing_breakdown,
                    )
                    raise
            finally:
                # See the identical guard in _execute_cell_statements: a
                # BaseException that bypasses `except Exception` (
                # KeyboardInterrupt, or asyncio.CancelledError from an
                # interrupted await -- the more exposed case on THIS path)
                # must still cancel a pending timer, or it fires later on
                # whatever cell happens to be running by then.
                self._magics._cancel_progress_badge()

        return (all_metrics, buffered_result_outputs, badge_render_time)
