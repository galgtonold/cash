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
        recover it from the original cell text."""
        if not isinstance(node, ast.Expr):
            return False
        end_line = getattr(node, "end_lineno", None)
        end_col = getattr(node, "end_col_offset", None)
        if end_line is None or end_col is None:
            return False
        lines = raw_cell.splitlines()
        if end_line > len(lines):
            return False
        rest = lines[end_line - 1][end_col:]
        if end_line < len(lines):
            rest = rest + "\n" + "\n".join(lines[end_line:])
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
    ) -> list:
        """Process one non-control statement with caching; return updated buffer."""
        metrics = self._statement_processor.process_statement(
            stmt_code, self._magics._global_ttl, silent=True,
            annotation=annotation,
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
                        stmt_code, annotation, occ, is_last, all_metrics, buffered_result_outputs,
                    )

                self._magics._cancel_progress_badge()
                t_badge = time.time()
                self._magics._maybe_progress_badge(
                    all_metrics, display_id=badge_display_id,
                    step=unified_step + 1, total=total_steps_unified, code=None,
                )
                badge_render_time += time.time() - t_badge

            except Exception as e:  # noqa: BLE001 - intentionally broad: catches user code exceptions
                if isinstance(e, KeyboardInterrupt):
                    raise
                self._finalize_error_badge(
                    e, raw_cell, node, all_metrics, badge_display_id,
                    hook_start, timing_breakdown,
                )
                raise

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
                        stmt_code, annotation, occ, is_last, all_metrics, buffered_result_outputs,
                    )

                self._magics._cancel_progress_badge()
                t_badge = time.time()
                self._magics._maybe_progress_badge(
                    all_metrics, display_id=badge_display_id,
                    step=unified_step + 1, total=total_steps_unified, code=None,
                )
                badge_render_time += time.time() - t_badge

            except Exception as e:  # noqa: BLE001 - intentionally broad: catches user code exceptions
                if isinstance(e, KeyboardInterrupt):
                    raise
                self._finalize_error_badge(
                    e, raw_cell, node, all_metrics, badge_display_id,
                    hook_start, timing_breakdown,
                )
                raise

        return (all_metrics, buffered_result_outputs, badge_render_time)
