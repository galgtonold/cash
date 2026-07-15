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
import sys
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from IPython.display import display, publish_display_data

from ...exceptions import AmbiguousCellError, UpstreamStateError
from .._protocols import ShellProtocol
from ..analysis import CodeAnalyzer
from ..annotations import get_statement_annotations
from ..cache_status import CacheStatus
from ..consumables import consumable_state, is_consumable_unrestorable
from ..control_structures import is_control_structure
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

        # 5. AST parse
        try:
            tree = ast.parse(raw_cell)
        except SyntaxError:
            self._magics._render_interactive_badge([], display_id=badge_display_id, status="DONE")
            return _PipelineSyntaxError()

        # 6. Pre-execution notifications
        all_metrics = self._build_pre_execution_notifications(
            raw_cell, pre_upstream_metrics, upstream_metrics,
        )

        if self._debug:
            print("[TIMING_PROXY] Start executing statements...")

        # 7. Statement execution
        result = self._execute_cell_statements(
            raw_cell, tree, all_metrics, badge_display_id,
            hook_start, timing_breakdown,
        )
        if isinstance(result, _EarlyReturn):
            return result

        all_metrics, buffered_result_outputs, badge_render_time = result
        timing_breakdown['badge_progress'] = badge_render_time

        return _PipelineCompleted(
            all_metrics=all_metrics,
            buffered_outputs=buffered_result_outputs,
            badge_display_id=badge_display_id,
            hook_start=hook_start,
            timing_breakdown=timing_breakdown,
            badge_render_time=badge_render_time,
        )

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

        # 5. AST parse
        try:
            tree = ast.parse(raw_cell)
        except SyntaxError:
            self._magics._render_interactive_badge([], display_id=badge_display_id, status="DONE")
            return _PipelineSyntaxError()

        # 6. Pre-execution notifications
        all_metrics = self._build_pre_execution_notifications(
            raw_cell, pre_upstream_metrics, upstream_metrics,
        )

        if self._debug:
            print("[TIMING_PROXY] Start executing statements (async)...")

        # 7. Statement execution (awaited)
        result = await self._execute_cell_statements_async(
            raw_cell, tree, all_metrics, badge_display_id,
            hook_start, timing_breakdown,
        )
        if isinstance(result, _EarlyReturn):
            return result

        all_metrics, buffered_result_outputs, badge_render_time = result
        timing_breakdown['badge_progress'] = badge_render_time

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
            # and destroy the signal for the run after it. See CAS-118 / CAS-50.
            self._record_consumable_bases(inputs)

        except (RuntimeError, SyntaxError, AmbiguousCellError):
            raise
        except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.debug("[STATE] Error in state restoration logic: %s", e)
            raise

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

        try:
            upstream_metrics, total_restore_time, total_execution_time = self._ensure_state_for_inputs(
                raw_cell, progress_callback=_upstream_progress_cb,
            )
        except Exception as e:  # noqa: BLE001 - broad fallback for upstream simulation failures
            if isinstance(e, KeyboardInterrupt):
                raise
            if original_run_cell is None:
                # Magic path: SyntaxError from upstream sim is best surfaced as
                # a normal "log + return" (matches the executor's own AST-parse
                # SyntaxError path).  Any other exception propagates so the
                # magic's caller sees the real error.
                if isinstance(e, SyntaxError):
                    self._magics._render_interactive_badge([], display_id=badge_display_id, status="DONE")
                    return _EarlyReturn(None)
                raise
            if isinstance(e, SyntaxError):
                self._magics._render_interactive_badge([], display_id=badge_display_id, status="DONE")
                return _EarlyReturn(original_run_cell(raw_cell, *args, **kwargs))
            if isinstance(e, (RuntimeError, AmbiguousCellError, UpstreamStateError)):
                # Re-raise inside the user's cell so IPython renders the traceback
                # as if the cell itself raised.  Import the exception class
                # explicitly because the user's namespace may not have it.
                cls = type(e)
                error_code = (
                    f"from {cls.__module__} import {cls.__name__}; "
                    f"raise {cls.__name__}('''{str(e)}''')"
                )
                self._magics._render_interactive_badge([], display_id=badge_display_id, status="DONE")
                return _EarlyReturn(original_run_cell(error_code, *args, **kwargs))
            logger.error("Cash auto-caching failed: %s. Falling back to normal execution.", e)
            self._magics._render_interactive_badge([], display_id=badge_display_id, status="DONE")
            return _EarlyReturn(original_run_cell(raw_cell, *args, **kwargs))

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
            # doesn't emit a phantom repr (CAS-96).
            if self._expr_has_trailing_semicolon(raw_cell, node):
                stmt_code = stmt_code + ";"

            occ = stmt_occurrence_counts.get(stmt_code, 0)
            stmt_occurrence_counts[stmt_code] = occ + 1
            annotation = get_statement_annotations(raw_cell, node)
            is_last = (i == len(tree.body) - 1)
            unified_step = upstream_step_count + i + 1

            t_badge_pre = time.time()
            if self._magics._badge_mode == 'html':
                self._magics._render_interactive_badge(
                    all_metrics, display_id=badge_display_id,
                    status="RUNNING", current_step=unified_step,
                    total_steps=total_steps_unified, current_code=stmt_code,
                )
            badge_render_time += time.time() - t_badge_pre

            try:
                if is_control_structure(node):
                    if self._debug:
                        print("[CONTROL] Detected control structure, delegating to ControlStructureProcessor")
                    ctrl_result = self._control_structure_processor.process(
                        node, ttl=self._magics._global_ttl, silent=True,
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
            if self._magics._badge_mode == 'html':
                self._magics._render_interactive_badge(
                    all_metrics, display_id=badge_display_id,
                    status="RUNNING", current_step=unified_step,
                    total_steps=total_steps_unified, current_code=stmt_code,
                )
            badge_render_time += time.time() - t_badge_pre

            try:
                if is_control_structure(node):
                    if self._debug:
                        print("[CONTROL] Detected control structure, delegating to ControlStructureProcessor")
                    ctrl_result = self._control_structure_processor.process(
                        node, ttl=self._magics._global_ttl, silent=True,
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
