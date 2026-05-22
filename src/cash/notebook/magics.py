from __future__ import annotations

"""IPython magic commands for transparent cell caching in Jupyter notebooks."""

import ast
import contextlib
import logging
import sys
import time
import uuid
from collections.abc import Callable
from datetime import datetime

# Any is used at IPython API boundaries where types come from the shell's dynamic
# namespace (user_ns, execution info objects).  These cannot be typed more precisely
# without declaring a hard IPython dependency in production code.
from typing import Any, TypedDict

from IPython.core.magic import Magics, cell_magic, line_magic, magics_class
from IPython.display import HTML, display, publish_display_data

from ..core import Cash
from ..utils import safe_text
from . import badge_renderer as _badge
from ._protocols import ShellProtocol
from .audit import AuditLogger
from .cache_status import CacheStatus
from .cell_executor import (
    CellExecutor,
    _EarlyReturn,
    _PipelineCompleted,
    _PipelineSyntaxError,
)
from .control_structures import ControlStructureProcessor
from .error_display import show_clean_error as _show_clean_error_impl
from .magic_admin import CashAdminMagicsMixin
from .module_invalidator import ModuleInvalidator
from .object_hashing import compute_hash
from .restore import Restorer
from .provenance import ProvenanceTracker
from .statement_processor import ProcessResult, StatementProcessor
from .upstream import UpstreamChecker

__all__ = ["CashMagics"]

# ---------------------------------------------------------------------------
# TypedDicts for internal data structures
# ---------------------------------------------------------------------------

class TimingBreakdown(TypedDict, total=False):
    """Phase-level timing accumulated during ``_execute_cell``."""

    badge_init: float
    total_restore_time: float
    total_execution_time: float
    upstream_check: float
    upstream_check_raw: float
    badge_progress: float

class StatementSummary(TypedDict):
    """Per-statement summary stored in ``CellMetrics.statements``."""

    code: str
    status: str | None
    execution_time: float
    saved_time: float
    outputs: list[str]
    is_upstream: bool

class CellMetrics(TypedDict):
    """Structure of ``_last_cell_metrics`` exposed by ``%cash_status``."""

    statements: list[StatementSummary]
    total_time: float
    total_restored_time: float
    total_computed_time: float
    upstream_metrics: list[ProcessResult]
    status: str | None

logger = logging.getLogger(__name__)


class CashSession:
    """Groups session-level concerns owned by a single CashMagics instance.

    Separating these from execution-level state (backend, shell, tracking
    dictionaries) makes the sub-boundary explicit and each component
    independently addressable.
    """

    __slots__ = ('stats', 'provenance', 'audit')

    def __init__(self) -> None:
        self.stats: dict[str, Any] = {
            'cells_executed': 0,
            'statements_computed': 0,
            'statements_restored': 0,
            'statements_skipped': 0,
            'total_compute_time': 0.0,
            'total_restored_time': 0.0,
            'total_time_saved': 0.0,
        }
        self.provenance: ProvenanceTracker = ProvenanceTracker()
        self.audit: AuditLogger = AuditLogger()


_OP_MAP = {
    CacheStatus.COMPUTED: 'cache_miss',
    CacheStatus.RESTORED: 'cache_hit',
    CacheStatus.SKIPPED: 'cache_skip',
}


@magics_class
class CashMagics(CashAdminMagicsMixin, Magics):
    def __init__(self, shell: ShellProtocol, cash_instance: Cash) -> None:
        """Initialise CashMagics in three phases (ordering matters):

        1. **State setup** — mode flags, tracking dicts, convenience aliases.
        2. **Processing components** — upstream checker, statement processor,
           control structure processor, module invalidator (requires state
           from phase 1).
        3. **Session state** — badge throttle, cell tracking, event hooks
           (requires components from phase 2).
        """
        super().__init__(shell)
        self._cash_instance = cash_instance
        self._execution_lock = False
        self._debug = cash_instance.debug

        # Auto-caching mode state
        self._auto_cache_enabled = False
        self._global_ttl = None

        # Badge display mode: 'html' (interactive display_id badges), 'print' (text summary), 'off' (no badge)
        self._badge_mode = 'html'

        # Execution history tracking for fallback matching
        self._execution_history = []  # List of cell contents executed this session
        self._executed_cell_raw_codes = set()  # Set of raw cell codes executed this session (used in repair/reset)

        # Shared tracking state — single owner of all lineage/dependency dicts
        from ._protocols import TrackingState
        self._tracking_state = TrackingState()

        self._init_processing_components(shell, cash_instance)
        self._init_session_state(shell)

    def _init_processing_components(self, shell: ShellProtocol, cash_instance: Cash) -> None:
        """Create upstream checker, statement processor, control structure processor,
        and module invalidator.

        Wires shared tracking state and function tracker so all components
        use the same lineage dictionaries and source hashes.
        """
        self._upstream_checker = UpstreamChecker(
            shell,
            cash_instance=cash_instance,
            debug=self._debug,
            compute_hash_fn=compute_hash,
            tracking_state=self._tracking_state,
        )

        self._statement_processor = StatementProcessor(
            shell,
            cash_instance,
            debug=self._debug,
            compute_hash_fn=compute_hash,
            tracking_state=self._tracking_state,
        )

        # Share function_tracker so the upstream simulation computes cache keys
        # with the same func_source_hashes as the runtime statement processor.
        self._upstream_checker.function_tracker = self._statement_processor.function_tracker

        self._control_structure_processor = ControlStructureProcessor(
            shell,
            self._statement_processor,
            debug=self._debug,
        )

        self._module_invalidator = ModuleInvalidator(shell, debug=self._debug)

        self._restorer = Restorer(
            shell,
            backend=cash_instance.backend,
            tracking_state=self._tracking_state,
            debug=self._debug,
        )

        self._cell_executor = CellExecutor(
            shell,
            cash_instance=cash_instance,
            magics=self,
            tracking_state=self._tracking_state,
            statement_processor=self._statement_processor,
            upstream_checker=self._upstream_checker,
            restorer=self._restorer,
            module_invalidator=self._module_invalidator,
            control_structure_processor=self._control_structure_processor,
            debug=self._debug,
        )

    def _init_session_state(self, shell: ShellProtocol) -> None:
        """Initialise badge throttle, cell ID tracking, session stats, and event hooks."""
        # Badge throttle state
        self._badge_cell_start_time = 0.0
        self._last_badge_render_time = 0.0
        self._BADGE_MIN_RENDER_INTERVAL = 0.3

        # Cell ID tracking (available since IPython 8.3)
        self._current_cell_id = None

        # Last cell execution metrics (for %cash_status)
        self._last_cell_metrics: CellMetrics = {
            'statements': [],
            'total_time': 0.0,
            'total_restored_time': 0.0,
            'total_computed_time': 0.0,
            'upstream_metrics': [],
            'status': None,
        }

        # Session-level concerns (statistics, provenance, audit) grouped in one object
        self._session = CashSession()

        # Structured log handler (set by %cash_debug json/file)
        self._log_handler = None

        # Benchmark config (one-shot, set by %cash_benchmark)
        self._benchmark_config = None

        # Register event handler to capture cell_id before execution
        try:
            shell.events.register('pre_run_cell', self._capture_cell_id)
        except (AttributeError, TypeError) as e:
            logger.warning(
                "Could not register pre_run_cell event handler: %s. "
                "Cell ID tracking will be disabled — upstream change detection "
                "and VS Code cell-level caching may not work correctly.",
                e,
            )

        # Monkey-patch run_cell to intercept execution
        self._original_run_cell = shell.run_cell
        shell.run_cell = self._execute_cell

    @line_magic
    def cash_on(self, line: str) -> None:
        """
        Enable automatic caching for all subsequent cells.
        Usage:
            %cash_on              # Enable with no TTL
            %cash_on ttl=3600     # Enable with 1-hour TTL
        """
        # Parse optional TTL
        ttl = None
        if line:
            parts = line.split('=')
            if len(parts) == 2 and parts[0].strip() == 'ttl':
                try:
                    ttl = int(parts[1].strip())
                except ValueError:
                    logger.warning("Invalid TTL value")
                    return

        # Invalidate notebook path cache so we re-discover the current notebook
        # (fixes Issue 23: switching notebooks within the same kernel session)
        from .server_discovery import invalidate_notebook_path_cache
        invalidate_notebook_path_cache()

        # Clear upstream checker's simulation and AST caches to prevent stale
        # data from a previous notebook from interfering (Issue 23 part 2)
        self._upstream_checker.reset_caches()

        self._auto_cache_enabled = True
        self._global_ttl = ttl
        ttl_msg = f" (TTL: {ttl}s)" if ttl else ""
        print(safe_text(f"✅ Cash enabled.{ttl_msg} Your computations will be cached automatically."))
        print("   Run %cash_help for available commands.")
        # Report existing cache state if available
        try:
            backend = self._cash_instance.backend
            if hasattr(backend, 'list_entries'):
                entries = backend.list_entries()
                if entries:
                    print(f"   Found existing cache with {len(entries)} entries.")
        except (OSError, AttributeError, TypeError):
            pass
        # One-time hint: cash reads upstream cells from the .ipynb file on
        # disk.  If the user edits an upstream cell but doesn't save, cash
        # won't see the change.  Recommend enabling VS Code auto-save.
        if not getattr(self, '_save_hint_shown', False):
            self._save_hint_shown = True
            print("[Tip] Cash reads upstream cells from the saved notebook file.")
            print("   Save (Ctrl+S) after editing upstream cells, or enable auto-save:")
            print('   Settings -> "files.autoSave": "afterDelay"')
        if self._debug:
            print(f"TTL: {ttl or 'None'}")

    @line_magic
    def cash_off(self, line: str) -> None:
        """
        Disable automatic caching.
        Usage: %cash_off
        """
        self._auto_cache_enabled = False
        self._global_ttl = None
        print("[OK] Auto-caching disabled")

    @line_magic
    def cash_debug(self, line: str) -> None:
        """Toggle debug output on/off.

        Usage:
            %cash_debug on          - Enable debug logging
            %cash_debug off         - Disable debug logging
            %cash_debug json        - Enable JSON-formatted debug output
            %cash_debug file path   - Also log to file in JSON format
        """
        parts = line.strip().lower().split()
        mode = parts[0] if parts else ''

        if mode in ('on', 'true', '1', 'enable'):
            self._debug = True
            logger.setLevel(logging.DEBUG)
            print("Cache debug output enabled.")
        elif mode in ('off', 'false', '0', 'disable'):
            self._debug = False
            logger.setLevel(logging.INFO)
            print("Cache debug output disabled.")
        elif mode == 'json':
            self._debug = True
            from ..logging import setup_logging
            self._log_handler = setup_logging(
                level=logging.DEBUG, json_output=True)
            print("Cache debug output enabled (JSON format).")
        elif mode == 'file' and len(parts) > 1:
            log_path = parts[1]
            self._debug = True
            from ..logging import setup_logging
            self._log_handler = setup_logging(
                level=logging.DEBUG, log_file=log_path)
            print(f"Cache debug output enabled (logging to {log_path}).")
        else:
            # Toggle if no argument
            self._debug = not self._debug
            print(f"Cache debug output: {'enabled' if self._debug else 'disabled'}")

        # Propagate to global cash logger to capture all component logs (backends, etc.)
        cash_logger = logging.getLogger("cash")
        cash_logger.setLevel(logging.DEBUG if self._debug else logging.INFO)
        # Also set local logger
        logger.setLevel(logging.DEBUG if self._debug else logging.INFO)

        # Propagate to components
        self._statement_processor.debug = self._debug
        self._upstream_checker.debug = self._debug
        self._cash_instance.debug = self._debug

    @line_magic
    def cash_help(self, line: str) -> None:
        """Display a quick-reference card for Cash magic commands."""
        topic = line.strip().lower() if line else ''
        if topic in ('badge', 'badges'):
            print(
                "Badge Display\n"
                "─────────────\n"
                "  %cash_badge html    Interactive HTML badges (default)\n"
                "  %cash_badge print   Text summary after cell completes\n"
                "  %cash_badge off     No badge output\n"
                "\n"
                "Badge status icons:\n"
                "  [C] COMPUTED  — statement was executed (cache miss)\n"
                "  [R] RESTORED  — result loaded from cache (cache hit)\n"
                "  [S] SKIPPED   — unchanged, no work needed"
            )
        elif topic in ('debug', 'debugging'):
            print(
                "Debugging\n"
                "─────────\n"
                "  %cash_debug on      Enable debug output\n"
                "  %cash_debug off     Disable debug output\n"
                "  %cash_debug json    JSON-formatted debug output\n"
                "  %cash_debug file p  Log debug output to file path p\n"
                "  %cash_verify        Check cache integrity\n"
                "  %cash_verify --fix  Check and remove corrupted entries\n"
                "  %cash_repair        Remove corrupted cache entries\n"
                "  %cash_repair --state  Reset tracking (keep cache)\n"
                "  %cash_repair --full   Clear all cache and state"
            )
        elif topic in ('collab', 'collaboration', 'sharing'):
            print(
                "Collaboration & Sharing\n"
                "───────────────────────\n"
                "  %cash_export file      Export cache to file\n"
                "  %cash_export f --json  Export lineage as JSON\n"
                "  %cash_import file      Import cache from file\n"
                "  %cash_import f --merge Merge with existing cache\n"
                "  %cash_diff file        Compare with exported cache\n"
                "  %cash_diff f --vars    Show variable-level differences"
            )
        elif topic in ('inspect', 'provenance', 'audit'):
            print(
                "Inspection & Audit\n"
                "──────────────────\n"
                "  %cash_status        Last cell execution metrics\n"
                "  %cash_stats         Session-wide statistics\n"
                "  %cash_provenance v  History of variable v\n"
                "  %cash_provenance --time   Timeline of computations\n"
                "  %cash_provenance --graph  Dependency graph\n"
                "  %cash_audit on/off  Enable/disable audit logging\n"
                "  %cash_audit show    View audit log\n"
                "  %cash_log           View recent log events"
            )
        else:
            print(
                "Cash — Smart Caching for Jupyter Notebooks\n"
                "═══════════════════════════════════════════\n"
                "\n"
                "Quick Start:\n"
                "  %cash_on             Enable automatic caching\n"
                "  %cash_on ttl=3600    Enable with 1-hour TTL\n"
                "  %cash_off            Disable caching\n"
                "\n"
                "Essential Commands:\n"
                "  %cash_status         Show last cell metrics\n"
                "  %cash_stats          Session-wide statistics\n"
                "  %cash_badge html|print|off   Set badge display\n"
                "  %cash_debug on|off   Toggle debug output\n"
                "\n"
                "Cache Management:\n"
                "  %cash_verify         Check cache integrity\n"
                "  %cash_repair         Fix corrupted entries\n"
                "  %cash_export file    Export cache to file\n"
                "  %cash_import file    Import cache from file\n"
                "\n"
                "Module Tracking:\n"
                "  %cash_track module   Track a module for changes\n"
                "  %cash_track --list   Show tracked modules\n"
                "\n"
                "Annotations (in code comments):\n"
                "  # @cash:no-cache    Skip caching for a statement\n"
                "  # @cash:ttl=300     Set TTL for a statement\n"
                "  # @cash:persist     Force disk persistence\n"
                "\n"
                "Topics: %cash_help badge | debug | collab | inspect\n"
                "\n"
                "Docs: https://cash-lib.readthedocs.io/"
            )

    @line_magic
    def cash_feedback(self, line: str) -> None:
        """Show feedback instructions for beta users."""
        print(
            "We'd love to hear from you!\n"
            "───────────────────────────\n"
            "\n"
            "Bug reports & feature requests:\n"
            "  https://github.com/galgtonold/cash/issues\n"
            "\n"
            "Questions & discussion:\n"
            "  https://github.com/galgtonold/cash/discussions\n"
            "\n"
            "You can also run %cash_stats to see how much time\n"
            "Cash has saved you this session."
        )

    @line_magic
    def cash_badge(self, line: str) -> None:
        """Set the badge display mode.

        Usage:
            %cash_badge html   - Interactive HTML badges with live updates (default)
            %cash_badge print  - Text summary printed once after cell completes
            %cash_badge off    - No badge output at all
        """
        mode = line.strip().lower()
        if mode in ('html', 'print', 'off'):
            self._badge_mode = mode
            print(f"Badge mode set to: {mode}")
        else:
            print(f"Current badge mode: {self._badge_mode}")
            print("Usage: %cash_badge html|print|off")

    @line_magic
    def cash_status(self, line: str) -> dict[str, Any] | None:
        """
        Get machine-readable status of the last cell execution.

        Usage:
            %cash_status          # Print JSON status
            %cash_status json     # Return JSON string
            %cash_status dict     # Return Python dict

        Returns a dict/JSON with:
            - statements: List of statement metrics (status, code, outputs, times)
            - total_time: Total execution time
            - total_restored_time: Time saved by cache hits
            - total_computed_time: Time spent computing
            - upstream_metrics: Metrics from upstream re-executions
            - status: Overall status (COMPUTED, RESTORED, SKIPPED, MIXED)
            - lineage: Current variable lineage state
            - cache_stats: Backend statistics
        """
        import json

        mode = line.strip().lower() if line else 'print'

        # Build comprehensive status
        status = {
            'last_cell': self._last_cell_metrics.copy(),
            'lineage': dict(self._tracking_state.variable_lineage),
            'executed_codes': {k: v[:50] + '...' if len(v) > 50 else v
                              for k, v in self._tracking_state.executed_cell_codes.items()},
            'auto_cache_enabled': self._auto_cache_enabled,
            'debug_enabled': self._debug,
        }

        # Add cache stats if available
        try:
            backend = self._cash_instance.backend
            if hasattr(backend, 'stats'):
                status['cache_stats'] = backend.stats()
            else:
                status['cache_stats'] = {'keys': len(list(backend.keys())) if hasattr(backend, 'keys') else 'unknown'}
        except (AttributeError, TypeError, OSError) as exc:
            logger.debug("Failed to retrieve cache stats: %s", exc)
            status['cache_stats'] = {}

        if mode == 'dict':
            return status
        if mode == 'json':
            return json.dumps(status, default=str, indent=2)
        # Print formatted output
        print(json.dumps(status, default=str, indent=2))
        return status

    @staticmethod
    def _cell_id_from_parent_metadata(shell: Any) -> str | None:
        """Return cell_id from IPython parent-header metadata, or None.

        Checks the two locations VS Code and other frontends use.
        """
        if not hasattr(shell, 'get_parent'):
            return None
        parent = shell.get_parent()
        if not parent:
            return None
        metadata = parent.get('metadata', {})
        if 'cellId' in metadata:
            return metadata['cellId']
        if 'vscode' in metadata and 'cellId' in metadata['vscode']:
            return metadata['vscode']['cellId']
        return None

    @staticmethod
    def _maybe_seed_notebook_path(cell_id: str | None) -> None:
        """If cell_id is a VS Code URI, seed the notebook-path cache from it."""
        if not cell_id:
            return
        from .server_discovery import extract_notebook_path_from_vscode_cell_id, set_notebook_path
        nb_path = extract_notebook_path_from_vscode_cell_id(cell_id)
        if nb_path:
            set_notebook_path(nb_path)

    def _capture_cell_id(self, info: Any) -> None:
        """Capture cell_id from IPython's pre_run_cell event.

        This is called by IPython before each cell execution.
        The cell_id is available since IPython 8.3.
        In VS Code, it might be in metadata.

        As a side-effect, if the cell_id is a VS Code URI we extract the
        notebook file path from it and seed the notebook-path cache so that
        upstream checking can find the notebook even when
        ``__vsc_ipynb_file__`` is not injected.
        """
        try:
            self._current_cell_id = None

            # 1. Try standard info.cell_id (JupyterLab / IPython 8.3+)
            if hasattr(info, 'cell_id') and info.cell_id:
                self._current_cell_id = info.cell_id

            # 2. Try to get it from parent header metadata (VS Code / others)
            if not self._current_cell_id:
                self._current_cell_id = self._cell_id_from_parent_metadata(self.shell)

            # 3. Seed notebook-path cache from VS Code cell_id URI
            self._maybe_seed_notebook_path(self._current_cell_id)

            if self._debug:
                if self._current_cell_id:
                    print(f"[CELL_ID] Captured cell_id: {self._current_cell_id}")
                else:
                    print("[CELL_ID] No cell_id found in info or metadata")

        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            if self._debug:
                print(f"[CELL_ID] Could not capture cell_id: {e}")
            self._current_cell_id = None

    def _should_render_progress_badge(self) -> bool:
        """Check if enough time has passed to render a progress badge update.

        Badge throttling policy:
        Enforce a minimum interval between badge renders to prevent flicker
        when many fast statements execute in quick succession.  The first
        render after cell start is always allowed so the user immediately
        sees what is running.

        Returns True if a badge update should be rendered now.
        """
        now = time.time()

        # Throttle: skip if we rendered very recently
        if now - self._last_badge_render_time < self._BADGE_MIN_RENDER_INTERVAL:
            return False

        self._last_badge_render_time = now
        return True

    def _maybe_progress_badge(
        self,
        metrics: list[ProcessResult],
        display_id: str,
        step: int,
        total: int,
        code: str | None,
    ) -> None:
        """Render a RUNNING badge update if throttle allows.

        Consolidates the throttle check + render into a single call so the
        execution loop reads as: ``execute → _maybe_progress_badge(...)`` rather
        than the repeated ``if badge_mode == 'html' and _should_render…`` pattern.
        """
        if self._badge_mode == 'html' and self._should_render_progress_badge():
            self._render_interactive_badge(
                metrics,
                display_id=display_id,
                status="RUNNING",
                current_step=step,
                total_steps=total,
                current_code=code,
            )

    def _execute_cell(self, raw_cell: str, *args: Any, **kwargs: Any) -> Any:
        """Proxy for ``interactiveshell.run_cell`` to implement caching when
        ``%cash_on`` is active.  Delegates to :meth:`CellExecutor.execute_cell`."""
        if not self._auto_cache_enabled:
            return self._original_run_cell(raw_cell, *args, **kwargs)

        # Record raw cell text for bug-report history (before any processing)
        self._execution_history.append(raw_cell)

        # Benchmark dispatch (one-shot)
        benchmark_config = getattr(self, '_benchmark_config', None)
        if benchmark_config and benchmark_config.get('active'):
            self._benchmark_config['active'] = False
            self._run_benchmark(
                raw_cell,
                benchmark_config['iterations'],
                benchmark_config['cold_start'],
                benchmark_config['compare_mode'],
            )
            return self._original_run_cell("pass", *args, **kwargs)

        try:
            result = self._cell_executor.execute_cell(
                raw_cell, args, kwargs,
                original_run_cell=self._original_run_cell,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001 - intentionally broad: surfaces user code exceptions to IPython
            return self._synthesize_run_cell_raise(e, args, kwargs)

        if isinstance(result, _EarlyReturn):
            return result.value
        if isinstance(result, _PipelineSyntaxError):
            return self._original_run_cell(raw_cell, *args, **kwargs)

        return self._finalize_cell_execution(
            raw_cell, result.all_metrics, result.buffered_outputs,
            result.badge_display_id, result.hook_start, result.timing_breakdown,
            result.badge_render_time, args, kwargs,
        )

    def _synthesize_run_cell_raise(
        self,
        e: BaseException,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """Re-raise *e* through IPython's run_cell so the kernel reply status
        is "error" while suppressing IPython's duplicate traceback.

        Hook-path only: makes sense when ``_execute_cell`` is itself standing
        in for ``run_cell``.  The ``%%cash`` magic does **not** call this —
        it just lets the exception propagate so IPython's magic-error path
        handles it.
        """
        self.shell.user_ns['__cash_exception__'] = e
        orig_showtb = getattr(self.shell, 'showtraceback', None)
        try:
            self.shell.showtraceback = lambda *a, **kw: None
        except (AttributeError, TypeError):
            logger.debug("Could not suppress IPython showtraceback")
        ipython_error_result = None
        try:
            ipython_error_result = self._original_run_cell("raise __cash_exception__", *args, **kwargs)
        finally:
            try:
                if orig_showtb is not None:
                    self.shell.showtraceback = orig_showtb
                else:
                    with contextlib.suppress(AttributeError, TypeError):
                        del self.shell.showtraceback
            except (AttributeError, TypeError):
                logger.debug("Could not restore showtraceback")
        return ipython_error_result

    def _finalize_cell_execution(
        self,
        raw_cell: str,
        all_metrics: list[ProcessResult],
        buffered_result_outputs: list,
        badge_display_id: str,
        hook_start: float,
        timing_breakdown: TimingBreakdown,
        badge_render_time: float,
        args: tuple,
        kwargs: dict,
        delegate_to_run_cell: bool = True,
    ) -> Any:
        """Post-process a cell execution: flush analytics, record metrics, render final badge.

        Tail phase shared by ``_execute_cell`` (hook-driven `%cash_on`) and
        the ``cash`` cell magic (`%%cash`).  Handles analytics flushing,
        session statistics updates, provenance recording, audit logging,
        debug output, and the final badge render.

        ``delegate_to_run_cell``: when True (the default, used by the hook),
        ends by calling ``self._original_run_cell("pass", *args, **kwargs)``
        so IPython's internal bookkeeping (execution count, history) stays
        in sync.  ``%%cash`` passes False — it runs inside an IPython magic
        whose own dispatcher already keeps that bookkeeping consistent.
        """
        hook_total = time.time() - hook_start

        # Flush buffered analytics events to avoid stale data
        try:
            self._statement_processor.analytics_manager.flush()
        except (AttributeError, TypeError, OSError):
            logger.debug("Analytics flush failed in _execute_cell")

        # Post-execution: auto-track any newly imported local modules.
        # On the FIRST run of `import trackmod`, auto_track_local_imports couldn't
        # track it pre-execution because it wasn't in sys.modules yet.  Now that
        # statements have executed, the module IS in sys.modules and we can track
        # its file mtime so future runs detect source changes.
        ft = self._statement_processor.function_tracker
        try:
            ft.auto_track_local_imports(raw_cell)
        except (ImportError, AttributeError, OSError, TypeError) as exc:
            logger.debug("Post-execution auto-track failed: %s", exc)

        # Update last cell metrics for %cash_status
        self._update_last_cell_metrics(all_metrics, hook_total)

        # Update session-wide statistics
        self._update_session_stats(all_metrics)

        self._record_observability(all_metrics)

        if self._debug:
            print(f"[TIMING_PROXY] PROXY TOTAL: {hook_total*1000:.1f}ms")
            print(f"[TIMING_PROXY] Badge init: {timing_breakdown.get('badge_init', 0)*1000:.1f}ms")
            print(f"[TIMING_PROXY] Upstream check: {timing_breakdown.get('upstream_check', 0)*1000:.1f}ms")
            print(f"[TIMING_PROXY] Badge progress renders: {badge_render_time*1000:.1f}ms")

        # Now that all debug prints are done, show the Buffered Result (if any)
        for output in buffered_result_outputs:
            if isinstance(output, dict) and 'data' in output:
                publish_display_data(data=output['data'], metadata=output.get('metadata', {}))
            else:
                display(output)

        # Update Interactive Badge with final metrics
        if self._badge_mode == 'html':
            self._render_interactive_badge(all_metrics, display_id=badge_display_id, cell_total_time=hook_total, timing_breakdown=timing_breakdown)
        elif self._badge_mode == 'print':
            self._print_text_badge(all_metrics, cell_total_time=hook_total)

        # Delegate to original run_cell with "pass" so IPython keeps its
        # execution count + history consistent.  Skipped when called from
        # `%%cash` (its dispatcher already handles that bookkeeping).
        if delegate_to_run_cell:
            return self._original_run_cell("pass", *args, **kwargs)
        return None

    def _update_last_cell_metrics(self, all_metrics: list[ProcessResult], hook_total: float) -> None:
        """Compute and store ``_last_cell_metrics`` for ``%cash_status``."""
        statuses = [m.get('status') for m in all_metrics if m.get('status')]
        if all(s == CacheStatus.RESTORED for s in statuses) and statuses:
            overall_status = 'RESTORED'
        elif all(s == CacheStatus.COMPUTED for s in statuses) and statuses:
            overall_status = 'COMPUTED'
        elif all(s == CacheStatus.SKIPPED for s in statuses) and statuses:
            overall_status = 'SKIPPED'
        elif statuses:
            overall_status = 'MIXED'
        else:
            overall_status = None

        self._last_cell_metrics = {
            'statements': [
                {
                    'code': m.get('code', '')[:100],
                    'status': m.get('status'),
                    'execution_time': m.get('execution_time', 0.0),
                    'saved_time': m.get('saved_time', 0.0),
                    'outputs': m.get('restored_vars', []),
                    'is_upstream': m.get('is_upstream', False),
                }
                for m in all_metrics
            ],
            'total_time': hook_total,
            'total_restored_time': sum(m.get('saved_time', 0.0) for m in all_metrics),
            'total_computed_time': sum(m.get('execution_time', 0.0) for m in all_metrics if m.get('status') == CacheStatus.COMPUTED),
            'upstream_metrics': [m for m in all_metrics if m.get('is_upstream', False)],
            'status': overall_status,
        }

    def _update_session_stats(self, all_metrics: list[ProcessResult]) -> None:
        """Increment session-wide caching statistics from *all_metrics*."""
        stats = self._session.stats
        stats['cells_executed'] += 1
        for m in all_metrics:
            status = m.get('status')
            if status == CacheStatus.COMPUTED:
                stats['statements_computed'] += 1
                stats['total_compute_time'] += m.get('execution_time', 0.0)
            elif status == CacheStatus.RESTORED:
                stats['statements_restored'] += 1
                stats['total_restored_time'] += m.get('saved_time', 0.0)
                stats['total_time_saved'] += m.get('saved_time', 0.0)
            elif status == CacheStatus.SKIPPED:
                stats['statements_skipped'] += 1

    def _record_observability(self, all_metrics: list[ProcessResult]) -> None:
        """Record provenance + audit entries for each statement in *all_metrics*.

        Walks the metrics list once and fans out to both trackers, sharing
        the per-statement field extraction.  Provenance records one entry
        per output variable; audit records one entry per output variable
        (or a single placeholder keyed by code-prefix when a statement has
        no string-named outputs).
        """
        for m in all_metrics:
            code = m.get('code', '')
            status = m.get('status', 'computed')
            duration_ms = m.get('execution_time', 0.0) * 1000
            # NOTE: never fall back to ``m.get('rich_outputs')`` (or the legacy
            # ``m.get('outputs')`` it replaced) — those hold IPython rich-display
            # objects, NOT variable names. F-01 (commit e739134) fixed the same
            # anti-pattern in the badge view; the audit path needs the same care.
            outputs = m.get('restored_vars', []) or m.get('output_vars', []) or m.get('evaluated_vars', [])
            inputs_list = list(m.get('inputs', []))
            # Outputs may contain rich-display dicts; provenance/audit only
            # care about string variable names.
            var_names = [o for o in (outputs or []) if isinstance(o, str)]

            provenance_status = str(status).lower() if status else 'computed'
            for out_var in var_names:
                self._session.provenance.record(
                    variable=out_var,
                    code=code,
                    inputs=inputs_list,
                    status=provenance_status,
                    duration_ms=duration_ms,
                    lineage_hash=self._tracking_state.variable_lineage.get(out_var, ''),
                    file_deps=list(self._tracking_state.executed_file_deps.get(out_var, [])),
                )

            audit_op = _OP_MAP.get(status, 'cache_operation')
            for out_var in (var_names or [code[:30]]):
                self._session.audit.log(
                    operation=audit_op,
                    variable=out_var,
                    code=code,
                    status='success',
                    duration_ms=duration_ms,
                )

    def _print_text_badge(self, metrics_list: list[ProcessResult], cell_total_time: float | None = None) -> None:
        """Print a plain-text summary of the cell execution (for 'print' badge mode).

        Delegates to :func:`badge_renderer.print_text_badge`.
        """
        _badge.print_text_badge(metrics_list, cell_total_time=cell_total_time)

    def _get_bug_report_context(self) -> dict:
        """Collect runtime environment info for the pre-filled bug report URL."""
        import sys
        try:
            from cash import __version__ as _v
        except Exception:
            _v = "unknown"
        backend = getattr(self._cash_instance, 'backend', None)
        backend_name = type(backend).__name__ if backend else 'unknown'
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        # --- Execution history (what IPython actually ran) ---
        # Prefer input_hist_raw (untransformed magics like %cash_on) over
        # input_hist_parsed / In (which transforms magics to get_ipython() calls).
        # input_hist_raw lives on history_manager, not directly on the shell.
        hm = getattr(self.shell, 'history_manager', None)
        in_history = getattr(hm, 'input_hist_raw', None) if hm else None
        if in_history is None:
            in_history = getattr(self.shell, 'user_ns', {}).get('In', [])
        # Filter empty strings and the 'pass' pseudo-cells that cash injects,
        # then deduplicate consecutive identical cells (from re-running).
        filtered: list[str] = []
        for c in in_history:
            if not c.strip() or c.strip() == 'pass':
                continue
            if filtered and c == filtered[-1]:
                continue
            filtered.append(c)
        exec_history = filtered[-6:]

        # --- Notebook source (actual .ipynb cell contents on disk) ---
        notebook_cells: list[str] = []
        try:
            from .server_discovery import get_notebook_cells
            notebook_cells = get_notebook_cells() or []
        except Exception:
            pass

        return {
            'version': _v,
            'python_version': python_version,
            'backend': backend_name,
            'notebook_history': exec_history,
            'notebook_source': notebook_cells,
        }

    def _render_interactive_badge(self, metrics_list: list[ProcessResult], display_id: str | None = None, status: str = "DONE", current_step: int = 0, total_steps: int = 0, current_code: str | None = None, update_existing: bool = True, cell_total_time: float | None = None, timing_breakdown: dict[str, float] | None = None, _from_thread: bool = False) -> None:
        """Render a clickable interactive badge with detailed execution history.

        Delegates HTML generation to :func:`badge_renderer.render_interactive_badge`
        and handles the IPython display / publish lifecycle.
        """
        html = _badge.render_interactive_badge(
            metrics_list=metrics_list,
            badge_mode=self._badge_mode,
            debug=self._debug,
            display_id=display_id,
            status=status,
            current_step=current_step,
            total_steps=total_steps,
            current_code=current_code,
            cell_total_time=cell_total_time,
            timing_breakdown=timing_breakdown,
            bug_report_context=self._get_bug_report_context(),
        )
        if not html:
            return

        try:
            if _from_thread and display_id:
                # From a background thread, use publish_display_data directly
                # with update=True to send an ``update_display_data`` message.
                # This avoids display()'s bookkeeping which can create duplicate
                # output areas when called from non-main threads.
                publish_display_data(
                    {'text/html': html},
                    transient={'display_id': display_id},
                    update=True,
                )
            elif display_id:
                display(HTML(html), display_id=display_id, update=update_existing)
            else:
                display(HTML(html))
        except (TypeError, ValueError, AttributeError, OSError) as e:
            if self._debug:
                print(f"[BADGE RENDER ERROR] {e}")

    @staticmethod
    def _cash_parse_ttl(line: str) -> int | None:
        """Parse optional TTL value from a %%cash magic line. Returns None if not set."""
        if not line:
            return None
        parts = line.split('=')
        if len(parts) == 2 and parts[0].strip() == 'ttl':
            try:
                return int(parts[1].strip())
            except ValueError:
                logger.warning("Invalid TTL value")
        return None

    @cell_magic
    def cash(self, line: str, cell: str) -> None:
        """Cell magic to cache the execution of a single cell.

        Usage: ``%%cash [ttl=60]``

        Delegates to :meth:`CellExecutor.execute_cell` so this path runs
        the same caching logic as ``%cash_on`` — module-change detection,
        opaque-warning metrics, function-change metrics, and the
        early-return plumbing all apply here too.  The two used to drift;
        the shared executor makes drift structurally impossible.

        Does **not** pass ``original_run_cell`` to the executor — that
        opts out of the IPython-fallback paths.  Exceptions from upstream
        simulation propagate naturally so IPython's magic-error path
        handles them.
        """
        ttl = self._cash_parse_ttl(line)
        saved_ttl = self._global_ttl
        self._global_ttl = ttl
        try:
            result = self._cell_executor.execute_cell(cell)

            if isinstance(result, _EarlyReturn):
                return
            if isinstance(result, _PipelineSyntaxError):
                logger.error("Syntax Error in %%cash cell")
                return

            self._finalize_cell_execution(
                cell, result.all_metrics, result.buffered_outputs,
                result.badge_display_id, result.hook_start, result.timing_breakdown,
                result.badge_render_time, (), {},
                delegate_to_run_cell=False,
            )
        finally:
            self._global_ttl = saved_ttl

    def _show_clean_error(
        self,
        exc: Exception,
        raw_cell: str,
        node: ast.AST,
    ) -> None:
        """Display an exception with a clean traceback pointing to the user's cell.

        Delegates to :func:`error_display.show_clean_error`.
        """
        _show_clean_error_impl(exc, raw_cell, node, self.shell)


