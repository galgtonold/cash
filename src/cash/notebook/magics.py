from __future__ import annotations

"""IPython magic commands for transparent cell caching in Jupyter notebooks."""

import ast
import contextlib
import hashlib
import logging
import os
import pickle
import sys
import time
import types
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
from ..exceptions import AmbiguousCellError
from ..utils import resolve_file_dep_path, safe_text
from . import badge_renderer as _badge
from ._protocols import ShellProtocol
from .analysis import CodeAnalyzer
from .annotations import get_statement_annotations
from .audit import AuditLogger
from .cache_status import CacheStatus
from .control_structures import ControlStructureProcessor, is_control_structure
from .error_display import show_clean_error as _show_clean_error_impl
from .magic_admin import CashAdminMagicsMixin
from .module_invalidator import ModuleInvalidator
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

class _EarlyReturn:
    """Sentinel wrapper returned by ``_execute_cell_statements`` when a
    statement error forces an early exit through ``_original_run_cell``.

    ``_execute_cell`` checks for this wrapper and propagates the stored
    IPython result unchanged.
    """
    __slots__ = ('value',)

    def __init__(self, value: Any) -> None:
        self.value = value

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
        self._cell_code_map = {}  # cell_id -> cell code (for interactive toggles)
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
            compute_hash_fn=self._compute_hash,
            tracking_state=self._tracking_state,
        )

        self._statement_processor = StatementProcessor(
            shell,
            cash_instance,
            debug=self._debug,
            compute_hash_fn=self._compute_hash,
            calculate_memory_fn=self._calculate_memory_size,
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

    @staticmethod
    def _hash_dataframe_or_series(obj: Any, type_name: str) -> str:
        """Hash a pandas DataFrame or Series using shape + dtypes + data sample."""
        shape_str = f"{obj.shape}"
        dtypes_str = str(obj.dtypes.to_dict()) if type_name == 'DataFrame' else str(obj.dtype)
        try:
            sample = str(obj.head(5).values.tobytes() if len(obj) > 0 else b'')
        except (TypeError, ValueError, AttributeError):
            sample = str(obj.head(5))
        combined = f"{shape_str}:{dtypes_str}:{sample}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    @staticmethod
    def _hash_collection(obj: Any) -> str:
        """Hash a list/tuple/dict/set/frozenset — sampling large ones to avoid O(n) pickle."""
        n = len(obj)
        if n <= 200:
            return hashlib.sha256(pickle.dumps(obj)).hexdigest()
        if isinstance(obj, (list, tuple)):
            combined = f"list:{n}:{repr(obj[:5])}:{repr(obj[-5:])}"
        elif isinstance(obj, dict):
            combined = f"dict:{n}:{repr(sorted(obj.keys())[:10])}"
        else:
            combined = f"set:{n}:{repr(sorted(obj)[:10])}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def _compute_hash(self, obj: Any) -> str:
        """Compute a hash for an object using type-specific methods with explicit fallbacks.

        Strategy order:
        1. Type-specific fast hash (DataFrame/ndarray/collections)
        2. Generic pickle hash
        3. Identity hash (always succeeds)
        """
        _HASH_ERRORS = (TypeError, ValueError, AttributeError, pickle.PicklingError)
        type_name = type(obj).__name__

        # Strategy 1: type-specific fast hash
        try:
            if type_name in ('DataFrame', 'Series'):
                return self._hash_dataframe_or_series(obj, type_name)
            if type_name == 'ndarray':
                shape_str = str(obj.shape)
                dtype_str = str(obj.dtype)
                sample = str(obj.flat[:100].tobytes() if obj.size > 0 else b'')
                combined = f"{shape_str}:{dtype_str}:{sample}"
                return hashlib.sha256(combined.encode('utf-8')).hexdigest()
            if isinstance(obj, (list, tuple, dict, set, frozenset)):
                return self._hash_collection(obj)
            return hashlib.sha256(pickle.dumps(obj)).hexdigest()
        except _HASH_ERRORS as exc:
            logger.debug("Primary hash failed for %s: %s", type_name, exc)

        # Strategy 2: plain pickle fallback
        try:
            return hashlib.sha256(pickle.dumps(obj)).hexdigest()
        except (TypeError, pickle.PicklingError):
            pass

        # Strategy 3: identity hash (always succeeds)
        return hashlib.sha256(str(id(obj)).encode('utf-8')).hexdigest()

    def _calculate_memory_size(self, variables_dict: dict[str, Any]) -> int:
        """
        Calculate the total memory size of output variables using type-specific methods.
        This is much faster than pickle.dumps() on the entire payload.

        Args:
            variables_dict: Dictionary of variable names to their values

        Returns:
            Total memory size in bytes
        """
        total_size = 0

        for _var_name, value in variables_dict.items():
            try:
                # Check type and use appropriate method
                type_name = type(value).__name__

                # DataFrame: Use built-in memory_usage (fast and accurate)
                if type_name == 'DataFrame':
                    try:
                        total_size += value.memory_usage(deep=True).sum()
                        continue
                    except (TypeError, AttributeError):
                        pass  # Fall through to other methods

                # NumPy array: Use nbytes
                elif type_name == 'ndarray':
                    try:
                        total_size += value.nbytes
                        continue
                    except (TypeError, AttributeError):
                        pass

                # Series: Similar to DataFrame
                elif type_name == 'Series':
                    try:
                        total_size += value.memory_usage(deep=True)
                        continue
                    except (TypeError, AttributeError):
                        pass

                # For other types, use sys.getsizeof with recursion for containers
                total_size += self._recursive_getsizeof(value)

            except (TypeError, ValueError, RecursionError):
                # If all else fails, try pickle as last resort
                try:
                    total_size += len(pickle.dumps(value))
                except (TypeError, pickle.PicklingError):
                    # If even pickle fails, estimate with sys.getsizeof
                    total_size += sys.getsizeof(value)

        return total_size

    def _recursive_getsizeof(self, obj: Any, seen: set[int] | None = None) -> int:
        """
        Recursively calculate size of an object including its contents.
        More accurate than plain sys.getsizeof() for containers.
        """
        size = sys.getsizeof(obj)

        if seen is None:
            seen = set()

        obj_id = id(obj)
        if obj_id in seen:
            return 0

        seen.add(obj_id)

        # Handle containers
        if isinstance(obj, dict):
            size += sum(self._recursive_getsizeof(k, seen) + self._recursive_getsizeof(v, seen)
                       for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            size += sum(self._recursive_getsizeof(item, seen) for item in obj)

        return size

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

    def _check_and_reexecute_upstream_cells(self, cell_code: str, required_inputs: set, progress_callback: Callable[..., None] | None = None) -> tuple[list[ProcessResult], float, float]:
        """
        Check if any upstream cells (that define required inputs) have changed.
        If so, re-execute them before proceeding with current cell.

        Args:
            cell_code: The code content of the current cell
            required_inputs: Set of variable names this cell requires
            progress_callback: Optional callback(metrics_so_far, current_stmt_code)
                called during upstream re-execution so the badge can show progress.

        Returns a list of metrics for any executed or restored upstream statements.
        """
        # Delegate to upstream checker
        upstream_metrics, total_restore_time, total_execution_time = self._upstream_checker.check_and_reexecute(
            cell_code,
            required_inputs,
            self._statement_processor.process_statement,
            self._global_ttl,
            cell_id=self._current_cell_id,
            progress_callback=progress_callback,
            control_structure_callback=self._control_structure_processor.process,
        )

        return upstream_metrics, total_restore_time, total_execution_time

    def _ensure_state_for_inputs(self, cell_code: str, progress_callback: Callable[..., None] | None = None) -> tuple[list[ProcessResult], float, float]:
        """
        Ensure that all required inputs for a cell are available in memory.
        Handles upstream checking and state restoration.

        Args:
            cell_code: The code content of the current cell
            progress_callback: Optional callback(metrics_so_far, current_stmt_code)
                called during upstream re-execution for badge progress.

        Returns:
            Tuple of (upstream_metrics, total_restore_time, total_execution_time)
        """
        try:
            if self._debug:
                 print(f"[ENSURE_STATE_DEBUG] Cell code: {cell_code[:50]}...")

            # Analyze cell dependencies
            inputs, outputs = CodeAnalyzer.analyze_code_block(cell_code)

            if self._debug:
                print(f"[ENSURE_STATE_DEBUG] Analyzed inputs: {inputs}")
                print(f"[ENSURE_STATE_DEBUG] Analyzed outputs: {outputs}")
                print(f"[ENSURE_STATE_DEBUG] Current user_ns keys (first 10): {list(self.shell.user_ns.keys())[:10]}")

            # Attempt to restore missing inputs from cache first
            # This is a fast path to avoid re-execution if possible
            total_restore_time = 0
            upstream_metrics = []

            for var_name in inputs:
                if var_name not in self.shell.user_ns:
                    start_restore = time.time()
                    try:
                        metrics = self._restore_variable(var_name)
                        total_restore_time += (time.time() - start_restore)
                        if metrics:
                            upstream_metrics.extend(metrics)
                    except NameError:
                        # If _restore_variable raises NameError, it means it couldn't find a source
                        # and we should proceed, hoping upstream re-execution will provide it.
                        if self._debug:
                            print(f"[STATE] Could not restore '{var_name}' from cache. Hoping for upstream re-execution.")

            # Check if any upstream cells/statements have changed compared to what we have in memory.
            # If the notebook file dictates a different lineage than what we restored, we must re-execute.
            reexec_metrics, upstream_restore_time, total_execution_time = self._check_and_reexecute_upstream_cells(cell_code, inputs, progress_callback=progress_callback)
            total_restore_time += upstream_restore_time  # Add upstream restore time to our cache restore time

            # Usually check_and_reexecute returns re-executed stuff or NEWLY restored stuff via virtual restore.
            # So simple list extend is fine.
            upstream_metrics.extend(reexec_metrics)

        except (RuntimeError, SyntaxError, AmbiguousCellError):
            raise
        except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.debug("[STATE] Error in state restoration logic: %s", e)
            raise

        return upstream_metrics, total_restore_time, total_execution_time

    def _is_available_in_builtins(self, var_name: str) -> bool:
        """Check if variable is available in __builtins__."""
        builtins_ns = self.shell.user_ns.get('__builtins__')
        if not builtins_ns:
            return False

        if isinstance(builtins_ns, dict):
            return var_name in builtins_ns
        if isinstance(builtins_ns, types.ModuleType):
            return hasattr(builtins_ns, var_name)
        return False

    def _validate_file_deps(self, var_name: str, metadata: dict) -> None:
        """Raise NameError if any file dependency has changed since caching.

        Called before restoring a cached variable — if files have changed the
        cached value is stale and must be recomputed.
        """
        for fpath, stored in metadata.get('file_dependencies', {}).items():
            resolved = resolve_file_dep_path(fpath)
            if resolved is None:
                if self._debug:
                    print(f"[STATE] Cannot restore '{var_name}': file dependency missing: {fpath}")
                raise NameError(f"name '{var_name}' is not defined (file dependency missing)")
            # Tolerate both the new {'mtime': ..., 'size': ...} form and the
            # legacy bare-float form left over from older cache entries.
            if isinstance(stored, dict):
                stored_mtime = float(stored.get('mtime', 0.0))
                stored_size = stored.get('size')
            else:
                stored_mtime = float(stored)
                stored_size = None
            try:
                cur_stat = os.stat(resolved)
            except OSError:
                raise NameError(f"name '{var_name}' is not defined (file dependency missing)")
            delta = abs(cur_stat.st_mtime - stored_mtime)
            if delta > 0.01:
                if self._debug:
                    print(f"[STATE] Cannot restore '{var_name}': file dependency mtime changed: {resolved} (delta={delta:.4f}s)")
                raise NameError(f"name '{var_name}' is not defined (file dependency changed)")
            if stored_size is not None and cur_stat.st_size != stored_size:
                if self._debug:
                    print(f"[STATE] Cannot restore '{var_name}': file dependency size changed: {resolved}")
                raise NameError(f"name '{var_name}' is not defined (file dependency changed)")

    def _restore_tracking_state(self, var_name: str, metadata: dict, restored_vars: dict) -> None:
        """Update TrackingState after writing a restored variable into user_ns."""
        restored_hash = self._compute_hash(restored_vars[var_name])
        hashes = self._tracking_state.variable_hashes
        if var_name not in hashes:
            hashes[var_name] = set()
        hashes[var_name].add(restored_hash)

        output_lineages = metadata.get('output_lineages', {})
        if var_name in output_lineages:
            self._tracking_state.lineage.record(
                var_name,
                output_lineages[var_name],
                value=self.shell.user_ns.get(var_name),
            )

        stored_code = metadata.get('code', metadata.get('cell_code'))
        if stored_code:
            self._tracking_state.executed_cell_codes[var_name] = stored_code

        stored_hash = metadata.get('source_hash', metadata.get('cell_hash'))
        if stored_hash:
            cell_hashes = self._tracking_state.executed_cell_hashes
            if var_name not in cell_hashes:
                cell_hashes[var_name] = set()
            elif isinstance(cell_hashes[var_name], str):
                cell_hashes[var_name] = {cell_hashes[var_name]}
            cell_hashes[var_name].add(stored_hash)

        file_deps = metadata.get('file_dependencies', {})
        if file_deps:
            file_dep_set = self._tracking_state.executed_file_deps
            if var_name not in file_dep_set:
                file_dep_set[var_name] = set()
            file_dep_set[var_name].update(file_deps.keys())

    def _build_restore_metric(self, var_name: str, metadata: dict, restored_vars: dict) -> ProcessResult:
        """Build a ProcessResult entry for a successfully restored variable."""
        saved_time = metadata.get('execution_time', 0.0)
        source = metadata.get('source', metadata.get('storage', 'Disk'))
        if isinstance(source, list):
            source = source[0] if source else 'Disk'
        return {
            'code': metadata.get('code', f"# defined {var_name}"),
            'status': CacheStatus.RESTORED,
            'execution_time': 0.0,
            'total_time': saved_time,
            'saved_time': saved_time,
            'error': None,
            'restored_vars': list(restored_vars.keys()),
            'uncacheable_reasons': [],
            'source': source,
            'is_upstream': True,
            'storage': [source],
        }

    def _ensure_inputs_current(
        self, var_name: str, metadata: dict, restored_metrics: list[ProcessResult],
    ) -> None:
        """Recursively restore any stale input variables required by var_name."""
        for input_var in metadata.get('inputs', []):
            if input_var in (var_name, 'get_ipython', '__builtins__'):
                continue
            if input_var not in self.shell.user_ns:
                restored_metrics.extend(self._restore_variable(input_var))
            elif input_var in self._tracking_state.variable_hashes:
                current_hash = self._compute_hash(self.shell.user_ns.get(input_var))
                if current_hash not in self._tracking_state.variable_hashes[input_var]:
                    restored_metrics.extend(self._restore_variable(input_var))

    def _fetch_cached_payload(self, var_name: str) -> tuple[dict, dict] | None:
        """Look up cache entry for var_name and validate it.

        Returns ``(metadata, cached_data)`` if found and valid, or raises/returns
        ``None`` to signal that the caller should return early.

        Raises ``NameError`` when var_name has no known source and is not a builtin.
        """
        if var_name not in self._tracking_state.variable_sources:
            if self._is_available_in_builtins(var_name):
                if self._debug:
                    print(f"[STATE] '{var_name}' not in cache, but found in built-ins. Using built-in.")
                return None
            if self._debug:
                print(f"[STATE] Cannot restore '{var_name}': no cached source found")
            raise NameError(f"name '{var_name}' is not defined")

        cache_key = self._tracking_state.variable_sources[var_name]
        metadata, cached_data = self._cash_instance.backend.get(cache_key)
        if not cached_data:
            if self._debug:
                print(f"[STATE] Cannot restore '{var_name}': cache miss for key {cache_key[:16]}...")
            return None

        self._validate_file_deps(var_name, metadata)
        return metadata, cached_data

    def _restore_variable(self, var_name: str) -> list[ProcessResult]:
        """Restore a single variable from cache, recursively ensuring dependencies first."""
        result = self._fetch_cached_payload(var_name)
        if result is None:
            return []
        metadata, cached_data = result
        restored_metrics: list[ProcessResult] = []

        try:
            if not isinstance(cached_data, dict) or 'variables' not in cached_data:
                if self._debug:
                    print(f"[STATE] Invalid payload format for '{var_name}'")
                return []

            self._ensure_inputs_current(var_name, metadata, restored_metrics)

            restored_vars = cached_data['variables']
            if var_name in restored_vars:
                self.shell.user_ns[var_name] = restored_vars[var_name]
                self._restore_tracking_state(var_name, metadata, restored_vars)
                if self._debug:
                    print(f"[STATE] Restored '{var_name}' from cache")
                restored_metrics.append(self._build_restore_metric(var_name, metadata, restored_vars))
            elif self._debug:
                print(f"[STATE] Variable '{var_name}' not in cached payload")

        except (KeyError, TypeError, ValueError, AttributeError, OSError, pickle.UnpicklingError) as e:
            logger.debug("[STATE] Error restoring '%s': %s", var_name, e)
            if self._debug:
                print(f"[STATE] Error restoring '{var_name}': {e}")

        return restored_metrics

    # ------------------------------------------------------------------
    # _execute_cell helpers — each encapsulates one concern of the
    # cell execution pipeline.  Keeping these small and focused makes
    # _execute_cell itself a readable orchestration sequence.
    # ------------------------------------------------------------------

    def _extract_cell_id_and_notebook_path(self) -> None:
        """Resolve cell_id and notebook path from IPython kernel metadata.

        Must run BEFORE the upstream check so the notebook path is available
        for reading upstream cells.  Sets ``self._current_cell_id`` as a
        side-effect.
        """
        try:
            cell_id = self._cell_id_from_parent_metadata(self.shell)
            self._current_cell_id = cell_id
            self._maybe_seed_notebook_path(cell_id)

            if self._debug:
                if cell_id:
                    print(f"[PROXY_CELL_ID] Captured cell_id early: {cell_id}")
                else:
                    print("[PROXY_CELL_ID] No cell_id in parent metadata")
        except (AttributeError, TypeError, KeyError, RuntimeError) as e:
            if self._debug:
                print(f"[PROXY_CELL_ID] Could not capture cell_id early: {e}")

    def _init_cell_timing_and_badge(self, badge_display_id: str) -> TimingBreakdown:
        """Set up timing tracking and render the initial 'RUNNING' badge.

        Returns the ``timing_breakdown`` dict used to accumulate phase timings.
        """
        timing_breakdown: TimingBreakdown = {}
        cell_start = time.time()

        self._badge_cell_start_time = cell_start
        self._last_badge_render_time = 0.0

        t_badge_init = time.time()
        if self._badge_mode == 'html':
            self._render_interactive_badge(
                [], display_id=badge_display_id,
                status="RUNNING", update_existing=False,
            )
        timing_breakdown['badge_init'] = time.time() - t_badge_init
        return timing_breakdown

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
                notification = {
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

    def _resolve_upstream_state(
        self,
        raw_cell: str,
        pre_upstream_metrics: list[ProcessResult],
        badge_display_id: str,
        timing_breakdown: TimingBreakdown,
        args: tuple,
        kwargs: dict,
    ) -> tuple[list[ProcessResult], float, float] | _EarlyReturn:
        """Run upstream dependency checking and state restoration.

        Returns ``(upstream_metrics, restore_time, exec_time)`` on success, or
        an ``_EarlyReturn`` wrapping the original runner's result if an
        unrecoverable error forced a fallback.
        """
        t_ensure = time.time()

        # Progress callback for badge updates during upstream re-execution
        def _upstream_progress_cb(
            upstream_metrics_so_far: list,
            current_stmt_code: str,
            current_step: int | None = None,
            total_steps: int | None = None,
        ) -> None:
            combined = pre_upstream_metrics + upstream_metrics_so_far
            upstream_label = f"↑ {current_stmt_code}" if current_stmt_code else current_stmt_code
            self._maybe_progress_badge(
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
            if isinstance(e, SyntaxError):
                self._render_interactive_badge([], display_id=badge_display_id, status="DONE")
                return _EarlyReturn(self._original_run_cell(raw_cell, *args, **kwargs))
            if isinstance(e, (RuntimeError, AmbiguousCellError)):
                error_code = f"raise {type(e).__name__}('''{str(e)}''')"
                self._render_interactive_badge([], display_id=badge_display_id, status="DONE")
                return _EarlyReturn(self._original_run_cell(error_code, *args, **kwargs))
            logger.error("Cash auto-caching failed: %s. Falling back to normal execution.", e)
            self._render_interactive_badge([], display_id=badge_display_id, status="DONE")
            return _EarlyReturn(self._original_run_cell(raw_cell, *args, **kwargs))

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

    def _execute_cell(self, raw_cell: str, *args: Any, **kwargs: Any) -> Any:
        """Proxy for interactiveshell.run_cell to implement caching.

        Orchestrates the cell execution pipeline in distinct phases:
        1. Guard / benchmark dispatch
        2. Cell ID & notebook path resolution
        3. Badge & timing initialisation
        4. Module change detection & lineage invalidation
        5. Upstream dependency resolution
        6. Pre-execution notification assembly
        7. Statement-by-statement execution (delegated to _execute_cell_statements)
        8. Finalisation & badge rendering (delegated to _finalize_cell_execution)
        """
        if not self._auto_cache_enabled:
            return self._original_run_cell(raw_cell, *args, **kwargs)

        # Record raw cell text for bug-report history (before any processing)
        self._execution_history.append(raw_cell)

        # 1. Benchmark dispatch (one-shot)
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

        # 2. Early cell_id capture & notebook path discovery
        self._extract_cell_id_and_notebook_path()

        # 3. Badge & timing initialisation
        badge_display_id = str(uuid.uuid4())
        timing_breakdown = self._init_cell_timing_and_badge(badge_display_id)

        hook_start = time.time()
        if self._debug:
            print(f"[TIMING_PROXY] Start cached_run_cell: {datetime.now().strftime('%H:%M:%S.%f')}")

        # 4. Module change detection (must precede upstream check)
        pre_upstream_metrics = self._detect_module_changes(raw_cell)

        # 5. Upstream dependency resolution
        upstream_result = self._resolve_upstream_state(
            raw_cell, pre_upstream_metrics, badge_display_id,
            timing_breakdown, args, kwargs,
        )
        if isinstance(upstream_result, _EarlyReturn):
            return upstream_result.value

        upstream_metrics, _total_restore_time, _total_execution_time = upstream_result

        # 6. Parse AST & assemble pre-execution notifications
        try:
            tree = ast.parse(raw_cell)
        except SyntaxError:
            self._render_interactive_badge([], display_id=badge_display_id, status="DONE")
            return self._original_run_cell(raw_cell, *args, **kwargs)

        all_metrics = self._build_pre_execution_notifications(
            raw_cell, pre_upstream_metrics, upstream_metrics,
        )

        if self._debug:
            print("[TIMING_PROXY] Start executing statements...")

        # 7. Execute/cache each statement
        t_process = time.time()
        result = self._execute_cell_statements(
            raw_cell, tree, all_metrics, badge_display_id,
            hook_start, timing_breakdown, args, kwargs,
        )
        if isinstance(result, _EarlyReturn):
            return result.value
        all_metrics, buffered_result_outputs, badge_render_time = result

        time.time() - t_process
        timing_breakdown['badge_progress'] = badge_render_time

        # 8. Finalize — aggregate metrics, render badge, replay buffered outputs
        return self._finalize_cell_execution(
            raw_cell, all_metrics, buffered_result_outputs,
            badge_display_id, hook_start, timing_breakdown, badge_render_time,
            args, kwargs,
        )

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
            stmt_code, self._global_ttl, silent=True,
            render_badge=False, annotation=annotation,
            occurrence_index=occurrence_index,
        )
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

    def _handle_stmt_error(
        self,
        e: BaseException,
        raw_cell: str,
        node: ast.stmt,
        all_metrics: list[ProcessResult],
        badge_display_id: str,
        hook_start: float,
        timing_breakdown: TimingBreakdown,
        args: tuple,
        kwargs: dict,
    ) -> _EarlyReturn:
        """Show clean error, finalize badge, and re-raise via IPython kernel."""
        self._show_clean_error(e, raw_cell, node)
        hook_total = time.time() - hook_start
        if self._badge_mode == 'html':
            self._render_interactive_badge(
                all_metrics, display_id=badge_display_id,
                cell_total_time=hook_total, timing_breakdown=timing_breakdown,
                status="DONE",
            )
        elif self._badge_mode == 'print':
            self._print_text_badge(all_metrics, cell_total_time=hook_total)

        # Re-raise through IPython so the kernel reply status is "error".
        # Suppress IPython's own showtraceback to avoid a duplicate traceback.
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
        return _EarlyReturn(ipython_error_result)

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

    def _execute_cell_statements(
        self,
        raw_cell: str,
        tree: ast.Module,
        all_metrics: list[ProcessResult],
        badge_display_id: str,
        hook_start: float,
        timing_breakdown: TimingBreakdown,
        args: tuple,
        kwargs: dict,
    ) -> _EarlyReturn | tuple[list[ProcessResult], list, float]:
        """Iterate over AST statements, executing or caching each one.

        Returns ``(all_metrics, buffered_result_outputs, badge_render_time)``
        on success, or an ``_EarlyReturn`` if a statement raised an error.
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

            occ = stmt_occurrence_counts.get(stmt_code, 0)
            stmt_occurrence_counts[stmt_code] = occ + 1
            annotation = get_statement_annotations(raw_cell, node)
            is_last = (i == len(tree.body) - 1)
            unified_step = upstream_step_count + i + 1

            t_badge_pre = time.time()
            if self._badge_mode == 'html':
                self._render_interactive_badge(
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
                        node, ttl=self._global_ttl, silent=True,
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
                self._maybe_progress_badge(
                    all_metrics, display_id=badge_display_id,
                    step=unified_step + 1, total=total_steps_unified, code=None,
                )
                badge_render_time += time.time() - t_badge

            except Exception as e:  # noqa: BLE001 - intentionally broad: catches user code exceptions
                if isinstance(e, KeyboardInterrupt):
                    raise
                return self._handle_stmt_error(
                    e, raw_cell, node, all_metrics, badge_display_id,
                    hook_start, timing_breakdown, args, kwargs,
                )

        return (all_metrics, buffered_result_outputs, badge_render_time)

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
    ) -> Any:
        """Post-process a cell execution: flush analytics, record metrics, render final badge.

        This is the tail phase of ``_execute_cell`` extracted for readability.
        It handles analytics flushing, session statistics updates, provenance
        recording, audit logging, debug output, and the final badge render.
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

        # Delegate to original run_cell with "pass" to handle bookkeeping without executing code
        return self._original_run_cell("pass", *args, **kwargs)

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

    def _setup_cash_upstream(
        self, cell: str, line: str, badge_display_id: str,
    ) -> list[ProcessResult]:
        """Run upstream dependency check for %%cash, returning upstream metrics."""
        def _progress_cb(upstream_so_far, current_stmt_code, current_step=None, total_steps=None):
            upstream_label = f"↑ {current_stmt_code}" if current_stmt_code else current_stmt_code
            self._maybe_progress_badge(
                upstream_so_far, display_id=badge_display_id,
                step=current_step if current_step is not None else len(upstream_so_far),
                total=total_steps or 0, code=upstream_label,
            )

        try:
            inputs, _outputs = CodeAnalyzer.analyze_code_block(cell)
            full_cell_code = f"%%cash{(' ' + line) if line else ''}\n{cell}"
            upstream_metrics, _, _ = self._check_and_reexecute_upstream_cells(
                full_cell_code, inputs, progress_callback=_progress_cb,
            )
            return upstream_metrics
        except (SyntaxError, KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.debug("[DEBUG] Error checking upstream dependencies: %s", e)
            return []

    @staticmethod
    def _publish_outputs(rich_outputs: list) -> None:
        """Immediately publish each output via display/publish_display_data."""
        for output in rich_outputs:
            if isinstance(output, dict) and 'data' in output:
                publish_display_data(data=output['data'], metadata=output.get('metadata', {}))
            else:
                display(output)

    def _cash_execute_control_struct(
        self,
        node: Any,
        i: int,
        ttl: int | None,
        is_last_statement: bool,
        all_metrics: list,
        buffered_result_outputs: list,
    ) -> None:
        """Execute a control-structure node from %%cash and collect its metrics."""
        if self._debug:
            print("[CONTROL] Detected control structure in %%cash, delegating to ControlStructureProcessor")
        result = self._control_structure_processor.process(node, ttl=ttl, silent=True)

        for metrics in result.metrics:
            if metrics:
                metrics['statement_index'] = i
                all_metrics.append(metrics)
                if not metrics.get('_output_flushed'):
                    if metrics.get('stdout'):
                        print(metrics['stdout'], end='')
                    if metrics.get('stderr'):
                        print(metrics['stderr'], end='', file=sys.stderr)
                rich_outputs = metrics.get('rich_outputs', [])
                if is_last_statement:
                    buffered_result_outputs.extend(rich_outputs)
                else:
                    self._publish_outputs(rich_outputs)

        if self._debug:
            print(f"[CONTROL] Completed: {result.total_iterations} iterations, "
                  f"{result.cached_iterations} cached, {result.computed_iterations} computed")
        if not result.success:
            raise result.error or RuntimeError("Unknown error in control structure execution")

    def _cash_execute_regular_stmt(
        self,
        stmt_code: str,
        annotation: Any,
        occurrence_index: int,
        ttl: int | None,
        is_last_statement: bool,
        i: int,
        upstream_metrics: list,
        badge_display_id: str,
        cell_start: float,
        all_metrics: list,
        buffered_result_outputs: list,
    ) -> None:
        """Execute a regular (non-control-structure) statement from %%cash."""
        metrics = self._statement_processor.process_statement(
            stmt_code, ttl, silent=True, render_badge=False,
            annotation=annotation, occurrence_index=occurrence_index,
        )
        if self._debug:
            print(f"[TIMING_CELL] Statement process time: {(time.time() - cell_start)*1000:.2f}ms")

        if metrics:
            metrics['statement_index'] = i
            all_metrics.append(metrics)
            if metrics.get('stdout'):
                print(metrics['stdout'], end='')
            if metrics.get('stderr'):
                print(metrics['stderr'], end='', file=sys.stderr)
            if metrics.get('status') == CacheStatus.ERROR and metrics.get('error'):
                final_metrics = upstream_metrics + all_metrics
                if self._badge_mode == 'html':
                    self._render_interactive_badge(final_metrics, display_id=badge_display_id)
                elif self._badge_mode == 'print':
                    self._print_text_badge(final_metrics, cell_total_time=time.time() - cell_start)
                raise metrics['error']
            rich_outputs = metrics.get('rich_outputs', [])
            if is_last_statement:
                buffered_result_outputs.clear()
                buffered_result_outputs.extend(rich_outputs)
            else:
                self._publish_outputs(rich_outputs)

    def _cash_execute_stmt(
        self,
        node: Any,
        i: int,
        stmt_code: str,
        annotation: Any,
        occurrence_index: int,
        ttl: int | None,
        is_last_statement: bool,
        upstream_metrics: list,
        upstream_step_count: int,
        badge_display_id: str,
        cell_start: float,
        all_metrics: list,
        buffered_result_outputs: list,
        total_steps_unified: int,
    ) -> None:
        """Pre-badge render then dispatch to control-struct or regular-stmt handler."""
        unified_step = upstream_step_count + i + 1

        if self._badge_mode == 'html':
            self._render_interactive_badge(
                upstream_metrics + all_metrics, display_id=badge_display_id,
                status="RUNNING", current_step=unified_step,
                total_steps=total_steps_unified, current_code=stmt_code,
            )

        if is_control_structure(node):
            self._cash_execute_control_struct(
                node, i, ttl, is_last_statement, all_metrics, buffered_result_outputs,
            )
        else:
            self._cash_execute_regular_stmt(
                stmt_code, annotation, occurrence_index, ttl, is_last_statement,
                i, upstream_metrics, badge_display_id, cell_start,
                all_metrics, buffered_result_outputs,
            )

    @staticmethod
    def _cash_compute_overall_status(all_metrics: list) -> str | None:
        """Return the overall cell cache status from per-statement metrics."""
        statuses = [m.get('status') for m in all_metrics if m.get('status')]
        if not statuses:
            return None
        if all(s == CacheStatus.RESTORED for s in statuses):
            return 'RESTORED'
        if all(s == CacheStatus.COMPUTED for s in statuses):
            return 'COMPUTED'
        if all(s == CacheStatus.SKIPPED for s in statuses):
            return 'SKIPPED'
        return 'MIXED'

    def _cash_finalize_and_badge(
        self,
        final_metrics: list,
        buffered_result_outputs: list,
        badge_display_id: str,
        total_cell_time: float,
    ) -> None:
        """Flush buffered outputs and render the final cell badge."""
        for output in buffered_result_outputs:
            if isinstance(output, dict) and 'data' in output:
                publish_display_data(data=output['data'], metadata=output.get('metadata', {}))
            else:
                display(output)
        if self._badge_mode == 'html':
            self._render_interactive_badge(final_metrics, display_id=badge_display_id)
        elif self._badge_mode == 'print':
            self._print_text_badge(final_metrics, cell_total_time=total_cell_time)

    @cell_magic
    def cash(self, line: str, cell: str) -> None:
        """
        Cell magic to cache the execution of a cell.
        Usage: %%cash [ttl=60]
        """
        # NOTE: Do NOT clear recently_reloaded_modules here — see _execute_cell.

        cell_start = time.time()
        self._badge_cell_start_time = cell_start
        self._last_badge_render_time = 0.0

        t_parse_options = time.time()
        ttl = self._cash_parse_ttl(line)
        parse_options_time = time.time() - t_parse_options

        badge_display_id = str(uuid.uuid4())
        code_key = self._current_cell_id or badge_display_id
        self._cell_code_map[code_key] = cell

        if self._badge_mode == 'html':
            self._render_interactive_badge([], display_id=badge_display_id, status="RUNNING", update_existing=False)

        upstream_metrics = self._setup_cash_upstream(cell, line, badge_display_id)

        t_split = time.time()
        try:
            tree = ast.parse(cell)
        except SyntaxError as e:
            logger.error("Syntax Error: %s", e)
            return
        split_time = time.time() - t_split

        t_exec_all = time.time()
        all_metrics: list = []
        if self._debug:
            print("[TIMING_CELL] Start executing statements...")

        buffered_result_outputs: list = []
        upstream_step_count = len([
            m for m in upstream_metrics
            if m.get('is_upstream', False) and m.get('status') != 'SKIPPED'
        ])
        total_steps_unified = upstream_step_count + len(tree.body)
        stmt_occurrence_counts: dict = {}

        for i, node in enumerate(tree.body):
            try:
                stmt_code = ast.unparse(node)
            except (ValueError, TypeError) as exc:
                logger.debug("Failed to unparse AST node at index %d: %s", i, exc)
                continue

            occ = stmt_occurrence_counts.get(stmt_code, 0)
            stmt_occurrence_counts[stmt_code] = occ + 1
            annotation = get_statement_annotations(cell, node)
            is_last_statement = (i == len(tree.body) - 1)

            self._cash_execute_stmt(
                node, i, stmt_code, annotation, occ, ttl, is_last_statement,
                upstream_metrics, upstream_step_count, badge_display_id, cell_start,
                all_metrics, buffered_result_outputs, total_steps_unified,
            )
            self._maybe_progress_badge(
                upstream_metrics + all_metrics, display_id=badge_display_id,
                step=upstream_step_count + i + 1, total=total_steps_unified, code=stmt_code,
            )

        final_metrics = upstream_metrics + all_metrics
        exec_all_time = time.time() - t_exec_all
        total_cell_time = time.time() - cell_start

        try:
            self._statement_processor.analytics_manager.flush()
        except (AttributeError, TypeError, OSError):
            logger.debug("Analytics flush failed in cell magic")

        if self._debug:
            logger.debug("[CELL TIMING] Parse options: %.1fms | Split statements: %.1fms",
                         parse_options_time * 1000, split_time * 1000)
            logger.debug("[CELL TIMING] Execute all: %.1fms | CELL TOTAL: %.1fms",
                         exec_all_time * 1000, total_cell_time * 1000)

        overall_status = self._cash_compute_overall_status(all_metrics)
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
            'total_time': total_cell_time,
            'total_restored_time': sum(m.get('saved_time', 0.0) for m in all_metrics),
            'total_computed_time': sum(
                m.get('execution_time', 0.0) for m in all_metrics if m.get('status') == CacheStatus.COMPUTED
            ),
            'upstream_metrics': [m for m in all_metrics if m.get('is_upstream', False)],
            'status': overall_status,
        }

        self._cash_finalize_and_badge(final_metrics, buffered_result_outputs, badge_display_id, total_cell_time)

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


