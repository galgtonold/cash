from __future__ import annotations

"""Core statement processing: analysis, cache lookup, execution, and lineage tracking."""

import ast
import contextlib
import hashlib
import logging
import os
import pickle
import sys
import time
import types
from collections.abc import Callable, Generator
from contextlib import contextmanager
from io import StringIO
from typing import Any, TypedDict

from cash.exceptions import CacheBackendError, CacheKeyComputationError, CacheSerializationError
from cash.notebook._protocols import CashInstanceProtocol, ShellProtocol, TrackingState
from cash.notebook.cache_freshness import (
    CacheFreshnessChecker,
    snapshot_file_deps,
)
from cash.notebook.cache_key import CacheKeyContext, compute_cache_key
from cash.notebook.cache_status import CacheStatus, ExecutionResult
from cash.notebook.object_hashing import estimate_object_size
from cash.notebook.purity import is_known_pure, is_pure, is_stateful
from cash.notebook.server_discovery import get_notebook_path
from cash.notebook.statement_file_deps import (
    StatementFileDeps,
    compute_file_hash_component,
    read_module_source_hash,
)
from cash.notebook.statement_restore import StatementRestorer
from cash.utils import resolve_file_dep_path

__all__ = [
    "StatementCacheMetadata",
    "DecoratorCallMetric",
    "ProcessResult",
    "StatementProcessor",
]

# Debug log prefixes — module-level constants for filtering and consistency.
_LOG_PROCESSOR = "[PROCESSOR]"
_LOG_DEBUG = "[DEBUG]"
_LOG_MUTATION = "[MUTATION]"
_LOG_SIDE_EFFECT = "[SIDE_EFFECT]"
_LOG_CACHE_KEY = "[CACHE_KEY]"
_LOG_CACHE_HIT = "[CACHE_HIT_DEBUG]"
_LOG_CACHE = "[CACHE]"
_LOG_CACHE_DEBUG = "[CACHE DEBUG]"
_LOG_FILE_DEPS = "[FILE_DEPS]"
_LOG_FILE_HASH = "[FILE_HASH]"
_LOG_MODULE_HASH = "[MODULE_HASH]"
_LOG_TIMING = "[TIMING]"
_LOG_PURITY = "[PURITY]"
_LOG_OPTIMIZATION = "[OPTIMIZATION]"
_LOG_FORBIDDEN = "[FORBIDDEN]"
_LOG_ANNOTATION = "[ANNOTATION]"

_COST_MODEL_KEYS = (
    'cost_model_size_bytes',
    'cost_model_restore_seconds',
    'cost_model_type_name',
    'cost_model_family',
)

class _ProcessResultRequired(TypedDict):
    """Keys that are always present in a :class:`ProcessResult`."""

    status: CacheStatus
    execution_time: float
    total_time: float
    saved_time: float
    error: Exception | None
    restored_vars: list[str]
    code: str
    uncacheable_reasons: list[str]

class DecoratorCallMetric(TypedDict, total=False):
    """Metrics for a single ``@cash.cache`` decorated function call."""

    func_name: str
    cache_hit: bool
    execution_time: float

class StatementCacheMetadata(TypedDict, total=False):
    """Metadata stored alongside cached statement results."""

    timestamp: float
    inputs: list[str]
    outputs: list[str]
    execution_time: float
    source_hash: str
    code: str
    key: str
    # path -> {"mtime": float, "size": int}.  Older cache entries may use
    # the legacy bare-float form (path -> mtime); helpers below tolerate both.
    file_dependencies: dict[str, dict[str, float]]
    force_persist: bool
    output_lineages: dict[str, str]
    storage: list[str]
    source: str
    skipped_reason: str
    metadata_only: bool
    cost_model_size_bytes: int
    cost_model_restore_seconds: float
    cost_model_type_name: str
    cost_model_family: str

class ProcessResult(_ProcessResultRequired, total=False):
    """Typed dictionary for the return value of ``StatementProcessor.process_statement()``.

    Required keys (inherited from ``_ProcessResultRequired``) are always
    present.  Optional keys are added during execution depending on cache
    status.
    """

    # --- Set when available ---
    outputs: list[str]
    storage: list[str]
    _output_flushed: bool
    control_type: str
    body_statements: list[str]
    # --- Added by process_statement() at runtime ---
    source: str
    stdout: str
    stderr: str
    decorator_calls: list[DecoratorCallMetric]
    evaluated_vars: list[str]
    skipped_reason: str
    is_upstream: bool
    inputs: list[str]
    output_vars: list[str]
    loop_vars: dict[str, Any]
    control_context: str
    branch_label: str
    changed_functions: list[str]
    changed_modules: dict[str, str]
    cost_model_size_bytes: int
    cost_model_restore_seconds: float
    cost_model_type_name: str
    cost_model_family: str

logger = logging.getLogger(__name__)

_KNOWN_PICKLABLE_TYPE_NAMES = frozenset({
    'DataFrame', 'Series', 'ndarray',
    'int', 'float', 'str', 'bool', 'bytes', 'NoneType',
    'list', 'dict', 'tuple', 'set', 'frozenset',
    'int64', 'float64', 'int32', 'float32',
    'Timestamp', 'Timedelta', 'DatetimeIndex',
})

def _config_float(config: Any, attr: str, default: float) -> float:
    """Read a float-valued config attribute defensively.

    Returns ``default`` when the config is None, missing the attribute,
    or holding a value that can't be converted to float (e.g. a
    MagicMock in tests).
    """
    if config is None:
        return default
    try:
        return float(getattr(config, attr, default))
    except (TypeError, ValueError):
        return default


class _TeeWriter:
    """A writer that sends output to both a real stream and a list buffer.

    Output is forwarded to the real stream (e.g. Jupyter's IOPub) in batches
    controlled by a time-based flush policy.  This avoids the O(n²) behaviour
    of flushing after every single ``write()`` call — which, in Jupyter, sends
    one ZMQ message per flush, causing extreme slowdown for tight loops with
    many print calls.

    Instead we:
    * Always call ``self._real.write(s)`` so the data enters the kernel's
      output buffer immediately.
    * Only call ``self._real.flush()`` if ≥ ``_FLUSH_INTERVAL_S`` seconds
      have elapsed since the last flush.  Jupyter's own ``OutStream`` uses a
      similar strategy (~200 ms batching).
    * Accumulate text in a plain Python list (O(1) append) and join once at
      the end for the metrics/cache record.
    """

    _FLUSH_INTERVAL_S = 0.1  # seconds – matches ipykernel's default batch interval

    def __init__(self, real_stream: Any, chunks: list[str]) -> None:
        self._real = real_stream
        self._chunks = chunks
        self._last_flush = time.monotonic()

    def write(self, s: str) -> int:
        self._real.write(s)
        self._chunks.append(s)
        now = time.monotonic()
        if now - self._last_flush >= self._FLUSH_INTERVAL_S:
            self._real.flush()
            self._last_flush = now
        return len(s)

    def flush(self) -> None:
        self._real.flush()
        self._last_flush = time.monotonic()

    def getvalue(self) -> str:
        """Return all accumulated text."""
        return "".join(self._chunks)

    # Forward attribute access (encoding, fileno, etc.) to the real stream
    # so libraries that introspect the stream object still work.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

@contextmanager
def _tee_output() -> Generator[Any, None, None]:
    """Context manager that tees stdout/stderr to both the real stream and a buffer.

    Yields a ``CapturedOutput``-compatible object whose ``.stdout`` and
    ``.stderr`` attributes contain the recorded text, while everything
    written during the block also appears on the real streams immediately.

    Performance: uses batched flushes (~100 ms) so that 50 000+ print calls
    complete in roughly the same time as native Python/Jupyter output.
    """
    class TeedOutput:
        def __init__(self):
            self.stdout = ""
            self.stderr = ""
            self.outputs = []  # rich outputs not supported in tee mode

    teed = TeedOutput()
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    old_stdout, old_stderr = sys.stdout, sys.stderr

    sys.stdout = _TeeWriter(old_stdout, stdout_chunks)
    sys.stderr = _TeeWriter(old_stderr, stderr_chunks)
    try:
        yield teed
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        teed.stdout = "".join(stdout_chunks)
        teed.stderr = "".join(stderr_chunks)
        sys.stdout, sys.stderr = old_stdout, old_stderr

try:
    from IPython.display import HTML, display, publish_display_data
    from IPython.utils.io import capture_output
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False
    @contextmanager
    def capture_output(stdout: bool = True, stderr: bool = True, display: bool = True) -> Generator[Any, None, None]:
        """Fallback capture_output for when IPython is not available."""
        class CapturedOutput:
            def __init__(self):
                self.stdout = ""
                self.stderr = ""
                self.outputs = []
            def show(self):
                if self.stdout:
                    print(self.stdout, end='')
                if self.stderr:
                    print(self.stderr, end='', file=sys.stderr)

        captured = CapturedOutput()
        old_stdout, old_stderr = sys.stdout, sys.stderr

        if stdout:
            sys.stdout = StringIO()
        if stderr:
            sys.stderr = StringIO()

        try:
            yield captured
            if stdout:
                captured.stdout = sys.stdout.getvalue()
            if stderr:
                captured.stderr = sys.stderr.getvalue()
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    def publish_display_data(data, metadata=None):
        """Fallback publish_display_data."""

    def display(*objs, **kwargs):
        for obj in objs:
            print(obj)

from ..analytics import AnalyticsManager
from .analysis import CodeAnalyzer
from .annotations import CacheAnnotation
from .function_tracker import FunctionTracker
from .cacheability import StatementAnalysis, analyze_statement
from .cacheability_decision import decide_cacheability
from .purity import analyze_function_purity
from .randomness import (
    RandomnessDetector,
    capture_rng_state,
    restore_rng_state,
)


class StatementProcessor:
    """
    Processes and caches individual Python statements.

    This class handles the complete lifecycle of statement execution:
    1. Analyze code to detect inputs and outputs
    2. Compute cache key from code and input hashes
    3. Check cache for existing results
    4. Execute statement if cache miss
    5. Capture outputs and store in cache
    6. Track variable lineage for dependency checking

    Attributes:
        shell: IPython shell instance
        cash_instance: Cash backend for cache storage
        debug: Enable debug output
        compute_hash_fn: Function to compute variable hashes
    """

    def __init__(
        self,
        shell: ShellProtocol,
        cash_instance: CashInstanceProtocol,
        debug: bool = False,
        compute_hash_fn: Callable[[Any], str] | None = None,
        tracking_state: TrackingState | None = None,
    ) -> None:
        self.shell: ShellProtocol = shell
        self.cash_instance: CashInstanceProtocol = cash_instance
        self.debug = debug
        self.compute_hash: Callable[[Any], str] | None = compute_hash_fn

        self.analytics_manager = AnalyticsManager()

        self.randomness_detector = RandomnessDetector()

        # Document: function_tracker must be explicitly passed to UpstreamChecker
        self.function_tracker = FunctionTracker()

        self.set_tracking_state(tracking_state or TrackingState())

        # Cache-freshness checker (TTL / file-dep / input-file invalidation).
        # Holds the same tracking_state ref so set_tracking_state propagates.
        self._freshness = CacheFreshnessChecker(
            tracking_state=self._tracking_state,
            backend=cash_instance.backend if cash_instance is not None else None,
            debug=debug,
        )

        self.executed_file_mtimes = {}  # var_name -> {filepath: mtime} at time of last execution

        # Statement-level file-dep tracker. Shares executed_file_deps (from
        # tracking_state) and executed_file_mtimes (owned here) by reference.
        self._file_deps = StatementFileDeps(
            tracking_state=self._tracking_state,
            executed_file_mtimes=self.executed_file_mtimes,
            debug=debug,
        )

        # Statement-level cache restorer. Hydrates outputs from a cached
        # payload + replays stdout/stderr/rich-outputs.  Distinct from the
        # variable-granular Restorer in restore.py (owned by CashMagics);
        # see CONTEXT.md for the unit-of-work distinction.
        self._stmt_restorer = StatementRestorer(
            shell=shell,
            tracking_state=self._tracking_state,
            file_deps=self._file_deps,
            compute_hash=compute_hash_fn,
            debug=debug,
        )

        # Used to prevent the "redundant import" optimization from skipping
        # import statements for modules that need re-execution after source changes.
        self.recently_reloaded_modules: set[str] = set()

        # Per-variable tracking of which module attributes are used.
        # Maps var_name -> {module_name: {attr1, attr2, ...}}
        self.module_attribute_deps: dict[str, dict[str, set[str]]] = {}

        # Tracks variables that were granularly preserved during module invalidation.
        # Maps module_name -> set of var_names whose stored input lineages need
        # to be updated once the import statement re-executes.
        self._granular_preserved_vars: dict[str, set[str]] = {}

        # Track which variables came from which module via "from X import Y".
        # Maps var_name -> module_name.
        self.from_import_sources: dict[str, str] = {}

    def get_function_tracker(self) -> FunctionTracker:
        """Returns the function tracker instance.
        Must be shared with UpstreamChecker for cache key stability.
        """
        return self.function_tracker

    def _get_cash_instance(self) -> Any | None:
        """Return the Cash instance for decorator call tracking.

        Tries ``self.cash_instance`` first, then looks for the global
        ``cash._global_cash`` singleton so that ``@cash.cache`` calls are
        captured even when the user imports ``from cash import cache``.
        """
        if self.cash_instance is not None:
            return self.cash_instance
        try:
            import cash as _cash_mod
            return getattr(_cash_mod, '_global_cash', None)
        except ImportError:
            logger.debug("[PROCESSOR] Failed to import cash module for global instance")
            return None

    def set_tracking_state(self, state: TrackingState) -> None:
        """Wire all tracking dictionaries from a shared :class:`TrackingState`.

        This is the preferred way to configure tracking state.  All fields
        are aliases to the same mutable containers so mutations are visible
        across ``CashMagics``, ``StatementProcessor``, and ``UpstreamChecker``.
        Sibling sub-components (e.g. ``CacheFreshnessChecker``) receive the
        same reference via propagation here.
        """

        self._tracking_state = state
        self.executed_cell_codes = state.executed_cell_codes
        self.executed_cell_hashes = state.executed_cell_hashes
        self.variable_lineage = state.variable_lineage
        self.lineage = state.lineage
        self.variable_hashes = state.variable_hashes
        self.variable_sources = state.variable_sources
        self.current_session_hashes = state.current_session_hashes
        self.executed_file_deps = state.executed_file_deps
        self.vars_with_mutation_lineage = state.vars_with_mutation_lineage
        self.executed_input_lineages = state.executed_input_lineages
        # Propagate to sibling sub-components (created in __init__ after the
        # first set_tracking_state, so guard with hasattr).
        if hasattr(self, '_freshness'):
            self._freshness.set_tracking_state(state)
        if hasattr(self, '_file_deps'):
            self._file_deps.set_tracking_state(state)
        if hasattr(self, '_stmt_restorer'):
            self._stmt_restorer.set_tracking_state(state)

    def process_statement(self, code: str, ttl: int | None = None, silent: bool = False, render_badge: bool = True, annotation: CacheAnnotation | None = None, occurrence_index: int = 0, stream_output: bool = False) -> ProcessResult:
        """
        Process a single statement: Analyze -> Check Cache -> Execute/Restore.

        **Side effects**: updates ``shell.user_ns`` with output variables
        on cache hit (restore) or successful execution (compute).

        Args:
            code: Python code to execute
            ttl: Time-to-live for cache entry (seconds)
            silent: If True, suppress output display
            render_badge: If True, render status badge immediately (default True)
            annotation: Optional CacheAnnotation for cache control directives
            occurrence_index: Zero-based occurrence index for duplicate statements
                within the same cell. Used to generate unique cache keys when
                the same statement appears multiple times.
            stream_output: If True, output is teed to the real stream in
                real-time AND recorded in metrics.  Useful for long-running
                statements (e.g. single-unit for loops) where the user needs
                to see progress.  When True, ``metrics['_output_flushed']``
                is set so callers don't replay the output a second time.

        Returns:
            ProcessResult with keys: 'status', 'execution_time', 'total_time',
            'saved_time', 'restored_vars', 'code', plus optional keys depending
            on cache status.
        """
        effective_ttl, force_persist, skip_cache = self._parse_annotation(annotation, ttl)
        metrics: ProcessResult = {
            'status': CacheStatus.UNKNOWN,
            'execution_time': 0.0,
            'total_time': 0.0,
            'saved_time': 0.0,
            'error': None,
            'restored_vars': [],
            'code': code.strip(),
            'uncacheable_reasons': []
        }
        if self.debug:
            logger.debug("%s Processing statement: %s...", _LOG_DEBUG, code[:50])

        process_start = time.time()

        try:
            _parsed_tree = ast.parse(code.strip())
        except SyntaxError:
            _parsed_tree = None

        inputs, outputs, source_hash, cache_key, analysis_time, hash_time = self._analyze_and_hash(code, occurrence_index=occurrence_index, tree=_parsed_tree)
        # Expose the cache key on metrics so the badge can show a short
        # prefix in the row-detail "Key" field. Lets users see at a glance
        # when two runs of the same statement land in the same vs. a
        # different cache slot.
        metrics['cache_key'] = cache_key

        early_result, skip_cache = self._check_redundant_import(
            code, _parsed_tree, skip_cache, inputs, outputs, metrics, source_hash, cache_key, process_start,
        )
        if early_result is not None:
            return early_result

        # Compute the pure-AST StatementAnalysis once. Used both by the
        # cacheability decision and (on the cache-miss path) by
        # _post_execute for in-place-mutation tracking.
        statement_analysis = analyze_statement(code, _parsed_tree)

        if not skip_cache:
            cacheable, reasons = decide_cacheability(
                code=code,
                tree=_parsed_tree,
                inputs=inputs,
                outputs=outputs,
                annotation=annotation,
                analysis=statement_analysis,
                user_ns=self.shell.user_ns,
                variable_lineage=self.variable_lineage,
                is_stateful_call=self._check_callable_stateful,
                scan_forbidden=CodeAnalyzer.scan_for_forbidden_functions,
                should_skip_variable=self._should_skip_variable,
            )
            if not cacheable:
                metrics['uncacheable_reasons'].extend(reasons)
                skip_cache = True
        metadata, cached_data, cache_check_time = self._do_cache_lookup(skip_cache, cache_key, effective_ttl, inputs)

        if self.debug:
            self._print_cache_debug(code, cache_key, inputs, cached_data, analysis_time, hash_time, cache_check_time)

        if cached_data:
            hit_result = self._handle_cache_hit(cached_data, metadata, silent, render_badge, cache_key, inputs, metrics, process_start)
            if hit_result is not None:
                return hit_result

        error_metrics, result, captured, execution_time, accessed_files = self._execute_and_drain(
            code, stream_output, skip_cache, _parsed_tree, metrics, process_start, silent, render_badge,
        )
        if error_metrics is not None:
            return error_metrics

        metrics['status'] = CacheStatus.COMPUTED
        metrics['evaluated_vars'] = list(outputs) if outputs else []
        # Surface input variable names so downstream consumers (provenance,
        # audit, badge tooltips) can reconstruct the dependency graph.
        # Filter out the no-name inputs the AST sometimes emits.
        metrics['inputs'] = [v for v in (inputs or []) if isinstance(v, str)]
        # Attribute the miss for the badge's row-detail drawer when we can do
        # it cheaply. ``CacheFreshnessChecker`` sets ``last_miss_reason`` as a
        # side effect for TTL / file invalidations (already-computed
        # information).  We *don't* fall back to a backend-wide scan for the
        # empty-key path — that diagnostic was O(N²) in cache size and
        # dominated cold-run cost.
        if not skip_cache and self._freshness.last_miss_reason:
            metrics['miss_reason'] = self._freshness.last_miss_reason

        self._post_execute(
            code, result, inputs, outputs, accessed_files,
            execution_time, effective_ttl, cache_key, source_hash,
            captured, skip_cache, force_persist, metrics, process_start,
            _parsed_tree, statement_analysis,
        )

        return metrics

    def _parse_annotation(
        self,
        annotation: CacheAnnotation | None,
        ttl: int | None,
    ) -> tuple[int | None, bool, bool]:
        """Return ``(effective_ttl, force_persist, skip_cache)`` from annotation."""
        effective_ttl = ttl
        force_persist = False
        skip_cache = False
        if annotation:
            if annotation.ttl is not None:
                effective_ttl = annotation.ttl
            force_persist = annotation.persist
            skip_cache = annotation.no_cache
        return effective_ttl, force_persist, skip_cache

    def _do_cache_lookup(
        self,
        skip_cache: bool,
        cache_key: str,
        ttl: int | None,
        inputs: set[str],
    ) -> tuple[StatementCacheMetadata | None, Any | None, float]:
        """Run cache lookup unless *skip_cache* is set."""
        if not skip_cache:
            return self._freshness.check_cache(cache_key, ttl, inputs)
        if self.debug:
            logger.debug("%s Skipping cache lookup due to missing input lineage or @cash:no-cache", _LOG_ANNOTATION)
        return None, None, 0.0

    def _execute_and_drain(
        self,
        code: str,
        stream_output: bool,
        skip_cache: bool,
        tree: ast.Module | None,
        metrics: ProcessResult,
        process_start: float,
        silent: bool,
        render_badge: bool,
    ) -> tuple[ProcessResult | None, Any, Any, float, set[str]]:
        """Execute the statement, drain decorator calls, populate stdout/stderr in metrics.

        Returns ``(error_metrics, result, captured, execution_time, accessed_files)``.
        *error_metrics* is non-None only when execution fails; callers should return it.
        """
        if self.debug:
            logger.debug("%s Executing (cache miss)", _LOG_CACHE_DEBUG)

        result, captured, execution_time, accessed_files = self._execute_statement(
            code, stream_output=stream_output, tree=tree,
            skip_capture=(skip_cache and stream_output),
        )

        decorator_calls: list = []
        try:
            cash_instance = self._get_cash_instance()
            if cash_instance is not None:
                decorator_calls = cash_instance.drain_decorator_calls()
        except (AttributeError, TypeError, RuntimeError):
            logger.debug("%s Failed to drain decorator call log", _LOG_PROCESSOR)

        metrics['stdout'] = captured.stdout
        metrics['stderr'] = captured.stderr
        # IPython rich-display capture (RichOutput objects). Distinct from
        # ``metadata['outputs']`` and ``evaluated_vars`` (which hold variable
        # NAMES from AST analysis). Keeping them under different keys avoids
        # the F-01-style fallback chain that printed ``<RichOutput at 0x..>``
        # into badge fields.
        metrics['rich_outputs'] = captured.outputs
        if decorator_calls:
            metrics['decorator_calls'] = decorator_calls

        self._display_execution_output(captured, execution_time, silent, render_badge, stream_output, metrics)
        metrics['execution_time'] = execution_time

        if not result.success:
            metrics['status'] = CacheStatus.ERROR
            metrics['error'] = result.error
            metrics['total_time'] = time.time() - process_start
            self._handle_execution_error(result, silent)
            return metrics, result, captured, execution_time, accessed_files

        return None, result, captured, execution_time, accessed_files

    def _post_execute(
        self,
        code: str,
        result: Any,
        inputs: set[str],
        outputs: set[str],
        accessed_files: set[str],
        execution_time: float,
        effective_ttl: int | None,
        cache_key: str,
        source_hash: str,
        captured: Any,
        skip_cache: bool,
        force_persist: bool,
        metrics: ProcessResult,
        process_start: float,
        tree: ast.Module | None,
        statement_analysis: StatementAnalysis,
    ) -> None:
        """Auto-track imports, capture vars, detect mutations, save to cache, record analytics."""
        # Auto-track newly imported local modules so _capture_variables includes
        # the module source hash in the lineage on first execution.
        try:
            self.function_tracker.auto_track_local_imports(code)
        except (ImportError, AttributeError, OSError):
            logger.debug("%s Failed to auto-track local imports", _LOG_PROCESSOR)

        captured_vars = self._capture_and_track_variables(
            outputs, inputs, code, source_hash,
            cache_key=cache_key, accessed_files=accessed_files, tree=tree,
        )

        # Detect in-place mutations (detection-only; do not modify lineage).
        # Reuses the StatementAnalysis from process_statement to avoid a
        # second pass of AST visitors over the same tree.
        pure_mutations = statement_analysis.all_mutated_vars - outputs
        if pure_mutations:
            self.vars_with_mutation_lineage.update(pure_mutations)
            if self.debug:
                logger.debug("%s Detected in-place mutations on: %s", _LOG_MUTATION, pure_mutations)

        saved_metadata = None
        if not skip_cache:
            saved_metadata = self._save_to_cache(
                cache_key, code, result, inputs, outputs, accessed_files,
                execution_time, effective_ttl, captured, process_start,
                source_hash, captured_vars, force_persist=force_persist,
            )
        elif self.debug:
            logger.debug("%s Skipping cache save due to @cash:no-cache", _LOG_ANNOTATION)

        if saved_metadata and 'storage' in saved_metadata:
            metrics['storage'] = saved_metadata['storage']
        if saved_metadata and 'skipped_reason' in saved_metadata:
            metrics['skipped_reason'] = saved_metadata['skipped_reason']
        if saved_metadata:
            for k in _COST_MODEL_KEYS:
                if k in saved_metadata:
                    metrics[k] = saved_metadata[k]

        metrics['total_time'] = time.time() - process_start
        self.analytics_manager.record_event(
            status='MISS',
            execution_time=metrics['total_time'],
            saved_time=0.0,
            code_hash=cache_key,
        )

    def _check_callable_stateful(self, name: str) -> bool:
        """Return True if *name* resolves to a @stateful callable; continue-safe for known-pure."""
        if is_known_pure(name):
            return False
        func_obj = self.shell.user_ns.get(name)
        if func_obj is None:
            return False
        if is_stateful(func_obj):
            return True
        if is_pure(func_obj):
            return False
        if callable(func_obj) and analyze_function_purity(func_obj, self.shell.user_ns):
            if self.debug:
                logger.debug("%s Auto-detected '%s' as pure function", _LOG_PURITY, name)
            return False
        return False

    def _handle_cache_hit(
        self,
        cached_data: Any,
        metadata: StatementCacheMetadata | None,
        silent: bool,
        render_badge: bool,
        cache_key: str,
        inputs: set[str],
        metrics: ProcessResult,
        process_start: float,
    ) -> ProcessResult | None:
        """Restore from cache and populate *metrics* for a cache-hit path.

        Returns the completed *metrics* dict on success, or ``None`` if
        restoration fails (caller should fall through to execution).
        """
        try:
            if self.debug:
                logger.debug("%s Cache hit for key: %s...", _LOG_CACHE_HIT, cache_key[:20])
                logger.debug("%s Input lineages used: %s", _LOG_CACHE_HIT, [(v, self.variable_lineage.get(v, 'NONE')[:16] + '...') for v in inputs if v not in ['get_ipython', '__builtins__', 'print']])
                if metadata:
                    logger.debug("%s Stored lineages in cache: %s", _LOG_CACHE_HIT, [(k, v[:16]+'...') for k,v in metadata.get('output_lineages', {}).items()])
            self._stmt_restorer.restore_from_cache(cached_data, metadata, silent, process_start, render_badge)

            metrics['status'] = CacheStatus.RESTORED
            metrics['saved_time'] = metadata.get('execution_time', 0.0) if metadata else 0.0
            metrics['restored_vars'] = metadata.get('outputs', []) if metadata else []
            # Carry the stored input list through so provenance/audit can
            # reconstruct the dependency graph on a cache hit, not just on
            # a fresh compute.
            metrics['inputs'] = list(metadata.get('inputs', []) if metadata else [])
            metrics['total_time'] = time.time() - process_start

            if metadata:
                if 'source' in metadata:
                    metrics['source'] = metadata['source']
                    metrics['storage'] = [metadata['source']]
                elif 'storage' in metadata:
                    metrics['storage'] = metadata['storage']
                for k in _COST_MODEL_KEYS:
                    if k in metadata:
                        metrics[k] = metadata[k]

            self.analytics_manager.record_event(
                status='HIT',
                execution_time=metrics['total_time'],
                saved_time=metrics['saved_time'],
                code_hash=cache_key,
            )

            payload = cached_data
            if isinstance(payload, dict) and 'variables' in payload:
                metrics['stdout'] = payload.get('stdout', '')
                metrics['stderr'] = payload.get('stderr', '')
                # Read both the new key and the legacy 'outputs' key so old
                # on-disk caches keep restoring their rich display data.
                metrics['rich_outputs'] = payload.get('rich_outputs', payload.get('outputs', []))
            else:
                metrics['stdout'] = ''
                metrics['stderr'] = ''
                metrics['rich_outputs'] = []

            return metrics
        except (CacheBackendError, CacheSerializationError, KeyError, TypeError, ValueError, AttributeError, OSError, pickle.UnpicklingError) as e:
            logger.warning("%s Restoration failed (%s), falling back to execution.", _LOG_CACHE, e, exc_info=True)
            return None

    @staticmethod
    def _collect_import_source_modules(tree: ast.Module) -> set[str]:
        """Return the set of top-level module names referenced by import nodes in *tree*."""
        names: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name)
                    names.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
                names.add(node.module.split('.')[0])
        return names

    def _check_redundant_import(
        self,
        code: str,
        tree: ast.Module | None,
        skip_cache: bool,
        inputs: set[str],
        outputs: set[str],
        metrics: ProcessResult,
        source_hash: str,
        cache_key: str,
        process_start: float,
    ) -> tuple[ProcessResult | None, bool]:
        """Detect redundant (already-imported) import statements.

        Returns ``(early_result, skip_cache)``.  If the import is genuinely
        redundant *early_result* is the completed metrics dict.  If the module
        was recently reloaded, ``skip_cache`` is set to ``True`` so the caller
        forces re-execution.  Returns ``(None, skip_cache)`` when normal
        processing should continue.
        """
        try:
            tree_check = tree if tree is not None else ast.parse(code.strip())
            import_names = self._get_redundant_import_names(tree_check)
            if not import_names:
                return None, skip_cache

            source_module_names = self._collect_import_source_modules(tree_check)
            has_reloaded = bool((import_names | source_module_names) & self.recently_reloaded_modules)
            all_present = all(name in self.shell.user_ns for name in import_names)

            if has_reloaded:
                self.recently_reloaded_modules -= source_module_names
                if self.debug:
                    logger.debug(
                        "%s Import involves recently-reloaded module, disabling cache for: %s",
                        _LOG_OPTIMIZATION, code.strip()
                    )
                return None, True  # updated skip_cache

            if all_present:
                if self.debug:
                    logger.debug("%s SKIPPING redundant import: %s", _LOG_OPTIMIZATION, code.strip())
                metrics['status'] = CacheStatus.SKIPPED
                metrics['total_time'] = time.time() - process_start
                self._update_state_tracking(
                    code, ExecutionResult(success=True, skipped=True),
                    inputs, outputs, set(), source_hash, cache_key, tree=tree,
                )
                return metrics, skip_cache

        except (ImportError, AttributeError, SyntaxError) as e:
            if self.debug:
                logger.debug("%s Error checking imports: %s", _LOG_OPTIMIZATION, e)

        return None, skip_cache

    def _publish_rich_outputs(self, outputs: list) -> None:
        """Replay a list of rich display outputs."""
        for output in outputs:
            if isinstance(output, dict) and 'data' in output:
                publish_display_data(data=output['data'], metadata=output.get('metadata', {}))
            else:
                display(output)

    def _display_execution_output(self, captured: Any, execution_time: float, silent: bool, render_badge: bool, stream_output: bool, metrics: ProcessResult) -> None:
        """Display captured stdout/stderr/rich outputs after execution."""
        if stream_output:
            # User already saw output in real-time via _TeeWriter.
            metrics['_output_flushed'] = True
            if not silent:
                if render_badge:
                    self._render_status_badge(CacheStatus.COMPUTED, execution_time=execution_time)
                self._publish_rich_outputs(captured.outputs)
        elif not silent:
            if captured.stdout:
                print(captured.stdout, end='')
            if captured.stderr:
                print(captured.stderr, end='', file=sys.stderr)
            if render_badge:
                self._render_status_badge(CacheStatus.COMPUTED, execution_time=execution_time)
            self._publish_rich_outputs(captured.outputs)

    def _execute_statement(self, code: str, stream_output: bool = False, tree: ast.Module | None = None, skip_capture: bool = False) -> tuple[Any, Any, float, set[str]]:
        """Execute statement with output capture and file tracking.

        Args:
            code: Python code to execute.
            stream_output: When True, use _tee_output() so output goes to the
                real terminal in real-time while still being recorded.
            tree: Optional pre-parsed AST to avoid redundant parsing.
            skip_capture: When True AND stream_output is True, bypass output
                capture entirely.  Output goes directly to the real streams
                with zero interception overhead.  Use when the result will not
                be cached (skip_cache=True) so recorded stdout/stderr are not
                needed.
        """
        start_time = time.time()
        accessed_files = set()

        try:
            # Fast path: when streaming and we won't cache, execute with the
            # real stdout/stderr directly — no TeeWriter interception at all.
            if stream_output and skip_capture:
                class _EmptyCaptured:
                    stdout = ''
                    stderr = ''
                    outputs = []
                captured = _EmptyCaptured()
                ctx_manager = contextlib.nullcontext(captured)
            elif stream_output:
                ctx_manager = _tee_output()
            else:
                ctx_manager = capture_output(stdout=True, stderr=True, display=True)
            with ctx_manager as captured:
                with FileAccessTracker(self.shell.user_ns) as file_tracker:
                    if tree is None:
                        try:
                            tree = ast.parse(code)
                        except SyntaxError:
                             # Fallback to standard exec if parse fails (though it shouldn't if compiled worked, but good for safety)
                             tree = None

                    if tree and tree.body and isinstance(tree.body[-1], ast.Expr):
                         body_nodes = tree.body[:-1]
                         last_node = tree.body[-1]

                         if body_nodes:
                             mod = ast.Module(body=body_nodes, type_ignores=[])
                             # Locations must be fixed for some python versions/ast nodes
                             # but usually parse provides them.
                             c_body = compile(mod, '<cash>', 'exec')
                             exec(c_body, self.shell.user_ns, self.shell.user_ns)

                         expr_val = last_node.value
                         mod_expr = ast.Expression(body=expr_val)
                         ast.fix_missing_locations(mod_expr)
                         c_expr = compile(mod_expr, '<cash>', 'eval')
                         result_val = eval(c_expr, self.shell.user_ns, self.shell.user_ns)

                         if result_val is not None:
                             display(result_val)
                    else:
                        compiled_code = compile(code, '<cash>', 'exec')
                        exec(compiled_code, self.shell.user_ns, self.shell.user_ns)
                        result_val = None

                accessed_files = file_tracker.get_accessed_files()

                result = ExecutionResult(success=True)

        except Exception as e:  # noqa: BLE001 - broad fallback wrapping arbitrary user code
            result = self._create_error_result(e)
            if 'captured' not in dir():
                class _EmptyCaptured:
                    stdout = ''
                    stderr = ''
                    outputs = []
                captured = _EmptyCaptured()

        execution_time = time.time() - start_time
        return result, captured, execution_time, accessed_files

    def _update_state_tracking(self, code: str, result: Any, inputs: set[str], outputs: set[str], accessed_files: set[str], source_hash: str, cache_key: str, tree: ast.Module | None = None) -> None:
        """Update lineage and variable tracking."""
        self._capture_and_track_variables(outputs, inputs, code, source_hash, cache_key=cache_key, accessed_files=accessed_files, tree=tree)

    def _save_to_cache(self, cache_key: str, code: str, result: Any, inputs: set[str], outputs: set[str], accessed_files: set[str], execution_time: float, ttl: int | None, captured: Any, process_start: float, source_hash: str, captured_vars: dict[str, Any], force_persist: bool = False) -> StatementCacheMetadata | None:
        if getattr(result, 'skipped', False):
             return None

        all_file_deps = set(accessed_files) if accessed_files else set()

        # CRITICAL: Include inherited file dependencies from input variables
        # This ensures that when Cell 3 (`df`) is cached, it stores the CSV file's mtime
        # even though Cell 3 didn't directly read the file. This allows proper invalidation.
        if hasattr(self, 'executed_file_deps'):
            for input_var in inputs:
                if input_var in self.executed_file_deps:
                    all_file_deps.update(self.executed_file_deps[input_var])

        return self._store_in_cache(
            cache_key,
            captured_vars,
            captured,
            ttl,
            inputs,
            outputs,
            execution_time,
            process_start,
            source_hash=source_hash,
            code=code,
            file_dependencies=all_file_deps,
            force_persist=force_persist
        )

    def _build_input_lineages(self, inputs: set[str], user_ns: dict) -> tuple[list[str], dict[str, str]]:
        """Build input lineage hashes list and map for a set of input variables."""
        input_lineage_hashes: list[str] = []
        input_lineage_map: dict[str, str] = {}
        for input_var in inputs:
            if input_var in self.variable_lineage:
                lineage = self.variable_lineage[input_var]
                input_lineage_hashes.append(lineage)
                input_lineage_map[input_var] = lineage
            elif input_var in user_ns:
                try:
                    lineage = self.compute_hash(user_ns[input_var])
                    input_lineage_hashes.append(lineage)
                    input_lineage_map[input_var] = lineage
                except (TypeError, ValueError, AttributeError, pickle.PicklingError) as e:
                    if self.debug:
                        logger.warning("Warning: Could not hash input '%s' for lineage: %s", input_var, e)
        return input_lineage_hashes, input_lineage_map

    def _apply_granular_module_update(self, var_name: str, value: Any, output_lineage_hash: str) -> None:
        """Apply deferred granular lineage update when a tracked module is re-imported."""
        if isinstance(value, types.ModuleType) and var_name in self._granular_preserved_vars:
            preserved = self._granular_preserved_vars.pop(var_name)
            for pv in preserved:
                pv_inputs = self.executed_input_lineages.get(pv)
                if pv_inputs is not None and var_name in pv_inputs:
                    pv_inputs[var_name] = output_lineage_hash
                    if self.debug:
                        logger.debug("[GRANULAR] Deferred update: '%s'.'%s' -> %s...", pv, var_name, output_lineage_hash[:12])

    def _update_module_attribute_deps(self, var_name: str, code: str, user_ns: dict) -> None:
        """Update granular module attribute dependency tracking for *var_name*."""
        try:
            attr_accesses = self.function_tracker.extract_module_attribute_accesses(code)
            mod_deps: dict[str, set[str]] = {}
            for input_name, attrs in attr_accesses.items():
                input_val = user_ns.get(input_name)
                if isinstance(input_val, types.ModuleType) and input_name in self.function_tracker._tracked_modules:
                    mod_deps[input_name] = attrs
            if mod_deps:
                self.module_attribute_deps[var_name] = mod_deps
            elif var_name in self.module_attribute_deps:
                del self.module_attribute_deps[var_name]
        except (AttributeError, TypeError, ValueError, SyntaxError):
            logger.debug("[PROCESSOR] Module attribute tracking failed for '%s', falling back to full invalidation", var_name)
            self.module_attribute_deps.pop(var_name, None)

    def _update_variable_content_hashes(self, var_name: str, value: Any, output_lineage_hash: str) -> None:
        """Update variable_hashes and current_session_hashes for *var_name*."""
        type_name = type(value).__name__
        if type_name in ('DataFrame', 'Series', 'ndarray'):
            # For large objects, use lineage hash as proxy for content hash
            if var_name not in self.variable_hashes:
                self.variable_hashes[var_name] = set()
            self.variable_hashes[var_name].add(output_lineage_hash)
            self.current_session_hashes[var_name] = output_lineage_hash
        elif self.compute_hash:
            try:
                content_hash = self.compute_hash(value)
                if var_name not in self.variable_hashes:
                    self.variable_hashes[var_name] = set()
                self.variable_hashes[var_name].add(content_hash)
                self.current_session_hashes[var_name] = content_hash
            except (TypeError, ValueError, AttributeError, pickle.PicklingError) as e:
                if self.debug:
                    logger.debug("[CACHE DEBUG] Could not hash captured variable '%s': %s", var_name, e)

    def _capture_and_track_variables(
        self,
        outputs: set[str],
        inputs: set[str],
        code: str,
        source_hash: str,
        cache_key: str,
        accessed_files: set[str] | None = None,
        tree: ast.Module | None = None
    ) -> dict[str, Any]:
        """
        Capture output variables and update tracking dictionaries.

        Args:
            outputs: Set of variable names identified as outputs by CodeAnalyzer.
            inputs: Set of variable names identified as inputs by CodeAnalyzer.
            code: The original code statement.
            source_hash: SHA256 hash of the code statement.
            cache_key: The cache key for this statement.
            accessed_files: Optional set of files accessed during execution.

        Returns:
            A dictionary of captured output variables and their values.
        """
        captured_vars = {}
        user_ns = self.shell.user_ns

        file_hash_component = ""
        if accessed_files:
            file_hash_component = compute_file_hash_component(accessed_files)

        for var_name in outputs:
            if var_name not in user_ns:
                continue
            value = user_ns[var_name]
            captured_vars[var_name] = value

            input_lineage_hashes, input_lineage_map = self._build_input_lineages(inputs, user_ns)
            self.executed_input_lineages[var_name] = input_lineage_map

            func_lineage_component = ""
            func_source_hashes = self.function_tracker.get_callable_source_hashes(inputs, user_ns)
            if func_source_hashes:
                func_parts = [f"{k}:{v}" for k, v in sorted(func_source_hashes.items())]
                func_lineage_component = ":" + ":".join(func_parts)

            # Include tracked module source hash in lineage.
            module_lineage_component = self._compute_module_lineage_component(
                value, var_name, code, tree
            )

            # Compute lineage hash for the output variable
            lineage_str = f"{source_hash}:{':'.join(sorted(input_lineage_hashes))}{file_hash_component}{func_lineage_component}{module_lineage_component}"
            output_lineage_hash = hashlib.sha256(lineage_str.encode('utf-8')).hexdigest()

            # Record via LineageStore so the dict entry and ``_cash_lineage_hash``
            # are written together and cannot drift.
            self.lineage.record(var_name, output_lineage_hash, value=value)

            self._apply_granular_module_update(var_name, value, output_lineage_hash)

            if var_name not in self.executed_cell_hashes:
                self.executed_cell_hashes[var_name] = set()
            elif isinstance(self.executed_cell_hashes[var_name], str):
                self.executed_cell_hashes[var_name] = {self.executed_cell_hashes[var_name]}
            self.executed_cell_hashes[var_name].add(source_hash)

            self.executed_cell_codes[var_name] = code

            self._update_module_attribute_deps(var_name, code, user_ns)
            self._update_variable_content_hashes(var_name, value, output_lineage_hash)

            self.variable_sources[var_name] = cache_key

            self._file_deps.update_for_var(var_name, accessed_files, inputs, value)

        return captured_vars


    def _compute_module_lineage_component(
        self,
        value: Any,
        var_name: str,
        code: str,
        tree: ast.Module | None = None,
    ) -> str:
        """Compute the module source hash component for lineage tracking.

        Handles three cases:
        1. Direct module import (``import X``): hash the module source + deps.
        2. Callable from a tracked module (``from X import func``): hash its
           source module.
        3. Non-callable from a tracked module (``from X import CONST``): parse
           the AST to discover the source module and hash it.

        Returns a lineage string fragment like ``:mod_src:<hash>`` or ``""``.
        """
        import sys as _sys

        if isinstance(value, types.ModuleType):
            mod_file = getattr(value, '__file__', None)
            if not (mod_file and os.path.isfile(mod_file) and var_name in self.function_tracker._tracked_modules):
                return ""
            dep_files = {
                dep_path
                for dep_path, _ in self.function_tracker._dep_file_to_parents.items()
                if var_name in self.function_tracker._dep_file_to_parents[dep_path]
            }
            mod_source_hash = read_module_source_hash(mod_file, dep_files)
            return f":mod_src:{mod_source_hash}" if mod_source_hash else ""

        if callable(value):
            obj_module = getattr(value, '__module__', None)
            if not (obj_module and obj_module in self.function_tracker._tracked_modules):
                return ""
            self.from_import_sources[var_name] = obj_module
            mod_obj = _sys.modules.get(obj_module)
            mod_file = getattr(mod_obj, '__file__', None) if mod_obj else None
            if not (mod_file and os.path.isfile(mod_file)):
                return ""
            mod_source_hash = read_module_source_hash(mod_file)
            if not mod_source_hash:
                return ""
            if self.debug:
                logger.debug("%s Including module source hash for '%s' from '%s': %s...", _LOG_CACHE_DEBUG, var_name, obj_module, mod_source_hash[:12])
            return f":from_mod_src:{mod_source_hash}"

        # Non-callable: parse the AST to find the source module.
        return self._resolve_import_module_lineage(var_name, code, tree)

    def _lookup_from_import_mod_hash(self, var_name: str, from_mod: str) -> str:
        """Return a ``:from_mod_src:<hash>`` lineage fragment for a tracked from-import module."""
        import sys as _sys
        self.from_import_sources[var_name] = from_mod
        if from_mod not in self.function_tracker._tracked_modules:
            return ""
        mod_obj = _sys.modules.get(from_mod)
        mod_file = getattr(mod_obj, '__file__', None) if mod_obj else None
        if not (mod_file and os.path.isfile(mod_file)):
            return ""
        mod_source_hash = read_module_source_hash(mod_file)
        if not mod_source_hash:
            return ""
        if self.debug:
            logger.debug("%s Including module source hash for constant '%s' from '%s': %s...", _LOG_CACHE_DEBUG, var_name, from_mod, mod_source_hash[:12])
        return f":from_mod_src:{mod_source_hash}"

    def _resolve_import_module_lineage(
        self,
        var_name: str,
        code: str,
        tree: ast.Module | None = None,
    ) -> str:
        """Resolve module lineage for a non-callable ``from X import Y``."""
        try:
            tree_check = tree if tree is not None else ast.parse(code.strip())
        except SyntaxError:
            if self.debug:
                logger.debug("%s Failed to parse import for '%s'", _LOG_PROCESSOR, var_name)
            return ""

        for node in tree_check.body:
            if not (isinstance(node, ast.ImportFrom) and node.module):
                continue
            for alias in node.names:
                imported_name = alias.asname or alias.name
                if imported_name != var_name:
                    continue
                return self._lookup_from_import_mod_hash(var_name, node.module)
        return ""


    def _should_skip_large_object_caching(
        self,
        captured_vars: dict[str, Any],
        execution_time: float,
        force_persist: bool = False,
        has_file_dependencies: bool = False
    ) -> tuple[bool, str | None, dict[str, Any] | None]:
        """Decide whether caching a set of output variables is worthwhile.

        The guiding principle is **expected time savings**: caching should only
        be skipped when the overhead of storing and later restoring the result
        is so high relative to re-computing that the user wouldn't benefit.

        The cost model is **backend-aware**:

        * **RAM (InMemoryBackend)** – no serialisation, only ``deepcopy``.
          Estimated at ~2 GB/s for pandas types, ~500 MB/s for generic objects.
          Overhead = 2 × copy time (store + restore).  This is very cheap, so
          RAM caching is almost always worthwhile.

        * **Disk / remote (FileBackend, Redis, S3, TieredBackend)** – needs
          pickle + I/O.  Estimated at ``estimated_serialization_speed`` from
          config (~200 MB/s).  Overhead = 2 × serialise time (store + restore).

        Caching is skipped when the expected time savings would be less than
        ``min_cache_savings_pct`` (default 20 %) of the original execution time::

            expected_savings = execution_time - est_restore_time
            skip  ⟺  expected_savings < min_cache_savings_pct × execution_time

        Equivalently::

            skip  ⟺  est_restore_time > (1 - min_cache_savings_pct) × execution_time

        Special cases that always allow caching (never skip):
        - ``force_persist`` is set (user explicitly wants caching via ``@cash:persist``)
        - The statement has direct file dependencies (I/O-bound; caching avoids
          re-reading from disk)

        Returns:
            ``(should_skip, reason, prediction)`` where *reason* is a human-readable
            explanation when skipping (else ``None``), and *prediction* is the cost-model
            dict for the largest variable seen (keys: ``size_bytes``, ``restore_seconds``,
            ``type_name``, ``family``), or ``None`` when estimation failed for all vars.

        The early-return paths (``force_persist``, ``has_file_dependencies``) still
        compute and return the prediction so downstream observability (cost-model
        validation, residual reports) sees consistent family attribution regardless
        of which gate fired the cache decision.
        """
        # NOTE: the "too cheap to cache" floor (statements whose own compute
        # is below ``min_execution_time_to_cache_seconds``) is checked
        # earlier in ``_store_in_cache`` — earlier than this method — so
        # that no metadata entry is written at all. That keeps subsequent
        # warm lookups as fast cache misses rather than slow
        # metadata-only hits.

        # --- Determine cost model based on backend type -----------------------
        backend = getattr(self.cash_instance, 'backend', None)
        backend_type = type(backend).__name__ if backend else ''

        # For TieredBackend the first (fastest) tier determines the restore cost
        # because that's where the data will be read from on cache hit.
        if backend_type == 'TieredBackend' and hasattr(backend, 'backends') and backend.backends:
            primary_backend_type = type(backend.backends[0]).__name__
        else:
            primary_backend_type = backend_type

        is_ram_backend = primary_backend_type == 'InMemoryBackend'

        config = getattr(self.cash_instance, 'config', None)
        min_savings_pct = _config_float(config, 'min_cache_savings_pct', 0.20)
        fixed_budget = _config_float(config, 'min_cache_fixed_budget_seconds', 0.05)

        # Track the prediction for the largest variable (by size_bytes) seen so
        # far. Computed even on the early-return paths so observability is
        # consistent — the file_dependencies / force_persist gates decide
        # whether to cache, not whether to predict.
        largest_prediction: dict[str, Any] | None = None

        # Compute predictions for all vars; collect skip-causing var separately.
        skip_decision: tuple[str | None, dict[str, Any] | None] | None = None
        for var_name, var_value in captured_vars.items():
            skip, reason, prediction = self._check_var_restore_budget(
                var_name, var_value, execution_time,
                is_ram_backend, min_savings_pct, fixed_budget,
            )
            # Keep track of the largest variable's prediction for exposure.
            if prediction is not None:
                if (largest_prediction is None
                        or prediction['size_bytes'] > largest_prediction['size_bytes']):
                    largest_prediction = prediction
            # Only the FIRST skip-causing var matters for the decision; remember it.
            if skip and skip_decision is None:
                skip_decision = (reason, prediction)

        # Never skip caching for statements that directly read files (file I/O
        # is inherently expensive; the "fast computation" heuristic doesn't apply)
        # or when force_persist is set by a user annotation. The prediction is
        # still returned so observability stays consistent.
        if force_persist or has_file_dependencies:
            return False, None, largest_prediction

        if skip_decision is not None:
            reason, skip_prediction = skip_decision
            return True, reason, skip_prediction

        return False, None, largest_prediction

    def _check_var_restore_budget(
        self,
        var_name: str,
        var_value: Any,
        execution_time: float,
        is_ram_backend: bool,
        min_savings_pct: float,
        fixed_budget: float,
    ) -> tuple[bool, str | None, dict[str, Any] | None]:
        """Return (skip, reason, prediction) for a single variable based on the
        predicted restore cost from the fitted cost model.

        The skip decision uses ``max(fixed_budget, (1 - min_savings_pct) ×
        execution_time)`` as the budget. The fixed budget covers the
        per-call overhead of cheap caches (e.g. opening a file) so trivial
        cells aren't refused; the ratio kicks in once compute is large
        enough to dominate. This matches the policy framing: small fixed
        overhead is fine, what we want to avoid is doubling a long cell.

        ``prediction`` is a dict with keys ``size_bytes``, ``restore_seconds``,
        ``type_name``, ``family``; or ``None`` if size estimation raises.
        """
        from cash.notebook import cost_model
        try:
            obj_size = estimate_object_size(var_value)
            type_name = type(var_value).__name__
            backend_kind = "ram" if is_ram_backend else "disk"
            family = cost_model.resolve_family(type_name)
            est_restore_time = cost_model.estimated_restore_time(
                type_name, obj_size, backend_kind
            )
            prediction: dict[str, Any] = {
                'size_bytes': obj_size,
                'restore_seconds': est_restore_time,
                'type_name': type_name,
                'family': family,
            }
            max_acceptable_restore = max(
                fixed_budget,
                (1.0 - min_savings_pct) * execution_time,
            )

            if execution_time > 0 and est_restore_time > max_acceptable_restore:
                size_mb = obj_size / (1024 * 1024)
                backend_label = "copying" if is_ram_backend else "serializing"
                pct_label = f"{min_savings_pct * 100:.0f}%"
                reason = (
                    f"Restoring '{var_name}' ({size_mb:.0f} MB {type_name}) would take "
                    f"~{est_restore_time:.2f}s vs {execution_time:.2f}s compute "
                    f"({backend_label}, <{pct_label} savings) — "
                    f"use @cash:persist to force"
                )
                if self.debug:
                    logger.debug("[SIZE_AWARE] %s", reason)
                return True, reason, prediction
            if self.debug and obj_size > 10 * 1024 * 1024:
                size_mb = obj_size / (1024 * 1024)
                backend_label = "copy" if is_ram_backend else "serialize"
                logger.debug(
                    "[SIZE_AWARE] Caching '%s' (%.1fMB %s) — est. %s %.2fs vs %.2fs compute",
                    var_name, size_mb, type_name, backend_label, est_restore_time, execution_time
                )
            return False, None, prediction
        except (TypeError, ValueError, AttributeError, OSError, RecursionError):
            logger.debug("[SIZE_AWARE] Failed to estimate object size, allowing caching")
        return False, None, None

    def _build_output_lineages(self, outputs: set[str]) -> dict[str, str]:
        """Collect ``{var: lineage_hash}`` for all outputs that have a lineage."""
        return {v: self.variable_lineage[v] for v in outputs if v in self.variable_lineage}

    def _filter_safe_vars(self, captured_vars: dict[str, Any]) -> dict[str, Any]:
        """Drop module objects; keep everything else (unknown types assumed picklable)."""
        if self.debug:
            logger.debug("[CACHE DEBUG] Filtering %s variables for pickleability...", len(captured_vars))
        safe: dict[str, Any] = {}
        for k, v in captured_vars.items():
            try:
                if isinstance(v, types.ModuleType):
                    continue
                # Unknown types are assumed picklable; backend handles failures.
                if type(v).__name__ in _KNOWN_PICKLABLE_TYPE_NAMES or True:  # noqa: SIM210
                    safe[k] = v
            except (TypeError, AttributeError, pickle.PicklingError) as e:
                if self.debug:
                    logger.debug("[CACHE DEBUG] Variable '%s' cannot be pickled (%s), skipping cache storage.", k, e)
        return safe

    def _store_in_cache(
        self,
        cache_key: str,
        captured_vars: dict[str, Any],
        captured_output: Any,
        ttl: int | None,
        inputs: set[str],
        outputs: set[str],
        execution_time: float,
        process_start: float,
        source_hash: str,
        code: str,
        file_dependencies: set[str],
        force_persist: bool = False
    ) -> StatementCacheMetadata | None:
        """Store execution results and metadata in the cache. Returns
        metadata, or ``None`` when the statement was so cheap to compute
        that we don't even write a metadata-only entry (the next lookup
        will miss cleanly rather than hit a metadata-only entry and
        pay a per-file read just to decide 'recompute')."""
        t_store = time.time()

        # "Too cheap to cache" floor — checked here (not inside
        # ``_should_skip_large_object_caching``) so we can skip writing
        # even a metadata-only entry. Without this, a notebook with many
        # trivial statements (e.g. 100 `a_i = i + 1`) would write 100
        # metadata-only files on the first run; every subsequent run
        # would pay ~1ms/statement of cache-lookup overhead reading
        # them only to discover they're skipped entries. By writing
        # nothing, the next lookup is a fast clean miss.
        if not force_persist and not file_dependencies:
            config_obj = getattr(self.cash_instance, 'config', None)
            min_exec_time = _config_float(
                config_obj, 'min_execution_time_to_cache_seconds', 0.01
            )
            # On Windows, time.perf_counter() can report exactly 0.0 for
            # genuinely instantaneous statements (a = 1) because the timer
            # resolution is coarser than the operation. Treat 0 the same
            # as "below the floor" — both mean "too cheap to cache".
            if execution_time < min_exec_time:
                if self.debug:
                    logger.debug(
                        "[SIZE_AWARE] Compute took only %.1fms, below "
                        "%.0fms floor — not writing cache entry",
                        execution_time * 1000, min_exec_time * 1000,
                    )
                return None

        # Size-aware caching: skip storing large objects when serialization overhead dominates
        should_skip, skip_reason, prediction = self._should_skip_large_object_caching(
            captured_vars, execution_time, force_persist,
            has_file_dependencies=bool(file_dependencies),
        )
        if should_skip:
            skip_metadata: StatementCacheMetadata = {
                'timestamp': time.time(),
                'inputs': list(inputs),
                'outputs': list(outputs),
                'execution_time': execution_time,
                'source_hash': source_hash,
                'code': code,
                'key': cache_key,
                'skipped_reason': skip_reason,
                'metadata_only': True,
                'output_lineages': self._build_output_lineages(outputs),
            }
            if prediction is not None:
                skip_metadata['cost_model_size_bytes'] = prediction['size_bytes']
                skip_metadata['cost_model_restore_seconds'] = prediction['restore_seconds']
                skip_metadata['cost_model_type_name'] = prediction['type_name']
                skip_metadata['cost_model_family'] = prediction['family']
            try:
                backend = self.cash_instance.backend if self.cash_instance else None
                if backend is not None:
                    self._stmt_restorer.persist_metadata_only(backend, cache_key, skip_metadata)
            except (OSError, TypeError, ValueError, AttributeError):
                logger.debug("[PROCESSOR] Best-effort metadata persistence failed")
            return skip_metadata

        metadata: StatementCacheMetadata = {
            'timestamp': time.time(),
            'inputs': list(inputs),
            'outputs': list(outputs),
            'execution_time': execution_time,
            'source_hash': source_hash,
            'code': code,
            'key': cache_key,
            'file_dependencies': snapshot_file_deps(file_dependencies),
            'force_persist': force_persist,
            'output_lineages': self._build_output_lineages(outputs),
        }
        if prediction is not None:
            metadata['cost_model_size_bytes'] = prediction['size_bytes']
            metadata['cost_model_restore_seconds'] = prediction['restore_seconds']
            metadata['cost_model_type_name'] = prediction['type_name']
            metadata['cost_model_family'] = prediction['family']

        payload = {
            'variables': self._filter_safe_vars(captured_vars),
            'stdout': captured_output.stdout,
            'stderr': captured_output.stderr,
            # Rich-display output (RichOutput objects). The 'outputs' key in
            # the sibling ``metadata`` dict holds variable NAMES — two distinct
            # concepts; keep them on different keys here too.
            'rich_outputs': captured_output.outputs,
            'rng_state': capture_rng_state(),
        }

        if ttl is not None:
            metadata['ttl'] = ttl

        try:
            self.cash_instance.backend.set(cache_key, payload, metadata)
        except (OSError, TypeError, ValueError, pickle.PicklingError, RuntimeError) as e:
            logger.warning("[CACHE] Failed to write to cache backend: %s", e)

        try:
            backend = self.cash_instance.backend
            if backend is not None:
                self._stmt_restorer.persist_metadata_only(backend, cache_key, metadata)
        except (OSError, TypeError, ValueError, AttributeError):
            logger.debug("[PROCESSOR] Best-effort metadata persistence failed")

        store_time = time.time() - t_store
        total_time = time.time() - process_start

        if self.debug:
            logger.debug("[TIMING] Store: %.1fms | OVERALL: %.1fms", store_time*1000, total_time*1000)
            logger.debug("[CACHE DEBUG] Stored in cache: %s", cache_key)

        return metadata

    def _analyze_and_hash(self, code: str, occurrence_index: int = 0, tree: ast.Module | None = None) -> tuple[set[str], set[str], str, str, float, float]:
        """Analyze code and compute hashes.

        Delegates cache key computation to the unified ``compute_cache_key``
        function in ``cash.notebook.cache_key`` to ensure key consistency
        across runtime and simulation.

        Args:
            code: Python source code to analyze.
            occurrence_index: Index for disambiguating duplicate code blocks.
            tree: Optional pre-parsed AST to avoid redundant parsing.

        Raises:
            CacheKeyComputationError: If the cache key cannot be computed.
        """
        t1 = time.time()
        inputs, outputs = CodeAnalyzer.analyze_code_block(code, tree=tree)
        analysis_time = time.time() - t1

        t2 = time.time()

        try:
            cache_key, source_hash, _, _, _ = compute_cache_key(
                code,
                inputs,
                ctx=CacheKeyContext(
                    variable_lineage=self.variable_lineage,
                    user_ns=self.shell.user_ns,
                    function_tracker=self.function_tracker,
                    compute_hash_fn=self.compute_hash,
                    debug=self.debug,
                    debug_print_fn=lambda msg: logger.debug(msg),
                ),
                outputs=outputs,
                occurrence_index=occurrence_index,
            )
        except Exception as exc:
            raise CacheKeyComputationError(
                f"Failed to compute cache key for: {code[:80]!r}"
            ) from exc

        hash_time = time.time() - t2
        return inputs, outputs, source_hash, cache_key, analysis_time, hash_time

    def _should_skip_variable(self, var_name: str, val: Any) -> bool:
        """Check if a variable should be skipped for hashing."""
        if isinstance(val, types.ModuleType) or var_name == 'get_ipython':
            return True

        return bool(callable(val) and (var_name.startswith('_') or hasattr(val, '__self__')))

    def _print_cache_debug(
        self,
        code: str,
        cache_key: str,
        inputs: set[str],
        cached_data: Any,
        analysis_time: float,
        hash_time: float,
        cache_check_time: float
    ) -> None:
        logger.debug("[CACHE DEBUG] Statement: %s...", code[:50])
        logger.debug("[CACHE DEBUG] Key: %s...", cache_key[:40])
        logger.debug("[CACHE DEBUG] Inputs: %s", inputs)
        logger.debug("[CACHE DEBUG] Cache hit: %s", cached_data is not None)
        logger.debug("[TIMING] Analysis: %.1fms | Hash: %.1fms | Lookup: %.1fms", analysis_time*1000, hash_time*1000, cache_check_time*1000)

    def _get_redundant_import_names(self, tree: ast.AST) -> set[str] | None:
        """
        Check if AST represents ONLY imports and return the set of defined variable names.
        Returns None if it contains non-import statements.
        """
        defined_names = set()

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.asname:
                            defined_names.add(alias.asname)
                        else:
                            defined_names.add(alias.name.split('.')[0])
                else: # ImportFrom
                    for alias in node.names:
                        if alias.asname:
                            defined_names.add(alias.asname)
                        else:
                            if alias.name == '*':
                                return None
                            defined_names.add(alias.name)

            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue
            else:
                return None

        return defined_names

    def _create_error_result(self, exception: Exception) -> Any:
        """Create error result with cleaned traceback.

        Filters out cash framework frames, keeping only user code frames.
        The traceback starts from the first ``<cash>`` frame (where user
        code is compiled and executed) and includes all subsequent frames
        (e.g., user-defined function calls).
        """
        import traceback

        exc_type, exc_value, exc_tb = sys.exc_info()

        # This preserves the user's call chain (e.g., user code calling
        # a user-defined function) while dropping cash internals above.
        clean_tb = None
        tb = exc_tb
        while tb is not None:
            frame = tb.tb_frame
            filename = frame.f_code.co_filename
            if filename == '<cash>':
                clean_tb = tb
                break
            tb = tb.tb_next

        if clean_tb is None:
            clean_tb = exc_tb

        e_with_clean_tb = exc_value.with_traceback(clean_tb)
        formatted_tb = ''.join(traceback.format_exception(exc_type, exc_value, clean_tb))

        return ExecutionResult(
            success=False,
            error=e_with_clean_tb,
            tb_string=formatted_tb,
        )

    def _handle_execution_error(self, result: Any, silent: bool) -> bool | None:
        if not silent:
            raise result.error from None
        if self.debug:
            logger.debug("[SILENT] Error in statement: %s", result.error)
        return False

    def _render_status_badge(
        self,
        status: str,
        execution_time: float = 0.0,
        time_saved: float = 0.0,
        source: str | None = None,
        storage: Any = None,
    ) -> None:
        """Render a simple status badge (delegates to badge_renderer)."""
        from .badge_renderer import render_status_badge
        html = render_status_badge(
            status,
            execution_time=execution_time,
            time_saved=time_saved,
            source=source,
            storage=storage,
        )
        try:
            display(HTML(html))
        except (ImportError, TypeError, ValueError):
            logger.debug("[PROCESSOR] Failed to display status badge HTML")

from .file_tracker import FileAccessTracker

