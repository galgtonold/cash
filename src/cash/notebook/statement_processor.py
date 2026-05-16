from __future__ import annotations

"""Core statement processing: analysis, cache lookup, execution, and lineage tracking."""

import ast
import builtins
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
from cash.notebook.cache_key import CacheKeyContext, compute_cache_key
from cash.notebook.cache_status import CacheStatus, ExecutionResult
from cash.notebook.purity import is_known_pure, is_pure, is_stateful
from cash.notebook.server_discovery import get_notebook_path
from cash.utils import normalize_path, resolve_file_dep_path

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

logger = logging.getLogger(__name__)

_SCALAR_TYPES = (int, float, str, bool, complex)
_KNOWN_PICKLABLE_TYPE_NAMES = frozenset({
    'DataFrame', 'Series', 'ndarray',
    'int', 'float', 'str', 'bool', 'bytes', 'NoneType',
    'list', 'dict', 'tuple', 'set', 'frozenset',
    'int64', 'float64', 'int32', 'float32',
    'Timestamp', 'Timedelta', 'DatetimeIndex',
})

def _get_statement_code_and_hash(metadata: StatementCacheMetadata | None) -> tuple[str | None, str | None]:
    """Return stored statement code and hash."""
    if not metadata:
        return None, None
    stored_code = metadata.get('code')
    stored_hash = metadata.get('source_hash')
    return stored_code, stored_hash

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
        calculate_memory_fn: Function to calculate memory size
    """

    def __init__(
        self,
        shell: ShellProtocol,
        cash_instance: CashInstanceProtocol,
        debug: bool = False,
        compute_hash_fn: Callable[[Any], str] | None = None,
        calculate_memory_fn: Callable[[dict[str, Any]], int] | None = None,
        tracking_state: TrackingState | None = None,
    ) -> None:
        self.shell: ShellProtocol = shell
        self.cash_instance: CashInstanceProtocol = cash_instance
        self.debug = debug
        self.compute_hash: Callable[[Any], str] | None = compute_hash_fn
        self.calculate_memory: Callable[[dict[str, Any]], int] | None = calculate_memory_fn

        self.analytics_manager = AnalyticsManager()

        self.randomness_detector = RandomnessDetector()

        # Document: function_tracker must be explicitly passed to UpstreamChecker
        self.function_tracker = FunctionTracker()

        self.set_tracking_state(tracking_state or TrackingState())

        self.executed_file_mtimes = {}  # var_name -> {filepath: mtime} at time of last execution

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
        """

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
        if skip_cache and annotation and annotation.no_cache:
            metrics['uncacheable_reasons'].append('@cash:no-cache annotation')

        if self.debug:
            logger.debug("%s Processing statement: %s...", _LOG_DEBUG, code[:50])

        process_start = time.time()

        try:
            _parsed_tree = ast.parse(code.strip())
        except SyntaxError:
            _parsed_tree = None

        skip_cache = self._check_forbidden_functions_skip(code, skip_cache, metrics, _parsed_tree)
        inputs, outputs, source_hash, cache_key, analysis_time, hash_time = self._analyze_and_hash(code, occurrence_index=occurrence_index, tree=_parsed_tree)

        early_result, skip_cache = self._check_redundant_import(
            code, _parsed_tree, skip_cache, inputs, outputs, metrics, source_hash, cache_key, process_start,
        )
        if early_result is not None:
            return early_result

        skip_cache = self._check_skip_conditions(code, skip_cache, inputs, outputs, metrics, _parsed_tree)
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

        self._post_execute(
            code, result, inputs, outputs, accessed_files,
            execution_time, effective_ttl, cache_key, source_hash,
            captured, skip_cache, force_persist, metrics, process_start,
            _parsed_tree,
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

    def _check_forbidden_functions_skip(
        self,
        code: str,
        skip_cache: bool,
        metrics: ProcessResult,
        tree: ast.Module | None,
    ) -> bool:
        """Return updated *skip_cache*; populates metrics if forbidden functions found."""
        if skip_cache:
            return skip_cache
        try:
            forbidden_reasons = CodeAnalyzer.scan_for_forbidden_functions(code, self.shell.user_ns, tree=tree)
            if forbidden_reasons:
                metrics['uncacheable_reasons'] = forbidden_reasons
                if self.debug:
                    logger.debug("%s Disabling cache due to forbidden functions: %s", _LOG_FORBIDDEN, forbidden_reasons)
                return True
        except (TypeError, AttributeError, SyntaxError) as e:
            if self.debug:
                logger.debug("%s Error scanning for forbidden functions: %s", _LOG_FORBIDDEN, e)
        return skip_cache

    def _do_cache_lookup(
        self,
        skip_cache: bool,
        cache_key: str,
        ttl: int | None,
        inputs: set[str],
    ) -> tuple[StatementCacheMetadata | None, Any | None, float]:
        """Run cache lookup unless *skip_cache* is set."""
        if not skip_cache:
            return self._check_cache(cache_key, ttl, inputs)
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
        metrics['outputs'] = captured.outputs
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
        _post_analysis = analyze_statement(code, tree)
        pure_mutations = _post_analysis.all_mutated_vars - outputs
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

    def _detect_stateful_call(self, analysis: StatementAnalysis) -> bool:
        """Return True if any bare-name call target resolves to a @stateful callable."""
        try:
            for name in analysis.called_names:
                if self._check_callable_stateful(name):
                    return True
        except (TypeError, AttributeError):
            logger.debug("%s Error checking function purity for statement", _LOG_PURITY)
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
            self._restore_from_cache(cached_data, metadata, silent, process_start, render_badge)

            metrics['status'] = CacheStatus.RESTORED
            metrics['saved_time'] = metadata.get('execution_time', 0.0) if metadata else 0.0
            metrics['restored_vars'] = metadata.get('outputs', []) if metadata else []
            metrics['total_time'] = time.time() - process_start

            if metadata:
                if 'source' in metadata:
                    metrics['source'] = metadata['source']
                    metrics['storage'] = [metadata['source']]
                elif 'storage' in metadata:
                    metrics['storage'] = metadata['storage']

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
                metrics['outputs'] = payload.get('outputs', [])
            else:
                metrics['stdout'] = ''
                metrics['stderr'] = ''
                metrics['outputs'] = []

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

    def _check_input_lineage_skip(self, inputs: set[str]) -> bool:
        """Return True if any input variable lacks lineage tracking."""
        builtin_names = set(dir(builtins))
        for var_name in inputs:
            if var_name in ['get_ipython', '__builtins__', 'print', '__name__', '__doc__']:
                continue
            if var_name in builtin_names:
                continue
            if var_name not in self.shell.user_ns:
                if self.debug:
                    logger.debug("%s Skipping cache: input '%s' missing from memory", _LOG_CACHE_KEY, var_name)
                return True
            if var_name not in self.variable_lineage:
                val = self.shell.user_ns[var_name]
                if not self._should_skip_variable(var_name, val):
                    if self.debug:
                        logger.debug("%s Skipping cache: input '%s' has no tracked lineage", _LOG_CACHE_KEY, var_name)
                    return True
        return False

    def _check_skip_conditions(self, code: str, skip_cache: bool, inputs: set[str], outputs: set[str], metrics: ProcessResult, tree: ast.Module | None) -> bool:
        """Check purity, mutation, side-effect, and input lineage conditions.

        Returns updated skip_cache flag (True if caching should be skipped).
        """
        analysis = analyze_statement(code, tree)

        # PURITY CHECK: @stateful functions must never be skipped.
        if not skip_cache and self._detect_stateful_call(analysis):
            metrics['uncacheable_reasons'].append("Calls @stateful function")
            skip_cache = True
            if self.debug:
                logger.debug("%s Skipping cache for statement calling @stateful function", _LOG_PURITY)

        # MUTATION + SIDE-EFFECT: delegate skip-reason computation to the analysis.
        if not skip_cache:
            reasons = analysis.skip_reasons(outputs)
            if reasons:
                metrics['uncacheable_reasons'].extend(reasons)
                skip_cache = True
                if self.debug:
                    logger.debug("%s Skipping cache: %s", _LOG_MUTATION, reasons)

        # Input lineage check: skip cache if any input lacks lineage.
        if not skip_cache and self._check_input_lineage_skip(inputs):
            metrics['uncacheable_reasons'].append('Input variable missing lineage')
            skip_cache = True

        return skip_cache

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
            file_hash_component = self._compute_file_hash_component(accessed_files)

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

            self._update_file_deps_for_var(var_name, accessed_files, inputs, value)

        return captured_vars

    def _inherit_file_deps_from_inputs(self, var_name: str, inputs: set[str], value: Any) -> None:
        """Propagate file deps from *inputs* to *var_name*, skipping scalar outputs."""
        is_scalar = isinstance(value, _SCALAR_TYPES)
        if not is_scalar:
            for input_var in inputs:
                if input_var not in self.executed_file_deps:
                    continue
                if var_name not in self.executed_file_deps:
                    self.executed_file_deps[var_name] = set()
                self.executed_file_deps[var_name].update(self.executed_file_deps[input_var])
                if input_var in self.executed_file_mtimes:
                    if var_name not in self.executed_file_mtimes:
                        self.executed_file_mtimes[var_name] = {}
                    self.executed_file_mtimes[var_name].update(self.executed_file_mtimes[input_var])
                if self.debug:
                    logger.debug(
                        "[CACHE DEBUG] Propagated file deps from '%s' to '%s': %s",
                        input_var, var_name, self.executed_file_deps[input_var],
                    )
        elif self.debug and any(iv in self.executed_file_deps for iv in inputs):
            logger.debug(
                "[FILE_DEPS] Skipping file dep propagation for scalar '%s' (type: %s)",
                var_name, type(value).__name__,
            )

    def _update_file_deps_for_var(
        self,
        var_name: str,
        accessed_files: set[str] | None,
        inputs: set[str],
        value: Any,
    ) -> None:
        """Record direct and inherited file dependencies for *var_name*.

        Two sources of file deps are handled here so that the logic is not
        duplicated:

        1. **Direct deps** — files that were read during the current
           statement's execution (``accessed_files``).
        2. **Inherited deps** — file deps already carried by input variables
           (e.g. ``df = df.sort_values()`` inherits ``df``'s source file so
           downstream cells still invalidate when that file changes).
           Scalar outputs are excluded from inheritance because a scalar
           derived from a DataFrame (``n_rows = len(df)``) should not be
           invalidated when the source CSV changes.
        """
        if not hasattr(self, 'executed_file_deps'):
            return

        # 1. Direct file dependencies from this statement's execution.
        if accessed_files:
            if var_name not in self.executed_file_deps:
                self.executed_file_deps[var_name] = set()
            self.executed_file_deps[var_name].update(accessed_files)
            if var_name not in self.executed_file_mtimes:
                self.executed_file_mtimes[var_name] = {}
            for fpath in accessed_files:
                    with contextlib.suppress(OSError):  # File may have been deleted between execution and capture
                        self.executed_file_mtimes[var_name][fpath] = os.path.getmtime(fpath)
        # 2. Propagate file dependencies from input variables (unless output is scalar).
        self._inherit_file_deps_from_inputs(var_name, inputs, value)

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
            mod_source_hash = self._read_module_source_hash(mod_file, dep_files)
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
            mod_source_hash = self._read_module_source_hash(mod_file)
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
        mod_source_hash = self._read_module_source_hash(mod_file)
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

    def _compute_file_hash_component(self, accessed_files: set[str]) -> str:
        """Compute a hash component from accessed file paths and their stats."""
        notebook_dir = None
        try:
            notebook_path = get_notebook_path()
            if notebook_path:
                notebook_dir = os.path.dirname(os.path.realpath(notebook_path))
        except (OSError, ValueError):
            logger.debug("[PROCESSOR] Failed to get notebook directory for file hash")

        file_components = []
        for f in sorted(accessed_files):
            if os.path.exists(f):
                canonical_path = normalize_path(os.path.realpath(f))
                display_path = canonical_path
                if notebook_dir:
                    try:
                        rel_path = normalize_path(os.path.relpath(canonical_path, notebook_dir))
                        if not rel_path.startswith('../../../'):
                            display_path = rel_path
                    except (ValueError, OSError):
                        pass  # Cross-drive relpath fails on Windows; fall back to absolute
                try:
                    stat = os.stat(canonical_path)
                    file_components.append(f"{display_path}:{stat.st_mtime}:{stat.st_size}")
                except OSError:
                    pass  # File may have been removed between exists() and stat()
        if file_components:
            component = ":" + hashlib.sha256(",".join(file_components).encode('utf-8')).hexdigest()
            logger.debug("[FILE_HASH] Final hash component: %s...", component[:50])
            return component
        return ""

    def _read_module_source_hash(self, mod_file: str, dep_files: set[str] | None = None) -> str | None:
        """Read a module file (and optional dependency files) and return their combined SHA256 hash."""
        try:
            hasher = hashlib.sha256()
            with open(mod_file, 'rb') as mf:
                hasher.update(mf.read())
            if dep_files:
                for dep_path in sorted(dep_files):
                    if os.path.isfile(dep_path):
                        try:
                            with open(dep_path, 'rb') as df:
                                hasher.update(df.read())
                        except OSError:
                            logger.debug("[MODULE_HASH] Could not read dependency file: %s", dep_path)
            return hasher.hexdigest()
        except OSError:
            logger.debug("[MODULE_HASH] Could not read module file: %s", mod_file)
            return None

    def _estimate_object_size(self, obj: Any) -> int:
        """Estimate the memory size of an object in bytes.

        Uses sys.getsizeof for basic types and specialized estimators
        for common data science types (DataFrame, ndarray).
        Falls back to sys.getsizeof for unknown types.
        """
        try:
            type_name = type(obj).__name__
            if type_name == 'DataFrame':
                return int(obj.memory_usage(deep=True).sum())
            if type_name == 'Series':
                return int(obj.memory_usage(deep=True))
            if type_name == 'ndarray':
                return obj.nbytes
            return sys.getsizeof(obj)
        except (TypeError, AttributeError):
            return sys.getsizeof(obj)

    def _should_skip_large_object_caching(
        self,
        captured_vars: dict[str, Any],
        execution_time: float,
        force_persist: bool = False,
        has_file_dependencies: bool = False
    ) -> tuple[bool, str | None]:
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
            ``(should_skip, reason)`` where *reason* is a human-readable
            explanation when skipping, else ``None``.
        """
        if force_persist:
            return False, None

        # Never skip caching for statements that directly read files.
        # File I/O (pd.read_csv, np.load, etc.) is inherently expensive and
        # the "fast computation" heuristic doesn't apply — the time is spent
        # on disk I/O, not CPU computation, so caching genuinely saves time.
        if has_file_dependencies:
            return False, None

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

        if is_ram_backend:
            default_speed = 2 * 1024 * 1024 * 1024  # 2 GB/s for known fast types
            generic_speed = 500 * 1024 * 1024         # 500 MB/s for generic objects
        else:
            config = getattr(self.cash_instance, 'config', None)
            default_speed = getattr(config, 'estimated_serialization_speed', 200 * 1024 * 1024) if config else 200 * 1024 * 1024
            generic_speed = default_speed  # same for all types

        config = getattr(self.cash_instance, 'config', None)
        min_savings_pct = getattr(config, 'min_cache_savings_pct', 0.20) if config else 0.20

        for var_name, var_value in captured_vars.items():
            skip, reason = self._check_var_restore_budget(
                var_name, var_value, execution_time,
                is_ram_backend, default_speed, generic_speed, min_savings_pct,
            )
            if skip:
                return True, reason

        return False, None

    def _check_var_restore_budget(
        self,
        var_name: str,
        var_value: Any,
        execution_time: float,
        is_ram_backend: bool,
        default_speed: int,
        generic_speed: int,
        min_savings_pct: float,
    ) -> tuple[bool, str | None]:
        """Return (skip, reason) for a single variable based on estimated restore cost."""
        ram_fast_types = frozenset({'DataFrame', 'Series', 'ndarray'})
        try:
            obj_size = self._estimate_object_size(var_value)
            type_name = type(var_value).__name__
            speed = (default_speed if type_name in ram_fast_types else generic_speed) if is_ram_backend else default_speed
            est_restore_time = obj_size / speed
            max_acceptable_restore = (1.0 - min_savings_pct) * execution_time

            if execution_time > 0 and est_restore_time > max_acceptable_restore:
                size_mb = obj_size / (1024 * 1024)
                backend_label = "copying" if is_ram_backend else "serializing"
                pct_label = f"{min_savings_pct * 100:.0f}%"
                reason = (
                    f"Restoring '{var_name}' ({size_mb:.0f} MB) would take "
                    f"~{est_restore_time:.2f}s vs {execution_time:.2f}s compute "
                    f"({backend_label}, <{pct_label} savings) — "
                    f"use @cash:persist to force"
                )
                if self.debug:
                    logger.debug("[SIZE_AWARE] %s", reason)
                return True, reason
            if self.debug and obj_size > 10 * 1024 * 1024:
                size_mb = obj_size / (1024 * 1024)
                backend_label = "copy" if is_ram_backend else "serialize"
                logger.debug(
                    "[SIZE_AWARE] Caching '%s' (%.1fMB) — est. %s %.2fs vs %.2fs compute",
                    var_name, size_mb, backend_label, est_restore_time, execution_time
                )
        except (TypeError, ValueError, AttributeError, OSError, RecursionError):
            logger.debug("[SIZE_AWARE] Failed to estimate object size, allowing caching")
        return False, None

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
    ) -> StatementCacheMetadata:
        """Store execution results and metadata in the cache. Returns metadata."""
        t_store = time.time()

        # Size-aware caching: skip storing large objects when serialization overhead dominates
        should_skip, skip_reason = self._should_skip_large_object_caching(captured_vars, execution_time, force_persist, has_file_dependencies=bool(file_dependencies))
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
            try:
                backend = self.cash_instance.backend if self.cash_instance else None
                if backend is not None:
                    self._persist_metadata_only(backend, cache_key, skip_metadata)
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
            'file_dependencies': self._snapshot_file_deps(file_dependencies),
            'force_persist': force_persist,
            'output_lineages': self._build_output_lineages(outputs),
        }

        payload = {
            'variables': self._filter_safe_vars(captured_vars),
            'stdout': captured_output.stdout,
            'stderr': captured_output.stderr,
            'outputs': captured_output.outputs,
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
                self._persist_metadata_only(backend, cache_key, metadata)
        except (OSError, TypeError, ValueError, AttributeError):
            logger.debug("[PROCESSOR] Best-effort metadata persistence failed")

        store_time = time.time() - t_store
        total_time = time.time() - process_start

        if self.debug:
            logger.debug("[TIMING] Store: %.1fms | OVERALL: %.1fms", store_time*1000, total_time*1000)
            logger.debug("[CACHE DEBUG] Stored in cache: %s", cache_key)

        return metadata

    def _persist_metadata_only(self, backend: Any, cache_key: str, metadata: dict[str, Any]) -> None:
        """Persist only metadata (no data payload) to disk for badge display after restart.

        Walks the backend chain to find backends that support metadata-only writes
        (e.g. FileBackend). This ensures timing info survives kernel restarts even
        when the actual data was too large / too cheap to cache.
        """
        if hasattr(backend, 'set_metadata_only'):
            backend.set_metadata_only(cache_key, metadata)

    def _record_restored_var_hash(self, var_name: str, value: Any, metadata: StatementCacheMetadata | None) -> None:
        """Update variable_hashes / current_session_hashes for a single restored variable."""
        type_name = type(value).__name__
        if type_name in ('DataFrame', 'Series', 'ndarray'):
            lineage_hash = (metadata.get('output_lineages', {}) if metadata else {}).get(var_name)
            if lineage_hash:
                self.variable_hashes.setdefault(var_name, set()).add(lineage_hash)
                self.current_session_hashes[var_name] = lineage_hash
        elif self.compute_hash:
            try:
                content_hash = self.compute_hash(value)
                self.variable_hashes.setdefault(var_name, set()).add(content_hash)
                self.current_session_hashes[var_name] = content_hash
            except (TypeError, ValueError, AttributeError, RecursionError) as e:
                if self.debug:
                    logger.debug("[CACHE DEBUG] Could not hash restored variable '%s': %s", var_name, e)

    def _restore_one_var(self, var_name: str, value: Any, metadata: StatementCacheMetadata | None) -> None:
        """Write one restored variable into the shell namespace and update tracking state."""
        self.shell.user_ns[var_name] = value

        if metadata:
            output_lineages = metadata.get('output_lineages', {})
            if var_name in output_lineages:
                self.lineage.record(var_name, output_lineages[var_name], value=value)

            stored_code, stored_hash = _get_statement_code_and_hash(metadata)
            if stored_hash:
                if var_name not in self.executed_cell_hashes:
                    self.executed_cell_hashes[var_name] = set()
                elif isinstance(self.executed_cell_hashes[var_name], str):
                    self.executed_cell_hashes[var_name] = {self.executed_cell_hashes[var_name]}
                self.executed_cell_hashes[var_name].add(stored_hash)
            if stored_code:
                self.executed_cell_codes[var_name] = stored_code

        self._record_restored_var_hash(var_name, value, metadata)

        if metadata and 'key' in metadata:
            self.variable_sources[var_name] = metadata['key']

    def _restore_file_deps_from_metadata(self, restored_vars: dict, metadata: StatementCacheMetadata | None) -> None:
        """Propagate file deps from cached metadata back into executed_file_deps/mtimes."""
        if not metadata:
            return
        file_deps = metadata.get('file_dependencies', {})
        if not file_deps:
            return
        # `executed_file_mtimes` historically holds {path: float}; flatten the
        # new {'mtime': ..., 'size': ...} form back to a bare mtime here.
        mtime_map = {
            path: self._split_file_dep_value(stored)[0]
            for path, stored in file_deps.items()
        }
        for var_name in restored_vars:
            self.executed_file_deps.setdefault(var_name, set()).update(file_deps.keys())
            if not hasattr(self, 'executed_file_mtimes'):
                self.executed_file_mtimes = {}
            self.executed_file_mtimes.setdefault(var_name, {}).update(mtime_map)

    def _replay_cached_outputs(
        self,
        stdout: str,
        stderr: str,
        rich_outputs: list,
        metadata: StatementCacheMetadata | None,
        render_badge: bool,
    ) -> float:
        """Replay stdout/stderr/rich outputs and render badge. Returns elapsed seconds."""
        t_output = time.time()
        if stdout:
            print(stdout, end='')
        if stderr:
            print(stderr, end='', file=sys.stderr)

        saved_time = metadata.get('execution_time', 0.0) if metadata else 0.0
        source = metadata.get('source') if metadata else None
        storage = metadata.get('storage') if metadata else None
        if render_badge:
            self._render_status_badge(CacheStatus.RESTORED, time_saved=saved_time, source=source, storage=storage)

        for output in rich_outputs:
            if isinstance(output, dict) and 'data' in output:
                publish_display_data(data=output['data'], metadata=output.get('metadata', {}))
            else:
                display(output)
        return time.time() - t_output

    def _restore_from_cache(self, cached_data: Any, metadata: StatementCacheMetadata | None, silent: bool, process_start: float, render_badge: bool = True) -> ProcessResult:
        t_restore = time.time()

        try:
            payload = cached_data
            if isinstance(payload, dict) and 'variables' in payload:
                restored_vars = payload['variables']
                stdout = payload.get('stdout', '')
                stderr = payload.get('stderr', '')
                rich_outputs = payload.get('outputs', [])
                rng_state = payload.get('rng_state')
                if rng_state:
                    if self.debug:
                        logger.debug("[CACHE DEBUG] Restoring RNG state")
                    restore_rng_state(rng_state)
            else:
                restored_vars = payload
                stdout = stderr = ""
                rich_outputs = []

            t_var = time.time()
            for var_name, value in restored_vars.items():
                self._restore_one_var(var_name, value, metadata)

            self._restore_file_deps_from_metadata(restored_vars, metadata)
            var_restore_time = time.time() - t_var

            output_replay_time = 0.0
            if not silent:
                output_replay_time = self._replay_cached_outputs(stdout, stderr, rich_outputs, metadata, render_badge)

            restore_time = time.time() - t_restore
            total_time = time.time() - process_start

            if self.debug:
                logger.debug("[TIMING] Var restore: %.1fms | Output: %.1fms", var_restore_time*1000, output_replay_time*1000)
                logger.debug("[TIMING] Total restore: %.1fms | OVERALL: %.1fms", restore_time*1000, total_time*1000)
                logger.debug("[CACHE DEBUG] ✓ Restored from cache")

        except (KeyError, TypeError, ValueError, AttributeError, OSError) as e:
            if self.debug:
                logger.debug("[CACHE DEBUG] Error restoring cache: %s", e)
            raise

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

    def _invalidate_if_ttl_expired(self, metadata: StatementCacheMetadata, cached_data: Any, ttl: int) -> Any:
        """Return None if the cache entry has exceeded *ttl* seconds, else return *cached_data*."""
        timestamp = metadata.get('timestamp', 0)
        if time.time() - timestamp > ttl:
            if self.debug:
                logger.debug("[CACHE DEBUG] Cache expired (TTL)")
            return None
        return cached_data

    @staticmethod
    def _snapshot_file_deps(paths: set[str]) -> dict[str, dict[str, float]]:
        """Return ``{path: {'mtime': mtime, 'size': size}}`` for paths that exist."""
        snapshot: dict[str, dict[str, float]] = {}
        for f in paths:
            try:
                st = os.stat(f)
            except OSError:
                continue
            snapshot[f] = {'mtime': st.st_mtime, 'size': st.st_size}
        return snapshot

    @staticmethod
    def _split_file_dep_value(value: Any) -> tuple[float, int | None]:
        """Return ``(mtime, size_or_None)`` for either the new or legacy form.

        New form: ``{'mtime': ..., 'size': ...}``.
        Legacy:   bare float (mtime only).  Cache entries from ≤0.5.0 use
        the legacy form; we still honour them but skip the size check.
        """
        if isinstance(value, dict):
            return float(value.get('mtime', 0.0)), value.get('size')
        return float(value), None

    def _invalidate_if_direct_file_changed(self, metadata: StatementCacheMetadata, cached_data: Any) -> Any:
        """Return None if any direct file dep in *metadata* is missing or modified."""
        file_deps = metadata.get('file_dependencies', {})
        for fpath, stored in file_deps.items():
            resolved = resolve_file_dep_path(fpath)
            if resolved is None:
                if self.debug:
                    logger.debug("[CACHE DEBUG] File dependency missing: %s", fpath)
                return None
            stored_mtime, stored_size = self._split_file_dep_value(stored)
            try:
                cur_stat = os.stat(resolved)
            except OSError:
                return None
            mtime_delta = abs(cur_stat.st_mtime - stored_mtime)
            if mtime_delta > 0.01:
                if self.debug:
                    logger.debug("[CACHE DEBUG] File dependency mtime changed: %s (delta=%.4fs)", resolved, mtime_delta)
                return None
            # Filesystems with coarse mtime granularity (HFS+/APFS, some
            # ext4 configs) can produce identical mtimes for back-to-back
            # writes.  Falling back to size catches that case for the
            # common "rewrote the CSV" scenario.
            if stored_size is not None and cur_stat.st_size != stored_size:
                if self.debug:
                    logger.debug(
                        "[CACHE DEBUG] File dependency size changed: %s (was %d, now %d)",
                        resolved, stored_size, cur_stat.st_size,
                    )
                return None
        return cached_data

    def _input_file_changed(self, input_var: str, fpath: str) -> bool:
        """Return True if *fpath* (a dep of *input_var*) has been modified since it was cached."""
        resolved = resolve_file_dep_path(fpath)
        if resolved is None:
            if self.debug:
                logger.debug("[CACHE DEBUG] Input '%s' file dependency missing: %s", input_var, fpath)
            return True
        source_cache_key = self.variable_sources.get(input_var)
        if not source_cache_key:
            return False
        source_meta, _ = self.cash_instance.backend.get(source_cache_key)
        if not source_meta:
            return False
        source_file_deps = source_meta.get('file_dependencies', {})
        if fpath not in source_file_deps:
            return False
        stored_mtime, stored_size = self._split_file_dep_value(source_file_deps[fpath])
        try:
            cur_stat = os.stat(resolved)
        except OSError:
            return True
        mtime_delta = abs(cur_stat.st_mtime - stored_mtime)
        if mtime_delta > 0.01:
            if self.debug:
                logger.debug("[CACHE DEBUG] Input '%s' source file mtime changed: %s (delta=%.4fs)", input_var, resolved, mtime_delta)
            return True
        if stored_size is not None and cur_stat.st_size != stored_size:
            if self.debug:
                logger.debug("[CACHE DEBUG] Input '%s' source file size changed: %s (was %d, now %d)", input_var, resolved, stored_size, cur_stat.st_size)
            return True
        return False

    def _invalidate_if_input_file_changed(self, inputs: set[str], cached_data: Any) -> Any:
        """Return None if any file dep of an input variable has changed since it was computed."""
        if not hasattr(self, 'executed_file_deps'):
            return cached_data
        for input_var in inputs:
            for fpath in self.executed_file_deps.get(input_var, ()):
                if self._input_file_changed(input_var, fpath):
                    return None
        return cached_data

    def _check_cache(self, cache_key: str, ttl: int | None, inputs: set[str] | None = None) -> tuple[StatementCacheMetadata | None, Any | None, float]:
        """Check cache for existing entry.

        Also checks file dependencies inherited from input variables.
        """
        t3 = time.time()
        metadata, cached_data = self.cash_instance.backend.get(cache_key)
        cache_check_time = time.time() - t3

        if cached_data and metadata:
            if ttl:
                cached_data = self._invalidate_if_ttl_expired(metadata, cached_data, ttl)
            if cached_data:
                cached_data = self._invalidate_if_direct_file_changed(metadata, cached_data)
            # CRITICAL: Also check file dependencies inherited from INPUT variables.
            # Fixes the bug where `df` cell was cached even when the source CSV changed.
            if cached_data and inputs:
                cached_data = self._invalidate_if_input_file_changed(inputs, cached_data)
            if cached_data is not None and 'output_lineages' not in metadata:
                cached_data = None
                if self.debug:
                    logger.debug("[CACHE DEBUG] Cache entry missing lineage metadata (stale format), invalidating.")

        return metadata, cached_data, cache_check_time

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

