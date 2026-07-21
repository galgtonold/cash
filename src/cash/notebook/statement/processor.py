from __future__ import annotations

"""Core statement processing: analysis, cache lookup, execution, and lineage tracking."""

import ast
import contextlib
import hashlib
import inspect
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
from cash.notebook.cache_key import CacheKeyContext, compute_cache_key, write_provenance_key
from cash.notebook.cache_status import CacheStatus, ExecutionResult
from cash.notebook.file_dep_snapshot import snapshot_file_deps
from cash.notebook.object_hashing import estimate_object_size
from cash.notebook.purity import is_known_pure, is_pure, is_stateful
from cash.notebook.statement._metadata import StatementCacheMetadata
from cash.notebook.statement.file_deps import StatementFileDeps
from cash.notebook.statement.freshness import CacheFreshnessChecker
from cash.notebook.statement.lineage import StatementLineageBuilder
from cash.notebook.statement.miss_guard import (
    GUARD_SKIP_REASON,
    MissGuard,
    resolve_cache_dir,
)
from cash.notebook.statement.restore import StatementRestorer

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
_LOG_CACHE_HIT = "[CACHE_HIT_DEBUG]"
_LOG_CACHE = "[CACHE]"
_LOG_CACHE_DEBUG = "[CACHE DEBUG]"
_LOG_PURITY = "[PURITY]"
_LOG_OPTIMIZATION = "[OPTIMIZATION]"
_LOG_FORBIDDEN = "[FORBIDDEN]"
_LOG_ANNOTATION = "[ANNOTATION]"

# --- Loop-persist amplification guard (CAS-160) ----------------------------
# ``# @cash:persist`` inside (or on) a loop makes EVERY iteration a persist
# target. When the loop grows one object -- the classic "add a column per
# iteration" frame build -- each iteration snapshots the whole object at its
# current width, so a 40 MB final frame costs sum(widths) on disk: 13x for 25
# columns, and quadratic in the iteration count thereafter. CAS-142's caps are
# structurally blind to this: its per-object refusal compares ONE value against
# half the tier cap (40 MB vs >=4 GiB -> fine) and its evict-after-write warning
# needs the total to exceed the cap (520 MB vs >=8 GiB -> never evicts). Neither
# looks at *cumulative writes for one statement*, which is the dimension that
# actually blows up.
#
# So track that dimension directly. Once one statement's cumulative persisted
# bytes exceed both an absolute floor and a multiple of the value's CURRENT
# size, stop value-persisting it (metadata-only, exactly like the size-aware
# skip) and warn once. Skipping beats evict-after-write here: it also stops
# paying the rising per-iteration serialisation cost, which is the "re-runs got
# slower" half of the symptom.
#
# The floor keeps the guard off small loops entirely (nobody's disk is at risk
# from a few MB), and the accounting is only ever done for statements carrying
# an iteration/branch context marker, so an ordinary single-statement
# ``# @cash:persist`` can never trip it -- it writes once, and is not in a loop.
_PERSIST_AMPLIFICATION_FLOOR_BYTES = 64 * 1024 * 1024
_PERSIST_AMPLIFICATION_LIMIT = 4

# Badge/metadata reason for a write refused by the guard above. A constant so
# consumers compare identity rather than pattern-matching the wording.
AMPLIFICATION_SKIP_REASON = (
    "loop caching a growing object would store every intermediate state; "
    "further iterations kept metadata-only (see the emitted warning)"
)

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

# ``capture_output`` is the ONLY IPython name imported at module scope, and it
# keeps its try/except because the fallback below is a genuine working
# equivalent (it really does capture stdout/stderr), not a silent drop.  The
# module MUST stay importable without IPython: base ``cash`` declares
# ``dependencies = []`` — IPython lives in the ``[notebook]`` extra — and this
# module sits on the ``import cash`` chain, so a module-level unguarded IPython
# import makes a bare ``pip install cash-lib`` unimportable (CAS-129).
#
# ``display`` / ``publish_display_data`` deliberately do NOT get the same
# treatment: there is no honest fallback for "render rich output" without
# IPython, and a no-op stub would make a display call silently vanish — the
# cell appears to succeed while producing nothing.  They are imported
# function-locally at each use site instead, so a genuine display attempt
# fails loudly with a clear ImportError.  Same rule as
# ``StatementRestorer._replay_cached_outputs`` (CAS-129/CAS-132).
try:
    from IPython.utils.io import capture_output
except ImportError:
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

from ...analytics import AnalyticsManager
from ..analysis import CodeAnalyzer
from ..annotations import CacheAnnotation
from ..compiled_source import is_cash_filename, register_cell_source
from ..function_tracker import FunctionTracker
from ..cacheability import (
    KNOWN_PURE_METHODS,
    RECEIVER_READONLY_WRITE_METHODS,
    StatementAnalysis,
    analyze_statement,
    assigned_method_call_receivers,
    standalone_method_call_receivers,
    standalone_method_mutation_receivers,
)
from ..cacheability_decision import (
    decide_cacheability,
    identity_coupled_reason,
    receiver_is_identity_coupled,
)
from ..purity import analyze_function_purity
from ..randomness import (
    RandomnessDetector,
    capture_object_rng_states,
    capture_rng_state,
    get_drawing_rng_modules,
    get_seeding_rng_modules,
    rng_epoch_fingerprint,
    check_and_warn_randomness,
    warn_stale_estimator_fit,
    warn_stale_randomness,
    warn_unseeded_estimator_fit,
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
        # When True, force-persist every statement (bypass the cost-aware
        # floors), as if every statement carried ``# @cash:persist``. Seeded
        # from config; flippable at runtime (``%cash_persist`` magic). Read
        # defensively because tests pass a MagicMock cash_instance.
        try:
            self.persist_all = bool(getattr(cash_instance.config, 'persist_all', False))
        except (AttributeError, TypeError):
            self.persist_all = False
        self.compute_hash: Callable[[Any], str] | None = compute_hash_fn

        # Loop-persist amplification guard (CAS-160). Cumulative value-persisted
        # bytes per loop-body statement (keyed on the body's real source, with
        # the per-iteration discriminator comment stripped, so all iterations of
        # one statement share a counter), plus the set of statements already
        # warned about so a 1000-iteration loop warns once, not 1000 times.
        self._persist_bytes_by_stmt: dict[str, int] = {}
        self._warned_persist_amplification: set[str] = set()

        self.analytics_manager = AnalyticsManager()

        self.randomness_detector = RandomnessDetector()

        # Document: function_tracker must be explicitly passed to UpstreamChecker
        self.function_tracker = FunctionTracker()
        # module -> cache key of the seeding statement in force (CAS-223).
        self._rng_seed_epochs: dict[str, str] = {}

        self.set_tracking_state(tracking_state or TrackingState())

        # Cache-freshness checker (TTL / file-dep / input-file invalidation).
        # Stateless w.r.t. tracking state — receives it per call.
        self._freshness = CacheFreshnessChecker(
            backend=cash_instance.backend if cash_instance is not None else None,
            debug=debug,
        )

        # Perpetual-miss guard (CAS-172): learns which statements can never hit
        # (unstable cache key -> a new key every run -> zero hits) and stops
        # SERIALISING them, while keeping the hash + the lookup. Verdicts persist
        # to the cache dir so a restart doesn't re-pay the learning. Resolved
        # defensively: the backend is a MagicMock in a good number of tests, and
        # an unresolvable dir just means session-scoped verdicts.
        try:
            _guard_dir = resolve_cache_dir(
                cash_instance.backend if cash_instance is not None else None
            )
        except (AttributeError, TypeError):
            _guard_dir = None
        self._miss_guard = MissGuard(_guard_dir)

        # Statement-level file-dep tracker. Stateless w.r.t. tracking state —
        # receives it per call. ``executed_file_deps`` and ``executed_file_mtimes``
        # live on TrackingState.
        self._file_deps = StatementFileDeps(debug=debug)

        # Statement-level cache restorer. Hydrates outputs from a cached
        # payload + replays stdout/stderr/rich-outputs.  Distinct from the
        # variable-granular Restorer in restore.py (owned by CashMagics);
        # see CONTEXT.md for the unit-of-work distinction.  Stateless w.r.t.
        # tracking state — receives it per call.
        self._stmt_restorer = StatementRestorer(
            shell=shell,
            file_deps=self._file_deps,
            compute_hash=compute_hash_fn,
            debug=debug,
            rng_seed_epochs=self._rng_seed_epochs,
        )

        # Used to prevent the "redundant import" optimization from skipping
        # import statements for modules that need re-execution after source changes.
        self.recently_reloaded_modules: set[str] = set()

        # Statement-level lineage builder. Owns the lineage hash + content
        # hash + module-source hash bookkeeping for each output variable.
        # Stateless w.r.t. tracking state — receives it per call.  The three
        # dicts it writes (``granular_preserved_vars``, ``module_attribute_deps``,
        # ``from_import_sources``) live on TrackingState.
        self._lineage = StatementLineageBuilder(
            shell=shell,
            function_tracker=self.function_tracker,
            file_deps=self._file_deps,
            compute_hash=compute_hash_fn,
            debug=debug,
        )

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

        The four sibling sub-components (``_freshness``, ``_file_deps``,
        ``_stmt_restorer``, ``_lineage``) receive ``TrackingState`` as a
        method parameter and hold no aliased dict references, so no
        propagation step is required here.
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
        self.mutation_verdicts = state.mutation_verdicts

    def process_statement(self, code: str, ttl: int | None = None, silent: bool = False, annotation: CacheAnnotation | None = None, occurrence_index: int = 0, stream_output: bool = False, force_outputs: set[str] | None = None, is_last: bool = True) -> ProcessResult:
        """
        Process a single statement: Analyze -> Check Cache -> Execute/Restore.

        **Side effects**: updates ``shell.user_ns`` with output variables
        on cache hit (restore) or successful execution (compute).

        Args:
            code: Python code to execute
            ttl: Time-to-live for cache entry (seconds)
            silent: If True, suppress output display
            annotation: Optional CacheAnnotation for cache control directives
            occurrence_index: Zero-based occurrence index for duplicate statements
                within the same cell. Used to generate unique cache keys when
                the same statement appears multiple times.
            stream_output: If True, output is teed to the real stream in
                real-time AND recorded in metrics.  Useful for long-running
                statements (e.g. single-unit for loops) where the user needs
                to see progress.  When True, ``metrics['_output_flushed']``
                is set so callers don't replay the output a second time.
            force_outputs: Extra output variable names to capture/restore on top
                of those AST analysis discovers, and to treat as expected writes
                so an in-place mutation on them does NOT block caching. Used by
                the accumulator-loop fast path (CAS-145) to route ``out = []`` +
                ``for e in it: out.append(f(e))`` through the cache as one unit,
                capturing the accumulator AND the leaked loop variable. Does NOT
                affect the cache key (outputs only enter it when they are
                modules), so the key still tracks the loop source + input
                lineages.

        Returns:
            ProcessResult with keys: 'status', 'execution_time', 'total_time',
            'saved_time', 'restored_vars', 'code', plus optional keys depending
            on cache status.
        """
        effective_ttl, force_persist, skip_cache, allow_random, cache_fit = self._parse_annotation(annotation, ttl)
        unseeded_calls = self._warn_unseeded_randomness(code, allow_random)
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
        # Caller-forced outputs (accumulator-loop fast path, CAS-145): capture
        # and restore these on top of the AST-discovered outputs, and mark them
        # as expected writes so an in-place accumulator mutation (``out.append``)
        # is not read as a caching blocker. Added AFTER the cache key is computed
        # so it is unaffected — the key already tracks the loop source + input
        # lineages (a forced output only ever enters the key when it is a module,
        # which an accumulator never is).
        if force_outputs:
            outputs = outputs | force_outputs
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
        statement_analysis = analyze_statement(code, _parsed_tree, self.shell.user_ns)

        # A standalone bare-Expr method call (``lst.append(x)``, ``bus.on(fn)``)
        # has no Store target, so AST analysis never surfaces the receiver as an
        # output and its lineage stays frozen -> a cached downstream consumer
        # serves a stale value once the mutation is edited. The broad-precise
        # classifier decides which receivers actually mutate (statically known,
        # a prior runtime verdict, or assume-mutate); the rest are observed by
        # content after execution (see _post_execute). Routed receivers go into
        # the output set so capture_and_track bumps their lineage (source-based,
        # matching the upstream simulation) and the statement is skip-cached so
        # the mutated receiver is never round-tripped.
        # Control-structure BODY statements are dispatched here individually with
        # an injected marker comment, but the upstream simulation treats the whole
        # loop/branch as one unit (its mutations flow through the loop-mutation
        # lineage path, not per-body classification). Classifying a body statement
        # here would bump the receiver with a per-statement source the simulation
        # never reproduces -> cross-cell desync. Skip them; the control structure
        # owns its body's mutation lineage.
        if '# __iteration_context__:' in code or '# control_context:' in code:
            mut_pre_route, mut_observe, mut_assumed, mut_record = set(), set(), set(), False
            est_fit: set[str] = set()
            # ...with ONE exception: a draw on a live Figure/Axes (CAS-220).
            draw_only = self._identity_coupled_call_receivers(_parsed_tree)
        else:
            mut_pre_route, mut_observe, mut_assumed, mut_record = self._classify_method_mutations(
                _parsed_tree, source_hash, outputs,
            )
            est_fit = self._estimator_fit_receivers(_parsed_tree, outputs) if cache_fit else set()
            draw_only = set()
        # OPT-IN ONLY (``# @cash:cache-fit``, CAS-170). A bare ``estimator.fit(X, y)``
        # mutates its receiver in place, so the classifier above routes it to
        # skip-caching: the statement re-executes and is never serialised, which is
        # net-NEUTRAL -- a fit that keeps missing cannot cost more than it saves.
        #
        # It does NOT make aliases safe. CAS-170 originally claimed skipping the fit
        # made ``backup = clf`` correct "by construction"; CAS-184 disproved that.
        # ``backup = clf`` is an ORDINARY ASSIGNMENT that cash caches on its own, and
        # restoring it rebinds ``backup`` to a pre-fit deserialised copy -- the fit
        # statement has no bearing on it either way. Do not restore that reasoning.
        #
        # Caching a bare fit instead (CAS-138) is the OPT-IN path, kept because it
        # is a large win when it lands but demoted from the default because its
        # correctness surface exceeds what per-statement restore can guarantee:
        #   * a cache HIT may REBIND the receiver, leaving an alias pointing at the
        #     pre-fit object. Not fixable per-statement -- on a warm run-all the
        #     CONSTRUCTOR statement's own hit-restore rebinds the receiver before
        #     the fit's in-place transfer lands, so the alias graph is already
        #     broken upstream; and
        #   * the duck-type gate admits the whole sklearn-compatible universe
        #     (xgboost/lightgbm/custom), each with its own ``__getstate__``
        #     contract, and several never restore -- re-serialising every run for
        #     a net LOSS.
        # For reliable ML caching, wrap training in a returning function under
        # ``@cash.cache`` instead (verified 9-11x, no identity caveat).
        #
        # When opted in: add the receiver to ``outputs`` (so its source-based
        # lineage is bumped AND the fitted value is captured/saved) but do NOT
        # skip-cache it, so the normal lookup runs (hit -> in-place restore; miss
        # -> execute + save). A receiver that is BOTH an estimator fit AND another
        # genuine skip receiver still skips (the skip wins for that receiver).
        # ``est_fit`` also threads to the cache-hit path so its restore is IN
        # PLACE (CAS-138). Without the directive ``est_fit`` is empty and every
        # site below degrades to the pre-CAS-138 skip-cache behaviour.
        skip_pre_route = mut_pre_route - est_fit
        if mut_pre_route or est_fit:
            outputs = outputs | mut_pre_route | est_fit
        if skip_pre_route:
            skip_cache = True
            metrics['uncacheable_reasons'].append(
                f"In-place mutation on: {', '.join(sorted(skip_pre_route))} "
                "(receiver lineage bumped; statement re-executes)"
            )
        # CAS-220: a draw inside a loop/branch body. Skip the CACHE without
        # touching ``outputs`` -- the statement must re-execute so the artists
        # actually land on the Axes, but bumping its lineage from a per-statement
        # source is precisely what the control-body skip above exists to avoid.
        if draw_only:
            skip_cache = True
            metrics['uncacheable_reasons'].append(
                f"Draws on: {', '.join(sorted(draw_only))} "
                "(live Figure/Axes; statement re-executes)"
            )
        # An UNSEEDED estimator fit routed to caching above is frozen on re-run
        # with no warning -- cash's AST detector cannot see the randomness inside
        # sklearn's compiled .fit(). Warn now (compute time); the same set drives
        # the restore-time warning on a cache hit below (CAS-167).
        unseeded_fits = self._warn_unseeded_estimator_fit(code, est_fit, allow_random)

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
            )
            if not cacheable:
                metrics['uncacheable_reasons'].extend(reasons)
                skip_cache = True
        effective_ttl = self._ttl_floor_from_called_functions(inputs, effective_ttl)
        metadata, cached_data, cache_check_time = self._do_cache_lookup(skip_cache, cache_key, effective_ttl, inputs)
        self._observe_miss_guard(skip_cache, code, source_hash, cache_key, cached_data)

        if self.debug:
            self._print_cache_debug(code, cache_key, inputs, cached_data, analysis_time, hash_time, cache_check_time)

        if cached_data and not self._import_needs_reexecution(_parsed_tree):
            hit_result = self._handle_cache_hit(cached_data, metadata, silent, cache_key, inputs, metrics, process_start, est_fit)
            if hit_result is not None:
                # The restore SUCCEEDED, so the value handed back is a replay.
                self._warn_stale_randomness(code, unseeded_calls, allow_random)
                self._warn_stale_estimator_fit(code, unseeded_fits, allow_random)
                return hit_result

        error_metrics, result, captured, execution_time, accessed_files = self._execute_and_drain(
            code, stream_output, skip_cache, _parsed_tree, metrics, process_start, silent, is_last,
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
            mut_observe, mut_assumed, mut_record, est_fit,
        )

        return metrics

    async def process_statement_async(self, code: str, ttl: int | None = None, silent: bool = False, annotation: CacheAnnotation | None = None, occurrence_index: int = 0, stream_output: bool = False, is_last: bool = True) -> ProcessResult:
        """Async twin of :meth:`process_statement` for top-level-await cells.

        Line-for-line the same pipeline — analysis, cache lookup, cache-hit
        restore, cacheability decision, mutation classification, post-execute
        capture + store — as :meth:`process_statement`.  The ONLY difference is
        that the cache-*miss* execution goes through
        :meth:`_execute_and_drain_async` (which awaits the compiled unit) so a
        statement containing a top-level ``await`` runs on IPython's live loop.

        The cache-*hit* path (``_handle_cache_hit``) returns BEFORE any
        coroutine is built, so an identical second run skips the await entirely
        (CAS-116).  CAS-96 trailing-semicolon suppression and CAS-115/89
        live-alias edge-recording (in ``_post_execute``) apply unchanged because
        this method routes through the same ``_analyze_and_hash`` /
        ``_handle_cache_hit`` / ``_post_execute`` helpers.
        """
        effective_ttl, force_persist, skip_cache, allow_random, cache_fit = self._parse_annotation(annotation, ttl)
        unseeded_calls = self._warn_unseeded_randomness(code, allow_random)
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
            logger.debug("%s Processing statement (async): %s...", _LOG_DEBUG, code[:50])

        process_start = time.time()

        try:
            _parsed_tree = ast.parse(code.strip())
        except SyntaxError:
            _parsed_tree = None

        inputs, outputs, source_hash, cache_key, analysis_time, hash_time = self._analyze_and_hash(code, occurrence_index=occurrence_index, tree=_parsed_tree)
        metrics['cache_key'] = cache_key

        early_result, skip_cache = self._check_redundant_import(
            code, _parsed_tree, skip_cache, inputs, outputs, metrics, source_hash, cache_key, process_start,
        )
        if early_result is not None:
            return early_result

        statement_analysis = analyze_statement(code, _parsed_tree, self.shell.user_ns)

        if '# __iteration_context__:' in code or '# control_context:' in code:
            mut_pre_route, mut_observe, mut_assumed, mut_record = set(), set(), set(), False
            est_fit: set[str] = set()
            # ...with ONE exception: a draw on a live Figure/Axes (CAS-220).
            draw_only = self._identity_coupled_call_receivers(_parsed_tree)
        else:
            mut_pre_route, mut_observe, mut_assumed, mut_record = self._classify_method_mutations(
                _parsed_tree, source_hash, outputs,
            )
            est_fit = self._estimator_fit_receivers(_parsed_tree, outputs) if cache_fit else set()
            draw_only = set()
        # OPT-IN ONLY (``# @cash:cache-fit``, CAS-170). A bare ``estimator.fit(X, y)``
        # mutates its receiver in place, so the classifier above routes it to
        # skip-caching: the statement re-executes and is never serialised, which is
        # net-NEUTRAL -- a fit that keeps missing cannot cost more than it saves.
        #
        # It does NOT make aliases safe. CAS-170 originally claimed skipping the fit
        # made ``backup = clf`` correct "by construction"; CAS-184 disproved that.
        # ``backup = clf`` is an ORDINARY ASSIGNMENT that cash caches on its own, and
        # restoring it rebinds ``backup`` to a pre-fit deserialised copy -- the fit
        # statement has no bearing on it either way. Do not restore that reasoning.
        #
        # Caching a bare fit instead (CAS-138) is the OPT-IN path, kept because it
        # is a large win when it lands but demoted from the default because its
        # correctness surface exceeds what per-statement restore can guarantee:
        #   * a cache HIT may REBIND the receiver, leaving an alias pointing at the
        #     pre-fit object. Not fixable per-statement -- on a warm run-all the
        #     CONSTRUCTOR statement's own hit-restore rebinds the receiver before
        #     the fit's in-place transfer lands, so the alias graph is already
        #     broken upstream; and
        #   * the duck-type gate admits the whole sklearn-compatible universe
        #     (xgboost/lightgbm/custom), each with its own ``__getstate__``
        #     contract, and several never restore -- re-serialising every run for
        #     a net LOSS.
        # For reliable ML caching, wrap training in a returning function under
        # ``@cash.cache`` instead (verified 9-11x, no identity caveat).
        #
        # When opted in: add the receiver to ``outputs`` (so its source-based
        # lineage is bumped AND the fitted value is captured/saved) but do NOT
        # skip-cache it, so the normal lookup runs (hit -> in-place restore; miss
        # -> execute + save). A receiver that is BOTH an estimator fit AND another
        # genuine skip receiver still skips (the skip wins for that receiver).
        # ``est_fit`` also threads to the cache-hit path so its restore is IN
        # PLACE (CAS-138). Without the directive ``est_fit`` is empty and every
        # site below degrades to the pre-CAS-138 skip-cache behaviour.
        skip_pre_route = mut_pre_route - est_fit
        if mut_pre_route or est_fit:
            outputs = outputs | mut_pre_route | est_fit
        if skip_pre_route:
            skip_cache = True
            metrics['uncacheable_reasons'].append(
                f"In-place mutation on: {', '.join(sorted(skip_pre_route))} "
                "(receiver lineage bumped; statement re-executes)"
            )
        # CAS-220: a draw inside a loop/branch body. Skip the CACHE without
        # touching ``outputs`` -- the statement must re-execute so the artists
        # actually land on the Axes, but bumping its lineage from a per-statement
        # source is precisely what the control-body skip above exists to avoid.
        if draw_only:
            skip_cache = True
            metrics['uncacheable_reasons'].append(
                f"Draws on: {', '.join(sorted(draw_only))} "
                "(live Figure/Axes; statement re-executes)"
            )
        # An UNSEEDED estimator fit routed to caching above is frozen on re-run
        # with no warning -- cash's AST detector cannot see the randomness inside
        # sklearn's compiled .fit(). Warn now (compute time); the same set drives
        # the restore-time warning on a cache hit below (CAS-167).
        unseeded_fits = self._warn_unseeded_estimator_fit(code, est_fit, allow_random)

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
            )
            if not cacheable:
                metrics['uncacheable_reasons'].extend(reasons)
                skip_cache = True
        effective_ttl = self._ttl_floor_from_called_functions(inputs, effective_ttl)
        metadata, cached_data, cache_check_time = self._do_cache_lookup(skip_cache, cache_key, effective_ttl, inputs)
        self._observe_miss_guard(skip_cache, code, source_hash, cache_key, cached_data)

        if self.debug:
            self._print_cache_debug(code, cache_key, inputs, cached_data, analysis_time, hash_time, cache_check_time)

        # CACHE HIT — returns before any coroutine is built, so an identical
        # second run of a top-level-await cell skips the await entirely.
        if cached_data and not self._import_needs_reexecution(_parsed_tree):
            hit_result = self._handle_cache_hit(cached_data, metadata, silent, cache_key, inputs, metrics, process_start, est_fit)
            if hit_result is not None:
                # The restore SUCCEEDED, so the value handed back is a replay.
                self._warn_stale_randomness(code, unseeded_calls, allow_random)
                self._warn_stale_estimator_fit(code, unseeded_fits, allow_random)
                return hit_result

        error_metrics, result, captured, execution_time, accessed_files = await self._execute_and_drain_async(
            code, stream_output, skip_cache, _parsed_tree, metrics, process_start, silent, is_last,
        )
        if error_metrics is not None:
            return error_metrics

        metrics['status'] = CacheStatus.COMPUTED
        metrics['evaluated_vars'] = list(outputs) if outputs else []
        metrics['inputs'] = [v for v in (inputs or []) if isinstance(v, str)]
        if not skip_cache and self._freshness.last_miss_reason:
            metrics['miss_reason'] = self._freshness.last_miss_reason

        self._post_execute(
            code, result, inputs, outputs, accessed_files,
            execution_time, effective_ttl, cache_key, source_hash,
            captured, skip_cache, force_persist, metrics, process_start,
            _parsed_tree, statement_analysis,
            mut_observe, mut_assumed, mut_record, est_fit,
        )

        return metrics

    def _parse_annotation(
        self,
        annotation: CacheAnnotation | None,
        ttl: int | None,
    ) -> tuple[int | None, bool, bool, bool, bool]:
        """Return ``(effective_ttl, force_persist, skip_cache, allow_random, cache_fit)``.

        ``persist_all`` (config / ``%cash_persist`` magic) forces persistence
        for every statement, as if each carried ``# @cash:persist``.

        ``allow_random`` (``# @cash:allow-random``) is *advisory only* — it
        suppresses the unseeded-randomness warning and nothing else.  It must
        never reach the cacheability decision: an unseeded random statement is
        cacheable by design, with or without the directive.

        ``cache_fit`` (``# @cash:cache-fit``) opts a bare ``estimator.fit(X, y)``
        statement IN to the estimator-fit caching path (CAS-138).  It is off by
        default: without it a bare fit is skip-cached and simply re-executes,
        which is net-neutral (CAS-170).  It does NOT, as this comment used to
        claim, keep aliases correct: ``backup = model`` is an ordinary assignment
        whose own restore rebinds a pre-fit copy, independently of the fit
        (CAS-184 — fixed by refusing to cache a bare alias bind).
        """
        effective_ttl = ttl
        force_persist = self.persist_all
        skip_cache = False
        allow_random = False
        cache_fit = False
        if annotation:
            if annotation.ttl is not None:
                effective_ttl = annotation.ttl
            force_persist = force_persist or annotation.persist
            skip_cache = annotation.no_cache
            allow_random = annotation.allow_random
            cache_fit = annotation.cache_fit
        return effective_ttl, force_persist, skip_cache, allow_random, cache_fit

    def _ttl_floor_from_called_functions(self, inputs: set[str], effective_ttl: int | None) -> int | None:
        """Lower *effective_ttl* to the TTL of any ``@cash.cache`` function called here.

        A statement ``x = f()`` where ``f`` is decorated ``@cash.cache(ttl=0)`` was
        cached with no TTL under %cash_on, so the statement restore froze ``x`` at
        the first result — silently overriding the freshness the decorator
        promised (CAS-224). The call target appears in ``inputs`` (the analyzer
        lists ``f`` for ``x = f()``); if it is a cash wrapper with a smaller
        declared TTL, the statement must expire at least as often. ``ttl=0`` then
        rides the existing immediate-expiry path (CAS-221), so every run is a miss
        and the decorated body runs every time, as ``ttl=0`` asks.

        Only LOWERS the TTL and only for a wrapper carrying an explicit TTL, so a
        plain ``@cash.cache`` (ttl=None) call is completely unaffected — the
        statement caches exactly as before.
        """
        user_ns = self.shell.user_ns
        floor = effective_ttl
        for name in inputs:
            fn = user_ns.get(name)
            if fn is None or not getattr(fn, '_cash_cached', False):
                continue
            declared = getattr(fn, '_cash_declared_ttl', None)
            if declared is None:
                continue
            floor = declared if floor is None else min(floor, declared)
        return floor

    @staticmethod
    def _strip_control_markers(code: str) -> str:
        """Drop the per-iteration / per-branch cache-key discriminator comments.

        Control-structure body statements arrive with an ``# __iteration_context__``
        / ``# control_context`` comment prepended, which differs per iteration.
        Stripping it keeps the detector memo AND the per-statement dedupe keyed on
        the body's real source, so a random draw inside a 1000-iteration loop
        warns once, not 1000 times.
        """
        if '# __iteration_context__:' not in code and '# control_context:' not in code:
            return code
        return '\n'.join(
            line for line in code.split('\n')
            if not line.startswith('# __iteration_context__:')
            and not line.startswith('# control_context:')
        )

    def _warn_unseeded_randomness(self, code: str, allow_random: bool) -> list:
        """Warn when *code* draws from an unseeded RNG (CAS-114).

        Called on the common path of both ``process_statement`` twins, BEFORE the
        cache lookup, for two reasons:

        * the warning describes the *source*, so it must not depend on whether
          this particular run hit or missed the cache; and
        * ``analyze_code`` doubles as the detector's seed-tracking hook — a
          ``np.random.seed(42)`` statement must mark its module seeded even on a
          cache hit, or the next cell warns spuriously.

        Returns the unseeded calls it found, so the cache-hit path can report the
        replay without re-running the scan (which would double the seed-tracking
        side effect above).

        Never allowed to break execution: this is advisory output, so a detector
        fault must not take the statement down with it.
        """
        code = self._strip_control_markers(code)
        try:
            unseeded_calls, _has_seed = check_and_warn_randomness(
                code, self.randomness_detector, suppress_warning=allow_random,
            )
            return list(unseeded_calls)
        except (SyntaxError, ValueError, AttributeError, RecursionError):
            logger.debug("%s Randomness detection failed for statement", _LOG_PROCESSOR)
            return []

    def _warn_stale_randomness(
        self, code: str, unseeded_calls: list, allow_random: bool,
    ) -> None:
        """Announce that a cached unseeded random value was just replayed (CAS-135).

        Called ONLY after a restore has actually succeeded, because that is the
        event being reported: not "this statement contains randomness" (true on
        every run, and already covered by ``_warn_unseeded_randomness``), but
        "the number you are looking at is a replay of an earlier run".

        That claim can only be made here.  ``_warn_unseeded_randomness`` runs
        before the cache lookup, where the hit/miss outcome does not exist yet —
        so it can only ever say "may not be reproducible".  On a restore that
        understates it: the value *is* frozen.  CAS-114 warned on the COLD run,
        when the value is freshly computed and correct, and said nothing on the
        restores, when it is not.  This is the missing half.

        Gated on a successful restore specifically: ``_handle_cache_hit``
        returns None when restoration fails and the caller falls through to real
        execution, in which case the value is fresh and "replay" would be a lie.

        The dedupe is deliberately kept.  It is keyed on ``(code, message)``, and
        this message is a different claim from the compute-time one, so it lands
        in its own slot: the replay is announced once per statement per session,
        on the first restore.  Removing the dedupe instead would flood a
        re-run — the very thing CAS-114 avoided so users don't learn to filter
        the whole class away.
        """
        if allow_random or not unseeded_calls:
            return
        try:
            warn_stale_randomness(
                self._strip_control_markers(code), unseeded_calls,
                self.randomness_detector, suppress_warning=allow_random,
            )
        except (SyntaxError, ValueError, AttributeError, RecursionError):
            logger.debug("%s Stale-randomness warning failed for statement", _LOG_PROCESSOR)

    def _unseeded_estimator_fits(self, est_fit: set[str]) -> list[str]:
        """Return the sorted subset of *est_fit* receivers that are UNSEEDED (CAS-167).

        A bare ``estimator.fit(X, y)`` under ``# @cash:cache-fit`` caches via the
        CAS-138 path, but cash's AST
        randomness detector cannot see the randomness inside sklearn's compiled
        ``.fit()``. An estimator built without a ``random_state`` draws fresh
        entropy each fit, so the cached fitted model is a frozen replay -- two
        honest fits would differ. This flags exactly those receivers.

        UNSEEDED iff ``get_params()`` contains ``random_state`` AND it is ``None``.
        An int / ``RandomState`` / ``Generator`` seed -> SEEDED (no warning). No
        ``random_state`` param at all (e.g. ``LinearRegression``) -> deterministic
        (no warning). Any ``get_params`` failure -> no warning: advisory output
        must never crash the statement.
        """
        unseeded: list[str] = []
        for rf in est_fit:
            try:
                est = self.shell.user_ns.get(rf)
                if est is None:
                    continue
                params = est.get_params()
                if 'random_state' in params and params['random_state'] is None:
                    unseeded.append(rf)
            except (AttributeError, TypeError, ValueError, KeyError):
                continue
        return sorted(unseeded)

    def _warn_unseeded_estimator_fit(
        self, code: str, est_fit: set[str], allow_random: bool,
    ) -> list[str]:
        """Warn that an UNSEEDED estimator ``.fit()`` is cached as a frozen replay (CAS-167).

        The estimator-fit analogue of :meth:`_warn_unseeded_randomness`, emitted at
        COMPUTE time on the common path (before the cache lookup) so the warning
        describes the *source* independently of this run's hit/miss outcome. Goes
        out on the SAME ``CashRandomnessWarning`` path as CAS-135, so users'
        existing filters catch it.

        Returns the unseeded receivers found, so the cache-hit path can announce
        the replay without re-deriving them. ``# @cash:allow-random`` suppresses
        the warning (but the set is still returned; the restore twin re-checks the
        directive and stays silent too). Never allowed to break execution.
        """
        if not est_fit:
            return []
        unseeded = self._unseeded_estimator_fits(est_fit)
        if not unseeded:
            return []
        try:
            warn_unseeded_estimator_fit(
                self._strip_control_markers(code), unseeded,
                self.randomness_detector, suppress_warning=allow_random,
            )
        except (ValueError, AttributeError, RecursionError):
            logger.debug("%s Estimator-fit randomness warning failed", _LOG_PROCESSOR)
        return unseeded

    def _warn_stale_estimator_fit(
        self, code: str, unseeded_fits: list[str], allow_random: bool,
    ) -> None:
        """Announce that a cached UNSEEDED estimator fit was just replayed (CAS-167).

        The restore-time twin of :meth:`_warn_unseeded_estimator_fit`, mirroring
        how :meth:`_warn_stale_randomness` follows :meth:`_warn_unseeded_randomness`:
        it makes the stronger claim only a successful restore licenses -- the
        fitted model on screen IS a replay, not merely "may differ". Called ONLY
        after ``_handle_cache_hit`` reports a successful restore.
        """
        if allow_random or not unseeded_fits:
            return
        try:
            warn_stale_estimator_fit(
                self._strip_control_markers(code), unseeded_fits,
                self.randomness_detector, suppress_warning=allow_random,
            )
        except (ValueError, AttributeError, RecursionError):
            logger.debug("%s Stale estimator-fit warning failed", _LOG_PROCESSOR)

    def _do_cache_lookup(
        self,
        skip_cache: bool,
        cache_key: str,
        ttl: int | None,
        inputs: set[str],
    ) -> tuple[StatementCacheMetadata | None, Any | None, float]:
        """Run cache lookup unless *skip_cache* is set."""
        if not skip_cache:
            return self._freshness.check_cache(self._tracking_state, cache_key, ttl, inputs)
        if self.debug:
            logger.debug("%s Skipping cache lookup due to missing input lineage or @cash:no-cache", _LOG_ANNOTATION)
        return None, None, 0.0

    def _observe_miss_guard(
        self,
        skip_cache: bool,
        code: str,
        source_hash: str,
        cache_key: str,
        cached_data: Any,
    ) -> None:
        """Feed one lookup outcome to the perpetual-miss guard (CAS-172).

        Called on every run that actually performed a lookup — a skipped lookup
        never serialises either, so it carries no evidence about whether
        serialising pays back.

        A *hit* is "the key matched an entry", i.e. ``cached_data`` is not None.
        A key that matched but whose entry was invalidated (TTL / changed file
        dep) reads as a miss here, and correctly so: the key was STABLE, so it
        registers no churn and cannot move the counter. That workflow —
        recompute because a file changed, cache for the next unchanged run — is
        exactly the one that must never be guarded.

        Control-structure BODY statements are excluded, following the same
        precedent as mutation classification: they arrive with a per-iteration
        marker comment, so every iteration is a different ``source_hash``. The
        "identical source, run repeatedly" signature is meaningless for them, and
        recording one per iteration would grow the store by the loop's trip
        count.
        """
        if skip_cache:
            return
        if '# __iteration_context__:' in code or '# control_context:' in code:
            return
        self._miss_guard.observe(source_hash, cache_key, hit=cached_data is not None)

    def _execute_and_drain(
        self,
        code: str,
        stream_output: bool,
        skip_cache: bool,
        tree: ast.Module | None,
        metrics: ProcessResult,
        process_start: float,
        silent: bool,
        is_last: bool = True,
    ) -> tuple[ProcessResult | None, Any, Any, float, set[str]]:
        """Execute the statement, drain decorator calls, populate stdout/stderr in metrics.

        Returns ``(error_metrics, result, captured, execution_time, accessed_files)``.
        *error_metrics* is non-None only when execution fails; callers should return it.
        """
        if self.debug:
            logger.debug("%s Executing (cache miss)", _LOG_CACHE_DEBUG)

        result, captured, execution_time, accessed_files = self._execute_statement(
            code, stream_output=stream_output, tree=tree,
            skip_capture=(skip_cache and stream_output), is_last=is_last,
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

        self._display_execution_output(captured, execution_time, silent, stream_output, metrics)
        metrics['execution_time'] = execution_time

        if not result.success:
            metrics['status'] = CacheStatus.ERROR
            metrics['error'] = result.error
            metrics['total_time'] = time.time() - process_start
            self._handle_execution_error(result, silent)
            return metrics, result, captured, execution_time, accessed_files

        return None, result, captured, execution_time, accessed_files

    async def _execute_and_drain_async(
        self,
        code: str,
        stream_output: bool,
        skip_cache: bool,
        tree: ast.Module | None,
        metrics: ProcessResult,
        process_start: float,
        silent: bool,
        is_last: bool = True,
    ) -> tuple[ProcessResult | None, Any, Any, float, set[str]]:
        """Async twin of :meth:`_execute_and_drain`.

        Identical to the sync version except it awaits
        :meth:`_execute_statement_async` so a top-level-await statement runs on
        IPython's live loop.  All post-execution bookkeeping (decorator drain,
        stdout/stderr/rich capture, output display, error handling) is byte-for-
        byte the same.
        """
        if self.debug:
            logger.debug("%s Executing (cache miss)", _LOG_CACHE_DEBUG)

        result, captured, execution_time, accessed_files = await self._execute_statement_async(
            code, stream_output=stream_output, tree=tree,
            skip_capture=(skip_cache and stream_output), is_last=is_last,
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
        metrics['rich_outputs'] = captured.outputs
        if decorator_calls:
            metrics['decorator_calls'] = decorator_calls

        self._display_execution_output(captured, execution_time, silent, stream_output, metrics)
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
        mut_observe: set[str] = frozenset(),
        mut_assumed: set[str] = frozenset(),
        mut_record: bool = False,
        est_fit: set[str] = frozenset(),
    ) -> None:
        """Auto-track imports, capture vars, detect mutations, save to cache, record analytics."""
        # Broad-precise mutation observation: for a standalone method call whose
        # method is not statically known, compare each candidate receiver's
        # content after execution against its pre-statement hash. Must run BEFORE
        # capture_and_track so a newly-detected mutation is in ``outputs`` (its
        # lineage gets bumped) and skip-caches the statement. The verdict is
        # recorded for the upstream simulation, which cannot observe execution.
        if mut_record:
            newly_mutated = {b for b in mut_observe if self._receiver_mutated(b)}
            if newly_mutated:
                # ``est_fit`` is non-empty only under ``# @cash:cache-fit``
                # (CAS-170). Those receivers still enter ``outputs`` (source-based
                # lineage bump + fitted value capture) and are still recorded in
                # ``mutation_verdicts`` below (so the upstream simulation bumps
                # downstream lineage on a data edit), but they are NOT
                # skip-cached -- they cache + restore in place (CAS-138). Every
                # other observed mutation -- including a bare fit WITHOUT the
                # directive -- still skip-caches its receiver.
                outputs = outputs | newly_mutated
                skip_observed = newly_mutated - est_fit
                if skip_observed:
                    skip_cache = True
                    metrics.setdefault('uncacheable_reasons', []).append(
                        f"In-place mutation on: {', '.join(sorted(skip_observed))} "
                        "(observed; receiver lineage bumped; statement re-executes)"
                    )
            self.mutation_verdicts[source_hash] = set(mut_assumed) | newly_mutated

        # Auto-track newly imported local modules so _capture_variables includes
        # the module source hash in the lineage on first execution.
        try:
            self.function_tracker.auto_track_local_imports(code)
        except (ImportError, AttributeError, OSError):
            logger.debug("%s Failed to auto-track local imports", _LOG_PROCESSOR)

        captured_vars = self._lineage.capture_and_track_variables(
            self._tracking_state, outputs, inputs, code, source_hash,
            cache_key=cache_key, accessed_files=accessed_files, tree=tree,
        )

        # A statement producing a live-alias object (numpy view, pandas
        # groupby/rolling ref-holder) must NOT be cached: pickling and restoring
        # such an object decouples it from its live base, so a later base
        # mutation would be lost after restore. Force re-derivation from the live
        # base instead (CAS-115 / CAS-89). ``.copy()`` produces no alias and stays
        # cacheable (over-invalidation guard).
        if not skip_cache:
            from .derivation_edges import is_uncacheable_alias
            for out in outputs:
                val = captured_vars.get(out)
                if val is not None and is_uncacheable_alias(val, self.shell.user_ns):
                    skip_cache = True
                    metrics.setdefault('uncacheable_reasons', []).append(
                        f"Live-alias object '{out}' (view/ref-holder); re-derived "
                        "from live base, not cached (CAS-115/89)."
                    )
                    break

        # A statement producing an object that is identity-coupled to a library
        # global (a matplotlib Figure/Axes vs pyplot's ``Gcf`` current-figure
        # registry) must NOT be cached. The RAM tier deep-copies on store, and a
        # Figure's ``__setstate__`` re-registers the COPY as pyplot's current
        # figure -- so the user draws on their figure while ``plt.savefig()``
        # writes the cache's snapshot: a blank PNG on the FIRST run, silently.
        # This must run here (post-execution) rather than in decide_cacheability:
        # the object does not exist yet when that runs. Refusing BEFORE
        # _save_to_cache is what prevents the deep-copy from ever happening
        # (CAS-144).
        if not skip_cache:
            for out in outputs:
                val = captured_vars.get(out)
                if val is None:
                    continue
                reason = identity_coupled_reason(out, val)
                if reason is not None:
                    skip_cache = True
                    metrics.setdefault('uncacheable_reasons', []).append(reason)
                    break

        # Record executed file-WRITING statements by code text (CAS-81/82):
        # writes have no variable edge, so the upstream simulation needs this
        # to tell an edited/new writer from one that already ran.
        try:
            if any(e.kind == 'file_write' for e in statement_analysis.side_effects):
                self._tracking_state.executed_write_stmt_codes.add(code)
                # Persist write provenance so a post-restart isolated reader can
                # tell an already-on-disk writer effect (skip it) from a stale
                # one (re-fire it) — ``executed_write_stmt_codes`` is empty after
                # a restart, which used to force every writer to re-fire and
                # re-run its non-idempotent side effect (CAS-153 round-3).
                self._persist_write_provenance(code, inputs, tree)
        except AttributeError:
            pass

        # Detect in-place mutations (detection-only; do not modify lineage).
        # Reuses the StatementAnalysis from process_statement to avoid a
        # second pass of AST visitors over the same tree.
        pure_mutations = statement_analysis.all_mutated_vars - outputs
        if pure_mutations:
            self.vars_with_mutation_lineage.update(pure_mutations)
            if self.debug:
                logger.debug("%s Detected in-place mutations on: %s", _LOG_MUTATION, pure_mutations)

        # Perpetual-miss guard (CAS-172): this statement's key has churned for
        # ``GUARD_AFTER_CONSECUTIVE_CHURN_MISSES`` runs with zero hits, so
        # serialising it again buys nothing. Routed through the SAME
        # metadata-only path as the size-aware skip rather than through
        # ``skip_cache``: output lineages must still persist for the upstream
        # simulation, and only the value payload is the wasted cost.
        # ``force_persist`` (``# @cash:persist`` / ``%cash_persist``) wins — a
        # user who explicitly asks for persistence gets it; the guard is a
        # default, not a veto.
        miss_guarded = (
            not skip_cache
            and not force_persist
            and not self._miss_guard.should_serialise(source_hash)
        )

        saved_metadata = None
        if not skip_cache:
            saved_metadata = self._save_to_cache(
                cache_key, code, result, inputs, outputs, accessed_files,
                execution_time, effective_ttl, captured, process_start,
                source_hash, captured_vars, force_persist=force_persist,
                miss_guarded=miss_guarded,
            )
        elif self.debug:
            logger.debug("%s Skipping cache save due to @cash:no-cache", _LOG_ANNOTATION)

        if saved_metadata and saved_metadata.storage is not None:
            metrics['storage'] = saved_metadata.storage
        if saved_metadata and saved_metadata.skipped_reason is not None:
            metrics['skipped_reason'] = saved_metadata.skipped_reason
        if saved_metadata:
            for k in _COST_MODEL_KEYS:
                value = getattr(saved_metadata, k)
                if value is not None:
                    metrics[k] = value

        metrics['total_time'] = time.time() - process_start
        self.analytics_manager.record_event(
            status='MISS',
            execution_time=metrics['total_time'],
            saved_time=0.0,
            code_hash=cache_key,
        )

    def _persist_write_provenance(
        self,
        code: str,
        inputs: set[str],
        tree: ast.Module | None,
    ) -> None:
        """Record what file(s) a just-executed writer statement produced (CAS-153).

        Persists ``{paths, file_deps snapshot, input lineages}`` to the backend
        under a writer-specific key derived from the statement source, so a
        post-restart isolated downstream reader can short-circuit an
        already-fresh writer (:meth:`ReexecutionPlanner._writer_output_already_fresh`)
        instead of re-firing a non-idempotent side effect and re-deriving stale
        data. Best-effort and CONSERVATIVE: an unresolvable output path (f-string
        / computed) records nothing, so that writer keeps re-firing as before.
        """
        try:
            from cash.notebook.cacheability import statement_written_paths

            raw_paths = statement_written_paths(code, tree, self.shell.user_ns)
            if not raw_paths:
                return  # path(s) not statically resolvable -> stay conservative
            paths = sorted({os.path.abspath(p) for p in raw_paths})
            file_deps = snapshot_file_deps(set(paths))
            # Every recorded path must be readable now, else there is nothing to
            # vouch for (and a later freshness check would fail anyway).
            if any(p not in file_deps for p in paths):
                return
            input_lineages = {
                v: self.variable_lineage[v]
                for v in inputs
                if v in self.variable_lineage
            }
            record = {
                'write_provenance': True,
                'paths': paths,
                'file_deps': file_deps,
                'input_lineages': input_lineages,
                'code': code,
                'ttl': None,  # provenance must not expire out from under a reader
            }
            backend = self.cash_instance.backend if self.cash_instance else None
            if backend is not None:
                self._stmt_restorer.persist_metadata_only(
                    backend, write_provenance_key(code), record,
                )
        except (OSError, TypeError, ValueError, AttributeError):
            logger.debug("%s write-provenance persistence failed", _LOG_PROCESSOR)

    def _identity_coupled_call_receivers(self, tree: ast.Module | None) -> set[str]:
        """Receiver names in *tree* that are live matplotlib Figures/Axes (CAS-220).

        The narrow companion to :meth:`_classify_method_mutations`, for the one
        case that must survive the control-body skip. A body statement carries an
        injected marker comment and its method-mutation classification is skipped
        wholesale, because bumping a receiver with a per-statement source the
        upstream simulation never reproduces desyncs the loop. That is right for
        ordinary receivers and wrong for a live Axes: ``ax.bar(...)`` in a loop
        body has NO outputs, so it is cached as an ordinary no-output call and
        restored as a no-op, while the sibling ``fig.savefig(...)`` still
        executes because it writes a file. The draw is skipped, the write is
        not, and the deliverable PNG is blank.

        Identity-coupling is the same single discriminator CAS-194 used to tell
        ``ax.hist()`` (draws on an Axes) from ``df.hist()`` (receiver-pure), and
        CAS-199 used to widen scope to captured-return draws without
        over-invalidating. It imports no matplotlib and is False for everything
        that is not a Figure/Axes, so no loop that caches today stops caching.

        Callers use this to skip the CACHE only. It deliberately does NOT feed
        ``outputs``: the lineage bump is the part the control-body skip exists to
        prevent, and re-executing a draw needs none of it.
        """
        if tree is None:
            return set()
        receivers: set[str] = set()
        for base, _method in (
            standalone_method_call_receivers(tree) | assigned_method_call_receivers(tree)
        ):
            value = self.shell.user_ns.get(base)
            if isinstance(value, types.ModuleType):
                continue  # ``plt.savefig()`` is a module call, not a receiver draw
            if receiver_is_identity_coupled(value):
                receivers.add(base)
        return receivers

    def _classify_method_mutations(
        self,
        tree: ast.Module | None,
        source_hash: str,
        outputs: set[str],
    ) -> tuple[set[str], set[str], set[str], bool]:
        """Classify a statement's standalone method-call receivers.

        Returns ``(pre_route, observe, assumed, record_verdict)``:

        * ``pre_route`` — receivers to route into outputs + skip-cache now
          (statically known-mutating; a prior runtime verdict says it mutates;
          or assume-mutate because the receiver can't be reliably content-hashed
          — minus anything already in ``outputs``).
        * ``observe`` — tier-3 receivers to content-observe post-execution
          (verdict unknown, receiver reliably hashable).
        * ``assumed`` — tier-3 receivers assumed-mutating without observation
          (recorded into the verdict so the simulation reproduces them).
        * ``record_verdict`` — True when this statement's verdict is being learned.
        """
        candidates = standalone_method_call_receivers(tree)
        # CAS-199: captured-return draws (``counts, bins, _ = ax.hist(...)``) are
        # assignments, so they never appear in the bare-``Expr`` candidate set;
        # they are classified by the identity-coupled pass below and must not be
        # short-circuited by the ``not candidates`` guard.
        assigned = assigned_method_call_receivers(tree)
        if not candidates and not assigned:
            return set(), set(), set(), False
        tier1 = standalone_method_mutation_receivers(tree)
        verdict = self.mutation_verdicts.get(source_hash)
        pre_route: set[str] = set()
        observe: set[str] = set()
        assumed: set[str] = set()
        for base, method in candidates:
            receiver = self.shell.user_ns.get(base)
            if isinstance(receiver, types.ModuleType):
                continue  # ``time.sleep()`` / ``np.foo()`` is a module function
                          # call, not a method mutation of the receiver.
            if base in tier1:
                pre_route.add(base)
                continue
            if method in RECEIVER_READONLY_WRITE_METHODS:
                # ``df.to_csv(path)`` READS the frame and writes a file; it does
                # not mutate ``df``, so it must not bump its lineage (CAS-196).
                # The file-write side effect is scheduled elsewhere. (``savefig``
                # is intentionally NOT here — see the identity-coupled branch.)
                continue
            if receiver_is_identity_coupled(receiver):
                # A method call on a live matplotlib Axes/Figure DRAWS on it — it
                # adds artists / sets state — whatever it returns. ``ax.hist(...)``
                # returns a data tuple yet mutates the Axes just like ``ax.bar()``;
                # route it to the mutation path so the figure's fill statements are
                # rebuilt with it and never cached as an ordinary value (CAS-194).
                # ``fig.savefig(...)`` lands here too: bumping an identity-coupled
                # Figure is idempotent + load-bearing for CAS-175 chart coherence.
                pre_route.add(base)
                continue
            if method in KNOWN_PURE_METHODS:
                continue
            if verdict is not None:
                if base in verdict:
                    pre_route.add(base)
                continue
            # tier-3, verdict unknown: observe if cheaply+reliably hashable,
            # otherwise assume-mutate (conservative, correctness-first).
            if self._receiver_observable(base):
                observe.add(base)
            else:
                assumed.add(base)
                pre_route.add(base)
        # CAS-199: the CAPTURED-return form ``counts, bins, _ = ax.hist(...)`` is
        # an ``ast.Assign``, so the bare-``Expr`` candidate set above never saw
        # it. Route its receiver as a draw too — but ONLY when it is identity-
        # coupled (a live Axes/Figure). That single discriminator is what keeps a
        # genuine pure capture (``m = df.mean()``, DataFrame receiver) on the
        # caching path: the general tier-3 "assume-mutate" logic above is
        # deliberately NOT applied here, so a non-coupled captured receiver is
        # never over-invalidated.
        for base, _method in assigned:
            if base in pre_route:
                continue
            receiver = self.shell.user_ns.get(base)
            if isinstance(receiver, types.ModuleType):
                continue
            if receiver_is_identity_coupled(receiver):
                pre_route.add(base)
        record_verdict = verdict is None and bool(observe or assumed)
        return pre_route - outputs, observe, assumed, record_verdict

    def _estimator_fit_receivers(
        self,
        tree: ast.Module | None,
        outputs: set[str],
    ) -> set[str]:
        """Receivers of a standalone ``est.fit(...)`` / ``est.partial_fit(...)``
        whose live value is a duck-typed sklearn estimator (CAS-138).

        Called ONLY for a statement carrying ``# @cash:cache-fit`` (CAS-170); the
        default is to leave a bare fit on the skip-cache path, where it
        re-executes.  That re-execution does NOT by itself make aliases correct —
        ``backup = model`` breaks on its own restore, not on the fit (CAS-184).

        A bare ``model.fit(X, y)`` mutates its receiver in place, so the general
        mutation classifier routes it to skip-caching. But a fit is the most
        expensive cell in an ML notebook and its cache key is already
        input-lineage-based (the estimator is an input), so a user who asks for it
        can have it cached. This narrow gate selects ONLY sklearn-style
        estimators: the ``fit`` / ``partial_fit`` method name plus the
        ``BaseEstimator`` duck-type contract -- a callable ``fit`` AND a callable
        ``get_params``. ``get_params`` is what excludes ``list.append`` /
        ``dict.update`` and a generic object that merely happens to expose a
        ``fit`` method, so the estimator-caching path never loosens general
        mutation caching.

        Modules are excluded (mirroring ``_classify_method_mutations``): a
        ``pkg.fit(...)`` module-function call is not a receiver mutation. Names
        already surfaced as AST outputs are excluded too -- those are produced by
        an assignment (a fresh binding each run), so an in-place transfer onto a
        pre-existing object would be wrong for them.
        """
        candidates = standalone_method_call_receivers(tree)
        if not candidates:
            return set()
        receivers: set[str] = set()
        for base, method in candidates:
            if method not in ('fit', 'partial_fit'):
                continue
            v = self.shell.user_ns.get(base)
            if isinstance(v, types.ModuleType):
                continue
            if callable(getattr(v, 'fit', None)) and callable(getattr(v, 'get_params', None)):
                receivers.add(base)
        return receivers - outputs

    def _receiver_observable(self, base: str) -> bool:
        """Return True if *base*'s value can be reliably content-hashed.

        ``compute_hash`` *samples* large objects (DataFrame/Series/ndarray, and
        collections over 200 elements), so an unchanged sample can't prove the
        object wasn't mutated outside the sample. Such receivers are excluded
        here and assume-mutated instead.
        """
        val = self.shell.user_ns.get(base)
        if val is None:
            return False
        if type(val).__name__ in ('DataFrame', 'Series', 'ndarray'):
            return False
        if isinstance(val, (list, tuple, dict, set, frozenset)) and len(val) > 200:
            return False
        return True

    def _receiver_mutated(self, base: str) -> bool:
        """Observe whether *base*'s content changed during this statement.

        Compares the post-execution content hash against the pre-statement hash
        in ``current_session_hashes``. Conservative (returns True) when the value
        is absent, has no prior recorded hash, or is unpicklable — in which case
        ``compute_hash`` returns an identity hash that can't reflect an in-place
        mutation.
        """
        val = self.shell.user_ns.get(base)
        if val is None:
            return True
        try:
            after = self.compute_hash(val)
        except (TypeError, ValueError, AttributeError, pickle.PicklingError):
            return True
        identity_hash = hashlib.sha256(str(id(val)).encode('utf-8')).hexdigest()
        if after == identity_hash:
            return True  # unpicklable -> identity hash -> mutation undetectable
        before = self.current_session_hashes.get(base)
        return before is None or after != before

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
        cache_key: str,
        inputs: set[str],
        metrics: ProcessResult,
        process_start: float,
        inplace_restore: set[str] = frozenset(),
    ) -> ProcessResult | None:
        """Restore from cache and populate *metrics* for a cache-hit path.

        Returns the completed *metrics* dict on success, or ``None`` if
        restoration fails (caller should fall through to execution).

        *inplace_restore* names the estimator-fit receivers whose fitted state
        must be transferred onto the EXISTING object rather than rebound, so
        every alias observes the fit (CAS-138). It is recomputed each call from
        the live namespace (never read from ``mutation_verdicts``, which is empty
        right after a kernel restart).
        """
        try:
            if self.debug:
                logger.debug("%s Cache hit for key: %s...", _LOG_CACHE_HIT, cache_key[:20])
                logger.debug("%s Input lineages used: %s", _LOG_CACHE_HIT, [(v, self.variable_lineage.get(v, 'NONE')[:16] + '...') for v in inputs if v not in ['get_ipython', '__builtins__', 'print']])
                if metadata:
                    logger.debug("%s Stored lineages in cache: %s", _LOG_CACHE_HIT, [(k, v[:16]+'...') for k,v in (metadata.output_lineages or {}).items()])
            self._stmt_restorer.restore_from_cache(self._tracking_state, cached_data, metadata, silent, process_start, inplace_restore)

            metrics['status'] = CacheStatus.RESTORED
            metrics['saved_time'] = (metadata.execution_time or 0.0) if metadata else 0.0
            metrics['restored_vars'] = (metadata.outputs or []) if metadata else []
            # Carry the stored input list through so provenance/audit can
            # reconstruct the dependency graph on a cache hit, not just on
            # a fresh compute.
            metrics['inputs'] = list((metadata.inputs or []) if metadata else [])
            metrics['total_time'] = time.time() - process_start

            if metadata:
                if metadata.source is not None:
                    metrics['source'] = metadata.source
                    metrics['storage'] = [metadata.source]
                elif metadata.storage is not None:
                    metrics['storage'] = metadata.storage
                for k in _COST_MODEL_KEYS:
                    value = getattr(metadata, k)
                    if value is not None:
                        metrics[k] = value

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
                metrics['rich_outputs'] = payload.get('rich_outputs', [])
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
        """Replay a list of rich display outputs.

        Raises ImportError without IPython — see the module-header note: a
        caller replaying rich output expects it to render, so failing loudly
        beats silently dropping it.  The ``if not outputs`` guard keeps the
        common no-rich-output path off the import entirely.
        """
        if not outputs:
            return

        from IPython.display import display, publish_display_data

        for output in outputs:
            if isinstance(output, dict) and 'data' in output:
                publish_display_data(data=output['data'], metadata=output.get('metadata', {}))
            else:
                display(output)

    def _display_execution_output(self, captured: Any, execution_time: float, silent: bool, stream_output: bool, metrics: ProcessResult) -> None:
        """Display captured stdout/stderr/rich outputs after execution."""
        if stream_output:
            # User already saw output in real-time via _TeeWriter.
            metrics['_output_flushed'] = True
            if not silent:
                self._publish_rich_outputs(captured.outputs)
        elif not silent:
            if captured.stdout:
                print(captured.stdout, end='')
            if captured.stderr:
                print(captured.stderr, end='', file=sys.stderr)
            self._publish_rich_outputs(captured.outputs)

    @staticmethod
    def _make_capture_ctx(stream_output: bool, skip_capture: bool) -> Any:
        """Return the output-capture context manager for an execution.

        Shared by the sync and async executors so both wrap user code in the
        exact same stdout/stderr/display capture (only the exec primitive
        inside differs).
        """
        if stream_output and skip_capture:
            class _EmptyCaptured:
                stdout = ''
                stderr = ''
                outputs = []
            return contextlib.nullcontext(_EmptyCaptured())
        if stream_output:
            return _tee_output()
        return capture_output(stdout=True, stderr=True, display=True)

    def _execute_statement(self, code: str, stream_output: bool = False, tree: ast.Module | None = None, skip_capture: bool = False, is_last: bool = True) -> tuple[Any, Any, float, set[str]]:
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
            ctx_manager = self._make_capture_ctx(stream_output, skip_capture)
            with ctx_manager as captured:
                with FileAccessTracker(self.shell.user_ns) as file_tracker:
                    if tree is None:
                        try:
                            tree = ast.parse(code)
                        except SyntaxError:
                             # Fallback to standard exec if parse fails (though it shouldn't if compiled worked, but good for safety)
                             tree = None

                    # One linecache-registered filename per statement, so a
                    # traceback inside a function DEFINED here shows its source
                    # instead of "<cash>" with no line (CAS-201).
                    cash_file = register_cell_source(code)

                    if tree and tree.body and isinstance(tree.body[-1], ast.Expr):
                         body_nodes = tree.body[:-1]
                         last_node = tree.body[-1]

                         if body_nodes:
                             mod = ast.Module(body=body_nodes, type_ignores=[])
                             # Locations must be fixed for some python versions/ast nodes
                             # but usually parse provides them.
                             c_body = compile(mod, cash_file, 'exec')
                             exec(c_body, self.shell.user_ns, self.shell.user_ns)

                         expr_val = last_node.value
                         mod_expr = ast.Expression(body=expr_val)
                         ast.fix_missing_locations(mod_expr)
                         c_expr = compile(mod_expr, cash_file, 'eval')
                         result_val = eval(c_expr, self.shell.user_ns, self.shell.user_ns)

                         # IPython echoes only the LAST expression of a CELL. Cash
                         # splits the cell into statements and executes each as its
                         # own unit, so without ``is_last`` every bare expression
                         # got displayed and cash silently changed notebook
                         # semantics -- `a+1 / a+2 / a+3` printed 2,3,4 where a
                         # plain kernel prints 4 (CAS-174). Gating the DISPLAY (not
                         # the cache-keyed source) is deliberate: the reverted
                         # attempt appended ';' to the keyed source in the runtime
                         # only, desyncing it from the simulator's unparse and
                         # blanking a chart.
                         # A trailing ``;`` suppresses the repr in IPython. The
                         # cell splitter re-attaches it after ``ast.unparse``
                         # (CAS-96); honour it so no repr is displayed OR captured
                         # (an empty capture then also restores cleanly).
                         if (is_last and result_val is not None
                                 and not code.rstrip().endswith(';')):
                             from IPython.display import display
                             display(result_val)
                    else:
                        compiled_code = compile(code, cash_file, 'exec')
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

    async def _execute_statement_async(self, code: str, stream_output: bool = False, tree: ast.Module | None = None, skip_capture: bool = False, is_last: bool = True) -> tuple[Any, Any, float, set[str]]:
        """Async twin of :meth:`_execute_statement` for top-level-await cells.

        Byte-for-byte the same output-capture / file-tracking / last-expr
        display / error handling as the sync path — the ONLY difference is the
        exec primitive: every unit is compiled under
        ``ast.PyCF_ALLOW_TOP_LEVEL_AWAIT`` and, when the compiled code object
        carries ``CO_COROUTINE`` (i.e. it contains a top-level ``await``), the
        coroutine returned by ``exec``/``eval`` is awaited on IPython's live
        loop.  A plain statement compiled under the flag does NOT get
        ``CO_COROUTINE``, so it runs through the identical synchronous
        ``exec``/``eval`` and behaves exactly like the sync path.

        This mirrors IPython's own ``run_code`` pattern (compile under the flag,
        ``await`` when ``CO_COROUTINE``), so a top-level-await statement executes
        exactly once, on the same loop IPython would have used.
        """
        start_time = time.time()
        accessed_files = set()
        _FLAG = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT

        try:
            ctx_manager = self._make_capture_ctx(stream_output, skip_capture)
            with ctx_manager as captured:
                with FileAccessTracker(self.shell.user_ns) as file_tracker:
                    if tree is None:
                        try:
                            tree = ast.parse(code)
                        except SyntaxError:
                            tree = None

                    # Same per-statement linecache registration as the sync path
                    # so an await-bearing cell's tracebacks resolve too (CAS-201).
                    cash_file = register_cell_source(code)

                    if tree and tree.body and isinstance(tree.body[-1], ast.Expr):
                        body_nodes = tree.body[:-1]
                        last_node = tree.body[-1]

                        if body_nodes:
                            mod = ast.Module(body=body_nodes, type_ignores=[])
                            c_body = compile(mod, cash_file, 'exec', flags=_FLAG)
                            coro = eval(c_body, self.shell.user_ns, self.shell.user_ns)
                            if c_body.co_flags & inspect.CO_COROUTINE:
                                await coro

                        expr_val = last_node.value
                        mod_expr = ast.Expression(body=expr_val)
                        ast.fix_missing_locations(mod_expr)
                        c_expr = compile(mod_expr, cash_file, 'eval', flags=_FLAG)
                        result_val = eval(c_expr, self.shell.user_ns, self.shell.user_ns)
                        if c_expr.co_flags & inspect.CO_COROUTINE:
                            result_val = await result_val

                        # Same last-expression-only rule as the sync path (CAS-174).
                        if (is_last and result_val is not None
                                and not code.rstrip().endswith(';')):
                            from IPython.display import display
                            display(result_val)
                    else:
                        compiled_code = compile(code, cash_file, 'exec', flags=_FLAG)
                        coro = eval(compiled_code, self.shell.user_ns, self.shell.user_ns)
                        if compiled_code.co_flags & inspect.CO_COROUTINE:
                            await coro
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
        self._lineage.capture_and_track_variables(self._tracking_state, outputs, inputs, code, source_hash, cache_key=cache_key, accessed_files=accessed_files, tree=tree)

    def _save_to_cache(self, cache_key: str, code: str, result: Any, inputs: set[str], outputs: set[str], accessed_files: set[str], execution_time: float, ttl: int | None, captured: Any, process_start: float, source_hash: str, captured_vars: dict[str, Any], force_persist: bool = False, miss_guarded: bool = False) -> StatementCacheMetadata | None:
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
            force_persist=force_persist,
            miss_guarded=miss_guarded,
        )


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
          pickle + I/O.  Restore cost is predicted by the fitted cost model
          in ``cash.notebook.cost_model`` (per-family ``a + b·size_bytes``).
          Overhead = 2 × serialise time (store + restore).

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

    @staticmethod
    def _amplification_size(prediction: dict[str, Any] | None) -> int:
        """Size of the largest output var, or 0 when it isn't usable.

        Reads the estimate ``_should_skip_large_object_caching`` already
        computed, so the guard adds no sizing work to the write path.
        """
        if prediction is None:
            return 0
        try:
            size = int(prediction.get('size_bytes') or 0)
        except (TypeError, ValueError):
            return 0
        return max(size, 0)

    def _amplification_stmt_id(self, code: str) -> str | None:
        """Per-statement accounting key, or ``None`` if it cannot amplify.

        Only a statement replayed under a control structure writes more than
        once per run, so only those are accounted. Stripping the per-iteration
        discriminator comment makes every iteration of one loop body share a
        counter; an ordinary ``# @cash:persist`` on a single statement has no
        marker, gets ``None`` here, and is untouched by the whole mechanism.
        """
        if ('# __iteration_context__:' not in code
                and '# control_context:' not in code):
            return None
        return self._strip_control_markers(code).strip()

    def _check_persist_amplification(
        self,
        code: str,
        prediction: dict[str, Any] | None,
    ) -> tuple[bool, str | None]:
        """Return ``(skip, reason)`` for the CAS-160 loop-persist guard.

        Consulted immediately before a value-persist: refuse once this
        statement's cumulative *durably stored* bytes are out of all proportion
        to the value being stored. The counter is fed by
        :meth:`_account_persisted_bytes` after the write actually lands.

        Two thresholds must BOTH be crossed, which is what keeps the guard off
        healthy notebooks: an absolute floor
        (``_PERSIST_AMPLIFICATION_FLOOR_BYTES``), so small loops never engage at
        all, and a ratio against the current value, so a loop that legitimately
        stores a lot of *distinct* results is judged on proportion rather than
        volume.

        The verdict LATCHES per statement: once a statement has demonstrated
        amplification, later iterations stay metadata-only. Without the latch the
        guard would disengage exactly when it matters -- the running total
        freezes while the object keeps growing, so ``LIMIT x size`` would
        eventually overtake it and the writes would resume mid-loop.
        """
        size = self._amplification_size(prediction)
        if size <= 0:
            return False, None
        stmt_id = self._amplification_stmt_id(code)
        if stmt_id is None:
            return False, None

        if stmt_id in self._warned_persist_amplification:
            return True, AMPLIFICATION_SKIP_REASON

        cumulative = self._persist_bytes_by_stmt.get(stmt_id, 0)
        if (cumulative > _PERSIST_AMPLIFICATION_FLOOR_BYTES
                and cumulative > _PERSIST_AMPLIFICATION_LIMIT * size):
            self._warned_persist_amplification.add(stmt_id)
            self._warn_persist_amplification(stmt_id, cumulative, size)
            return True, AMPLIFICATION_SKIP_REASON
        return False, None

    def _account_persisted_bytes(
        self,
        code: str,
        prediction: dict[str, Any] | None,
        wire: dict[str, Any],
    ) -> None:
        """Add a completed write to its statement's running total (CAS-160).

        Counts a write only when it reached a **persistent** tier. The backend
        reports the resolved destinations back on the metadata dict, so this is
        a read of information the write already produced.

        Excluding RAM-only writes is what makes the guard track the resource the
        user is actually losing. A loop body that misses the promotion floor is
        cached in RAM and never touches the disk at all; counting those would
        warn about "filling your disk" for a notebook whose disk cache is a few
        KB, which is both false and noisy.
        """
        stmt_id = self._amplification_stmt_id(code)
        if stmt_id is None:
            return
        size = self._amplification_size(prediction)
        if size <= 0:
            return
        destinations = wire.get('storage') or ()
        if not isinstance(destinations, (list, tuple)):
            return
        if not any(d != 'RAM' for d in destinations):
            return
        self._persist_bytes_by_stmt[stmt_id] = (
            self._persist_bytes_by_stmt.get(stmt_id, 0) + size
        )

    def _warn_persist_amplification(
        self, stmt_id: str, cumulative: int, size: int
    ) -> None:
        """Warn once that a looped persist is snapshotting a growing object.

        Names the amplification in the user's own terms -- what it has already
        written versus how big the value actually is -- and points at the fix,
        which is to persist the finished object once instead of every
        intermediate state of it.
        """
        import warnings

        from cash.backends.adaptive_caps import human_bytes
        from cash.exceptions import CashCacheIneffectiveWarning

        first_line = (stmt_id.splitlines() or [''])[0].strip()
        if len(first_line) > 60:
            first_line = first_line[:57] + '...'
        warnings.warn(
            f"Cash: `{first_line}` runs in a loop and has already cached "
            f"{human_bytes(cumulative)} of intermediate snapshots for a value "
            f"that is currently only {human_bytes(size)} -- caching a growing "
            f"object every iteration costs the SUM of every intermediate size, "
            f"not the final one. Further iterations are not being stored. Move "
            f"`# @cash:persist` off the loop and onto a statement that produces "
            f"the finished object, so it is cached once.",
            CashCacheIneffectiveWarning,
            stacklevel=2,
        )

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
        force_persist: bool = False,
        miss_guarded: bool = False,
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
            # ``execution_time`` is wall clock (``time.time()``), not CPU time,
            # so it charges the statement for any scheduling stall too. On
            # Windows the clock can report exactly 0.0 for a genuinely
            # instantaneous statement (a = 1) because its resolution is coarser
            # than the operation; treat 0 the same as "below the floor" — both
            # mean "too cheap to cache". The converse also holds: on a heavily
            # contended machine a trivial statement can measure tens of ms and
            # legitimately clear the floor, so nothing may assume this branch is
            # taken for a given statement (see the floor-exit test, which pins
            # the threshold rather than trusting the machine to be fast).
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

        # Statements whose outputs include a __main__-defined function or
        # class are never VALUE-cached (CAS-93): those values pickle BY
        # REFERENCE to a binding that won't exist in the next session, so a
        # value entry would either crash the lookup (dangling find_class ->
        # AttributeError) or "restore" nothing and wrongly skip the defining
        # statement. Route them through the metadata-only path (same as the
        # size-aware skip): output lineages persist for the upstream
        # simulation, while the value lookup misses cleanly and the cheap
        # statement (lambda assign, `g = f` alias) re-executes.
        if not should_skip:
            _unrestorable = sorted(
                _name for _name, _v in captured_vars.items()
                if (inspect.isfunction(_v) or inspect.isclass(_v))
                and getattr(_v, '__module__', None) == '__main__'
            )
            if _unrestorable:
                should_skip = True
                skip_reason = (
                    f"__main__ function/class output(s) "
                    f"{', '.join(_unrestorable)} are unrestorable by value; "
                    f"statement re-executes (lineage persists)"
                )
        # Perpetual-miss guard (CAS-172). Placed LAST so it can override the
        # exemptions above: ``has_file_dependencies`` waives the whole size-aware
        # cost model, and that waiver is precisely how CAS-165/171 shipped — a fit
        # on a CSV-derived frame inherits the read's file deps, so the cost model
        # never got a vote and the frame was re-serialised every run for a cache
        # that could never hit. An unstable key does not become stable because the
        # statement touched a file. ``force_persist`` is checked by the caller and
        # is the one thing that outranks this.
        if not should_skip and miss_guarded:
            should_skip = True
            skip_reason = GUARD_SKIP_REASON

        # Loop-persist amplification guard (CAS-160). Placed after every other
        # gate so it only accounts writes that would ACTUALLY have happened, and
        # so it can override ``force_persist`` -- which is the whole point: a
        # user asking to persist one value must not silently get every
        # intermediate state of it written to their disk. It is the last word
        # because it is a disk-safety guard, not a cost heuristic.
        if not should_skip:
            amplified, amplified_reason = self._check_persist_amplification(
                code, prediction
            )
            if amplified:
                should_skip = True
                skip_reason = amplified_reason

        # Cost-model prediction fields are shared by both the skip and the
        # full-store branches; build them once.
        cost_fields: dict[str, Any] = {}
        if prediction is not None:
            cost_fields = {
                'cost_model_size_bytes': prediction['size_bytes'],
                'cost_model_restore_seconds': prediction['restore_seconds'],
                'cost_model_type_name': prediction['type_name'],
                'cost_model_family': prediction['family'],
            }

        if should_skip:
            skip_metadata = StatementCacheMetadata(
                timestamp=time.time(),
                inputs=list(inputs),
                outputs=list(outputs),
                execution_time=execution_time,
                source_hash=source_hash,
                code=code,
                key=cache_key,
                skipped_reason=skip_reason,
                metadata_only=True,
                output_lineages=self._lineage.build_output_lineages(self._tracking_state, outputs),
                **cost_fields,
            )
            try:
                backend = self.cash_instance.backend if self.cash_instance else None
                if backend is not None:
                    self._stmt_restorer.persist_metadata_only(backend, cache_key, skip_metadata.to_dict())
            except (OSError, TypeError, ValueError, AttributeError):
                logger.debug("[PROCESSOR] Best-effort metadata persistence failed")
            return skip_metadata

        metadata = StatementCacheMetadata(
            timestamp=time.time(),
            inputs=list(inputs),
            outputs=list(outputs),
            execution_time=execution_time,
            source_hash=source_hash,
            code=code,
            key=cache_key,
            file_dependencies=snapshot_file_deps(file_dependencies),
            force_persist=force_persist,
            output_lineages=self._lineage.build_output_lineages(self._tracking_state, outputs),
            ttl=ttl,
            **cost_fields,
        )

        payload = {
            'variables': self._filter_safe_vars(captured_vars),
            'stdout': captured_output.stdout,
            'stderr': captured_output.stderr,
            # Rich-display output (RichOutput objects). The 'outputs' key in
            # the sibling ``metadata`` dict holds variable NAMES — two distinct
            # concepts; keep them on different keys here too.
            'rich_outputs': captured_output.outputs,
            'rng_state': capture_rng_state(),
            # The seeding regime this state was captured under, so a later
            # restore can tell whether replaying it would clobber a re-seed
            # rather than continue the stream (CAS-223).
            'rng_epochs': dict(self._rng_seed_epochs),
        }

        # CAS-90: the module-global RNG post-state above misses generators the
        # user holds in a variable (``rng = np.random.default_rng(42)``).
        # Capture those too, scoped to this statement's inputs so the cost
        # stays proportional to what the statement actually reads.  Omitted
        # entirely when there are none, keeping the payload shape unchanged for
        # the overwhelming majority of statements.
        try:
            object_rng_states = capture_object_rng_states(inputs, self.shell.user_ns)
            if object_rng_states:
                payload['rng_object_states'] = object_rng_states
        except (TypeError, AttributeError) as e:
            logger.debug("[RANDOMNESS] Object RNG capture skipped: %s", e)

        # Dict-on-the-wire: the backend round-trips a plain dict and may
        # inject the resolved ``storage`` destinations back into it. We
        # re-wrap that mutated dict at the end so the returned view carries
        # the storage info on to the badge metrics.
        wire = metadata.to_dict()

        try:
            self.cash_instance.backend.set(cache_key, payload, wire)
        except (OSError, TypeError, ValueError, pickle.PicklingError, RuntimeError) as e:
            logger.warning("[CACHE] Failed to write to cache backend: %s", e)
        else:
            # Charge this write to its statement's amplification budget, now
            # that the backend has reported which tiers actually took it
            # (CAS-160). Only durable destinations count.
            self._account_persisted_bytes(code, prediction, wire)

        try:
            backend = self.cash_instance.backend
            if backend is not None:
                self._stmt_restorer.persist_metadata_only(backend, cache_key, wire)
        except (OSError, TypeError, ValueError, AttributeError):
            logger.debug("[PROCESSOR] Best-effort metadata persistence failed")

        store_time = time.time() - t_store
        total_time = time.time() - process_start

        if self.debug:
            logger.debug("[TIMING] Store: %.1fms | OVERALL: %.1fms", store_time*1000, total_time*1000)
            logger.debug("[CACHE DEBUG] Stored in cache: %s", cache_key)

        return StatementCacheMetadata.from_dict(wire)

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
                rng_fingerprint=rng_epoch_fingerprint(
                    get_drawing_rng_modules(code), self._rng_seed_epochs,
                ),
            )
        except Exception as exc:
            raise CacheKeyComputationError(
                f"Failed to compute cache key for: {code[:80]!r}"
            ) from exc

        # A statement that seeds opens a new epoch for its module, identified by
        # this statement's own cache key -- which folds in both its source and
        # its input lineage, so `seed(0)` -> `seed(1)` and `seed(cfg.seed)` are
        # both caught. Set AFTER the key above so a statement that seeds and
        # draws at once is keyed on the epoch it INHERITS, not the one it opens
        # (its own draw consumes the state its seed call just established, and
        # that dependency is already carried by its source). CAS-223.
        for module in get_seeding_rng_modules(code):
            self._rng_seed_epochs[module] = cache_key

        hash_time = time.time() - t2
        return inputs, outputs, source_hash, cache_key, analysis_time, hash_time

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

    def _import_needs_reexecution(self, tree: ast.Module | None) -> bool:
        """True for a pure-import statement whose bound name(s) are absent from
        ``user_ns``.

        A cache hit for an import would take the restore path, but restoring
        cannot rebind a *module* object (modules aren't cacheable values). On a
        fresh kernel (e.g. after a restart) the bound name is therefore missing,
        and a later statement in the same cell that uses it raises ``NameError``.
        Forcing re-execution re-imports and rebinds the name (cheap, idempotent)
        and re-stores the entry, so lineage tracking is preserved.
        """
        if tree is None:
            return False
        names = self._get_redundant_import_names(tree)
        if not names:
            return False
        return not all(name in self.shell.user_ns for name in names)

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
            if is_cash_filename(filename):
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


from ..file_tracker import FileAccessTracker

