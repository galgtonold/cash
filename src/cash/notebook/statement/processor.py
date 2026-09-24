"""Core statement processing: analysis, cache lookup, execution, and lineage tracking."""

from __future__ import annotations

import ast
import logging
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

import cash
from cash.backends.persistence_policy import PersistencePolicy
from cash.control_markers import has_marker
from cash.exceptions import (
    CacheKeyComputationError,
)
from cash.notebook._protocols import CashInstanceProtocol, ShellProtocol, TrackingState
from cash.notebook.cache_key import (
    CacheKeyContext,
    compute_cache_key,
)
from cash.notebook.cache_status import CacheStatus, ExecutionResult
from cash.notebook.statement._metadata import StatementCacheMetadata
from cash.notebook.statement.amplification import AmplificationGuard
from cash.notebook.statement.call_routing import CallRouting
from cash.notebook.statement.capture import display_execution_output, make_capture_ctx
from cash.notebook.statement.file_deps import StatementFileDeps
from cash.notebook.statement.freshness import CacheFreshnessChecker
from cash.notebook.statement.hit import CacheHitServer
from cash.notebook.statement.imports import (
    import_bindings_hold,
    import_needs_reexecution,
    import_source_modules,
    redundant_import_names,
)
from cash.notebook.statement.lineage import StatementLineageBuilder
from cash.notebook.statement.miss_guard import GUARD_SKIP_REASON, MissGuard
from cash.notebook.statement.mutations import MutationClassifier
from cash.notebook.statement.randomness import StatementRandomness
from cash.notebook.statement.rebuild_cost import RebuildCostLedger
from cash.notebook.statement.records import StatementRecords
from cash.notebook.statement.restore import StatementRestorer
from cash.notebook.statement.results import COST_MODEL_KEYS, ProcessResult
from cash.notebook.statement.run import CodeRunner, StatementExecution, StatementRun, error_result
from cash.notebook.statement.store import StatementStore
from cash.notebook.versioned_json_store import resolve_cache_dir
from cash.purity import is_known_pure, is_stateful

from ...analysis.annotations import CacheAnnotation
from ...analysis.cacheability import analyze_statement, statement_writes_files
from ...analysis.cacheability_decision import (
    decide_cacheability,
    identity_coupled_reason,
)
from ...analysis.code_analyzer import CodeAnalyzer
from ...analysis.mutation_effects import (
    StatementEffects,
    live_function_source,
    statement_effects,
)
from ...analysis.namespace_effects import statement_calls_user_writer
from ...analytics import AnalyticsManager
from ...tracking.file_dep_snapshot import file_state_epoch
from ...tracking.file_tracker import FileAccessTracker
from ...tracking.function_tracker import FunctionTracker
from ..consumables import is_consumable_unrestorable
from ..lineage_formula import key_hidden_reads
from ..write_observer import observe_writes
from .derivation_edges import is_uncacheable_alias

__all__ = ["StatementProcessor", "is_control_body"]

# Debug log prefixes — module-level constants for filtering and consistency.
_LOG_PROCESSOR = "[PROCESSOR]"
_LOG_DEBUG = "[DEBUG]"
_LOG_MUTATION = "[MUTATION]"
_LOG_CACHE_HIT = "[CACHE_HIT_DEBUG]"
_LOG_CACHE = "[CACHE]"
_LOG_CACHE_DEBUG = "[CACHE DEBUG]"
_LOG_OPTIMIZATION = "[OPTIMIZATION]"
_LOG_FORBIDDEN = "[FORBIDDEN]"
_LOG_ANNOTATION = "[ANNOTATION]"


logger = logging.getLogger(__name__)


def is_control_body(code: str) -> bool:
    """True when *code* is one statement out of a loop or branch BODY, not a
    statement the user wrote at cell level.

    ``for_handler`` / the control-structure processor dispatch a body statement
    here individually, with an injected marker comment carrying the iteration
    or branch context. The upstream simulation, in contrast, treats the whole
    loop or branch as ONE unit -- so any per-statement bookkeeping that names a
    variable (lineage bumps, mutation routing, callee-global capture) has to be
    withheld here and owned by the control structure instead, or the two
    engines disagree about who wrote what.

    Named rather than repeated inline: the same test now gates three separate
    decisions, and three copies of a marker string is three chances for one of
    them to silently stop matching.
    """
    return has_marker(code)


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
        compute_hash_fn: Function to compute variable hashes
    """

    def __init__(
        self,
        shell: ShellProtocol,
        cash_instance: CashInstanceProtocol,
        compute_hash_fn: Callable[[Any], str] | None = None,
        tracking_state: TrackingState | None = None,
        function_tracker: FunctionTracker | None = None,
    ) -> None:
        self.shell: ShellProtocol = shell
        self.cash_instance: CashInstanceProtocol = cash_instance
        self.compute_hash: Callable[[Any], str] | None = compute_hash_fn

        self._amplification = AmplificationGuard()

        # The Cash instance's own, which the dashboard reads (`Cash.show_stats`).
        analytics = getattr(cash_instance, "analytics", None)
        self.analytics_manager = (
            analytics
            if isinstance(analytics, AnalyticsManager)
            else AnalyticsManager(
                enabled=getattr(getattr(cash_instance, "config", None), "analytics", True) is not False
            )
        )

        # Shared with the upstream checker (the magics pass the same one to
        # both), so the simulation keys calls with the same source hashes.
        self.function_tracker = function_tracker if function_tracker is not None else FunctionTracker()

        # The one shared record of lineage and dependency state (see
        # TrackingState); the processor never aliases its fields.
        self.tracking_state: TrackingState = tracking_state or TrackingState()
        self._randomness = StatementRandomness(shell, self.tracking_state)
        self._rebuild_cost = RebuildCostLedger(shell, self.tracking_state, cash_instance)
        self._mutations = MutationClassifier(shell, self.tracking_state, compute_hash_fn)
        self._calls = CallRouting(
            shell,
            self.tracking_state,
            self.function_tracker,
            compute_hash_fn,
            cash_instance=self.get_cash_instance,
            is_stateful_call=self._check_callable_stateful,
        )

        # Cache-freshness checker (TTL / file-dep / input-file invalidation).
        # Stateless w.r.t. tracking state — receives it per call.
        self._freshness = CacheFreshnessChecker(
            backend=cash_instance.backend if cash_instance is not None else None,
        )

        # Perpetual-miss guard: learns which statements can never hit
        # (unstable cache key -> a new key every run -> zero hits) and stops
        # SERIALISING them, while keeping the hash + the lookup. Verdicts persist
        # to the cache dir so a restart doesn't re-pay the learning. Resolved
        # defensively: the backend is a MagicMock in a good number of tests, and
        # an unresolvable dir just means session-scoped verdicts.
        try:
            _guard_dir = resolve_cache_dir(cash_instance.backend if cash_instance is not None else None)
        except (AttributeError, TypeError):
            _guard_dir = None
        self._miss_guard = MissGuard(_guard_dir)

        # Statement-level file-dep tracker. Stateless w.r.t. tracking state —
        # receives it per call. ``executed_file_deps`` lives on TrackingState.
        self._file_deps = StatementFileDeps()

        # Statement-level cache restorer. Hydrates outputs from a cached
        # payload + replays stdout/stderr/rich-outputs.  Distinct from the
        # variable-granular Restorer in restore.py (owned by CashMagics) —
        # see that module's docstring for the unit-of-work distinction.
        # Stateless w.r.t. tracking state — receives it per call.
        self._stmt_restorer = StatementRestorer(shell=shell, compute_hash=compute_hash_fn)
        self._records = StatementRecords(shell, self.tracking_state, cash_instance, self.function_tracker)

        # Used to prevent the "redundant import" optimization from skipping
        # import statements for modules that need re-execution after source changes.
        self.recently_reloaded_modules: set[str] = set()

        # Statement-level lineage builder. Owns the lineage hash + content
        # hash + module-source hash bookkeeping for each output variable.
        # Stateless w.r.t. tracking state — receives it per call.  The three
        # dicts it writes (``granular_preserved_vars``, ``module_attribute_deps``,
        # ``from_import_sources``) live on TrackingState.
        self.lineage_builder = StatementLineageBuilder(
            shell=shell,
            function_tracker=self.function_tracker,
            file_deps=self._file_deps,
            compute_hash=compute_hash_fn,
        )
        self._hits = CacheHitServer(self.tracking_state, self._stmt_restorer, self._rebuild_cost)
        self._store = StatementStore(
            shell,
            self.tracking_state,
            cash_instance,
            lineage_builder=self.lineage_builder,
            amplification=self._amplification,
            rebuild_cost=self._rebuild_cost,
            calls=self._calls,
        )

    def get_cash_instance(self) -> Any | None:
        """Return the Cash instance for decorator call tracking.

        Tries ``self.cash_instance`` first, then looks for the global
        ``cash._global_cash`` singleton so that ``@cash.cache`` calls are
        captured even when the user imports ``from cash import cache``.
        """
        if self.cash_instance is not None:
            return self.cash_instance
        return getattr(cash, "_global_cash", None)

    def _attribute_input_change(self, metrics: dict, inputs, outputs) -> None:
        """Name the input whose change forced this statement to recompute.

        An upstream input changing is the most common reason a notebook
        statement re-runs, and it was the one reason the badge could not name:
        the row rendered EXECUTED with no attribution. A user with a
        reproducible slow re-run had nowhere to look but cash's source.

        Cheap by construction, and it must stay that way. ``TrackingState``
        already records, per output variable, the input lineages the statement
        last RAN with -- so the comparison is that record against the current
        lineages, an O(inputs) dict walk. It never touches the backend:
        answering the same question by scanning the cache is O(N^2) in cache
        size over a run and dominates cold-run wall time.

        Ordering is load-bearing: ``executed_input_lineages`` is rewritten by
        ``_post_execute``, which runs AFTER this. Reading it here therefore
        sees the previous run's inputs, which is the whole point -- once the
        statement has run, its inputs agree again and the reason is gone.

        Silent when it has nothing to say: a first run has no prior record, a
        statement whose inputs all match did not re-run because of them, and a
        name the statement also WRITES is excluded outright -- see the comment
        on ``wanted`` below for why that comparison cannot be trusted.

        A wrong reason is worse than no reason here. The row rendered EXECUTED
        with no attribution before this method existed, so failing closed to
        silence costs a diagnostic; failing open sends the user to inspect a
        variable that is not the problem.
        """
        try:
            state = self.tracking_state
            current = state.variable_lineage
            # A name this statement WRITES is not evidence about what it read.
            # ``executed_input_lineages`` is keyed by output variable name
            # alone, so every statement writing the same variable shares one
            # slot and reads back whichever of them ran last. For a chain of
            # ``df = df[...]`` filters that is always a different statement,
            # and the mismatch is guaranteed -- 17 of 21 attributions on
            # 01_nyc_taxi_analysis named an input that was also the
            # statement's own output, on a run whose cache keys and lineage
            # sequence were byte-identical to the previous one.
            #
            # Dropping only the self-referential names keeps the reason for
            # the half of the statement that is still sound:
            # ``df = df.join(other)`` may honestly blame ``other``.
            wanted = {v for v in (inputs or []) if isinstance(v, str) and v not in set(outputs or [])}
            for out in outputs or []:
                previous = state.executed_input_lineages.get(out)
                if not previous:
                    continue
                stale = sorted(
                    name
                    for name, was in previous.items()
                    if name in wanted and name in current and current[name] != was
                )
                if stale:
                    names = ", ".join(stale[:3])
                    more = f" +{len(stale) - 3} more" if len(stale) > 3 else ""
                    metrics["miss_reason"] = f"input changed: {names}{more}"
                    return
        except (AttributeError, TypeError):  # pragma: no cover - defensive
            return

    def forget_variable(self, name: str) -> None:
        """Drop everything recorded about how *name* was computed.

        Its lineage, defining code, input lineages, session hash, narrowed
        from-import component and module-attribute accesses all go, so the
        next statement that reads *name* treats it as having no lineage. The
        value in ``user_ns`` is left alone.
        """
        self.tracking_state.lineage.discard(name)
        self.tracking_state.executed_cell_codes.pop(name, None)
        self.tracking_state.executed_input_lineages.pop(name, None)
        self.tracking_state.current_session_hashes.pop(name, None)
        self.tracking_state.from_import_components.pop(name, None)
        self.tracking_state.module_attribute_deps.pop(name, None)

    def persistence_policy(self) -> PersistencePolicy:
        """The persistence policy statements are stored under right now
        (see :meth:`StatementStore.policy`)."""
        return self._store.policy()

    def set_written_later_in_cell(self, names: frozenset[str]) -> None:
        """Tell the store which names a later top-level statement of the cell
        writes (see :meth:`StatementStore.set_written_later_in_cell`)."""
        self._store.set_written_later_in_cell(names)

    def mark_module_reloaded(self, module_name: str) -> None:
        """Note that *module_name* was just reloaded, so the next import of it
        runs instead of being skipped as redundant."""
        self.recently_reloaded_modules.add(module_name)

    def loop_vars_scope(self, loop_vars: dict[str, Any], loop_var_digests: dict[str, str] | None = None):
        """Push one loop iteration's variables for the calls in its body
        (see :meth:`CallRouting.loop_vars_scope`)."""
        return self._calls.loop_vars_scope(loop_vars, loop_var_digests)

    def begin_structure_cost(self) -> None:
        """A control structure starts (see :meth:`RebuildCostLedger.begin_structure`)."""
        self._rebuild_cost.begin_structure()

    def end_structure_cost(self, reads, changed, success: bool) -> None:
        """A control structure ended (see :meth:`RebuildCostLedger.end_structure`)."""
        self._rebuild_cost.end_structure(reads, changed, success)

    def end_cell_persistence(self) -> None:
        """Write to disk what this cell left that would be costly to rebuild
        (see :meth:`RebuildCostLedger.end_cell_persistence`)."""
        self._rebuild_cost.end_cell_persistence()

    def begin_control_log(self, code: str):
        """Start logging control structure *code* as one statement
        (see :meth:`StatementRecords.begin_control_log`)."""
        return self._records.begin_control_log(code)

    def end_control_log(self, mark) -> None:
        """Replace what the structure's body logged with the structure itself."""
        self._records.end_control_log(mark)

    def persist_read_provenance(self, code: str, accessed_files: set[str]) -> None:
        """Record, across restarts, which files *code* read."""
        self._records.persist_read_provenance(code, accessed_files)

    def persist_write_provenance(
        self, code: str, inputs: set[str], tree: ast.Module | None, written: frozenset[str] | set[str] = frozenset()
    ) -> None:
        """Record what file(s) the writer statement *code* produced."""
        self._records.persist_write_provenance(code, inputs, tree, written)

    def persist_metadata_only(self, key: str, metadata: dict[str, Any]) -> None:
        """Write *metadata* under *key* with no value, for a later kernel to
        read. Only a tier that keeps metadata alone (a file tier) stores it."""
        self.cash_instance.backend.set_metadata_only(key, metadata)

    def user_written_paths(self, paths) -> frozenset[str]:
        """*paths* without cash's own storage (its cache directories)."""
        return self._records.user_written_paths(paths)

    def begin_cell_rng_observation(self) -> None:
        """Open a fresh per-cell RNG accumulation, before the cell's statements run."""
        self._randomness.begin_cell()

    def cell_rng_observation(self) -> tuple[set[str], dict | None, dict | None]:
        """What this cell's statements changed in the RNG streams, and the
        positions either side (see :meth:`StatementRandomness.cell_observation`)."""
        return self._randomness.cell_observation()

    def process_statement(
        self,
        code: str,
        ttl: int | None = None,
        silent: bool = False,
        annotation: CacheAnnotation | None = None,
        display_code: str | None = None,
        exec_source: str | None = None,
        occurrence_index: int = 0,
        stream_output: bool = False,
        force_outputs: set[str] | None = None,
        is_last: bool = True,
    ) -> ProcessResult:
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
                the accumulator-loop fast path to route ``out = []`` +
                ``for e in it: out.append(f(e))`` through the cache as one unit,
                capturing the accumulator AND the leaked loop variable. Does NOT
                affect the cache key (outputs only enter it when they are
                modules), so the key still tracks the loop source + input
                lineages.
            exec_source: The statement's ORIGINAL text (comments intact), when
                the caller could recover one -- see ``_statement_source`` in
                ``cell_executor.py``. Compiled in place of ``code`` so a
                function defined here keeps its comments for
                ``inspect.getsource`` (and therefore for a per-line
                ``# @cash:assume-safe`` waiver). Never affects the cache key:
                ``code`` (the unparsed form) is what is hashed. Discarded
                whenever call interception rewrote an eligible call in
                ``code`` -- see ``CallRouting.code_and_tree_for_execution``.

        Returns:
            ProcessResult with keys: 'status', 'execution_time', 'total_time',
            'saved_time', 'restored_vars', 'code', plus optional keys depending
            on cache status.
        """
        run = StatementRun(
            code,
            ttl=ttl,
            silent=silent,
            annotation=annotation,
            display_code=display_code,
            exec_source=exec_source,
            occurrence_index=occurrence_index,
            stream_output=stream_output,
            force_outputs=force_outputs,
            is_last=is_last,
        )
        done = self._prepare(run)
        if done is not None:
            return done
        with self._executing(run) as runner:
            runner.run()
        return self._finish(run, runner.execution)

    async def process_statement_async(
        self,
        code: str,
        ttl: int | None = None,
        silent: bool = False,
        annotation: CacheAnnotation | None = None,
        display_code: str | None = None,
        exec_source: str | None = None,
        occurrence_index: int = 0,
        stream_output: bool = False,
        force_outputs: set[str] | None = None,
        is_last: bool = True,
    ) -> ProcessResult:
        """:meth:`process_statement` for a statement holding a top-level ``await``.

        The same pipeline, step for step; only the execution differs, which
        awaits the compiled code on IPython's running loop. A cache hit returns
        before any coroutine is built, so an identical second run skips the
        await entirely.
        """
        run = StatementRun(
            code,
            ttl=ttl,
            silent=silent,
            annotation=annotation,
            display_code=display_code,
            exec_source=exec_source,
            occurrence_index=occurrence_index,
            stream_output=stream_output,
            force_outputs=force_outputs,
            is_last=is_last,
        )
        done = self._prepare(run)
        if done is not None:
            return done
        with self._executing(run) as runner:
            await runner.run_async()
        return self._finish(run, runner.execution)

    def _prepare(self, run: StatementRun) -> ProcessResult | None:
        """Analyse, key and look up *run*'s statement, before it may execute.

        Returns the finished result when the statement needs no execution -- a
        redundant import, or a cache hit whose restore succeeded -- and
        ``None`` when it must run, with *run* filled in for :meth:`_executing`
        and :meth:`_finish`.
        """
        self._begin(run)
        effects, analysis_time, hash_time = self._analyze(run)

        done = self._check_redundant_import(run)
        if done is not None:
            return done

        # Computed once: used by the cacheability decision and (on the
        # cache-miss path) by _post_execute for in-place-mutation tracking.
        run.analysis = analyze_statement(run.code, run.tree, self.shell.user_ns)
        self._route_mutations(run, effects)
        self._decide_cacheability(run)
        # An UNSEEDED estimator fit routed to caching above is frozen on re-run
        # with no warning -- cash's AST detector cannot see the randomness inside
        # sklearn's compiled .fit(). Warn now (compute time); the same set drives
        # the restore-time warning on a cache hit below. Only once the statement
        # is known to be cached: one that re-executes fits afresh every run.
        fits_cached = run.est_fit if not run.skip_cache else set()
        unseeded_fits = self._randomness.warn_unseeded_estimator_fit(run.code, fits_cached, run.allow_random)
        self._randomness.stamp_random_effect(run.metrics, run.code, run.unseeded_calls, unseeded_fits)

        metadata, cached_data = self._lookup(run, analysis_time, hash_time)
        if cached_data and not import_needs_reexecution(run.tree, self.shell.user_ns):
            hit_result = self._serve_hit(run, cached_data, metadata, unseeded_fits)
            if hit_result is not None:
                return hit_result

        self._route_calls(run)
        return None

    def _begin(self, run: StatementRun) -> None:
        """Read *run*'s annotation, warn about its randomness, start its
        metrics and parse it."""
        code = run.code
        run.effective_ttl, run.force_persist, run.skip_cache, run.allow_random, run.cache_fit = self._parse_annotation(
            run.annotation, run.ttl
        )
        self._calls.begin_statement(run.effective_ttl, run.force_persist)
        run.unseeded_calls = self._randomness.warn_unseeded(code, run.allow_random, skip_cache=run.skip_cache)
        self._randomness.warn_entropy_reseed(code)
        run.metrics = {
            "status": CacheStatus.UNKNOWN,
            "execution_time": 0.0,
            "total_time": 0.0,
            "saved_time": 0.0,
            "error": None,
            "restored_vars": [],
            "code": code.strip(),
            # The badge shows the user's own layout. NEVER part of the cache
            # key: `code` above is what is hashed, always.
            "display_code": run.display_code,
            "uncacheable_reasons": [],
        }
        self._randomness.stamp_random_effect(run.metrics, code, run.unseeded_calls)
        logger.debug("%s Processing statement: %s...", _LOG_DEBUG, code[:50])

        run.process_start = time.time()

        try:
            run.tree = ast.parse(code.strip())
        except SyntaxError:
            run.tree = None

    def _analyze(self, run: StatementRun) -> tuple[StatementEffects, float, float]:
        """Key *run* and settle its inputs and outputs.

        Returns ``(effects, analysis_time, hash_time)``.
        """
        effects, run.source_hash, run.cache_key, analysis_time, hash_time = self._analyze_and_hash(
            run.code, occurrence_index=run.occurrence_index, tree=run.tree
        )
        outputs = set(effects.outputs)
        callee_globals = set(effects.callee_globals)
        # Caller-forced outputs (accumulator-loop fast path): capture
        # and restore these on top of the AST-discovered outputs, and mark them
        # as expected writes so an in-place accumulator mutation (``out.append``)
        # is not read as a caching blocker. Added AFTER the cache key is computed
        # so it is unaffected — the key already tracks the loop source + input
        # lineages (a forced output only ever enters the key when it is a module,
        # which an accumulator never is).
        if run.force_outputs:
            outputs = outputs | run.force_outputs
        # A callee's writes to globals join ``outputs`` so their lineage is
        # bumped (a downstream consumer of the accumulator must re-key), and the
        # statement is skip-cached (`_route_mutations`) so the write actually
        # happens.
        #
        # NOT captured and restored: that is unsound the moment a global has
        # more than ONE writer, because an absolute end state does not compose
        # with a prefix that was itself skipped. Two calls to the same
        # appending helper in one cell, cold run::
        #
        #     expected  ['ok:3.3', 'cleanup', 'err:zero_div', 'cleanup']
        #     observed  ['err:zero_div', 'cleanup']
        #
        # The first writer's contribution is simply gone. ``force_outputs``
        # above CAN restore an accumulator, but only under
        # ``cacheable_accumulator_loop``'s conditions -- fresh empty seed, a
        # single accumulator call -- which are precisely the guarantees that
        # make one writer's snapshot sufficient.
        if callee_globals:
            outputs = outputs | callee_globals
        run.inputs, run.outputs = set(effects.inputs), outputs
        # Exposed so the badge can show a short prefix in the row-detail "Key"
        # field: two runs of the same statement in the same slot or not.
        run.metrics["cache_key"] = run.cache_key
        return effects, analysis_time, hash_time

    def _route_mutations(self, run: StatementRun, effects: StatementEffects) -> None:
        """Add the receivers *run* mutates to its outputs, and skip-cache it
        where the mutation must really happen on every run."""
        code, tree, metrics = run.code, run.tree, run.metrics
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
        if is_control_body(code):
            mut_pre_route: set[str] = set()
            # ...with ONE exception: a draw on a live Figure/Axes.
            draw_only = self._mutations.identity_coupled_call_receivers(tree)
            fit_only = self._mutations.fitted_receivers(tree)
        else:
            mut_pre_route, run.mut_observe, run.mut_assumed, run.mut_record = self._mutations.classify(
                tree,
                run.source_hash,
                run.outputs,
            )
            run.est_fit = self._mutations.estimator_fit_receivers(tree, run.outputs) if run.cache_fit else set()
            draw_only = set()
            fit_only = set()
        est_fit = run.est_fit
        # OPT-IN ONLY (``# @cash:cache-fit``). A bare ``estimator.fit(X, y)``
        # mutates its receiver in place, so the classifier above routes it to
        # skip-caching: the statement re-executes and is never serialised, which is
        # net-NEUTRAL -- a fit that keeps missing cannot cost more than it saves.
        #
        # It does NOT make aliases safe. ``backup = clf`` is an ORDINARY
        # ASSIGNMENT that cash caches on its own, and restoring it rebinds
        # ``backup`` to a pre-fit deserialised copy -- the fit statement has no
        # bearing on it either way.
        #
        # Caching a bare fit instead is the OPT-IN path, kept because it
        # is a large win when it lands but not the default because its
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
        # PLACE. Without the directive ``est_fit`` is empty and every
        # site below degrades to the skip-cache behaviour.
        fam = effects.arg_mutations - run.outputs
        skip_pre_route = mut_pre_route - est_fit
        if mut_pre_route or est_fit or fam:
            run.outputs = run.outputs | mut_pre_route | est_fit | fam
        if skip_pre_route:
            run.skip_cache = True
            metrics["uncacheable_reasons"].append(
                f"In-place mutation on: {', '.join(sorted(skip_pre_route))} "
                "(receiver lineage bumped; statement re-executes)" + self._mutations.cache_fit_hint(skip_pre_route)
            )
        # Deliberately the SAME treatment the inline spelling of the identical
        # mutation gets immediately above: the statement re-executes so the
        # callee's write to a global really happens.
        #
        # The expensive work is NOT lost. Call interception still serves the
        # call inside this statement, keyed on the mutated global's own
        # pre-call state, so what re-executes is the glue around it.
        callee_globals = set(effects.callee_globals)
        if callee_globals:
            run.skip_cache = True
            metrics["uncacheable_reasons"].append(
                f"Callee mutates: {', '.join(sorted(callee_globals))} "
                "(global lineage bumped; statement re-executes, call still cached)"
            )
        # a draw inside a loop/branch body. Skip the CACHE without
        # touching ``outputs`` -- the statement must re-execute so the artists
        # actually land on the Axes, but bumping its lineage from a per-statement
        # source is precisely what the control-body skip above exists to avoid.
        if draw_only:
            run.skip_cache = True
            metrics["uncacheable_reasons"].append(
                f"Draws on: {', '.join(sorted(draw_only))} (live Figure/Axes; statement re-executes)"
            )
        if fit_only:
            run.skip_cache = True
            metrics["uncacheable_reasons"].append(
                f"Fits: {', '.join(sorted(fit_only))} (estimator fitted in place; statement re-executes)"
            )

    def _decide_cacheability(self, run: StatementRun) -> None:
        """Skip-cache *run* when the static cacheability decision refuses it."""
        if run.skip_cache:
            return
        cacheable, reasons = decide_cacheability(
            code=run.code,
            tree=run.tree,
            inputs=run.inputs,
            outputs=run.outputs,
            annotation=run.annotation,
            analysis=run.analysis,
            user_ns=self.shell.user_ns,
            variable_lineage=self.tracking_state.variable_lineage,
            is_stateful_call=self._check_callable_stateful,
            scan_forbidden=CodeAnalyzer.scan_for_forbidden_functions,
        )
        if not cacheable:
            run.metrics["uncacheable_reasons"].extend(reasons)
            run.skip_cache = True

    def _lookup(
        self, run: StatementRun, analysis_time: float, hash_time: float
    ) -> tuple[StatementCacheMetadata | None, Any | None]:
        """Look *run* up in the cache: ``(metadata, cached_data)``, both None
        on a miss or a skipped lookup."""
        run.effective_ttl = self._ttl_floor_from_called_functions(run.inputs, run.effective_ttl)
        metadata, cached_data, cache_check_time = self._do_cache_lookup(
            run.skip_cache, run.cache_key, run.effective_ttl, run.inputs
        )
        self._observe_miss_guard(run.skip_cache, run.code, run.source_hash, run.cache_key, cached_data, run.inputs)

        if logger.isEnabledFor(logging.DEBUG):
            self._log_cache_lookup(
                run.code, run.cache_key, run.inputs, cached_data, analysis_time, hash_time, cache_check_time
            )
        return metadata, cached_data

    def _serve_hit(
        self,
        run: StatementRun,
        cached_data: Any,
        metadata: StatementCacheMetadata | None,
        unseeded_fits: Any,
    ) -> ProcessResult | None:
        """Restore *run* from its entry; the finished result, or None when the
        restore failed and the statement must run after all."""
        hit_result = self._hits.serve(run, cached_data, metadata, self._randomness.seed_epochs)
        if hit_result is None:
            return None
        self.analytics_manager.record_event(
            status="HIT",
            execution_time=hit_result["total_time"],
            saved_time=hit_result["saved_time"],
            code_hash=run.cache_key,
        )
        # The restore SUCCEEDED, so the value handed back is a replay.
        self._randomness.warn_stale(run.code, run.unseeded_calls, run.allow_random)
        self._randomness.warn_stale_estimator_fit(run.code, unseeded_fits, run.allow_random)
        self._randomness.flag_inline_unseeded_fit(
            hit_result, run.code, run.tree, run.outputs, run.allow_random, is_hit=True
        )
        return hit_result

    def _route_calls(self, run: StatementRun) -> None:
        """Settle what executes: *run*'s code with eligible calls routed
        through the call cache."""
        code = run.code
        run.exec_code, run.exec_tree = self._calls.code_and_tree_for_execution(code, run.tree, run.annotation)
        # `code_and_tree_for_execution` returns a NEW string, never `code`
        # itself, only when it routed an eligible call through the call cache.
        # Compiling the pre-rewrite original text after that would run a version
        # of the statement that never went through the indirection, so the
        # original text is used only when the rewrite made no change.
        if run.exec_code is not code:
            run.exec_source = None

    @contextmanager
    def _executing(self, run: StatementRun) -> Generator[CodeRunner, None, None]:
        """Run *run*'s code under output capture and cash's observers.

        Yields the :class:`CodeRunner`; the caller runs it (``run()``, or
        ``await run_async()``) inside the ``with`` block. Around it this
        captures stdout/stderr/display, observes file reads and writes and the
        global RNG streams, and records all of it on ``runner.execution``. An
        exception from the user's code becomes the execution's error result
        rather than propagating.

        A statement that may have written a file drops the freshness answers
        kept for the cell (``_forget_file_answers_if_it_wrote``).
        """
        logger.debug("%s Executing (cache miss)", _LOG_CACHE_DEBUG)
        code = run.exec_code
        source = run.exec_source if run.exec_source is not None else code
        # A tree parsed from the unparsed text has line numbers that do not
        # match the original source, so the runner re-parses that instead.
        tree = run.exec_tree if run.exec_source is None else None
        runner = CodeRunner(code, source, tree, run.is_last, self.shell.user_ns)
        execution = runner.execution
        marks = self._calls.cash_time_marks()
        start_time = time.time()
        # Snapshot the global RNG streams around execution so a before/after
        # diff catches a draw that static analysis and object-introspection
        # both miss -- one hidden inside a called function.
        pre_rng = self._randomness.begin_statement()
        try:
            with make_capture_ctx(run.stream_output, run.skip_cache and run.stream_output) as captured:
                execution.captured = captured
                with observe_writes() as written_paths, FileAccessTracker(self.shell.user_ns) as file_tracker:
                    yield runner
                execution.accessed_files = file_tracker.get_accessed_files()
                execution.written_paths = frozenset(written_paths)
                execution.accessed_remote = file_tracker.get_accessed_remote_urls()
                # `code` (the keyed form), not `source`: the observation is
                # about what the statement does, which is the same either way,
                # and this keeps it matched to the simulator's own unparse.
                self._randomness.observe_statement(pre_rng, code)
                execution.result = ExecutionResult(success=True)
        except Exception as e:  # noqa: BLE001 - broad fallback wrapping arbitrary user code
            execution.result = error_result(e)
        self._forget_file_answers_if_it_wrote(code, execution)
        execution.wall_time = time.time() - start_time
        execution.cost = self._calls.statement_cost(execution.wall_time, marks)
        execution.tax = self._calls.cash_tax_seconds(marks)

    def _finish(self, run: StatementRun, execution: StatementExecution) -> ProcessResult:
        """Record what the executed statement did, and store it."""
        metrics = run.metrics
        decorator_calls: list = []
        try:
            cash_instance = self.get_cash_instance()
            if cash_instance is not None:
                decorator_calls = cash_instance.drain_decorator_calls()
        except (AttributeError, TypeError, RuntimeError):
            logger.debug("%s Failed to drain decorator call log", _LOG_PROCESSOR)
        decorator_calls.extend(self._calls.drain_call_unit_events())
        self._calls.learn_call_wrapping(run.code, execution.wall_time, decorator_calls)

        captured = execution.captured
        metrics["stdout"] = captured.stdout
        metrics["stderr"] = captured.stderr
        # IPython rich-display capture (RichOutput objects). Distinct from
        # ``metadata['outputs']`` and ``evaluated_vars`` (which hold variable
        # NAMES from AST analysis), so they never mix in badge fields.
        metrics["rich_outputs"] = captured.outputs
        if decorator_calls:
            metrics["decorator_calls"] = decorator_calls

        display_execution_output(captured, run.silent, run.stream_output, metrics)
        metrics["execution_time"] = execution.wall_time
        metrics["compute_cost"] = execution.cost
        metrics["cash_tax"] = execution.tax

        result = execution.result
        if not result.success:
            metrics["status"] = CacheStatus.ERROR
            metrics["error"] = result.error
            metrics["total_time"] = time.time() - run.process_start
            self._handle_execution_error(result, run.silent)
            return metrics

        self._randomness.flag_inline_unseeded_fit(
            metrics, run.code, run.tree, run.outputs, run.allow_random, is_hit=False, skip_cache=run.skip_cache
        )
        self._randomness.flag_observed_hidden_draw(metrics, run.code, run.outputs, skip_cache=run.skip_cache)
        metrics["status"] = CacheStatus.COMPUTED
        metrics["evaluated_vars"] = list(run.outputs) if run.outputs else []
        # Input names, so provenance and badge tooltips can rebuild the
        # dependency graph. The no-name inputs the AST sometimes emits are dropped.
        metrics["inputs"] = [v for v in (run.inputs or []) if isinstance(v, str)]
        # Attribute the miss for the badge's row-detail drawer when it is cheap:
        # ``CacheFreshnessChecker`` sets ``last_miss_reason`` for TTL / file
        # invalidations. Never a backend-wide scan for the empty-key path.
        if not run.skip_cache and self._freshness.last_miss_reason:
            metrics["miss_reason"] = self._freshness.last_miss_reason
        elif not run.skip_cache:
            self._attribute_input_change(metrics, run.inputs, run.outputs)

        self._post_execute(run, execution)
        return metrics

    @property
    def persist_all(self) -> bool:
        """Force-persist every statement (bypass the cost-aware floors), as if
        each carried ``# @cash:persist``.

        Read from config at each statement, so ``cash.configure(persist_all=
        True)`` and ``%cash_persist`` (which writes config) both take effect
        on the next one. ``is True``, because tests pass a MagicMock
        cash_instance whose attributes are all truthy.
        """
        return getattr(getattr(self.cash_instance, "config", None), "persist_all", False) is True

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
        statement IN to the estimator-fit caching path.  It is off by
        default: without it a bare fit is skip-cached and simply re-executes,
        which is net-neutral.  It does NOT, as this comment used to
        claim, keep aliases correct: ``backup = model`` is an ordinary assignment
        whose own restore rebinds a pre-fit copy, independently of the fit
        (— fixed by refusing to cache a bare alias bind).
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
        promised. The call target appears in ``inputs`` (the analyzer
        lists ``f`` for ``x = f()``); if it is a cash wrapper with a smaller
        declared TTL, the statement must expire at least as often. ``ttl=0`` then
        rides the existing immediate-expiry path, so every run is a miss
        and the decorated body runs every time, as ``ttl=0`` asks.

        Only LOWERS the TTL and only for a wrapper carrying an explicit TTL, so a
        plain ``@cash.cache`` (ttl=None) call is completely unaffected — the
        statement caches exactly as before.
        """
        user_ns = self.shell.user_ns
        floor = effective_ttl
        for name in inputs:
            fn = user_ns.get(name)
            if fn is None or not getattr(fn, "_cash_cached", False):
                continue
            declared = getattr(fn, "_cash_declared_ttl", None)
            if declared is None:
                continue
            floor = declared if floor is None else min(floor, declared)
        return floor

    def begin_cell_statement_log(self) -> None:
        """Start this cell's statement log, before its statements run."""
        self._records.begin_cell()
        self._rebuild_cost.begin_cell()
        self._calls.begin_cell()

    def _do_cache_lookup(
        self,
        skip_cache: bool,
        cache_key: str,
        ttl: int | None,
        inputs: set[str],
    ) -> tuple[StatementCacheMetadata | None, Any | None, float]:
        """Run cache lookup unless *skip_cache* is set."""
        if not skip_cache:
            return self._freshness.check_cache(self.tracking_state, cache_key, ttl, inputs, epoch=file_state_epoch())
        logger.debug("%s Skipping cache lookup due to missing input lineage or @cash:no-cache", _LOG_ANNOTATION)
        return None, None, 0.0

    def _observe_miss_guard(
        self,
        skip_cache: bool,
        code: str,
        source_hash: str,
        cache_key: str,
        cached_data: Any,
        inputs: set[str] | None = None,
    ) -> None:
        """Feed one lookup outcome to the perpetual-miss guard.

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
        if has_marker(code):
            return
        self._miss_guard.observe(
            source_hash,
            cache_key,
            hit=cached_data is not None,
            components=self._records.lineages_read(inputs) if inputs else {},
        )

    def _post_execute(self, run: StatementRun, execution: StatementExecution) -> None:
        """Auto-track imports, capture vars, detect mutations, save to cache, record analytics.

        Updates ``run.outputs`` and ``run.skip_cache`` with what execution
        revealed (an observed mutation, an uncacheable value).
        """
        self._observe_mutations(run)

        # Auto-track newly imported local modules so _capture_variables includes
        # the module source hash in the lineage on first execution.
        try:
            self.function_tracker.auto_track_local_imports(run.code)
        except (ImportError, AttributeError, OSError):
            logger.debug("%s Failed to auto-track local imports", _LOG_PROCESSOR)
        self._records.persist_import_bindings(run.code, run.tree)

        captured_vars = self.lineage_builder.capture_and_track_variables(
            self.tracking_state,
            run.outputs,
            run.inputs,
            run.code,
            run.source_hash,
            cache_key=run.cache_key,
            accessed_files=execution.accessed_files,
            tree=run.tree,
            accessed_remote=execution.accessed_remote,
        )
        if not run.skip_cache:
            self._refuse_unrestorable_outputs(run, captured_vars)
        self._record_file_effects(run, execution)

        # Detect in-place mutations (detection-only; do not modify lineage).
        # Reuses the StatementAnalysis from process_statement to avoid a
        # second pass of AST visitors over the same tree.
        pure_mutations = run.analysis.all_mutated_vars - run.outputs
        if pure_mutations:
            self.tracking_state.vars_with_mutation_lineage.update(pure_mutations)
            logger.debug("%s Detected in-place mutations on: %s", _LOG_MUTATION, pure_mutations)

        miss_guarded = self._miss_guarded(run, execution, captured_vars)
        self._skip_a_newly_seen_draw(run)

        saved_metadata = None
        if not run.skip_cache:
            saved_metadata = self._store.save(
                run, execution, captured_vars, miss_guarded=miss_guarded, seed_epochs=self._randomness.seed_epochs
            )
        else:
            logger.debug("%s Skipping cache save due to @cash:no-cache", _LOG_ANNOTATION)
        self._report_saved(run, saved_metadata)
        storage = (saved_metadata.storage if saved_metadata else None) or ()
        self._rebuild_cost.note(
            run.cache_key, run.inputs, run.outputs, execution.cost, on_disk=any(s != "RAM" for s in storage)
        )

        run.metrics["total_time"] = time.time() - run.process_start
        self.analytics_manager.record_event(
            status="MISS",
            execution_time=run.metrics["total_time"],
            saved_time=0.0,
            code_hash=run.cache_key,
        )

    def _observe_mutations(self, run: StatementRun) -> None:
        """Add the receivers execution was seen mutating to *run*'s outputs.

        Broad-precise mutation observation: for a standalone method call whose
        method is not statically known, compare each candidate receiver's
        content after execution against its pre-statement hash. Runs BEFORE
        capture_and_track so a newly-detected mutation is in the outputs (its
        lineage gets bumped) and skip-caches the statement. The verdict is
        recorded for the upstream simulation, which cannot observe execution.
        """
        if not run.mut_record:
            return
        metrics, source_hash = run.metrics, run.source_hash
        newly_mutated = self._mutations.observed_mutations(run.mut_observe, source_hash)
        if newly_mutated:
            # ``run.est_fit`` is non-empty only under ``# @cash:cache-fit``.
            # Those receivers still enter the outputs (source-based lineage
            # bump + fitted value capture) and are still recorded in
            # ``mutation_verdicts`` below (so the upstream simulation bumps
            # downstream lineage on a data edit), but they are NOT
            # skip-cached -- they cache + restore in place. Every
            # other observed mutation -- including a bare fit WITHOUT the
            # directive -- still skip-caches its receiver.
            run.outputs = run.outputs | newly_mutated
            # The metrics hold their own copy of the outputs: without this the
            # badge row said "Produced -" for ``sc.pp.normalize_total(adata)``
            # on its first run.
            produced = metrics.setdefault("evaluated_vars", [])
            produced.extend(n for n in sorted(newly_mutated) if n not in produced)
            skip_observed = newly_mutated - run.est_fit
            if skip_observed:
                run.skip_cache = True
                metrics.setdefault("uncacheable_reasons", []).append(
                    f"In-place mutation on: {', '.join(sorted(skip_observed))} "
                    "(observed; receiver lineage bumped; statement re-executes)"
                    + self._mutations.cache_fit_hint(skip_observed)
                )
        self.tracking_state.mutation_verdicts[source_hash] = set(run.mut_assumed) | newly_mutated
        self._records.persist_mutation_verdict(source_hash, self.tracking_state.mutation_verdicts[source_hash])

    def _alias_refusal(self, name: str, value: Any) -> str | None:
        """A live-alias object (numpy view, pandas groupby/rolling ref-holder)
        is not cached: pickling and restoring it decouples it from its live
        base, so a later base mutation would be lost after restore. It is
        re-derived from the live base instead. ``.copy()`` produces no alias
        and stays cacheable."""
        if is_uncacheable_alias(value, self.shell.user_ns):
            return f"Live-alias object '{name}' (view/ref-holder); re-derived from live base, not cached."
        return None

    @staticmethod
    def _consumable_refusal(name: str, value: Any) -> str | None:
        """Nor a CONSUMABLE the cache cannot copy -- an open file handle, a
        generator. The RAM tier keeps such a value by reference, so a "hit"
        hands back the very object a reader already drained: on a second Run
        All `fh = open(p)` was served, and the cell reading `fh` printed []
        where Run All in plain Jupyter reads the file again
        (test_a_consumed_iterator_is_rebuilt_for_its_reader)."""
        if is_consumable_unrestorable(value):
            return (
                f"'{name}' is consumed as it is read (an open file or a "
                f"generator) and cannot be restored: it is re-created "
                f"every run"
            )
        return None

    def _refuse_unrestorable_outputs(self, run: StatementRun, captured_vars: dict[str, Any]) -> None:
        """Skip-cache *run* when one of its output values cannot be stored and
        restored faithfully, giving the first refusal found as the reason.

        Checked here, after execution, because the values do not exist when
        the cacheability decision runs; and before ``StatementStore.save``,
        because refusing then is what keeps the RAM tier from deep-copying
        them. ``identity_coupled_reason`` refuses an object identity-coupled to
        a library global: the RAM tier's deep copy of a matplotlib Figure
        re-registers the COPY as pyplot's current figure, so ``plt.savefig()``
        writes the cache's snapshot, a blank PNG on the first run.
        """
        refusals: tuple[Callable[[str, Any], str | None], ...] = (
            self._alias_refusal,
            identity_coupled_reason,
            self._consumable_refusal,
        )
        for refusal in refusals:
            for out in run.outputs:
                value = captured_vars.get(out)
                reason = refusal(out, value) if value is not None else None
                if reason is not None:
                    run.skip_cache = True
                    run.metrics.setdefault("uncacheable_reasons", []).append(reason)
                    return

    def _record_file_effects(self, run: StatementRun, execution: StatementExecution) -> None:
        """Record the files *run* wrote and read, for the upstream simulation
        and for a reader after a restart."""
        code = run.code
        # Record executed file-WRITING statements by code text:
        # writes have no variable edge, so the upstream simulation needs this
        # to tell an edited/new writer from one that already ran.
        # A write the code does not spell (``save_chart(kind)``, whose savefig
        # is in the helper) counts too: it was observed (``write_observer``).
        written = self._records.user_written_paths(execution.written_paths)
        try:
            if written or any(e.kind == "file_write" for e in run.analysis.side_effects):
                self.tracking_state.executed_write_stmt_codes.add(code)
                # The upstream check's per-file answers for this cell run were
                # taken before this write; nothing checked after it may use them.
                # Local: import cycle statement.processor -> upstream.virtual_lineage -> statement.processor.
                from ..upstream.virtual_lineage import forget_file_state_this_run

                forget_file_state_this_run()
                # Persist write provenance so a post-restart isolated reader can
                # tell an already-on-disk writer effect (skip it) from a stale
                # one (re-fire it): ``executed_write_stmt_codes`` is empty after
                # a restart. Not for a loop body statement: the simulation sees
                # the loop, which records its own (ControlStructureProcessor).
                if not is_control_body(code):
                    self._records.persist_write_provenance(code, run.inputs, run.tree, written)
        except AttributeError:
            pass
        if execution.accessed_files:
            self._records.persist_read_provenance(code, execution.accessed_files)

    def _miss_guarded(self, run: StatementRun, execution: StatementExecution, captured_vars: dict[str, Any]) -> bool:
        """Whether the perpetual-miss guard keeps *run*'s value from being written.

        This statement's key has churned for ``GUARD_AFTER_CONSECUTIVE_CHURN_MISSES``
        runs with zero hits, so serialising it again buys nothing. Routed
        through the SAME metadata-only path as the size-aware skip rather than
        through ``skip_cache``: output lineages must still persist for the
        upstream simulation, and only the value payload is the wasted cost.
        ``force_persist`` (``# @cash:persist`` / ``%cash_persist``) wins -- a
        user who explicitly asks for persistence gets it; the guard is a
        default, not a veto.
        """
        return (
            not run.skip_cache
            and not run.force_persist
            and not self._miss_guard.should_serialise(run.source_hash)
            and not self._store.write_is_cheap(run.outputs, captured_vars, execution.cost)
        )

    def _skip_a_newly_seen_draw(self, run: StatementRun) -> None:
        """Skip-cache *run* once when its execution revealed a hidden RNG draw.

        Its key was built without its RNG variable. Writing it creates an
        entry that a later run rebuilds and matches forever: after a kernel
        restart the ledger is empty, the same epoch-free key comes back, and
        the value computed under the OLD seed is restored -- silently, with a
        green badge. Persisting the ledger cannot fix that, because the key is
        consulted before any metadata is read.

        So the write is skipped exactly once. The value is correct for this
        run and is used normally; the next run builds the RNG-aware key,
        misses, and stores under it. Self-healing, one extra recompute per
        statement.
        """
        if self._randomness.draw_newly_seen and not run.skip_cache:
            run.skip_cache = True
            logger.debug(
                "%s Not storing %s: hidden RNG draw discovered after its key was built; next run keys it correctly",
                _LOG_ANNOTATION,
                run.source_hash[:12],
            )

    def _report_saved(self, run: StatementRun, saved_metadata: StatementCacheMetadata | None) -> None:
        """Copy what the store decided into *run*'s metrics for the badge."""
        if not saved_metadata:
            return
        metrics = run.metrics
        if saved_metadata.storage is not None:
            metrics["storage"] = saved_metadata.storage
        if saved_metadata.skipped_reason is not None:
            metrics["skipped_reason"] = saved_metadata.skipped_reason
            if saved_metadata.skipped_reason == GUARD_SKIP_REASON:
                # What kept changing the key.
                cause = self._miss_guard.cause(run.source_hash)
                if cause:
                    metrics["guard_cause"] = cause
        for k in COST_MODEL_KEYS:
            value = getattr(saved_metadata, k)
            if value is not None:
                metrics[k] = value

    def resolve_live_function_source(self, name: str) -> str | None:
        """The source of the function *name* is bound to in the user namespace now."""
        return live_function_source(name, self.shell.user_ns)

    def _check_callable_stateful(self, name: str) -> bool:
        """Return True if *name* resolves to a @stateful callable; continue-safe for known-pure."""
        if is_known_pure(name):
            return False
        func_obj = self.shell.user_ns.get(name)
        return func_obj is not None and is_stateful(func_obj)

    def _check_redundant_import(self, run: StatementRun) -> ProcessResult | None:
        """Detect redundant (already-imported) import statements.

        Returns the completed metrics when the import is genuinely redundant,
        else ``None``. An import that must run (a module recently reloaded, or
        a name that does not yet hold what the import binds) sets
        ``run.skip_cache``: an import runs, it is never stored.
        """
        code, tree = run.code, run.tree
        try:
            tree_check = tree if tree is not None else ast.parse(code.strip())
            import_names = redundant_import_names(tree_check)
            if not import_names:
                return None

            source_module_names = import_source_modules(tree_check)
            has_reloaded = bool((import_names | source_module_names) & self.recently_reloaded_modules)
            # Present is not enough: the name must hold what the import would
            # bind. `import array` then `from array import array` found `array`
            # present and skipped, leaving the module where the class belongs.
            all_present = all(name in self.shell.user_ns for name in import_names) and import_bindings_hold(
                tree_check, self.shell.user_ns
            )

            if has_reloaded:
                self.recently_reloaded_modules -= source_module_names
                logger.debug(
                    "%s Import involves recently-reloaded module, disabling cache for: %s",
                    _LOG_OPTIMIZATION,
                    code.strip(),
                )
                run.skip_cache = True
                return None

            if not all_present:
                # An import runs, it is never stored. Re-running one whose
                # module is loaded costs microseconds, and restoring one after
                # a restart imports the module anyway to unpickle what it
                # binds -- while a stored import hands back the objects it
                # bound THEN: `from helper import summary` restored after an
                # edit to helper.py put the pre-edit function back.
                run.skip_cache = True
                return None
            if all_present:
                logger.debug("%s SKIPPING redundant import: %s", _LOG_OPTIMIZATION, code.strip())
                run.metrics["status"] = CacheStatus.SKIPPED
                run.metrics["total_time"] = time.time() - run.process_start
                self._update_state_tracking(
                    code,
                    ExecutionResult(success=True, skipped=True),
                    run.inputs,
                    run.outputs,
                    set(),
                    run.source_hash,
                    run.cache_key,
                    tree=tree,
                )
                return run.metrics

        except (ImportError, AttributeError, SyntaxError) as e:
            logger.debug("%s Error checking imports: %s", _LOG_OPTIMIZATION, e)

        return None

    def _forget_file_answers_if_it_wrote(self, code: str, execution: StatementExecution) -> None:
        """Drop the cell's kept file answers (``CacheFreshnessChecker.
        forget_file_answers``) when the statement that just ran may have
        changed a file: it was seen writing one, cash's static writer check
        says it writes (its own text, or a user function it calls -- which
        covers writes made in C, like pyarrow's), or it failed part-way.

        Every executed statement dropped them at first, and a label loop
        (``ax.annotate`` per topic, reading a frame built from 10,000
        documents) re-checked all of them 52 times in one cell.
        """
        wrote = bool(execution.written_paths) or not getattr(execution.result, "success", False)
        if not wrote:
            try:
                wrote = (
                    statement_writes_files(code) or statement_calls_user_writer(code, self.shell.user_ns) is not None
                )
            except Exception:  # noqa: BLE001 - when unsure, check files again
                wrote = True
        if wrote:
            self._freshness.forget_file_answers(file_state_epoch())

    def _update_state_tracking(
        self,
        code: str,
        result: Any,
        inputs: set[str],
        outputs: set[str],
        accessed_files: set[str],
        source_hash: str,
        cache_key: str,
        tree: ast.Module | None = None,
    ) -> None:
        """Update lineage and variable tracking."""
        self.lineage_builder.capture_and_track_variables(
            self.tracking_state,
            outputs,
            inputs,
            code,
            source_hash,
            cache_key=cache_key,
            accessed_files=accessed_files,
            tree=tree,
        )

    def _analyze_and_hash(
        self, code: str, occurrence_index: int = 0, tree: ast.Module | None = None
    ) -> tuple[StatementEffects, str, str, float, float]:
        """Analyze *code* and compute its cache key.

        Returns ``(effects, source_hash, cache_key, analysis_time, hash_time)``.
        The key comes from the unified ``compute_cache_key``, the effects from
        :func:`~cash.analysis.mutation_effects.statement_effects` -- the same
        two functions the upstream simulation uses, with the live namespace as
        the source of called functions.

        Args:
            code: Python source code to analyze.
            occurrence_index: Index for disambiguating duplicate code blocks.
            tree: ``ast.parse(code.strip())``, or None when that fails.
        Raises:
            CacheKeyComputationError: If the cache key cannot be computed.
        """
        t1 = time.time()
        effects = statement_effects(
            code,
            tree,
            namespace=self.shell.user_ns,
            resolve_source=self.resolve_live_function_source,
            control_body=is_control_body(code),
        )
        inputs, outputs = set(effects.inputs), set(effects.outputs)
        self._records.log_statement_reads(code, inputs)
        analysis_time = time.time() - t1

        t2 = time.time()

        # A draw READS its module's hidden RNG variable; fold it into the key so a
        # re-seed re-keys the draw. Only the LOCAL key-input set gets it
        # -- the returned ``inputs`` stays clean, so cacheability/mutation never
        # see a phantom variable. Its lineage ALSO reaches the draw's OUTPUT
        # lineage (via capture_and_track_variables, which reads the same helper),
        # so a seed change propagates to everything cached downstream.
        # ...plus the modules a PRIOR run observed this statement drawing from
        # without saying so in its AST (``model.fit()``). Without this the
        # virtual RNG variable has no reader here, so an upstream re-seed cannot
        # propagate and the statement's consumers keep hitting.
        key_inputs = inputs | key_hidden_reads(code, self.tracking_state)

        try:
            cache_key, source_hash, _, _, _ = compute_cache_key(
                code,
                key_inputs,
                ctx=CacheKeyContext(
                    variable_lineage=self.tracking_state.variable_lineage,
                    user_ns=self.shell.user_ns,
                    function_tracker=self.function_tracker,
                    compute_hash_fn=self.compute_hash,
                ),
                outputs=outputs,
                occurrence_index=occurrence_index,
            )
        except Exception as exc:
            raise CacheKeyComputationError(f"Failed to compute cache key for: {code[:80]!r}") from exc

        self._randomness.record_seeds(code, cache_key)

        hash_time = time.time() - t2
        return effects, source_hash, cache_key, analysis_time, hash_time

    def _log_cache_lookup(
        self,
        code: str,
        cache_key: str,
        inputs: set[str],
        cached_data: Any,
        analysis_time: float,
        hash_time: float,
        cache_check_time: float,
    ) -> None:
        logger.debug("[CACHE DEBUG] Statement: %s...", code[:50])
        logger.debug("[CACHE DEBUG] Key: %s...", cache_key[:40])
        logger.debug("[CACHE DEBUG] Inputs: %s", inputs)
        logger.debug("[CACHE DEBUG] Cache hit: %s", cached_data is not None)
        logger.debug(
            "[TIMING] Analysis: %.1fms | Hash: %.1fms | Lookup: %.1fms",
            analysis_time * 1000,
            hash_time * 1000,
            cache_check_time * 1000,
        )

    def _handle_execution_error(self, result: Any, silent: bool) -> bool | None:
        if not silent:
            raise result.error from None
        logger.debug("[SILENT] Error in statement: %s", result.error)
        return False
