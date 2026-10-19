"""Core statement processing: analysis, cache lookup, execution, and lineage tracking."""

from __future__ import annotations

import ast
import contextlib
import io
import logging
import os
import secrets
import sys
import types
from collections.abc import Callable, Collection, Generator, Iterable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from cash import _plain_data, stored_pickle, value_hash
from cash._active import default_cash
from cash._clock import perf_counter as _perf_counter
from cash.backends import memory_backend
from cash.backends.persistence_policy import COMPUTE_FLOOR_S, PersistencePolicy
from cash.control_markers import has_marker
from cash.diagnostics import log_diagnostic, warn_diagnostic
from cash.exceptions import (
    CacheKeyComputationError,
    CashCacheIneffectiveWarning,
    CashWarning,
)
from cash.notebook import shadow as _shadow
from cash.notebook._protocols import CashInstanceProtocol, ShellProtocol
from cash.notebook.cache_key import (
    THE_STATEMENT,
    CacheKeyContext,
    CacheKeyResult,
    compute_cache_key,
    statement_source_hash,
    unkeyed_reach,
)
from cash.notebook.cache_status import CacheStatus, ExecutionResult
from cash.notebook.call_effects import DigestHandoff
from cash.notebook.compiled_source import register_cell_source
from cash.notebook.consumables import drawn_stream_inputs
from cash.notebook.per_item_reruns import MIN_COST_S as PER_ITEM_MIN_COST_S
from cash.notebook.read_handles import reaches_read_handle
from cash.notebook.statement import sparse_order
from cash.notebook.statement._metadata import StatementCacheMetadata
from cash.notebook.statement.amplification import AmplificationGuard
from cash.notebook.statement.call_routing import CallRouting
from cash.notebook.statement.capture import display_execution_output, make_capture_ctx, unshown_output
from cash.notebook.statement.carrier_advances import (
    advance_carriers,
    carrier_candidates,
    carriers_an_entry_advanced,
)
from cash.notebook.statement.control_body import is_control_body
from cash.notebook.statement.derivation_edges import (
    apply_share_record,
    logging_edges,
    record_shared_object_edges,
)
from cash.notebook.statement.directives import statement_directives, ttl_floor_from_called_functions
from cash.notebook.statement.evictions import EvictedRecomputes
from cash.notebook.statement.expression_memo import ExpressionMemo
from cash.notebook.statement.file_deps import StatementFileDeps
from cash.notebook.statement.freshness import CacheFreshnessChecker
from cash.notebook.statement.hit import CacheHitServer
from cash.notebook.statement.imports import (
    import_bindings_hold,
    import_needs_reexecution,
    import_source_modules,
    redundant_import_names,
)
from cash.notebook.statement.in_loop import InLoopChanges
from cash.notebook.statement.input_change import input_change_reason
from cash.notebook.statement.lineage import StatementLineageBuilder, clear_rebound_edges
from cash.notebook.statement.miss_guard import GUARD_SKIP_REASON, MissGuard
from cash.notebook.statement.mutation_routing import MutationRouting
from cash.notebook.statement.mutations import MutationClassifier
from cash.notebook.statement.output_refusals import is_history_name, live_shared_reason, unrestorable_output_reason
from cash.notebook.statement.randomness import StatementRandomness
from cash.notebook.statement.rebuild_cost import RebuildCostLedger
from cash.notebook.statement.records import StatementRecords
from cash.notebook.statement.restore import StatementRestorer
from cash.notebook.statement.results import COST_MODEL_KEYS, ProcessResult
from cash.notebook.statement.run import (
    ECHO_FIELD,
    LOOP_ITERABLE_NAME,
    NO_LOOP_ITERABLE,
    CodeRunner,
    StatementExecution,
    StatementRun,
    error_result,
    over_the_handed_iterable,
)
from cash.notebook.statement.store import StatementStore
from cash.notebook.tracking_state import TrackingState
from cash.notebook.versioned_json_store import resolve_cache_dir
from cash.purity import is_known_pure, is_stateful

from ...analysis.analysis_budget import (
    ANALYSIS_BUDGET_REASON,
    AnalysisBudget,
    AnalysisBudgetExceeded,
    analysis_budget,
    analysis_budget_seconds,
    no_analysis_budget,
)
from ...analysis.annotations import CacheAnnotation
from ...analysis.ast_util import resolve_dotted_name
from ...analysis.cacheability import analyze_statement, statement_writes_files
from ...analysis.cacheability_decision import (
    cacheability_verdict,
    identity_coupled_reason,
)
from ...analysis.code_analyzer import CodeAnalyzer, calls_ipython, statement_code
from ...analysis.handed_callables import answers_shared, new_analysis_round, stop_sharing_answers
from ...analysis.mutation_effects import (
    StatementEffects,
    live_function_source,
    statement_effects,
)
from ...analysis.namespace_effects import statement_calls_user_writer, statement_starting_workers
from ...analytics import AnalyticsManager
from ...tracking.file_dep_snapshot import file_state_epoch, probe_answers
from ...tracking.file_tracker import FileAccessTracker
from ...tracking.function_tracker import FunctionTracker
from ...tracking.randomness import (
    advanced_rng_lineage,
    carrier_positions,
    drawn_rng_vars,
    hidden_lineage_writes,
    moved_carrier_names,
    rng_carrier_kind,
    rng_virtual_var,
)
from ..anchors import anchor_digests
from ..borrowed_memory import borrowed_memory_names, lazy_frame_sources
from ..callee_reach import (
    CWD,
    ENVIRON,
    import_state_writes,
    module_globals,
    module_state_writes,
    rebound_modules,
)
from ..holder_patches import holder_patches
from ..lineage_formula import (
    held_lineage,
    keep_held_lineage,
    key_hidden_reads,
    no_cache_value_digest,
    use_lineage_alias_store,
)
from ..magic_effects import (
    is_magic_statement,
    is_rerun_magic,
    magic_base,
    magic_effects,
    magic_output_lineage,
    magic_rng_advances,
    simulation_cell,
)
from ..pyplot_draws import holds_artists, pyplot_draw_names
from ..pyplot_state import CHANGED_PYPLOT_REASON, pyplot_state
from ..recorded_reads import note_writes, snapshot
from ..restored_var import FORWARD_PROBE_PLACEHOLDER, apply_held_var
from ..run_memo import forget_file_state_this_run
from ..shared_objects import (
    VALUE_TYPES,
    GaveUp,
    WalkBudgetExceeded,
    anchors_held,
    is_value,
    library_value_types,
    output_history,
    referring_names,
    share_group,
    view_base_names,
    walk_budget,
)
from ..value_cutoff import LineageAliasStore
from ..write_observer import kept_writes, observe_writes
from .lineage import note_own_writes
from .observed_draws import ObservedDrawStore

__all__ = ["StatementProcessor", "is_control_body"]

# Debug log prefixes — module-level constants for filtering and consistency.
_LOG_PROCESSOR = "[PROCESSOR]"

#: How long the share check of a statement's outputs may take: a multiple of
#: what the statement cost, but not less than a tenth of a second. A second
#: let a 0.05 s loop over a big list of sessions take 1.1 s, every run: past
#: the budget the statement only re-runs, which costs what it is worth.
SHARE_CHECK_FACTOR = 2.0
SHARE_CHECK_FLOOR_S = 0.1
#: The least time the search for the variables sharing an object with a
#: statement's outputs takes when the share check could not say
#: (`referring_names`).
SHARE_FALLBACK_FLOOR_S = 1.0
#: The packages of the kernel's own objects, left out of the share check
#: (`StatementProcessor._kernel_object`).
_KERNEL_PACKAGES = frozenset({"IPython", "ipykernel"})


def _may_hold_memory(value: Any) -> bool:
    """Could another value borrow *value*'s memory? Not a module, class or
    function, nor a number or a string."""
    if isinstance(value, (types.ModuleType, type, types.FunctionType, types.BuiltinFunctionType)):
        return False
    return not isinstance(value, (int, float, complex, str, bool, type(None)))


def share_check_budget(cost: float) -> float:
    """Seconds the share check of a statement that took *cost* may take."""
    return max(SHARE_CHECK_FLOOR_S, SHARE_CHECK_FACTOR * cost)


_LOG_DEBUG = "[DEBUG]"
_LOG_MUTATION = "[MUTATION]"
_LOG_CACHE_HIT = "[CACHE_HIT_DEBUG]"
_LOG_CACHE = "[CACHE]"
_LOG_CACHE_DEBUG = "[CACHE DEBUG]"
_LOG_OPTIMIZATION = "[OPTIMIZATION]"
_LOG_FORBIDDEN = "[FORBIDDEN]"
_LOG_ANNOTATION = "[ANNOTATION]"

#: Why a statement that runs a magic or a shell command is not cached.
_IPYTHON_REASON = "Runs an IPython magic or shell command, which cash cannot see into, so it runs every time"

#: Why a statement that read a file replaced while it ran is not cached.
INPUT_CHANGED_REASON = (
    "A file it read changed while it ran, so this run is not cached: the next run reads the settled file"
)


logger = logging.getLogger(__name__)


#: The share of a statement's run time its store may spend pickling an
#: output whose change check ran out of budget (`_keeping_outputs`). A hit
#: reads those bytes back in about the time they took to make, or longer:
#: past half the statement's run, keeping them saves little, and the store
#: gives them up having spent at most that.
PAST_BUDGET_SHARE = 0.5

#: The level a hint's own error is logged at (`_note_per_item_rerun`,
#: `_note_round_trips`): debug, as a hint never breaks a statement. The test
#: suite raises it to warning, so an error there shows with a failing test
#: instead of passing for a hint that had nothing to say.
HINT_ERROR_LEVEL = logging.DEBUG

#: What `_keeping_outputs` reads an unbound name as.
_ABSENT = object()


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
        if _shadow.SHADOW is None and _shadow.requested():
            # What each statement ran is recorded and checked before it is
            # served (`cash.notebook.shadow`).
            _shadow.enable()

        self._amplification = AmplificationGuard()
        #: Statements already warned NOTEBOOK-SHARE-UNCHECKED about.
        self._share_unchecked_warned: set[str] = set()
        #: Statements already told about NOTEBOOK-ANALYSIS-BUDGET.
        self._past_budget_told: set[str] = set()
        #: What a statement's text last took to run, in seconds, or None when
        #: never measured (``CashSession.expected_compute``, set by the
        #: magics): its analysis budget (`analysis_budget_seconds`).
        self.expected_run_time: Callable[[str], float | None] = lambda code: None

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
        # What a cell's last line showed, for the next edit that computes it again.
        self._memo = ExpressionMemo()
        self._rebuild_cost = RebuildCostLedger(shell, self.tracking_state, cash_instance)
        # A statement's argument digests, handed between its in-place-change
        # check and the calls inside it, so each hashes a big argument once.
        digests = DigestHandoff()
        self._mutations = MutationClassifier(shell, self.tracking_state, compute_hash_fn, digests)
        self._calls = CallRouting(
            shell,
            self.tracking_state,
            self.function_tracker,
            compute_hash_fn,
            cash_instance=self.get_cash_instance,
            is_stateful_call=self._check_callable_stateful,
            digests=digests,
        )

        # Cache-freshness checker (TTL / file-dep / input-file invalidation).
        # Stateless w.r.t. tracking state — receives it per call. It reads the
        # backend statements are stored in, which `cash.configure` may replace.
        self._freshness = CacheFreshnessChecker(
            backend_of=(lambda: self.cash_instance.backend) if cash_instance is not None else None,
        )

        # Perpetual-miss guard: learns which statements can never hit
        # (unstable cache key -> a new key every run -> zero hits) and stops
        # SERIALISING them, while keeping the hash + the lookup. Verdicts persist
        # to the cache dir so a restart doesn't re-pay the learning. An
        # unresolvable dir just means session-scoped verdicts.
        try:
            _guard_dir = resolve_cache_dir(cash_instance.backend if cash_instance is not None else None)
        except (AttributeError, TypeError):
            _guard_dir = None
        self._miss_guard = MissGuard(_guard_dir)
        self._randomness.use_draw_store(ObservedDrawStore(_guard_dir) if _guard_dir else None)
        self._evicted = EvictedRecomputes()

        # Statement-level file-dep tracker. Stateless w.r.t. tracking state —
        # receives it per call. ``executed_file_deps`` lives on TrackingState.
        self._file_deps = StatementFileDeps()

        # Statement-level cache restorer. Hydrates outputs from a cached
        # payload + replays stdout/stderr/rich-outputs.
        # Stateless w.r.t. tracking state — receives it per call.
        self._stmt_restorer = StatementRestorer(shell=shell, compute_hash=compute_hash_fn)
        self._records = StatementRecords(shell, self.tracking_state, cash_instance, self.function_tracker)
        self._in_loop = InLoopChanges(shell, self.tracking_state, self._records)
        self._mutation_routing = MutationRouting(
            shell, self.tracking_state, self._mutations, self._records, self._in_loop
        )

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
        # Lineages found to name the same value as an earlier one (`value_cutoff`).
        use_lineage_alias_store(LineageAliasStore(_guard_dir) if _guard_dir else None)
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

    @property
    def in_loop(self) -> InLoopChanges:
        """What the statements of the loops running now change (`InLoopChanges`)."""
        return self._in_loop

    def get_cash_instance(self) -> CashInstanceProtocol | None:
        """Return the Cash instance for decorator call tracking.

        Tries ``self.cash_instance`` first, then the default instance, so
        that ``@cash.cache`` calls are captured even when the user imports
        ``from cash import cache``.
        """
        if self.cash_instance is not None:
            return self.cash_instance
        return default_cash()

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

    def note_share_edges(self, code: str, edges) -> None:
        """Record the edges the control structure *code* linked when it ran
        (``StatementRecords.note_share_edges``), under its source hash."""
        self._records.note_share_edges(statement_source_hash(code), edges)

    def recorded_share_edges(self, code: str):
        """The edges the control structure *code* linked when it last ran
        (``StatementRecords.recorded_share_edges``), or None."""
        return self._records.recorded_share_edges(statement_source_hash(code))

    def note_pyplot_draws(self, code: str, names: frozenset[str]) -> None:
        """Record the names the control structure *code* changed by drawing
        through pyplot (``StatementRecords.note_pyplot_draws``)."""
        self._records.note_pyplot_draws(statement_source_hash(code), names)

    def note_module_state(self, code: str, names, modules) -> None:
        """Record the local *modules*, held by *names*, the loop or branch
        *code* set state on (``StatementRecords.note_module_state``)."""
        self._records.note_module_state(code, names, modules)

    def _advance_carriers(self, source_hash: str, names: frozenset[str] | set[str] | None, key: str, code: str) -> None:
        """``advance_carriers`` for the statement *code*, keyed *key*, and the
        same record kept for a later kernel: a cheap draw is never stored, and
        without it the simulation after a restart does not know it drew."""
        advance_carriers(self.tracking_state, source_hash, names, key, code, self.shell.user_ns)
        self._records.persist_carrier_advances(source_hash, names)

    def advance_carriers_of_a_structure(self, code: str, positions: dict, before: dict[str, str]) -> None:
        """Move on the generators a top-level loop or branch *code* drew from.

        Its body runs statement by statement, and a body statement moves
        nothing (see :meth:`_executing`), so the structure does it as a whole,
        as the simulation sees it: one statement keyed on its lineages at
        entry. *positions* are the generators' positions before it ran
        (``carrier_positions``); *before* the lineages then. A generator the
        structure rebound, or that a statement run whole moved already, has
        its lineage.
        """
        user_ns = self.shell.user_ns
        lineage = self.tracking_state.variable_lineage
        moved = {name for name in moved_carrier_names(positions, user_ns) if lineage.get(name) == before.get(name)}
        try:
            key = self.key_as_one_statement(code, before)
        except Exception:  # noqa: BLE001 - unkeyable: a lineage no statement shares
            key = "unkeyable:" + secrets.token_hex(16)
        self._advance_carriers(statement_source_hash(code), moved, key, code)

    def advance_rng_of_a_structure(self, code: str, before: dict[str, str]) -> None:
        """Move on the RNG variables a top-level loop or branch *code* drew from.

        A body statement moves none (see :meth:`_advance_rng`), so the
        structure does it as a whole, as the simulation sees it: one statement
        keyed on its lineages at entry (*before*). A variable a seed in the
        body or a statement run whole moved already keeps its lineage.
        """
        lineage = self.tracking_state.variable_lineage
        drawn = drawn_rng_vars(key_hidden_reads(code, self.tracking_state), code, before)
        drawn = {var for var in drawn if lineage.get(var) == before.get(var)}
        if not drawn:
            return
        try:
            key = self.key_as_one_statement(code, before)
        except Exception:  # noqa: BLE001 - unkeyable: a lineage no statement shares
            key = "unkeyable:" + secrets.token_hex(16)
        for var in sorted(drawn):
            self.tracking_state.lineage.record(var, advanced_rng_lineage(key, var))

    def _advance_rng(self, run: StatementRun) -> None:
        """Move on the RNG variables *run*'s statement drew from, once it ran
        or was restored (a restore puts the stream where the run left it).

        A seeded draw reads its module's RNG variable, and leaves it at a new
        lineage derived from its own key (``advanced_rng_lineage``), so a
        change in the number or order of draws above re-keys every draw below.
        After the output lineages, which read the variable as it stood before
        the draw, as the simulation's do. Not for a loop or branch body: the
        structure moves it as a whole (:meth:`advance_rng_of_a_structure`).
        """
        if not run.cache_key or is_control_body(run.code) or run.magic_reads is not None:
            return  # a magic statement's draw moves it in record_magic
        state = self.tracking_state
        try:
            drawn = drawn_rng_vars(key_hidden_reads(run.code, state), run.code, state.variable_lineage)
        except (SyntaxError, ValueError, AttributeError, RecursionError):
            return
        key = run.rng_advance_key or run.cache_key
        for var in sorted(drawn):
            state.lineage.record(var, advanced_rng_lineage(key, var))

    def _reads_a_stream(self, run: StatementRun) -> bool:
        """Whether *run* reads, itself or through the functions it calls, a
        random generator or an iterator a variable holds: computing it
        again draws the next values, so what it showed is not kept."""
        user_ns = self.shell.user_ns
        try:
            names = carrier_candidates(run.inputs or (), user_ns, self.tracking_state.variable_lineage)
        except Exception:  # noqa: BLE001 - not told: assume it does
            return True
        for name in names:
            value = user_ns.get(name) if isinstance(name, str) else None
            if isinstance(value, Iterator) or rng_carrier_kind(value) is not None:
                return True
        return False

    def _memo_key(self, code: str) -> str | None:
        """*code*'s key as a statement of its own, now: what a value the
        expression memo kept must still have to be handed out."""
        try:
            return self.key_as_one_statement(code, self.tracking_state.variable_lineage)
        except Exception:  # no key, no reuse
            logger.debug("%s no key for the kept expression %r", _LOG_PROCESSOR, code[:80], exc_info=True)
            return None

    def key_as_one_statement(self, code: str, lineages: dict[str, str]) -> str:
        """The key the upstream simulation gives *code* as one statement, with
        the variables at *lineages* (``StatementLineage._key``)."""
        tree = ast.parse(code)
        effects = statement_effects(
            code,
            tree,
            namespace=self.shell.user_ns,
            resolve_source=self.resolve_live_function_source,
            control_body=False,
        )
        key, _, _, _, _ = compute_cache_key(
            code,
            set(effects.inputs) | key_hidden_reads(code, self.tracking_state),
            ctx=CacheKeyContext(
                variable_lineage=lineages,
                user_ns=self.shell.user_ns,
                function_tracker=self.function_tracker,
                compute_hash_fn=self.compute_hash,
                reads=self.tracking_state.reads,
            ),
            outputs=set(effects.outputs),
        )
        return key

    def adopt_unseen_seeds(self) -> None:
        """Record a seed no statement set, before the cell's upstream check
        (see :meth:`StatementRandomness.adopt_unseen_seeds`)."""
        self._randomness.adopt_unseen_seeds()

    def begin_cell_rng_observation(self) -> None:
        """Open a fresh per-cell RNG accumulation, before the cell's statements run."""
        self._randomness.begin_cell()

    def cell_rng_lineage(self) -> dict[str, str]:
        """The RNG variables' lineages where the cell's start position was taken
        (see :meth:`StatementRandomness.cell_pre_lineage`)."""
        return self._randomness.cell_pre_lineage()

    def rng_lineages(self) -> dict[str, str]:
        """The lineage of each RNG variable, now."""
        return self._randomness.rng_lineages()

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
        keep_displays: bool = False,
        force_outputs: set[str] | None = None,
        is_last: bool = True,
        loop_iterable: Any = NO_LOOP_ITERABLE,
        header_files: frozenset[str] = frozenset(),
        header_remote: frozenset[str] = frozenset(),
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
            keep_displays: With *stream_output*, the displays are still
                captured and held, so a hit shows them again (a top-level
                statement: its prints stream as in plain).
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
                the caller could recover one -- see ``statement_source`` in
                ``ipython/statement_source.py``. Compiled in place of ``code`` so a
                function defined here keeps its comments for
                ``inspect.getsource`` (and therefore for a per-line
                ``# @cash:assume-safe`` waiver). Never affects the cache key:
                ``code`` (the unparsed form) is what is hashed. Discarded
                whenever call interception rewrote an eligible call in
                ``code`` -- see ``CallRouting.code_and_tree_for_execution``.
            loop_iterable: For a ``for`` loop run as one unit whose header the
                caller already evaluated: that value, which the loop iterates
                instead of evaluating its header again (a random draw, a
                drained queue or a property would otherwise give a second,
                different value). Never affects the cache key: ``code`` keeps
                the header as written.
            header_files / header_remote: What evaluating that header read,
                recorded as the statement's own reads.

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
            keep_displays=keep_displays,
            force_outputs=force_outputs,
            is_last=is_last,
            loop_iterable=loop_iterable,
            header_files=header_files,
            header_remote=header_remote,
        )
        # The analyses before the statement runs share their walks through
        # what it reads (`answers_shared`), up to the moment it runs.
        shadow = _shadow.SHADOW
        with answers_shared():
            done = self._prepare(run)
            if done is not None:
                if shadow is not None:
                    shadow.statement_served(run, done)
                self._advance_rng(run)
                return done
            with self._executing(run) as runner:
                self._offer_digests(run, runner)
                unit = self._begin_recording(shadow, run)
                try:
                    runner.run()
                finally:
                    self._end_recording(shadow, run, unit)
                    self._mutations.digests.withdraw_offer()
        try:
            result = self._finish(run, runner.execution)
            if shadow is not None:
                shadow.statement_finished(run, unit, result)
            return result
        finally:
            self._advance_rng(run)

    @staticmethod
    def _begin_recording(shadow: Any, run: StatementRun) -> Any:
        """Start recording which code *run*'s statement runs (shadow mode)."""
        if shadow is None or run.skip_cache:
            return None
        spent = shadow.spent
        unit = shadow.begin("stmt")
        run.record_time = shadow.spent - spent
        if unit is not None and run.needs_trace:
            unit.lookups_followed = True
        return unit

    @staticmethod
    def _end_recording(shadow: Any, run: StatementRun, unit: Any) -> None:
        if unit is not None:
            spent = shadow.spent
            shadow.end(unit)
            run.record_time += shadow.spent - spent

    def _offer_digests(self, run: StatementRun, runner: CodeRunner) -> None:
        """Offer the arguments' digests taken before the statement to the
        ``@cash.cache`` call it is (`DigestHandoff.offer_to_decorator`):
        only when it runs as written (or as its original text, comments and
        quotes intact), not through the call cache."""
        if runner.source != (run.exec_source if run.exec_source is not None else run.code):
            return
        run.offered_as = register_cell_source(runner.source)
        self._mutations.digests.offer_to_decorator(run.offered_as, run.tree, self.shell.user_ns)

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
        keep_displays: bool = False,
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
            keep_displays=keep_displays,
            force_outputs=force_outputs,
            is_last=is_last,
        )
        shadow = _shadow.SHADOW
        with answers_shared():
            done = self._prepare(run)
            if done is not None:
                if shadow is not None:
                    shadow.statement_served(run, done)
                self._advance_rng(run)
                return done
            with self._executing(run) as runner:
                unit = self._begin_recording(shadow, run)
                try:
                    await runner.run_async()
                finally:
                    self._end_recording(shadow, run, unit)
        try:
            result = self._finish(run, runner.execution)
            if shadow is not None:
                shadow.statement_finished(run, unit, result)
            return result
        finally:
            self._advance_rng(run)

    def _prepare(self, run: StatementRun) -> ProcessResult | None:
        """Analyse, key and look up *run*'s statement, before it may execute.

        Returns the finished result when the statement needs no execution -- a
        redundant import, or a cache hit whose restore succeeded -- and
        ``None`` when it must run, with *run* filled in for :meth:`_executing`
        and :meth:`_finish`.
        """
        self._begin(run)
        run.analysis_budget = AnalysisBudget(analysis_budget_seconds(self._expected_run_time(run.code)))
        effects, analysis_time, hash_time = self._within_budget(run, lambda: self._analyze(run))
        if effects.unanalysed:
            run.metrics["uncacheable_reasons"].extend(effects.unanalysed)
            run.skip_cache = True
        if run.tree is not None and calls_ipython(run.tree):
            run.metrics["uncacheable_reasons"].append(_IPYTHON_REASON)
            run.skip_cache = True
            run.ipython_bindings = self.bindings()
            # One the user wrote at cell level: the simulation reads a loop or
            # a branch as one unit, so one in a body keeps no lineage, as before.
            if len(run.tree.body) == 1 and is_magic_statement(run.tree.body[0]) and not is_control_body(run.code):
                run.magic_reads = self.magic_reads(run.tree.body[0])
                run.magic_snapshots = self._mutations.magic_snapshots(run.tree.body[0], run.code)

        done = self._check_redundant_import(run)
        if done is not None:
            return done

        # Computed once: used by the cacheability decision and (on the
        # cache-miss path) by _post_execute for in-place-mutation tracking.
        run.analysis = analyze_statement(run.code, run.tree, self.shell.user_ns)
        self._within_budget(run, lambda: self._mutation_routing.route(run, effects))
        try:
            with analysis_budget(run.analysis_budget):
                self._decide_cacheability(run)
        except AnalysisBudgetExceeded:
            run.analysis_budget.tripped = True
        self._settle_budget(run)
        lineage = self.tracking_state.variable_lineage
        self.tracking_state.statement_entry_lineages[run.code] = {
            name: lineage[name] for name in run.inputs if name in lineage
        }
        # An UNSEEDED estimator fit routed to caching above is frozen on re-run
        # with no warning -- cash's AST detector cannot see the randomness inside
        # sklearn's compiled .fit(). Warn now (compute time); the same set drives
        # the restore-time warning on a cache hit below. Only once the statement
        # is known to be cached: one that re-executes fits afresh every run.
        fits_cached = run.est_fit if not run.skip_cache else set()
        unseeded_fits = self._randomness.warn_unseeded_estimator_fit(run.code, fits_cached, run.allow_random)
        self._randomness.stamp_random_effect(run.metrics, run.code, run.unseeded_calls, unseeded_fits)

        metadata, cached_data = self._lookup(run, analysis_time, hash_time)
        if run.must_run:
            # It learns what it changes as it runs: a restore would not.
            metadata, cached_data = None, None
        run.entry_holders = bool(metadata is not None and metadata.holders)
        if _shadow.SHADOW is not None:
            _shadow.SHADOW.statement_lookup(self, run, bool(cached_data))
            if cached_data and _shadow.SHADOW.stale(run.shadow):
                # Code it ran changed since it was stored: it runs again.
                cached_data = None
            elif cached_data and run.needs_trace and not _shadow.SHADOW.agreed(run.shadow):
                # Its key does not name what a string found: only a trace
                # checked unchanged serves it.
                cached_data = None
        elif run.needs_trace:
            cached_data = None
        if cached_data and self._live_objects_shared(run, metadata):
            cached_data = None
        if cached_data and not import_needs_reexecution(run.tree, self.shell.user_ns):
            hit_result = self._serve_hit(run, cached_data, metadata, unseeded_fits)
            if hit_result is not None:
                return hit_result

        if not run.mut_record and not is_control_body(run.code):
            run.rewatching = self._mutations.start_rewatch(run.source_hash, run.cache_key, run.tree)
        user_ns = self.shell.user_ns
        run.bound_before = {
            name: id(user_ns[name]) for name in run.outputs | run.mut_observe | run.mut_assumed if name in user_ns
        }
        self._route_calls(run)
        return None

    def _expected_run_time(self, code: str) -> float | None:
        """What *code* last took to run (`expected_run_time`); None when
        unknown or when asking fails."""
        try:
            return self.expected_run_time(code.strip())
        except Exception:  # noqa: BLE001 - no measurement: the floor applies
            return None

    def _within_budget(self, run: StatementRun, step: Callable[[], Any]) -> Any:
        """*step*, with *run*'s analysis budget (`analysis_budget`).

        The analyses answer "could not tell in time" themselves, each as for
        a callee it cannot read. One that does not, and lets the budget's
        exception out, is asked again without a budget: a step left half
        done would answer for less than the statement does.
        """
        budget = run.analysis_budget
        try:
            with analysis_budget(budget):
                return step()
        except AnalysisBudgetExceeded:
            budget.tripped = True
            with no_analysis_budget():
                return step()

    def _settle_budget(self, run: StatementRun) -> None:
        """Note what *run*'s analyses took, and when they ran past its budget,
        run it uncached with its outputs keyed by value, and watch what of
        the process it changes (`_observe_process_state`)."""
        budget = run.analysis_budget
        if budget is None:
            return
        run.analysis_spent = budget.spent()
        if not budget.tripped:
            return
        run.past_budget = True
        run.skip_cache = True
        reasons = run.metrics["uncacheable_reasons"]
        if ANALYSIS_BUDGET_REASON not in reasons:
            reasons.append(ANALYSIS_BUDGET_REASON)
        run.process_before = (os.getcwd(), dict(os.environ))

    def _budget_left(self, run: StatementRun) -> AnalysisBudget | None:
        """What is left of *run*'s analysis budget, for the analyses after it
        ran; None for a statement that has none."""
        budget = run.analysis_budget
        if budget is None:
            return None
        return AnalysisBudget(max(0.0, budget.seconds - run.analysis_spent))

    @contextmanager
    def _after_it_ran(self, run: StatementRun) -> Generator[None, None, None]:
        """Run the analyses below, after *run* ran, within what is left of
        its budget, and count what they took."""
        left = self._budget_left(run)
        if left is None:
            yield
            return
        try:
            with analysis_budget(left):
                yield
        finally:
            run.analysis_spent += left.spent()

    @staticmethod
    def _observe_process_state(run: StatementRun) -> None:
        """What of the process a statement past its budget changed, seen:
        what it hands over was not read for it (`callee_reach.handed`)."""
        if run.process_before is None:
            return
        cwd, environ = run.process_before
        run.process_before = None
        changed = set(run.process_state)
        try:
            if os.getcwd() != cwd:
                changed.add(CWD)
        except OSError:
            changed.add(CWD)
        if dict(os.environ) != environ:
            changed.add(ENVIRON)
        run.process_state = frozenset(changed)

    def _tell_past_budget(self, run: StatementRun, cost: float) -> None:
        """Say once per statement that it was not cached because working out
        how to key it took longer than it could save
        (``NOTEBOOK-ANALYSIS-BUDGET``); only for a statement slow enough that
        caching it would have mattered."""
        if not run.past_budget or cost < COMPUTE_FLOOR_S or run.source_hash in self._past_budget_told:
            return
        self._past_budget_told.add(run.source_hash)
        budget = run.analysis_budget.seconds if run.analysis_budget is not None else 0.0
        what = (
            f"this statement ran {cost:.2f}s and was not cached: working out what it can run through the "
            f"objects it hands over took more than its {budget:.1f}s budget (four times what it took the last "
            "time it ran, at least half a second). It runs every time, and what it binds is keyed by its value."
        )
        fix = (
            "nothing to do: when it runs, it measures itself, and its next budget follows. To cache it anyway, "
            "pass the function and the values it needs rather than an object graph holding them, or wrap the "
            "work in a function decorated with @cash.cache."
        )
        log_diagnostic(logger, "NOTEBOOK-ANALYSIS-BUDGET", what, fix)
        warn_diagnostic(CashCacheIneffectiveWarning, "NOTEBOOK-ANALYSIS-BUDGET", what, fix)

    def _begin(self, run: StatementRun) -> None:
        """Read *run*'s annotation, warn about its randomness, start its
        metrics and parse it."""
        code = run.code
        run.effective_ttl, run.force_persist, run.skip_cache, run.allow_random = statement_directives(
            run.annotation, run.ttl, self.persist_all
        )
        self._calls.begin_statement(run.effective_ttl, run.force_persist)
        # A seed set where no statement shows it (before `%cash_on`, inside an
        # imported function, or by the statement above) is in force from here.
        self._randomness.adopt_unseen_seeds()
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

        run.process_start = _perf_counter()

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
        # ``seeded_accumulator_loop``'s conditions -- each container seeded
        # empty in the same cell, written only inside this one loop -- which
        # are precisely the guarantees that make one writer's snapshot
        # sufficient.
        if callee_globals:
            outputs = outputs | callee_globals
        run.inputs, run.outputs = set(effects.inputs), outputs
        # Exposed so the badge can show a short prefix in the row-detail "Key"
        # field: two runs of the same statement in the same slot or not.
        run.metrics["cache_key"] = run.cache_key
        return effects, analysis_time, hash_time

    def _decide_cacheability(self, run: StatementRun) -> None:
        """Skip-cache *run* when the static cacheability decision refuses it."""
        if run.skip_cache:
            return
        cacheable, reasons, varies = cacheability_verdict(
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
            run.value_varies = varies
            return
        unfollowed, by_string, unkeyed_modules, followed = unkeyed_reach(
            run.code, run.inputs, self.shell.user_ns, self.tracking_state.variable_lineage
        )
        if unfollowed:
            # Its key cannot name what the notebook functions held in them
            # read (``held_functions``): served, it could be stale.
            names = ", ".join(repr(n) for n in sorted(unfollowed))
            run.metrics["uncacheable_reasons"].append(
                f"Reads {names}, which holds more objects than cash walks to find the notebook functions in it: "
                f"what those functions read cannot be keyed, so the statement runs every time"
            )
            run.skip_cache = True
            return
        if by_string and followed and _shadow.SHADOW is not None:
            # What the string found is recorded as it runs, and its trace is
            # checked before a hit is served (``cache_key.lookups_followed``).
            run.needs_trace = True
        elif by_string:
            # Its key cannot name the function a string spells at run time
            # (``string_lookups``): served, it could be stale.
            who = "The statement" if THE_STATEMENT in by_string else ", ".join(repr(n) for n in sorted(by_string))
            run.metrics["uncacheable_reasons"].append(
                f"{who} looks names up by a string (globals(), eval, vars(), getattr on a module): "
                f"which function that reaches, and what it reads, is known only when it runs, "
                f"so the statement runs every time"
            )
            run.skip_cache = True
            return
        if unkeyed_modules:
            # Its key names the code of the modules whose objects it calls
            # into (``held_modules``), and the file of these cannot be read:
            # served, it could be stale.
            names = ", ".join(repr(n) for n in sorted(unkeyed_modules))
            run.metrics["uncacheable_reasons"].append(
                f"Reads objects of classes from your module {names}, whose file cannot be read: "
                f"what their methods run cannot be keyed, so the statement runs every time"
            )
            run.skip_cache = True
            return
        drawn = drawn_stream_inputs(run.tree, run.inputs, run.outputs, self.shell.user_ns)
        if drawn:
            run.metrics["uncacheable_reasons"].append(
                f"Reads from {', '.join(repr(n) for n in drawn)}, an iterator or stream: "
                f"a cache hit would not advance it, so the statement runs every time"
            )
            run.skip_cache = True
            run.drew_from_stream = any(isinstance(self.shell.user_ns.get(name), io.IOBase) for name in drawn)
            return
        looked_at = reads_open_handle(run.inputs - run.outputs, self.shell.user_ns)
        if looked_at:
            # The handle's lineage names where it stood when it was opened,
            # not where it stands now: ``pos = fh.tell()`` after a read must
            # not be served the position an earlier run saw before it.
            run.metrics["uncacheable_reasons"].append(
                f"Reads {', '.join(repr(n) for n in looked_at)}, an open file: "
                f"where it stands changes as it is read, so the statement runs every time"
            )
            run.skip_cache = True
            run.drew_from_stream = True

    def _lookup(
        self, run: StatementRun, analysis_time: float, hash_time: float
    ) -> tuple[StatementCacheMetadata | None, Any | None]:
        """Look *run* up in the cache: ``(metadata, cached_data)``, both None
        on a miss or a skipped lookup."""
        run.effective_ttl = ttl_floor_from_called_functions(run.inputs, run.effective_ttl, self.shell.user_ns)
        metadata, cached_data, cache_check_time = self._do_cache_lookup(
            run.skip_cache, run.cache_key, run.effective_ttl, run.inputs
        )
        if cached_data is None and not run.skip_cache:
            run.stale_file = self._freshness.last_stale_file
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
        # Read before the restore moves the generators.
        advanced = None if is_control_body(run.code) else carriers_an_entry_advanced(cached_data, self.shell.user_ns)
        hit_result = self._hits.serve(run, cached_data, metadata, self._randomness.seed_epochs)
        if hit_result is None:
            return None
        if metadata is not None:
            sparse_order.put_in_order(metadata.sparse_ordered, self.shell.user_ns)
        # What changed through an alias moves on as a run moves it, past the
        # variables the entry restored with its outputs (moved on already).
        moved = set((metadata.holders or {}) if metadata is not None and metadata.holders_moved else ())
        clear_rebound_edges(self.tracking_state, run.outputs, run.inputs, self.shell.user_ns, run.tree)
        # The restore put back the objects its run shared with other
        # variables (the entry stores them together): link them as its run
        # did, as the simulation does for a hit.
        if not is_control_body(run.code):
            recorded = self._records.recorded_share_edges(run.source_hash)
            if recorded:
                apply_share_record(self.tracking_state.derivation_edges, (), recorded)
        self.lineage_builder.replay_derivation_bumps(
            self.tracking_state, run.outputs, run.inputs, skip=moved, code=run.code
        )
        # The restore put the generators where the run left them; their
        # lineages follow, as the run's did.
        self._advance_carriers(run.source_hash, advanced, run.cache_key, run.code)
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

    def _live_objects_shared(self, run: StatementRun, metadata: StatementCacheMetadata | None) -> bool:
        """Whether a hit of *run* must not restore over the objects its names
        are bound to now (`live_shared_reason`), with the miss reason set.

        Asked of the objects whose identity the run keeps: the outputs it
        changes in place (a fitted estimator among them) and the variables the
        entry stores with them. Nor is a name the upstream check's forward probe
        holds the place of (``FORWARD_PROBE_PLACEHOLDER``): no object of the
        notebook's, it is there for this very hit to replace.
        """
        held = set((metadata.holders or {}) if metadata is not None else ())
        mutated = set(run.analysis.all_mutated_vars) & run.outputs if run.analysis is not None else set()
        user_ns = self.shell.user_ns
        names = {name for name in run.outputs | held if user_ns.get(name) is not FORWARD_PROBE_PLACEHOLDER}
        keep = (held | mutated) & names
        if not keep:
            return False
        reason = live_shared_reason(
            names,
            keep,
            user_ns,
            cash_held=self._calls.held_call_results(),
            shell=self.shell,
        )
        if reason is None:
            return False
        self._freshness.last_miss_reason = reason
        return True

    def _route_calls(self, run: StatementRun) -> None:
        """Settle what executes: *run*'s code with eligible calls routed
        through the call cache."""
        code = run.code
        run.exec_code, run.exec_tree = self._calls.code_and_tree_for_execution(
            code, run.tree, run.annotation, key=run.cache_key
        )
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
        handed = run.loop_iterable is not NO_LOOP_ITERABLE
        if handed:
            source, tree = over_the_handed_iterable(source)
        memo = None
        if run.tree is not None and not handed:
            # Matched on the statement as written; it then runs as written,
            # with the kept value in place, not through the call cache.
            memo = self._memo.substitute(
                run.tree, self.tracking_state.variable_lineage, self.shell.user_ns, self._memo_key
            )
            if memo is not None:
                tree = memo[0]
                source = ast.unparse(tree)
                run.memo_used = True
        run.handed_iterable = handed
        runner = CodeRunner(code, source, tree, run.is_last, self.shell.user_ns)
        execution = runner.execution
        marks = self._calls.cash_time_marks()
        # Only the statement's own run is timed. Setting up the observation
        # around it is cash's cost, not the statement's, and the first setup
        # in a process installs the reader patches: counted, a trivial first
        # statement would look worth storing.
        wall_time = 0.0
        if run.annotation is not None and run.annotation.no_cache:
            self._randomness.resume_live_stream(run.code)
        self._randomness.before_compute(run.code)
        # Snapshot the global RNG streams around execution so a before/after
        # diff catches a draw that static analysis and object-introspection
        # both miss -- one hidden inside a called function.
        pre_rng = self._randomness.begin_statement()
        # Where each generator among the inputs stands, to see which ones the
        # statement draws from (`carrier_advances`). Not for a loop or branch
        # body: the structure is one statement to the simulation, as for
        # in-place mutations (`MutationRouting._classify`).
        positions = (
            {}
            if is_control_body(run.code)
            else carrier_positions(
                carrier_candidates(run.inputs, self.shell.user_ns, self.tracking_state.variable_lineage),
                self.shell.user_ns,
            )
        )
        # What the globals of the local modules it reaches hold, to see one
        # it sets state on that its text does not say (`note_module_state`).
        try:
            held = module_globals(run.code, self.shell.user_ns)
        except Exception:
            logger.debug("%s could not watch the modules %r reaches", _LOG_PROCESSOR, run.code[:80], exc_info=True)
            held = {}
        # pyplot's figures, to see a statement change them through a name
        # cash does not know (`pyplot_state`). Not read for one that already
        # runs every time.
        figures_before = None if run.skip_cache else pyplot_state()
        lineage = self.tracking_state.variable_lineage
        run.entry_lineages = {name: lineage[name] for name in run.inputs if name in lineage}
        unordered = sparse_order.unordered_sparse(run.inputs, self.shell.user_ns)
        try:
            with (
                self.watching_reads(run.code),
                make_capture_ctx(
                    run.stream_output, run.skip_cache and run.stream_output, run.keep_displays
                ) as captured,
            ):
                execution.captured = captured
                # Hashed as read, as the decorator does: a file replaced while
                # the statement runs is then caught at store time
                # (`StatementStore.save`), and the snapshot reuses the digest.
                with (
                    observe_writes() as written_paths,
                    FileAccessTracker(self.shell.user_ns, hash_on_read=not run.skip_cache) as file_tracker,
                ):
                    user_ns = self.shell.user_ns
                    shadowed = user_ns.get(LOOP_ITERABLE_NAME, NO_LOOP_ITERABLE) if handed else None
                    if handed:
                        user_ns[LOOP_ITERABLE_NAME] = run.loop_iterable
                    if memo is not None:
                        user_ns.update(memo[1])
                    stop_sharing_answers()
                    start_time = _perf_counter()
                    try:
                        yield runner
                    finally:
                        wall_time = max(0.0, _perf_counter() - start_time - run.record_time)
                        if memo is not None:
                            # Bound only while it ran; what the value cost
                            # to compute is what the statement is worth.
                            for name in memo[1]:
                                user_ns.pop(name, None)
                            wall_time += memo[2]
                        new_analysis_round()
                        if handed:
                            # Gone before the outputs are captured, and the
                            # value no longer held once the loop is done.
                            run.loop_iterable = NO_LOOP_ITERABLE
                            if shadowed is NO_LOOP_ITERABLE:
                                user_ns.pop(LOOP_ITERABLE_NAME, None)
                            else:
                                user_ns[LOOP_ITERABLE_NAME] = shadowed
                execution.accessed_files = file_tracker.get_accessed_files() | run.header_files
                execution.written_paths = frozenset(written_paths)
                execution.appended_paths = frozenset(written_paths.appended)
                execution.waived_paths = written_paths.waived
                execution.file_tracker = file_tracker
                execution.accessed_remote = file_tracker.get_accessed_remote_urls() | run.header_remote
                execution.probes = probe_answers(file_tracker)
                # The keyed form (`run.code`), not what ran: a call routed
                # through the call cache rewrites the text, and the key and
                # the simulation look the observation up by the statement's
                # own text.
                self._randomness.observe_statement(pre_rng, run.code)
                execution.rng_moved = self._randomness.moved_streams
                execution.result = ExecutionResult(success=True)
        except Exception as e:  # noqa: BLE001 - broad fallback wrapping arbitrary user code
            execution.result = error_result(e)
        except BaseException as e:
            # An interrupt leaves the statement here, past the step that
            # shows what it printed, which plain has shown by the time the
            # KeyboardInterrupt's traceback comes.
            unshown_output(e, execution.captured, run.silent, run.stream_output)
            raise
        if figures_before is not None and execution.result.success and pyplot_state() != figures_before:
            run.skip_cache = True
            run.metrics["uncacheable_reasons"].append(CHANGED_PYPLOT_REASON)
        if not run.skip_cache and execution.written_paths and execution.result.success:
            wrote = self._kept_writes_reason(run, execution)
            if wrote:
                run.skip_cache = True
                run.metrics["uncacheable_reasons"].append(wrote)
        if not run.skip_cache and execution.result.success and not getattr(run.annotation, "assume_safe", False):
            workers = self._workers_reason(run)
            if workers:
                run.skip_cache = True
                run.metrics["uncacheable_reasons"].append(workers)
        if positions:
            run.carriers_advanced = moved_carrier_names(positions, self.shell.user_ns)
        if unordered:
            run.sparse_ordered = sparse_order.were_put_in_order(unordered, self.shell.user_ns) or None
        if held:
            run.rebound_modules = rebound_modules(held)
            del held
        elif execution.result is not None and execution.result.success:
            # An import whose module's top level sets state on another local
            # module (``import plugin`` doing ``@mylib.register``): seen once
            # it ran and the module is loaded.
            try:
                run.rebound_modules = run.rebound_modules | import_state_writes(run.code, self.shell.user_ns)
            except Exception:
                logger.debug("%s could not read what %r's imports set", _LOG_PROCESSOR, run.code[:80], exc_info=True)
        with self._after_it_ran(run):
            self._forget_file_answers_if_it_wrote(code, execution)
        execution.wall_time = wall_time
        execution.cost, execution.store_cost, execution.tax = self._calls.price(execution.wall_time, marks)
        execution.cached_call_reads = self._calls.files_read_in_cached_calls(marks)

    def _workers_reason(self, run: StatementRun) -> str | None:
        """Why *run* may not be stored when it starts worker processes or
        threads, itself or in a function of the user's it calls, or None.

        The write observer sees the statement's own process and context: a
        file a worker wrote (``ProcessPoolExecutor``, ``multiprocessing.Pool``,
        ``joblib.Parallel``, ``ThreadPool``, ``threading.Thread``) was not
        seen, the statement was stored, and the next Run All skipped every
        write. Whether the work writes cannot be told, so the statement runs
        every time (``# @cash:assume-safe`` caches it anyway)."""
        try:
            started = statement_starting_workers(run.code, self.shell.user_ns, run.tree)
        except Exception:
            logger.debug("%s could not read whether %r starts workers", _LOG_PROCESSOR, run.code[:80], exc_info=True)
            return None
        if not started:
            return None
        return (
            f"Starts worker processes or threads ({started}): a file they write is not seen, "
            "and a cache hit would skip it, so the statement runs every time "
            "(# @cash:assume-safe caches it anyway)"
        )

    def _kept_writes_reason(self, run: StatementRun, execution: StatementExecution) -> str | None:
        """Why *run* may not be stored when it was seen writing a file of the
        user's that is still there, or None.

        Cash's static writer check reads the calls it knows
        (``df.to_csv``, ``open(p, 'w')``); ``np.savetxt``, ``joblib.dump``,
        ``gzip.open(p, 'wt')`` and ``os.open`` in a helper are none of
        those. Stored, the next Run All restored the value without the
        write: the folder cleared at the top stayed empty, and a staging file
        written twice was read with its later content. A file written and
        removed again within the statement is no effect a hit skips.
        ``# @cash:assume-safe`` on the statement, or a waiver on the write's
        way in a function it calls (`WrittenPaths.waived`), keeps it
        cacheable; a waiver on another line does not."""
        backend = self.cash_instance.backend if self.cash_instance else None
        written = kept_writes(
            execution.written_paths,
            getattr(backend, "local_dir", None),
            execution.appended_paths,
            execution.waived_paths,
        )
        if not written:
            return None
        if getattr(run.annotation, "assume_safe", False):
            return None
        more = f" and {len(written) - 1} more" if len(written) > 1 else ""
        return (
            f"Wrote {written[0]}{more} as it ran: a cache hit would skip the write, "
            "so the statement runs every time (# @cash:assume-safe caches it anyway)"
        )

    @contextmanager
    def watching_reads(self, code: str) -> Generator[None, None, None]:
        """Around running *code*: what it changes of the environment and the
        module data statements were keyed on, itself or in a function it
        calls, is the notebook's own (`recorded_reads`)."""
        reads = self.tracking_state.reads
        try:
            before = snapshot(reads.watched, code, self.shell.user_ns)
        except Exception:
            logger.debug("%s could not watch the reads %r may change", _LOG_PROCESSOR, code[:80], exc_info=True)
            before = {}
        try:
            yield
        finally:
            if before:
                try:
                    note_writes(before, code, reads)
                except Exception:
                    logger.debug("%s could not note what %r changed", _LOG_PROCESSOR, code[:80], exc_info=True)

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
        if execution.echo:
            metrics[ECHO_FIELD] = execution.echo[0]
        if decorator_calls:
            metrics["decorator_calls"] = decorator_calls

        display_execution_output(captured, run.silent, run.stream_output, metrics)
        metrics["execution_time"] = execution.wall_time
        metrics["compute_cost"] = execution.cost
        metrics["cash_tax"] = execution.tax

        result = execution.result
        if not result.success:
            # A draw before the error still moved the generator.
            if run.carriers_advanced:
                self._advance_carriers(run.source_hash, run.carriers_advanced, run.cache_key, run.code)
            self._forget_ipython_bindings(run)
            metrics["status"] = CacheStatus.ERROR
            metrics["error"] = result.error
            metrics["total_time"] = _perf_counter() - run.process_start
            self._handle_execution_error(result, run.silent)
            return metrics

        self._randomness.flag_inline_unseeded_fit(
            metrics, run.code, run.tree, run.outputs, run.allow_random, is_hit=False, skip_cache=run.skip_cache
        )
        self._randomness.flag_observed_hidden_draw(
            metrics, run.code, run.outputs, skip_cache=run.skip_cache, allow_random=run.allow_random
        )
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
        elif not run.skip_cache and not self._evicted.attribute(
            metrics, self.cash_instance.backend, run.cache_key, run.code, execution.cost
        ):
            reason = input_change_reason(self.tracking_state, run.inputs, run.outputs)
            if reason is not None:
                metrics["miss_reason"] = reason

        self._post_execute(run, execution)
        if execution.echo and run.cache_key and not run.skip_cache and not run.outputs:
            if (
                not execution.accessed_files
                and not execution.accessed_remote
                and not self._randomness.observed_draw
                # What it printed or warned would be lost where it is reused.
                and not execution.printed
            ):
                self._memo.keep(
                    run.tree,
                    execution.echo[0],
                    execution.wall_time,
                    self.tracking_state.variable_lineage,
                    self.shell.user_ns,
                    self._memo_key,
                    computed_under=run.cache_key,
                    reads_a_stream=lambda: self._reads_a_stream(run),
                )
        self._forget_ipython_bindings(run)
        if run.magic_reads is not None:
            self.note_magic_changes(run.code, run.magic_snapshots)
            self.record_magic(run.code, run.tree.body[0], run.magic_reads)
        return metrics

    def _forget_ipython_bindings(self, run: StatementRun) -> None:
        """Forget how every name a magic or shell command bound was computed
        (:meth:`forget_rebound`), its assignment targets among them."""
        if run.ipython_bindings is None:
            return
        self.forget_rebound(run.ipython_bindings)
        for name in run.outputs:
            self.forget_variable(name)

    def _is_module(self, name: str) -> bool:
        return isinstance(self.shell.user_ns.get(name), types.ModuleType)

    def magic_reads(self, node: ast.stmt) -> dict[str, str | None]:
        """The lineage of each name the magic statement *node* reads, now
        (``magic_effects``): taken before it runs, for :meth:`record_magic`."""
        _, read = magic_effects(node, self._is_module)
        lineage = self.tracking_state.variable_lineage
        return {name: lineage.get(name) for name in read}

    def magic_cell_snapshots(self, raw_cell: str) -> dict[str, dict[str, str | None]]:
        """:meth:`MutationClassifier.magic_snapshots` for every statement of
        *raw_cell*, a cell of magics only that IPython runs on its own, by
        statement, before it runs; for :meth:`record_magic_cell`."""
        cell = simulation_cell(raw_cell)
        if cell is None:
            return {}
        source, tree = cell
        snapshots: dict[str, dict[str, str | None]] = {}
        for node in tree.body:
            if is_magic_statement(node):
                code = statement_code(node, source)
                taken = self._mutations.magic_snapshots(node, code)
                if taken is not None:
                    snapshots[code] = taken
        return snapshots

    def note_magic_changes(self, code: str, snapshots: dict[str, str | None] | None) -> None:
        """Record what the magic statement *code* changed through a call's
        argument (``MutationClassifier.note_magic_changes``), across restarts too."""
        changed = self._mutations.note_magic_changes(code, snapshots)
        if changed is not None:
            self._records.persist_mutation_verdict(statement_source_hash(code), changed)

    def record_magic_cell(
        self,
        raw_cell: str,
        lineage_before: Mapping[str, str],
        snapshots: Mapping[str, dict[str, str | None]] | None = None,
    ) -> None:
        """:meth:`record_magic` for every statement of *raw_cell*, a cell of
        magics only that IPython ran on its own, in order, from the lineages
        before it ran (*lineage_before*) and the fingerprints taken before it
        ran (*snapshots*, :meth:`magic_cell_snapshots`)."""
        cell = simulation_cell(raw_cell)
        if cell is None:
            return
        source, tree = cell
        if not all(is_magic_statement(node) for node in tree.body):
            return
        running: dict[str, str | None] = dict(lineage_before)
        for node in tree.body:
            _, read = magic_effects(node, self._is_module)
            code = statement_code(node, source)
            self.note_magic_changes(code, (snapshots or {}).get(code))
            running.update(self.record_magic(code, node, {n: running.get(n) for n in read}))

    def record_magic(self, code: str, node: ast.stmt, reads: Mapping[str, str | None]) -> dict[str, str]:
        """Give each name the magic statement *node* (*code*) bound or changed
        the lineage of a magic's output (``magic_output_lineage``), from the
        lineages it read (*reads*, :meth:`magic_reads`) and its value now;
        returns them.

        After :meth:`forget_rebound`: the names the magic is not known to bind
        (``%run``) stay without one.
        """
        ns = self.shell.user_ns
        changed, read = magic_effects(node, self._is_module)
        # And what it was seen changing through a call's argument
        # (`%time train(model)`), as the simulation reads it back.
        changed |= {n for n in self._mutations.magic_changed_arguments(code) if n in read and not self._is_module(n)}
        rerun = is_rerun_magic(node)
        base = magic_base(code, reads)
        digests: dict[str, str] = {}
        for name in sorted(changed):
            if name not in ns:
                continue
            digests[name] = no_cache_value_digest(ns[name])
            lineage = magic_output_lineage(base, digests[name])
            if not rerun:
                self.tracking_state.magic_lineages.add(lineage)
            self.tracking_state.lineage.record(name, lineage, value=ns[name])
        if digests:
            self.tracking_state.magic_values[base] = digests
            self.tracking_state.magic_generation += 1
        for var, lineage in magic_rng_advances(node, code, self.tracking_state.variable_lineage).items():
            self.tracking_state.lineage.record(var, lineage)
        return {name: magic_output_lineage(base, digest) for name, digest in digests.items()}

    def bindings(self) -> dict[str, int]:
        """The identity of every binding in the namespace, for :meth:`forget_rebound`."""
        return {name: id(value) for name, value in self.shell.user_ns.items()}

    def forget_rebound(self, before: dict[str, int], lineage_before: Mapping[str, str] | None = None) -> None:
        """Forget how every name bound, rebound or deleted since *before*
        (:meth:`bindings`) was computed, unless cash recorded its lineage
        since (it differs from *lineage_before*, when given).

        IPython binds them (``files = !ls``, ``%time x = f()``, ``%run``),
        not code cash can read, so no lineage describes their values: the one
        they had would key a downstream statement to the value before. Without
        one, a statement reading them runs uncached.
        """
        ns = self.shell.user_ns
        lineage = self.tracking_state.variable_lineage
        for name in before.keys() | ns.keys():
            if name in ns and before.get(name) == id(ns[name]):
                continue
            if lineage_before is not None and lineage.get(name) != lineage_before.get(name):
                continue
            self.forget_variable(name)

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
            return self._freshness.check_cache(
                self.tracking_state, cache_key, ttl, inputs, epoch=file_state_epoch(), bound=self.shell.user_ns
            )
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

    def _binds_a_stream(self, run: StatementRun) -> bool:
        """Whether the statement bound an open file, or read from one: a
        handle is a new stream at its start on every run, and a read leaves it
        elsewhere, which a lineage made of the code and the handle's cannot
        tell, so a reader below (``lines = fh.readlines()``, ``n = text.count(x)``
        after ``text = fh.read()``) would be served what it computed from an
        earlier handle or an earlier read.

        A closed handle is no stream: ``with open(path) as f:`` leaves one
        bound, which nothing can read from, and the statement is cached as
        any other -- the file it read is its dependency. Taken as a stream,
        it made every value the block read uncacheable and hashed in full
        on every run (300,000 records: 2.2 s of CPU a run).
        """
        if run.drew_from_stream:
            return True
        user_ns = self.shell.user_ns
        return any(_open_stream(user_ns.get(name)) for name in run.outputs)

    def _post_execute(self, run: StatementRun, execution: StatementExecution) -> None:
        """Auto-track imports, capture vars, detect mutations, save to cache, record analytics.

        Updates ``run.outputs`` and ``run.skip_cache`` with what execution
        revealed (an observed mutation, an uncacheable value).
        """
        self._observe_process_state(run)
        self._mutation_routing.observe(run)
        self._mutation_routing.note_module_state(run)
        if run.carriers_advanced is not None:
            # A generator the statement rebinds or was seen changing in place
            # is an output, with an output's lineage.
            run.carriers_advanced -= run.outputs

        # Auto-track newly imported local modules so _capture_variables includes
        # the module source hash in the lineage on first execution.
        try:
            self.function_tracker.auto_track_local_imports(run.code, self.shell.user_ns)
        except (ImportError, AttributeError, OSError):
            logger.debug("%s Failed to auto-track local imports", _LOG_PROCESSOR)
        self._records.persist_import_bindings(run.code, run.tree)

        self._key_a_newly_seen_draw(run)
        self._route_pyplot_draws(run)
        # A file it read was replaced while it ran: its outputs are not what
        # that file holds now, and their lineage, which folds the file's
        # present stat, would say they are. Run uncached, and keyed by value
        # as a no-cache statement's are, so nothing below is stored under a
        # lineage a settled run gives again.
        inputs_moved = not run.skip_cache and self._store.inputs_moved(run, execution)
        if inputs_moved:
            run.skip_cache = True
            run.metrics["uncacheable_reasons"].append(INPUT_CHANGED_REASON)
        # From here to the store, only cash's code runs: the pickle a digest
        # of an output takes is kept for the store to take
        # (`stored_pickle.keeping`), instead of pickling the value again.
        returned = self._calls.returned_digest(run.code)
        if returned is not None:
            # ``x = f(...)``: the digest the call's store took of its result
            # is the one the statement would take of ``x`` now.
            run.settled_digests = {**run.settled_digests, id(returned[0]): returned}
        del returned
        # Before the outputs are recorded as this statement's: an input it
        # also binds (``n += len(f.read())``) inherits its files as the entry
        # it came from recorded them (`StatementStore._producer_file_snapshots`).
        sources = self.tracking_state.variable_sources
        run.entry_sources = {name: sources[name] for name in run.inputs & run.outputs if name in sources}
        with self._keeping_outputs(run, execution.echo, execution.cost or 0.0):
            # What it links, from its outputs' edges dropped to those found
            # now, is kept for the simulation (`note_share_edges`).
            with logging_edges() as linked:
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
                    probes=execution.probes,
                    no_cache=(run.annotation is not None and run.annotation.no_cache)
                    or run.value_varies
                    or run.past_budget
                    or inputs_moved
                    or self._binds_a_stream(run),
                    replay_bumps=False,
                    lineage_reads=run.pyplot_drawn,
                    settled=run.settled_digests,
                )
            # Let go of the values they hold: the checks below count references.
            run.settled_digests = {}
            # The share check, the closure check and the RAM tier each look into
            # the outputs; JSON-like ones are walked once for all of them.
            with _plain_data.one_look(captured_vars):
                holders = None
                if not run.skip_cache:
                    holders = self._refuse_unrestorable_outputs(run, captured_vars, execution.echo, execution.cost)
                with logging_edges() as shared:
                    self._record_shared_object_edges(run, captured_vars, holders, execution.echo, execution.cost)
                    self._link_inputs_kept_by(run, execution.echo, execution.cost)
                if not is_control_body(run.code):
                    self._records.note_share_edges(run.source_hash, linked | shared)
                with self._after_it_ran(run):
                    self._record_file_effects(run, execution)
                self._note_per_item_rerun(run, execution)

                # Detect in-place mutations (detection-only; do not modify lineage).
                # Reuses the StatementAnalysis from process_statement to avoid a
                # second pass of AST visitors over the same tree.
                pure_mutations = run.analysis.all_mutated_vars - run.outputs
                if pure_mutations:
                    logger.debug("%s Detected in-place mutations on: %s", _LOG_MUTATION, pure_mutations)

                miss_guarded = self._miss_guarded(run, execution, captured_vars)
                self._skip_a_newly_seen_draw(run)

                saved_metadata = None
                if not run.skip_cache:
                    saved_metadata = self._store.save(
                        run,
                        execution,
                        captured_vars,
                        miss_guarded=miss_guarded,
                        seed_epochs=self._randomness.seed_epochs,
                    )
                else:
                    logger.debug("%s Skipping cache save due to @cash:no-cache", _LOG_ANNOTATION)
        moved = self._move_holders(run, saved_metadata)
        # After the holders moved: a variable the entry stores moves as a hit
        # of it moves it (`held_lineage`), not through its edges, so a hit,
        # a run and the simulation leave it alike.
        self.lineage_builder.replay_derivation_bumps(
            self.tracking_state, run.outputs, run.inputs, skip=moved, code=run.code
        )
        # After the save: the entry records each input's lineage as the
        # statement read it, before its draw moved it on.
        self._advance_carriers(run.source_hash, run.carriers_advanced, run.cache_key, run.code)
        self._report_saved(run, saved_metadata)
        self._tell_past_budget(run, execution.cost or execution.wall_time or 0.0)
        storage = (saved_metadata.storage if saved_metadata else None) or ()
        self._rebuild_cost.note(
            run.cache_key, run.inputs, run.outputs, execution.store_cost, on_disk=any(s != "RAM" for s in storage)
        )

        run.metrics["total_time"] = _perf_counter() - run.process_start
        self.analytics_manager.record_event(
            status="MISS",
            execution_time=run.metrics["total_time"],
            saved_time=0.0,
            code_hash=run.cache_key,
        )

    def _keeping_outputs(
        self, run: StatementRun, echo: tuple[Any, ...] = (), cost: float = 0.0
    ) -> contextlib.AbstractContextManager[None]:
        """`stored_pickle.keeping` the pickles of *run*'s outputs, the values
        its store may hold; nothing for a statement that is not stored.
        The other outputs and the value it *echo*es are checked for objects
        an output's pickle shares with them (`stored_pickle.apart`). A
        pickle a change check's budget cut short is left to the RAM tier
        (`memory_backend.room_past_budget`), when the statement took longer
        than that budget (*cost*), to make if it keeps the output: in at
        most `PAST_BUDGET_SHARE` of *cost*, for a value slower to pickle
        than its statement is worth is not one its store keeps."""
        if run.skip_cache or not run.outputs:
            return contextlib.nullcontext()
        user_ns = self.shell.user_ns
        # ``x = f(...)``: the entry refers to the call's, whose store has made
        # its bytes already. Kept again, a 150 MB Bunch was copied for nothing.
        referred = self._calls.referred_result(run.code)

        def stored(value: Any) -> bool:
            if id(value) == referred:
                return False
            return any(user_ns.get(name, _ABSENT) is value for name in run.outputs)

        def others(_value: Any) -> list[Any]:
            found = [user_ns[name] for name in run.outputs if name in user_ns]
            return [*found, *echo]

        past = None
        if cost > value_hash.CHANGE_CHECK_BUDGET_S:
            past = memory_backend.room_past_budget(getattr(self.cash_instance, "backend", None))
        return stored_pickle.keeping(stored, others, past, cost * PAST_BUDGET_SHARE)

    def _refuse_unrestorable_outputs(
        self, run: StatementRun, captured_vars: dict[str, Any], echo: tuple[Any, ...] = (), cost: float = 0.0
    ) -> dict[str, Any] | None:
        """Skip-cache *run* when one of its output values cannot be stored and
        restored faithfully (:func:`unrestorable_output_reason`).

        The value a statement echoes is held by its entry too: `sns.heatmap(df)`
        echoes an Axes, and copying that into the RAM tier revives a second
        figure in pyplot's registry, which becomes the current one.

        The statement's own echo, in *echo* and in its metrics row, is cash's
        reference, not another holder: `clf.fit(X, y)` echoes `clf` itself.
        A fitted estimator (``run.est_fit``) is checked like any output: the
        variables holding it, or its steps and parameters, are stored with it.
        """
        holders: dict[str, Any] | None = None if is_control_body(run.code) else {}
        # The check may not take longer than the statement is worth: a hit
        # saves `cost`, and finding the holders of a loop variable that is
        # an element of a list of 40,000 sessions took 6 s after a 0.1 s loop.
        budget = share_check_budget(cost)
        found: dict[str, Any] | None = None
        gave_up: list[GaveUp] = []
        anchors: dict[str, Any] = {}
        anchor_links: set[str] = set()
        anchor_holds: dict[str, list[str]] = {}
        try:
            with walk_budget(budget):
                cash_held = [*self._calls.held_call_results(), echo, run.metrics, *self._mutations.digests.held()]
                reason = unrestorable_output_reason(
                    run.outputs,
                    captured_vars,
                    self.shell.user_ns,
                    cash_held=cash_held,
                    shell=self.shell,
                    holders=holders,
                    gave_up=gave_up,
                    inputs=run.inputs,
                )
                if reason is not None and holders is not None:
                    # Anchors only where holders alone refuse: an anchor binds
                    # the live object only while its digest is the one stored,
                    # which a holder restored with the outputs never needs
                    # (a restored ColumnTransformer can digest apart from the
                    # one it was stored from, and the entry would miss).
                    retry_holders: dict[str, Any] = {}
                    retry_gave_up: list[GaveUp] = []
                    retry = unrestorable_output_reason(
                        run.outputs,
                        captured_vars,
                        self.shell.user_ns,
                        cash_held=cash_held,
                        shell=self.shell,
                        holders=retry_holders,
                        gave_up=retry_gave_up,
                        inputs=run.inputs,
                        anchorable=self._anchorable(run),
                        anchors=anchors,
                        anchor_links=anchor_links,
                    )
                    if retry is None and anchors:
                        reason, holders, gave_up = None, retry_holders, retry_gave_up
                    else:
                        anchors.clear()
                        anchor_links.clear()
                if reason is None and echo:
                    reason = identity_coupled_reason("the value it echoes", echo[0])
                if reason is None:
                    reason = self._lazy_frame_refusal(run, captured_vars)
                if reason is None and holders:
                    found = dict(holders)
                    reason = self._take_holders(run, captured_vars, holders)
                if reason is None and anchors:
                    user_ns = self.shell.user_ns
                    anchor_holds = {
                        name: sorted(anchors_held(user_ns[name], anchors))
                        for name in anchor_links - set(anchors)
                        if name in user_ns
                    }
        except WalkBudgetExceeded:
            run.share_unchecked = True
            reason = (
                f"finding which other variables hold objects of its outputs would take longer than "
                f"{budget:.1f}s, more than the statement is worth, so it re-runs every time"
            )
        if reason is None and anchors:
            run.anchors = {name: self.tracking_state.current_session_hashes[name] for name in anchors}
            run.anchor_links = anchor_links | set(anchors)
            run.anchor_holds = anchor_holds
        if reason is None and found is not None:
            return found
        # Gave up on a holder that is not a variable: the variables it found
        # all the same are linked (`_holders_of`), and asking again would
        # cost the check twice.
        run.share_gave_up = gave_up[0] if gave_up else None
        if reason is not None:
            run.skip_cache = True
            run.metrics.setdefault("uncacheable_reasons", []).append(reason)
            return None
        return holders

    def _lazy_frame_refusal(self, run: StatementRun, captured_vars: dict[str, Any]) -> str | None:
        """A polars LazyFrame built over its inputs' data (``pl.LazyFrame({'a':
        arr})``) is not stored: a restored copy would not see the next write
        to ``arr`` (`lazy_frame_sources`)."""
        user_ns = self.shell.user_ns
        for name in sorted(run.outputs):
            if name in captured_vars:
                sources = lazy_frame_sources([captured_vars[name]], run.inputs - {name}, user_ns)
                if sources:
                    return (
                        f"Live-alias object '{name}' (a LazyFrame over {', '.join(sorted(sources))}); "
                        "re-derived from live data, not cached."
                    )
        return None

    def _route_pyplot_draws(self, run: StatementRun) -> None:
        """Take the names bound to pyplot's current figure and axes for
        changed by *run* when it drew through pyplot (`pyplot_draw_names`):
        outputs whose lineage folds in the one they had, so it moves with the
        statement's code, as ``ax.plot(...)`` moves ``ax``'s. Recorded for the
        simulation (``TrackingState.pyplot_draw_outputs``). A loop or branch
        body is left to its structure (``update_lineage_after_execution``).
        """
        if is_control_body(run.code):
            return
        names = pyplot_draw_names(run.tree, self.shell.user_ns)
        self._records.note_pyplot_draws(run.source_hash, names)
        if not names:
            return
        # Outputs, and inputs too, so their new lineage folds in the one
        # they had and the simulation chains the draws of a figure in order.
        run.outputs = run.outputs | names
        run.inputs = run.inputs | names
        run.pyplot_drawn = names
        if not run.skip_cache:
            run.skip_cache = True
            run.metrics["uncacheable_reasons"].append(
                f"Draws through pyplot on: {', '.join(sorted(names))} (statement re-executes)"
            )

    def _record_shared_object_edges(
        self,
        run: StatementRun,
        captured_vars: dict[str, Any],
        holders: dict[str, Any] | None,
        echo: tuple[Any, ...] = (),
        cost: float = 0.0,
        fallback_floor: float = SHARE_FALLBACK_FLOOR_S,
        held: tuple[Any, ...] = (),
    ) -> None:
        """Record an edge between each output and every other variable bound
        to its object, holding it or held in it (`record_shared_object_edges`),
        so a later change through one name moves the other's lineage too.

        *holders* is what the share check of a stored statement found
        already (`_refuse_unrestorable_outputs`); for any other statement --
        one that re-executes every run, like ``raw.append(x)`` -- the same
        check (`share_group`) is asked here, within the same time budget.
        When its budget ran out, the variables are looked for from the
        outputs up (`referring_names`); when that is cut short too, some may
        be missing, and the NOTEBOOK-SHARE-UNCHECKED warning says so. A
        holder that is not a variable (a closure, a library's registry, a
        function's defaults) links no variable itself, but the variables
        found beside it are linked (`GaveUp`). The arrays a numpy view among
        the outputs sits on are linked too (`view_base_names`): a count of
        the view does not show them. A loop or branch body is
        left to its structure, which moves what it changed when it ends
        (``update_lineage_after_execution``).
        """
        if is_control_body(run.code):
            return
        outputs = [
            name for name in run.outputs if name in captured_vars and not self._kernel_object(captured_vars[name])
        ]
        if not outputs:
            return
        complete = True
        if holders is not None:
            links = [(outputs, set(holders) | run.anchor_links)]
        else:
            links, complete = self._holders_of(run, outputs, captured_vars, echo, cost, fallback_floor, held)
        bases, bases_complete = self._view_bases_of(outputs, captured_vars, cost, run.inputs)
        links += bases
        complete = complete and bases_complete
        for owners, names in links:
            record_shared_object_edges(self.tracking_state.derivation_edges, owners, names)
        if not complete:
            self._warn_share_unchecked(run, outputs, cost)

    def _kernel_object(self, value: Any) -> bool:
        """Whether *value* is the kernel's own machinery: the shell
        (``ip = get_ipython()``), or an object of IPython's or ipykernel's. It
        holds the whole session, user namespace included, so the share check
        ran out of time on it and warned NOTEBOOK-SHARE-UNCHECKED on a
        harmless line; no cell's data is changed in place through it."""
        if value is self.shell:
            return True
        module = getattr(type(value), "__module__", None) or ""
        return module.partition(".")[0] in _KERNEL_PACKAGES

    def _link_inputs_kept_by(self, run: StatementRun, echo: tuple[Any, ...] = (), cost: float = 0.0) -> None:
        """Link each variable *run* handed to code with what that code now
        keeps it in, among the names *run* changed in place.

        ``K.hold(data["meta"])`` leaves a class attribute of ``K``, or a
        global of a module of the user's, holding a part of ``data``; the
        share check of the outputs looks down from ``K`` and
        never finds it (a class or a module is no part of a value). Looked
        for from ``data`` up, the holder is found (`share_group`), and
        ``K.bump(5)`` changing it later moves ``data`` as a change through
        an alias does (`bump_derived_lineages`). Only a holder the statement
        changed is linked: one that read ``data`` and kept nothing links
        nothing.
        """
        if is_control_body(run.code):
            return
        user_ns = self.shell.user_ns
        # A class or a module: an object of the user's, or a list, is looked
        # down into by the outputs' own check already.
        changed = {
            name
            for name in run.outputs & run.inputs
            if name in user_ns and isinstance(user_ns[name], (type, types.ModuleType))
        }
        if not changed:
            return
        value_types = VALUE_TYPES + library_value_types()
        handed = [
            name
            for name in sorted(run.inputs - run.outputs)
            if name in user_ns
            and not isinstance(user_ns[name], (type, types.ModuleType, types.FunctionType))
            and not is_value(user_ns[name], value_types)
        ]
        if not handed:
            return
        history, named = output_history(user_ns, self.shell)
        hidden = getattr(self.shell, "user_ns_hidden", None) or {}
        cash_held = [*self._calls.held_call_results(), echo, run.metrics, *history]
        gave_up = GaveUp()
        try:
            with walk_budget(share_check_budget(cost)):
                share_group(
                    handed,
                    user_ns,
                    user_ns,
                    cash_held,
                    named,
                    foreign=lambda name: name in hidden or is_history_name(name),
                    gave_up=gave_up,
                )
        except WalkBudgetExceeded:
            pass  # what was found by then is linked; the rest stays unlinked
        edges = self.tracking_state.derivation_edges
        for owners, holders in gave_up.links:
            kept_by = set(holders) & changed
            if kept_by:
                record_shared_object_edges(edges, set(owners) & set(handed), kept_by)
        if gave_up.modules & run.state_modules:
            # A global of a module the statement set state on holds it: the
            # names through which it did (``Pt`` of ``Pt.hold(...)``) stand
            # for the module, as they do for its rebuild.
            record_shared_object_edges(edges, handed, run.state_names & changed)

    def record_structure_share_edges(
        self,
        code: str,
        names: Iterable[str],
        cost: float = 0.0,
        static: Mapping[str, set[str]] | None = None,
        held: tuple[Any, ...] = (),
    ) -> None:
        """Link the variables *names* a control structure changed or bound to
        every other variable they share an object with, as a statement's own
        outputs are (`_record_shared_object_edges`).

        A body statement records no edge (it runs once per pass, and the
        simulation treats the structure as one statement), so ``for name, rows
        in pairs: splits[name] = rows`` left ``splits`` unlinked from the
        lists it holds, and a later change of ``train_rows`` inside a block
        did not move it. Asked once when the structure ends, within the share
        check's budget for *cost*, the fallback search included: a structure
        that ran 10 ms is not made to take a second. Cut short,
        NOTEBOOK-SHARE-UNCHECKED says so. *held* is what cash holds of the
        structure (a loop's iterable). *static* links each loop target to
        the variables whose items it takes, without a check.
        """
        user_ns = self.shell.user_ns
        value_types = VALUE_TYPES + library_value_types()
        edges = self.tracking_state.derivation_edges
        for target, roots in (static or {}).items():
            if target in user_ns and not is_value(user_ns[target], value_types):
                record_shared_object_edges(edges, [target], [root for root in roots if root in user_ns])
        captured = {name: user_ns[name] for name in names if name in user_ns}
        if not captured:
            return
        run = StatementRun(code=code, source_hash=statement_source_hash(code))
        run.outputs = set(captured)
        # Its checks look into the values a level at a time, once for all of
        # them, as a statement's do: a loop's list of 1.5 million parsed rows
        # was read four times over, 2.7 s.
        with _plain_data.one_look(captured):
            self._record_shared_object_edges(run, captured, None, (), cost, fallback_floor=0.0, held=held)
        del captured

    def _holders_of(
        self,
        run: StatementRun,
        outputs: list[str],
        captured_vars: dict[str, Any],
        echo: tuple[Any, ...],
        cost: float,
        fallback_floor: float = SHARE_FALLBACK_FLOOR_S,
        held: tuple[Any, ...] = (),
    ) -> tuple[list[tuple[Iterable[str], set[str]]], bool]:
        """``(links, complete)``: the variables holding an object of *outputs*
        too, or held in one, as ``(owners, holders)`` pairs (`GaveUp.links`
        when the check gave up on a holder that is not a variable). Keeps no
        output value in a local: the check counts references."""
        value_types = VALUE_TYPES + library_value_types()
        # A figure or axes is held by pyplot's registry, which the check
        # gives up on; how a figure's axes reach it is a derivation edge of
        # its own (`detect_derivation_edges`).
        names = [
            name
            for name in outputs
            if not is_value(captured_vars.get(name), value_types) and not holds_artists(captured_vars.get(name))
        ]
        if not names:
            return [], True
        user_ns = self.shell.user_ns
        history, named = output_history(user_ns, self.shell)
        hidden = getattr(self.shell, "user_ns_hidden", None) or {}
        # *held* are containers cash holds itself (a loop's iterable), each
        # one counted as such whatever else refers to it.
        cash_held = [
            *self._calls.held_call_results(),
            echo,
            run.metrics,
            *history,
            *held,
            *self._mutations.digests.held(),
        ]

        def foreign(name: str) -> bool:
            return name in hidden or is_history_name(name)

        budget = share_check_budget(cost)
        if run.share_gave_up is not None:
            return list(run.share_gave_up.links), run.share_gave_up.complete
        timed_out = run.share_unchecked
        if not timed_out:
            try:
                gave_up = GaveUp()
                with walk_budget(budget):
                    _found, shared = share_group(
                        names, captured_vars, user_ns, cash_held, named, foreign=foreign, gave_up=gave_up
                    )
                if not shared:
                    # Each name with the holders found for it, not every
                    # output with every holder: a block checks many names.
                    return list(gave_up.links), True
                # A holder that is not a variable (a library's cache, a
                # closure, a function's defaults): the variables found beside
                # it, not a search from the outputs up, which would climb
                # through the library until its budget ran out.
                return list(gave_up.links), gave_up.complete
            except WalkBudgetExceeded:
                timed_out = True
        found_names, complete = referring_names(
            [captured_vars[name] for name in names],
            user_ns,
            max(budget, self._fallback_seconds(run, fallback_floor)),
            skip_name=lambda name: foreign(name) or name in names,
            ignore=[captured_vars, run.metrics, echo, *cash_held],
        )
        return [(outputs, found_names)], complete

    def _fallback_seconds(self, run: StatementRun, fallback_floor: float) -> float:
        """How long the search from the outputs up (`referring_names`) may
        take: *fallback_floor*, or less when *run*'s analysis budget has
        less left (`analysis_budget`). The share check is the last of the
        analyses of a statement and draws on the same budget: a search that
        gives up says so (NOTEBOOK-SHARE-UNCHECKED), as one past its floor
        does."""
        left = self._budget_left(run)
        if left is None:
            return fallback_floor
        return min(fallback_floor, left.seconds)

    def _view_bases_of(
        self, outputs: list[str], captured_vars: dict[str, Any], cost: float, inputs: Collection[str] = frozenset()
    ) -> tuple[list[tuple[Iterable[str], set[str]]], bool]:
        """``(links, complete)``: each output with the variables bound to an
        array a numpy view in it sits on (`view_base_names`), or whose data
        shares memory with it across libraries (`borrowed_memory_names`:
        ``s = pl.Series(arr)``; for a polars LazyFrame, the *inputs* it may
        keep its data in, `lazy_frame_sources`), within the share check's
        budget."""
        if "numpy" not in sys.modules:
            return [], True
        user_ns = self.shell.user_ns
        hidden = getattr(self.shell, "user_ns_hidden", None) or {}
        links: list[tuple[Iterable[str], set[str]]] = []

        def skip_name(other: str) -> bool:
            return other in hidden or is_history_name(other)

        done = 0
        try:
            with walk_budget(share_check_budget(cost)):
                for name in outputs:
                    found = view_base_names([captured_vars[name]], user_ns, skip_name=skip_name)
                    found |= borrowed_memory_names(
                        [captured_vars[name]], user_ns, skip_name=skip_name, containers=inputs
                    )
                    found |= lazy_frame_sources([captured_vars[name]], inputs, user_ns)
                    found.discard(name)
                    if found:
                        links.append(([name], found))
                    done += 1
        except WalkBudgetExceeded:
            # Out of time: an output not looked at may sit on the memory of
            # any value the statement read (``s = pl.Series(arr)`` on a busy
            # machine), and unlinked, a write to it left what was cached from
            # the output answering with the old data. Linked to each of them,
            # such a write moves it on: a needless miss at worst.
            readable = {
                other
                for other in inputs
                if other in user_ns and not skip_name(other) and _may_hold_memory(user_ns[other])
            }
            for name in outputs[done:]:
                if readable - {name}:
                    links.append(([name], readable - {name}))
            return links, False
        return links, True

    def _warn_share_unchecked(self, run: StatementRun, outputs: list[str], cost: float) -> None:
        """Warn, once per statement, that the variables sharing an object
        with *outputs* may not all be known (``NOTEBOOK-SHARE-UNCHECKED``)."""
        if run.source_hash in self._share_unchecked_warned:
            return
        self._share_unchecked_warned.add(run.source_hash)
        listed = ", ".join(f"'{name}'" for name in sorted(outputs))
        what = (
            f"cash could not find, within {share_check_budget(cost):.1f}s, every other variable that holds "
            f"an object of {listed} or is held in one. A later change made in place through such a variable "
            f"may not re-run the cells that read {listed}, or the other way round."
        )
        fix = (
            "if a later cell changes these objects in place through another name, rebind instead "
            "(x = x + [v]) or re-run the cells that read them yourself."
        )
        log_diagnostic(logger, "NOTEBOOK-SHARE-UNCHECKED", what, fix)
        # At the cell, as cash's other notebook warnings are: with no
        # location the nearest frame outside cash is ipykernel's, and it
        # read ``ipkernel.py:460``.
        from cash.notebook.ipython.cell_count import running_cell_count  # the ipython package imports this one

        location = ("<cash>", running_cell_count(self.shell))
        warn_diagnostic(CashWarning, "NOTEBOOK-SHARE-UNCHECKED", what, fix, location=location)

    def _move_holders(self, run: StatementRun, metadata: StatementCacheMetadata | None) -> set[str]:
        """Move on the lineages of the variables *metadata*'s entry stores
        with its outputs (`held_lineage`), as a hit of it does, once the entry
        holds them; and record that for the simulation (``held_with``).

        Only for an entry with its value: the simulation moves them on where
        it sees the entry, and a metadata-only record carries none, nor is it
        kept by every backend. And only when the run changed their object in
        place (``holders_moved``): ``report = {'totals': totals}`` leaves
        ``totals`` as it was, and moving it would change the key of the very
        statement that reads it, on every run. A run that moves none leaves
        them where they are, as it always did, and the record says so: the
        entry another run left may still name some.
        """
        cache_key = run.cache_key
        holders = None
        if metadata is not None and metadata.storage and not metadata.metadata_only and metadata.holders_moved:
            holders = metadata.holders
        if holders or run.entry_holders or cache_key in self.tracking_state.held_with:
            self.tracking_state.held_with[cache_key] = dict(holders or {})
        if not holders:
            return set()
        for name in run.holders_unchanged & set(holders):
            keep_held_lineage(holders[name], cache_key)
        for name, before in holders.items():
            apply_held_var(
                self.tracking_state,
                name,
                self.shell.user_ns.get(name),
                held_lineage(before, cache_key),
                compute_hash=self.compute_hash,
            )
        return set(holders)

    def _anchorable(self, run: StatementRun) -> Callable[[str], bool]:
        """Whether a variable holding an object of *run*'s outputs can be
        their anchor (`share_group`): not an output, and with the content it
        had when its lineage was last recorded, by a content digest a restore
        can check again (`anchors`)."""
        session = self.tracking_state.current_session_hashes
        lineage = self.tracking_state.variable_lineage
        user_ns = self.shell.user_ns

        def anchorable(name: str) -> bool:
            if name in run.outputs or name not in user_ns:
                return False
            before = session.get(name)
            if before is None or before == lineage.get(name):
                return False
            digests = anchor_digests({name: user_ns[name]}, self.compute_hash)
            return digests is not None and digests[name] == before

        return anchorable

    def _unchanged_holders(self, holders: dict[str, Any]) -> set[str]:
        """The *holders* whose content is what it was when their lineage was
        last recorded: the statement changed the object they share with an
        output elsewhere (``pre.fit_transform(X)`` fits ``pre``, not the
        ``cat_pipeline`` it holds). Known only by a content digest: a lineage
        standing in for a big value's content, or an identity digest, says
        nothing."""
        session = self.tracking_state.current_session_hashes
        lineage = self.tracking_state.variable_lineage
        unchanged: set[str] = set()
        for name, value in holders.items():
            before = session.get(name)
            if before is None or before == lineage.get(name) or self.compute_hash is None:
                continue
            try:
                after = value_hash.budgeted_hash(value, self.compute_hash)
            except Exception:  # noqa: BLE001 - a value that cannot be read moves on
                continue
            if after == before and not value_hash.is_identity_fallback_hash(value, after):
                unchanged.add(name)
        return unchanged

    def _take_holders(self, run: StatementRun, captured_vars: dict[str, Any], holders: dict[str, Any]) -> str | None:
        """Store *holders*, the variables holding an output's object too, with
        *run*'s outputs (`share_group`); why not, when one has no lineage to
        check a later hit against.

        Not for a loop or branch body (``holders`` is None there): the
        simulation treats the structure as one statement, as it does for
        in-place mutations (`MutationRouting._classify`), and could not move
        a holder's lineage on per body statement.
        """
        lineage = self.tracking_state.variable_lineage
        unknown = sorted(name for name in holders if name not in lineage)
        if unknown:
            names = ", ".join(f"'{n}'" for n in unknown)
            return (
                f"{names} also holds an object of this statement's outputs and has no lineage to check "
                f"a restore against, so the statement re-runs every time"
            )
        run.holders = {name: lineage[name] for name in holders}
        run.moves_holders = any(
            id(captured_vars[name]) == run.bound_before.get(name) for name in run.outputs if name in captured_vars
        )
        if run.moves_holders:
            run.holders_unchanged = self._unchanged_holders(holders)
        # By where they hold the outputs' objects, when every place can be
        # set again (`holder_patches`): stored whole, `data` brought along
        # every other frame it holds into each entry. Not after a loop: the
        # objects its target took on earlier passes are not outputs, so a
        # patch would put back the last one and lose what the loop did to the
        # rest (``for r in rows: normalize(r)``).
        patches = None
        if not _is_loop(run.tree):
            patches = holder_patches(
                holders, {name: captured_vars[name] for name in run.outputs if name in captured_vars}
            )
        captured_vars.update(patches if patches is not None else holders)
        return None

    def _writes_through_user_code(self, run: StatementRun) -> bool:
        """Whether the upstream replay takes *run* for a file writer through
        a function of the user's (``FileWriterScheduler._is_file_writer``)
        though no write was seen as it ran: a helper handed to a timer or a
        thread writes after the statement returns (``threading.Timer(1.0,
        save)``). Recorded as run all the same, or the replay took it for an
        edited writer and fired it again before every cell below. Asked only
        of a statement that is not stored, as such a writer never is."""
        if not run.skip_cache or "(" not in run.code:
            return False
        try:
            return statement_calls_user_writer(run.code, self.shell.user_ns, run.tree, appends=False) is not None
        except Exception:  # noqa: BLE001 - unjudged: as before, not recorded
            return False

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
        if written:
            note_own_writes(self.tracking_state, written)
        try:
            if (
                written
                or any(e.kind == "file_write" for e in run.analysis.side_effects)
                or self._writes_through_user_code(run)
            ):
                self.tracking_state.executed_write_stmt_codes.add(code)
                self.tracking_state.executed_write_input_lineages[code] = run.entry_lineages
                if written and kept_writes(written, None, execution.appended_paths):
                    self.tracking_state.observed_writer_codes.add(code)
                else:
                    self.tracking_state.observed_writer_codes.discard(code)
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
        try:
            self.tracking_state.file_round_trips.after_run(
                run.display_code or code,
                run.source_hash,
                read=execution.accessed_files,
                written=written - execution.appended_paths,
                stale_file=run.stale_file,
                inputs=run.inputs,
                outputs=run.outputs,
                cost=execution.cost,
            )
        except Exception:  # a hint never breaks a statement
            logger.log(
                HINT_ERROR_LEVEL,
                "%s could not check %r for a file round trip",
                _LOG_PROCESSOR,
                code[:80],
                exc_info=True,
            )

    def _note_per_item_rerun(self, run: StatementRun, execution: StatementExecution) -> None:
        """Fingerprint the inputs made of rows of a costly statement that was
        just computed, and say so when it ran over mostly unchanged rows
        (``CACHE-PER-ITEM-RERUN``). Advice only."""
        seconds = execution.wall_time - execution.tax
        if seconds < PER_ITEM_MIN_COST_S:
            return
        if is_control_body(run.code):
            # Not checked (the loop owns its body's inputs), but its time
            # pays towards the session's hashing budget.
            self.tracking_state.per_item_reruns.credit(seconds)
            return
        try:
            skip = run.outputs | run.analysis.all_mutated_vars | run.in_place
            user_ns = self.shell.user_ns
            values = {name: user_ns[name] for name in run.inputs - skip if name in user_ns}
            records = self._records
            self.tracking_state.per_item_reruns.after_run(
                run.display_code or run.code,
                run.source_hash,
                values=values,
                lineages=run.entry_lineages,
                seconds=seconds,
                load=records.recorded_row_blocks,
                save=records.persist_row_blocks,
            )
        except Exception:  # a hint never breaks a statement
            logger.log(
                HINT_ERROR_LEVEL,
                "%s could not check %r for a per-item rerun",
                _LOG_PROCESSOR,
                run.code[:80],
                exc_info=True,
            )

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
            and not self._store.write_is_cheap(run.outputs, captured_vars, execution.store_cost)
        )

    def _key_a_newly_seen_draw(self, run: StatementRun) -> None:
        """Give *run* the key the next run builds when its execution revealed
        a hidden draw from a seeded RNG module for the first time.

        Its key was built without that module's RNG variable, which the next
        run's key reads (``key_hidden_reads``), so storing the value under it
        would only miss on the next run. The key is built again now with the
        variable: it is the next run's key when everything the first key read
        is as it was (that key, built again now, comes out the same) and the
        statement spells no seed of its own (a seed moves the variable after
        the key read it). Otherwise the write is skipped once
        (:meth:`_skip_a_newly_seen_draw`).
        """
        randomness = self._randomness
        if not randomness.draw_newly_seen or hidden_lineage_writes(run.code) or randomness.helper_seeds(run.code):
            return
        code = run.code
        try:
            effects = statement_effects(
                code,
                run.tree,
                namespace=self.shell.user_ns,
                resolve_source=self.resolve_live_function_source,
                control_body=is_control_body(code),
            )
            inputs = set(effects.inputs) | key_hidden_reads(code, self.tracking_state)
            learnt = {rng_virtual_var(module) for module in randomness.newly_seen_draws}
            outputs = set(effects.outputs)
            before = self._key(code, inputs - learnt, outputs, run.occurrence_index, record=False).cache_key
            if before != run.cache_key:
                return
            next_key = self._key(code, inputs, outputs, run.occurrence_index, record=not run.skip_cache).cache_key
        except Exception:
            logger.debug("%s Could not key %s with its hidden draw", _LOG_PROCESSOR, code[:80], exc_info=True)
            return
        # The streams it drew from move on from the next run's key, as the
        # simulation moves them (`_advance_rng`), stored or not.
        run.rng_advance_key = next_key
        if run.skip_cache:
            return
        run.cache_key = next_key
        run.metrics["cache_key"] = run.cache_key
        randomness.draw_newly_seen = False

    def _skip_a_newly_seen_draw(self, run: StatementRun) -> None:
        """Skip-cache *run* once when its execution revealed a hidden RNG draw
        and :meth:`_key_a_newly_seen_draw` could not key it as the next run will.

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
        """Return True if *name* resolves to a @stateful callable; continue-safe for known-pure.

        A dotted *name* (``helpers.announce``) is followed through modules
        only, never through an object: reading an attribute of an object could
        run a property.
        """
        if is_known_pure(name, self.shell.user_ns):
            return False
        func_obj = resolve_dotted_name(name, self.shell.user_ns) if "." in name else self.shell.user_ns.get(name)
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
                run.metrics["total_time"] = _perf_counter() - run.process_start
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
        """Drop the cell's kept file answers, the freshness checker's
        (``CacheFreshnessChecker.forget_file_answers``) and the upstream
        check's (``forget_file_state_this_run``), when the statement that just
        ran may have changed a file: it was seen writing one, cash's static
        writer check says it writes (its own text, or a user function it
        calls -- which covers writes made in C, like pyarrow's), or it failed
        part-way. Answers taken before the write are not answers for what is
        checked after it.

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
            forget_file_state_this_run()

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
        t1 = _perf_counter()
        effects = statement_effects(
            code,
            tree,
            namespace=self.shell.user_ns,
            resolve_source=self.resolve_live_function_source,
            control_body=is_control_body(code),
        )
        inputs, outputs = set(effects.inputs), set(effects.outputs)
        self._records.log_statement_reads(code, inputs)
        analysis_time = _perf_counter() - t1

        t2 = _perf_counter()

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
            cache_key, source_hash, _, _, _ = self._key(code, key_inputs, outputs, occurrence_index)
        except Exception as exc:
            raise CacheKeyComputationError(f"Failed to compute cache key for: {code[:80]!r}") from exc

        self._randomness.record_seeds(code, cache_key)
        self._record_module_state_writes(code)

        hash_time = _perf_counter() - t2
        return effects, source_hash, cache_key, analysis_time, hash_time

    def _key(
        self, code: str, key_inputs: set[str], outputs: set[str], occurrence_index: int, *, record: bool = True
    ) -> CacheKeyResult:
        """The runtime's key for *code* with *key_inputs*, read now. *record*
        records what the environment reads in it held (``record_reads``);
        without it nothing is recorded."""
        return compute_cache_key(
            code,
            key_inputs,
            ctx=CacheKeyContext(
                variable_lineage=self.tracking_state.variable_lineage,
                user_ns=self.shell.user_ns,
                function_tracker=self.function_tracker,
                compute_hash_fn=self.compute_hash,
                reads=self.tracking_state.reads if record else None,
                record_reads=record,
                traced=True,
            ),
            outputs=outputs,
            occurrence_index=occurrence_index,
        )

    def _record_module_state_writes(self, code: str) -> None:
        """Note *code* as a statement that sets state on a local module, so a
        reload of the module knows its state is to be rebuilt
        (``ModuleInvalidator``). Run again, it moves to the end: the order is
        the one the kernel last ran them in."""
        try:
            modules = module_state_writes(code, self.shell.user_ns)
        except Exception:
            logger.debug("%s could not tell which modules %r sets state on", _LOG_PROCESSOR, code[:80], exc_info=True)
            return
        for module in modules:
            writers = self.tracking_state.module_state_writers.setdefault(module, [])
            if code in writers:
                writers.remove(code)
            writers.append(code)

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


def reads_open_handle(names: Iterable[str], user_ns: Mapping[str, Any]) -> list[str]:
    """The *names* bound to a file open for reading, or to something that
    reaches one (`reaches_read_handle`)."""
    return sorted(name for name in names if reaches_read_handle(user_ns.get(name)))


def _open_stream(value: Any) -> bool:
    """Is *value* a file or stream that can still be read or written?"""
    if not isinstance(value, io.IOBase):
        return False
    try:
        return not value.closed
    except Exception:  # noqa: BLE001 - a handle that cannot say is taken as open
        return True


def _is_loop(tree: ast.AST | None) -> bool:
    """Whether *tree*, a statement's parse, is a ``for`` or ``while`` loop."""
    body = getattr(tree, "body", None)
    node = body[0] if isinstance(body, list) and len(body) == 1 else tree
    return isinstance(node, (ast.For, ast.AsyncFor, ast.While))
