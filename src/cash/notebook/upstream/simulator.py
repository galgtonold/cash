from __future__ import annotations

"""Notebook simulator: pure-AST + cache-probing replay of upstream cells.

Extracted from ``UpstreamChecker`` so the simulation logic has a clear test
surface independent of the orchestrator. See CONTEXT.md entry: *NotebookSimulator*
and ``docs/architecture_decisions.md`` ADR-009.

The simulator never executes user code via the IPython kernel. It simulates
statement-by-statement using AST analysis and the cache backend, producing a
plan of statements to re-execute and a list of restored statements. The
orchestrator (``UpstreamChecker``) takes that plan and runs it via the real
``process_statement_callback``.
"""

import logging
from collections.abc import Callable
from typing import Any

from .._protocols import CashInstanceProtocol, ShellProtocol, TrackingState
from .._trace import trace_event
from ..cacheability import analyze_statement
from ..control_structures import is_control_structure  # noqa: F401  re-exported for test patching (mocked via @patch in test_issue_reproduction)
from .mismatch_classifier import MismatchClassifier
from .reexecution_planner import ReexecutionPlanner
from ..server_discovery import get_notebook_cells  # noqa: F401  re-exported for test patching
from ._types import (  # noqa: F401  re-exported (private renames)
    IncrementalStartResult as _IncrementalStartResult,
    SimulationCacheEntry as _SimulationCacheEntry,
    TraceEntry as _TraceEntry,
    apply_collected_mutations,
)
from .virtual_lineage import (  # noqa: F401  re-exported (imported by upstream/checker.py via .simulator)
    VirtualLineage,
    _BUILTIN_NAMES,
    _FORWARD_PROBE_PLACEHOLDER,
    _normalize_stmt,
)

__all__ = ["NotebookSimulator"]

logger = logging.getLogger(__name__)


class NotebookSimulator:
    """Replays upstream cells via AST simulation and cache probing.

    Constructed and owned by :class:`UpstreamChecker`. Shares mutable state
    references (lineage dicts, ``executed_*`` trackers) so writes are visible
    to both. The Phase-1 forward simulation / cache-probing methods live on
    :class:`VirtualLineage`; ``NotebookSimulator`` delegates to it and exposes
    property/method forwarders so existing call sites keep their shape.
    """

    def __init__(
        self,
        shell: ShellProtocol,
        cash_instance: CashInstanceProtocol | None,
        tracking_state: TrackingState,
        compute_hash_fn: Callable[[Any], str] | None = None,
        debug: bool = False,
    ) -> None:
        self.shell = shell
        self.cash_instance = cash_instance
        self.compute_hash_fn = compute_hash_fn
        self.debug = debug

        # Shared state refs (same dicts as UpstreamChecker / StatementProcessor).
        self.set_tracking_state(tracking_state)

        # Phase-1 simulator. Shares ``shell``/``cash_instance``/tracking-state
        # references with us so writes are visible on both sides.
        self._virtual_lineage = VirtualLineage(
            shell=shell,
            cash_instance=cash_instance,
            tracking_state=tracking_state,
            compute_hash_fn=compute_hash_fn,
            debug=debug,
        )

        # Phase-2 classifier. Shares tracking-state references and routes
        # back to ``_virtual_lineage`` for cache-probing helpers.
        self._classifier = MismatchClassifier(
            virtual_lineage=self._virtual_lineage,
            tracking_state=tracking_state,
            debug=debug,
        )

        # Phase-3 planner. Routes into VL + Classifier for helpers that
        # still live on those phases.
        self._planner = ReexecutionPlanner(
            virtual_lineage=self._virtual_lineage,
            classifier=self._classifier,
            debug=debug,
        )

    def set_tracking_state(self, state: TrackingState) -> None:
        """Re-wire shared state refs (mirrors UpstreamChecker.set_tracking_state)."""
        self._tracking_state = state
        self.executed_cell_codes = state.executed_cell_codes
        self.executed_cell_hashes = state.executed_cell_hashes
        self.variable_lineage = state.variable_lineage
        self.lineage = state.lineage
        self.executed_file_deps = state.executed_file_deps
        self.vars_with_mutation_lineage = state.vars_with_mutation_lineage
        self.executed_input_lineages = state.executed_input_lineages
        # Propagate to the Phase-1 simulator so its dict refs stay in sync.
        if hasattr(self, '_virtual_lineage'):
            self._virtual_lineage.set_tracking_state(state)
        if hasattr(self, '_classifier'):
            self._classifier.set_tracking_state(state)

    def reset_caches(self) -> None:
        """Clear simulation and AST caches."""
        self._virtual_lineage.reset_caches()

    def _apply_phase_mutations(self) -> None:
        """Drain phase RestoreCollectors and apply buffered ops to TrackingState.

        Phases buffer mutations as ``CacheRestore`` / ``LineageReset`` ops and
        usually drain themselves at method boundaries (so direct callers and
        mid-simulation reads see writes immediately). This safety-net drain
        catches anything left over after the full pipeline runs.
        """
        apply_collected_mutations(self._virtual_lineage._restores, self._tracking_state)
        apply_collected_mutations(self._classifier._restores, self._tracking_state)

    # --- Class-level method aliases (tests access on the class) ---
    # Static helpers usable directly; instance-method aliases are present so
    # ``hasattr(NotebookSimulator, '...')`` checks succeed after the
    # VirtualLineage extraction (ADR-009) and the package extraction
    # (ADR-010). Instance-method aliases are unbound and not intended to be
    # called through NotebookSimulator at runtime — call them via the
    # underlying VirtualLineage instance (`self._virtual_lineage`).
    _validate_file_freshness = VirtualLineage._validate_file_freshness
    _stat_file_deps = VirtualLineage._stat_file_deps
    _iter_body_nodes = VirtualLineage._iter_body_nodes
    _update_virtual_lineage = VirtualLineage._update_virtual_lineage
    _resolve_input_lineage = VirtualLineage._resolve_input_lineage
    _compute_module_source_hash = VirtualLineage._compute_module_source_hash

    # --- Narrow public API for UpstreamChecker (avoid private reach-ins) ---

    def set_current_cell_id(self, cell_id: str | None) -> None:
        self._virtual_lineage._current_cell_id = cell_id

    def get_cached_ast(self, code: str):
        return self._virtual_lineage._get_cached_ast(code)

    def last_index_for_cell(self, cell_id: str) -> int | None:
        return self._virtual_lineage._cell_id_to_last_index.get(cell_id)

    def record_cell_id_index(self, cell_id: str, idx: int) -> None:
        self._virtual_lineage._cell_id_to_last_index[cell_id] = idx

    def simulation_cache_entry(self, idx: int) -> _SimulationCacheEntry | None:
        cache = self._virtual_lineage._simulation_cache
        if 0 <= idx < len(cache):
            return cache[idx]
        return None

    def simulation_cache_size(self) -> int:
        return len(self._virtual_lineage._simulation_cache)

    def simulate_upstream(self, *args, **kwargs):
        return self._simulate_and_find_changes(*args, **kwargs)

    def _mark_stale_value_inputs_broken(
        self,
        required_inputs: set[str] | None,
        current_cell_reassigned: set[str] | None,
        broken_vars: set[str],
        current_cell_mutated: set[str] | None = None,
        notebook_cells: list[str] | None = None,
        current_cell_idx: int | None = None,
        current_cell_method_receivers: set[str] | None = None,
        current_cell_selfref_vars: set[str] | None = None,
    ) -> None:
        """Flag self-modifying required inputs whose live value is stale.

        ``variable_lineage[var]`` can be reset to a pre-cell base (downstream
        advancement / forward simulation) WITHOUT touching the in-memory value,
        whose own ``_cash_lineage_hash`` still reflects the later (advanced)
        version. The recorded lineage then *looks* consistent with the virtual
        state, so Pass 2 does not flag the variable — yet the cell would
        re-execute on a stale value. This is the self-referential re-run case:
        ``df = df.sort(); ...; df = df.rename()`` re-run reads the
        already-renamed frame and raises ``KeyError``. Modelling ``df`` as
        distinct versions (df_in -> df_out), the cell consumes the input
        version; when the live value's lineage disagrees with the recorded one
        the input version is not actually present, so mark it broken and let the
        normal restore / upstream-re-execution machinery re-derive its base.

        Two regimes, split on whether the live value carries a
        ``_cash_lineage_hash``:

        * **Lineage-carrying objects** (DataFrame/Series — branches (a)/(b)
          below). Restricted to variables the current cell **reassigns** with a
          fresh value (a ``Name`` store, ``df = ...``). In-place mutation of a
          such a var (``df['c'] = ...`` / ``df.attr = ...``) is deliberately
          excluded: it replays through the mutation-lineage restoration path,
          and its live value legitimately runs ahead of the reset base —
          flagging it here would force needless recompute of the whole cell (and
          wrongly defeat per-statement cache hits, e.g. an unchanged
          ``df['VolAdj']`` when only a later ``df['SMA']`` window changed).
        * **No-lineage values** (primitives / builtin containers / ndarray —
          ``_mark_nolineage_self_write_broken``). These cannot hold the attribute
          and their self-modifying statements skip the per-statement cache for
          missing input lineage, so an isolated re-run computes on the cell's own
          prior output (``lst.append`` doubles, ``total = total + k`` doubles).
          Both pure reassignment *and* in-place mutation are eligible here —
          there is no per-statement cache to defeat.

        Builtins are skipped.
        """
        if not required_inputs:
            return
        reassigned = current_cell_reassigned or set()
        # A no-lineage var the current cell mutates in place (``lst.append`` /
        # ``arr += 1`` / ``d.update``) is *self-written* even though it is not a
        # ``Name``-store reassignment.  Treat both as self-modifying inputs.
        inplace_self = (current_cell_mutated or set()) & required_inputs
        self_written = reassigned | inplace_self
        if not self_written:
            return
        # Vars that an *upstream* cell also mutates in place. A no-lineage var
        # mutated across several cells (``results.append(..)`` once per cell) has
        # no recorded per-cell base — current_session_hashes never advances past
        # the first assignment — so the content-base check below would wrongly
        # reset it to that assignment and drop the intermediate cells' mutations.
        # Computed lazily (only when an in-place no-lineage candidate exists).
        upstream_inplace_mutated: set[str] | None = None
        for var_name in required_inputs:
            # A user variable shadowing a builtin name is tracked in
            # variable_lineage; only skip genuine (untracked) builtins (CAS-63).
            if var_name in _BUILTIN_NAMES and var_name not in self.variable_lineage:
                continue
            if var_name not in self_written:
                continue
            live_value = self.shell.user_ns.get(var_name)
            live_lineage = getattr(live_value, '_cash_lineage_hash', None)
            if live_lineage is None:
                # Primitives / builtin containers / ndarray carry no
                # ``_cash_lineage_hash`` and their self-modifying statements skip
                # the per-statement cache (missing input lineage), so an isolated
                # re-run would compute on the cell's own prior output. Restore the
                # cell-entry base by marking the input broken (producer re-runs).
                if upstream_inplace_mutated is None:
                    upstream_inplace_mutated = self._scan_upstream_inplace_mutations(
                        notebook_cells, current_cell_idx,
                    )
                self._mark_nolineage_self_write_broken(
                    var_name, live_value, broken_vars, upstream_inplace_mutated,
                )
                continue
            if var_name not in reassigned:
                # A lineage-carrying var mutated in place -- not a Name
                # reassignment -- whose isolated re-run is non-idempotent and would
                # accumulate unless restored to its cell-entry base. Two cases:
                #   * METHOD receivers (``b.items.append(..)``): no-output method
                #     statements skip the per-statement cache (``results.append(x)``
                #     + ``obj.total += x``). [CAS-53]
                #   * SELF-REFERENTIAL subscript/attr writes (``df['a']=df['a']*2``,
                #     ``df['a']+=1``, ``df.iloc[i,j]+=x``). [CAS-54]
                # Restore via the same content/lineage-base machinery used for
                # no-lineage self-writes. Scoped so that a write to a NEW column read
                # from OTHER columns (``df['VolAdj']=df.groupby('Close')..``) is NOT
                # included and keeps its per-statement cache (CAS-42 preserved).
                force_reset = (
                    (current_cell_method_receivers and var_name in current_cell_method_receivers)
                    or (current_cell_selfref_vars and var_name in current_cell_selfref_vars)
                )
                if force_reset:
                    before = var_name in broken_vars
                    # The live VALUE's own lineage (``_cash_lineage_hash``) reflects
                    # the object actually in memory. When the mutation is nested in a
                    # control structure (``if c: df['a']=df['a']*2``) the runtime
                    # advances it past the cell-entry base, but the downstream-
                    # advancement fallback collapses the recorded ``variable_lineage``
                    # back to the base via ``reset_to`` (which leaves the value's
                    # attribute intact). So compare the VALUE's lineage against the
                    # cell-entry base — it survives the collapse and still betrays the
                    # stale (own-prior-mutation) value on an isolated re-run, while a
                    # fresh forward run (producer restored the base) leaves them equal.
                    # [CAS-57]
                    base_lineage = self.executed_input_lineages.get(var_name, {}).get(var_name)
                    if base_lineage is not None and live_lineage != base_lineage:
                        broken_vars.add(var_name)
                    else:
                        if upstream_inplace_mutated is None:
                            upstream_inplace_mutated = self._scan_upstream_inplace_mutations(
                                notebook_cells, current_cell_idx,
                            )
                        self._mark_nolineage_self_write_broken(
                            var_name, live_value, broken_vars, upstream_inplace_mutated,
                        )
                    trace_event("force_reset", var=var_name,
                                broke=(var_name in broken_vars and not before))
                continue
            recorded = self.variable_lineage.get(var_name)
            if recorded is None:
                continue
            # (a) ``variable_lineage[var]`` was reset to a pre-cell base (the
            # downstream-advancement reset, e.g. test_134's multi-statement
            # chain) but the live value still carries the advanced lineage: the
            # recorded/live disagreement betrays the stale value directly.
            if live_lineage != recorded:
                if self.debug:
                    logger.debug(
                        "[UPSTREAM_DEBUG] '%s' has a stale in-memory value "
                        "(recorded lineage %s but value lineage %s); marking broken "
                        "so its input version is restored before the cell re-runs.",
                        var_name, recorded[:8], live_lineage[:8],
                    )
                broken_vars.add(var_name)
                continue
            # (b) Self-modifying single statement (``df = df.iloc[1:]``): the
            # forward simulation reproduces the advanced lineage exactly, so
            # recorded == live == virtual and the reset in (a) never fires.
            # ``executed_input_lineages[var][var]`` is the version this cell
            # *consumed* the last time it ran (its cell-entry base). When the
            # live value's lineage differs from that base, the namespace holds
            # the cell's own prior OUTPUT rather than the base it must
            # re-consume on an isolated re-run -- mark it broken so the same
            # restore machinery re-derives the base. Self-disables on the first
            # run (no recorded input version yet) and on legitimate forward runs
            # (live == base). Primitives carry no ``_cash_lineage_hash`` and were
            # already skipped above; in-place mutation is excluded via
            # ``current_cell_reassigned``.
            base_input = self.executed_input_lineages.get(var_name, {}).get(var_name)
            if base_input is not None and live_lineage != base_input:
                if self.debug:
                    logger.debug(
                        "[UPSTREAM_DEBUG] '%s' holds its own prior output on re-run "
                        "(value lineage %s but cell-entry base %s); marking broken "
                        "so its base is restored before the cell re-runs.",
                        var_name, live_lineage[:8], base_input[:8],
                    )
                broken_vars.add(var_name)

    def _mark_nolineage_self_write_broken(
        self,
        var_name: str,
        live_value: Any,
        broken_vars: set[str],
        upstream_inplace_mutated: set[str] | None = None,
    ) -> None:
        """Mark a no-lineage self-modifying input broken if its live value is stale.

        Values with no ``_cash_lineage_hash`` (int/str, builtin list/dict/set,
        ndarray) cannot be tracked by the lineage branches (a)/(b). Two staleness
        signals, each of which self-disables on a fresh forward run (where the
        producer has already restored the cell-entry base) and only fires on an
        isolated re-run (where it has not):

        * **Lineage-base** — the var is a self-modifying *output*
          (``total = total + k``, ``arr += 1``). ``executed_input_lineages[var]
          [var]`` is the lineage of the version this cell *consumed* last run (the
          cell-entry base). After the cell ran, ``variable_lineage[var]`` advanced
          to the output lineage; on an isolated re-run the producer has not reset
          it, so base != current betrays the stale value. (On ``run_all`` the
          producer runs first and resets ``variable_lineage[var]`` to the base, so
          base == current and nothing is flagged.)
        * **Content-base** — pure in-place mutation (``lst.append``) produces no
          output, so ``executed_input_lineages[var]`` is absent and
          ``current_session_hashes[var]`` still holds the upstream producer's
          *content* hash (the base, never advanced by the mutation). A live
          content hash that differs means the namespace holds this cell's own
          prior mutation.

        The content-base check is suppressed when an *upstream* cell also mutates
        the var in place: such a var is accumulated across cells
        (``results.append(..)`` once per cell), its ``current_session_hashes``
        entry never advances past the first assignment, and restoring it to that
        assignment would drop the intermediate cells' contributions.
        """
        base_lineage = self.executed_input_lineages.get(var_name, {}).get(var_name)
        if base_lineage is not None:
            current_lineage = self.variable_lineage.get(var_name)
            if current_lineage is not None and current_lineage != base_lineage:
                if self.debug:
                    logger.debug(
                        "[UPSTREAM_DEBUG] no-lineage self-write '%s' holds its own prior "
                        "output on re-run (cell-entry base lineage %s but current %s); "
                        "marking broken so its base is restored before the cell re-runs.",
                        var_name, base_lineage[:8], current_lineage[:8],
                    )
                broken_vars.add(var_name)
            return

        if upstream_inplace_mutated and var_name in upstream_inplace_mutated:
            return
        if self.compute_hash_fn is None:
            return
        session_hashes = getattr(self._tracking_state, 'current_session_hashes', {})
        base_content = session_hashes.get(var_name)
        if base_content is None:
            return
        try:
            live_content = self.compute_hash_fn(live_value)
        except (TypeError, ValueError, AttributeError, RecursionError):
            return
        if live_content != base_content:
            if self.debug:
                logger.debug(
                    "[UPSTREAM_DEBUG] no-lineage in-place mutation '%s' holds its own prior "
                    "output on re-run (cell-entry base content %s but live %s); marking "
                    "broken so its base is restored before the cell re-runs.",
                    var_name, base_content[:8], live_content[:8],
                )
            broken_vars.add(var_name)

    def _scan_upstream_inplace_mutations(
        self,
        notebook_cells: list[str] | None,
        current_cell_idx: int | None,
    ) -> set[str]:
        """Return the set of names mutated in place by any cell *before* the current one.

        Used to suppress the content-base staleness check for variables
        accumulated across several cells (see ``_mark_nolineage_self_write_broken``).
        """
        if not notebook_cells or not current_cell_idx:
            return set()
        muts: set[str] = set()
        for code in notebook_cells[:current_cell_idx]:
            try:
                muts |= set(analyze_statement(code, None).all_mutated_vars)
            except (SyntaxError, ValueError, TypeError):
                continue
        return muts

    def _simulate_and_find_changes(
        self,
        current_cell_idx: int,
        notebook_cells: list[str],
        required_inputs: set[str] | None = None,
        current_cell_outputs: set[str] | None = None,
        current_cell_reassigned: set[str] | None = None,
        current_cell_mutated: set[str] | None = None,
        current_cell_method_receivers: set[str] | None = None,
        current_cell_selfref_vars: set[str] | None = None,
        current_cell_nocache_vars: set[str] | None = None,
    ) -> tuple[list[str], list[dict], float]:
        """Simulate notebook execution statement-by-statement.

        Returns:
            Tuple of ``(statements_to_reexecute, restored_info, total_restore_time)``:
            list of statement codes that need re-execution, list of dicts
            with info about restored statements, and total disk-cache lookup
            time (seconds) accumulated during simulation.
        """
        trace_event("simulate_enter", cell_idx=current_cell_idx,
                    reassigned=current_cell_reassigned or set(),
                    mutated=current_cell_mutated or set(),
                    required_inputs=required_inputs or set(),
                    selfref=current_cell_selfref_vars or set(),
                    method_receivers=current_cell_method_receivers or set())
        # Pass 1: Simulate ALL statements to build final virtual state
        stmt_lookup_times = {}  # stmt_code -> cache_lookup_time (disk I/O during simulation)
        loop_target_vars = set()  # Track loop iteration variables (e.g., 'item' in 'for item in data')

        (first_changed_cell, had_prior_cache, cache_had_hash_mismatch,
         simulation_trace, virtual_lineage, virtual_modules,
         new_cache_entries, vars_mutated_by_loops, vars_with_stale_files) = \
            self._virtual_lineage._find_incremental_start(current_cell_idx, notebook_cells)

        self._virtual_lineage._simulate_cells_pass1(
            first_changed_cell, current_cell_idx, notebook_cells,
            simulation_trace, virtual_lineage, virtual_modules,
            new_cache_entries, vars_mutated_by_loops, vars_with_stale_files,
            stmt_lookup_times, loop_target_vars,
        )

        # Detect whether any upstream cell was actually modified since last simulation.
        # This is True only when we had a prior simulation cache AND a cached cell's hash
        # changed (actual code modification).  NOT true when cells are simply not cached
        # yet (e.g., first time cell 3 runs, cache only has cell 1 â€” cell 2 is new to the
        # cache but wasn't modified).
        upstream_has_modifications = had_prior_cache and cache_had_hash_mismatch

        if self.debug:
            logger.debug("[UPSTREAM_DEBUG] upstream_has_modifications=%s "
                  "(had_prior_cache=%s, cache_had_hash_mismatch=%s, first_changed_cell=%s)",
                  upstream_has_modifications, had_prior_cache, cache_had_hash_mismatch, first_changed_cell)

        vars_derived_from_loops = self._virtual_lineage._propagate_loop_derived_vars(
            vars_mutated_by_loops, simulation_trace
        )

        if self.debug and loop_target_vars:
            logger.debug("[UPSTREAM_DEBUG] Loop target variables (iteration vars): %s", loop_target_vars)

        broken_vars, simulation_trace_codes, vars_tainted = self._classifier._run_pass2_identify_broken_vars(
            simulation_trace, virtual_lineage, virtual_modules, vars_mutated_by_loops,
            vars_with_stale_files, vars_derived_from_loops, loop_target_vars,
            upstream_has_modifications, required_inputs, current_cell_outputs,
            notebook_cells, current_cell_idx,
        )
        trace_event("broken_after_pass2", broken=broken_vars, tainted=vars_tainted)

        self._mark_stale_value_inputs_broken(
            required_inputs, current_cell_reassigned, broken_vars,
            current_cell_mutated=current_cell_mutated,
            notebook_cells=notebook_cells, current_cell_idx=current_cell_idx,
            current_cell_method_receivers=current_cell_method_receivers,
            current_cell_selfref_vars=current_cell_selfref_vars,
        )
        trace_event("broken_after_guard", broken=broken_vars)

        # A ``# @cash: no-cache`` statement opts the whole statement out of cash,
        # so its self-modified vars must behave like an uncached Jupyter cell:
        # re-running ACCUMULATES (advances), never resets. Pass 2 still flags such
        # a var stale (its runtime lineage advanced past the simulation's), which
        # would re-execute its producer and reset it -- so drop no-cache-written
        # vars from broken_vars here (the CAS-47 self-write-set exclusion only
        # covered the stale-value guard, not the pass-2 lineage mismatch).
        if current_cell_nocache_vars:
            removed = broken_vars & set(current_cell_nocache_vars)
            if removed:
                broken_vars -= removed
                trace_event("broken_drop_nocache", dropped=removed, broken=broken_vars)

        if not broken_vars:
            self._apply_phase_mutations()
            return [], [], 0.0

        # Optimization: probe the current cell's statements to see if cache
        # hits would restore broken variables, making upstream re-execution
        # unnecessary.  For example, if df is broken but the current cell's
        # first df-consuming statement is a disk cache hit that restores df,
        # we don't need to re-execute upstream cells that produce df.
        self._virtual_lineage._eliminate_broken_vars_via_current_cell_probe(
            broken_vars, notebook_cells, current_cell_idx,
            virtual_lineage, virtual_modules,
        )

        if not broken_vars:
            if self.debug:
                logger.debug("[UPSTREAM] All broken vars resolved by current cell cache hits â€” skipping upstream")
            self._apply_phase_mutations()
            return [], [], 0.0

        result = self._planner._build_reexecution_plan(
            simulation_trace, broken_vars, vars_tainted, simulation_trace_codes,
            virtual_lineage, virtual_modules, vars_derived_from_loops,
            vars_mutated_by_loops, upstream_has_modifications,
            stmt_lookup_times, notebook_cells,
        )
        self._apply_phase_mutations()
        return result

