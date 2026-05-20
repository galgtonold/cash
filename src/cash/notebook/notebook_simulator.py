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

from ._protocols import CashInstanceProtocol, ShellProtocol, TrackingState
from .control_structures import is_control_structure  # noqa: F401  re-exported for test patching (mocked via @patch in test_issue_reproduction)
from .mismatch_classifier import MismatchClassifier
from .reexecution_planner import ReexecutionPlanner
from .server_discovery import get_notebook_cells  # noqa: F401  re-exported for test patching
from .simulator_types import (  # noqa: F401  re-exported (private renames)
    IncrementalStartResult as _IncrementalStartResult,
    SimulationCacheEntry as _SimulationCacheEntry,
    TraceEntry as _TraceEntry,
    apply_collected_mutations,
)
from .virtual_lineage import (  # noqa: F401  re-exported (imported by upstream.py via .notebook_simulator)
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
    property/method forwarders for the legacy seams.
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

    # --- Class-level static-method aliases (tests access on the class) ---
    _validate_file_freshness = VirtualLineage._validate_file_freshness
    _stat_file_deps = VirtualLineage._stat_file_deps
    _iter_body_nodes = VirtualLineage._iter_body_nodes

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

    def _simulate_and_find_changes(
        self,
        current_cell_idx: int,
        notebook_cells: list[str],
        required_inputs: set[str] | None = None,
        current_cell_outputs: set[str] | None = None
    ) -> tuple[list[str], list[dict], float]:
        """Simulate notebook execution statement-by-statement.

        Returns:
            Tuple of ``(statements_to_reexecute, restored_info, total_restore_time)``:
            list of statement codes that need re-execution, list of dicts
            with info about restored statements, and total disk-cache lookup
            time (seconds) accumulated during simulation.
        """
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

