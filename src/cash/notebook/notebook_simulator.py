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
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ._protocols import CashInstanceProtocol, ShellProtocol, TrackingState
from .analysis import CodeAnalyzer
from .control_structures import is_control_structure  # re-exported for test patching
from .mismatch_classifier import MismatchClassifier
from .server_discovery import get_notebook_cells  # re-exported for test patching
from .simulator_types import (
    IncrementalStartResult as _IncrementalStartResult,
    SimulationCacheEntry as _SimulationCacheEntry,
    TraceEntry as _TraceEntry,
)
from .virtual_lineage import (
    VirtualLineage,
    _BUILTIN_NAMES,
    _FORWARD_PROBE_PLACEHOLDER,
    _normalize_stmt,
)

if TYPE_CHECKING:
    from .statement_processor import ProcessResult

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

    def set_tracking_state(self, state: TrackingState) -> None:
        """Re-wire shared state refs (mirrors UpstreamChecker.set_tracking_state)."""
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

    # --- Property forwarders for state owned by VirtualLineage ---------

    @property
    def function_tracker(self):
        return self._virtual_lineage.function_tracker

    @function_tracker.setter
    def function_tracker(self, v):
        self._virtual_lineage.function_tracker = v

    @property
    def _current_cell_id(self):
        return self._virtual_lineage._current_cell_id

    @_current_cell_id.setter
    def _current_cell_id(self, v):
        self._virtual_lineage._current_cell_id = v

    @property
    def _simulation_cache(self):
        return self._virtual_lineage._simulation_cache

    @_simulation_cache.setter
    def _simulation_cache(self, v):
        self._virtual_lineage._simulation_cache = v

    @property
    def _cell_id_to_last_index(self):
        return self._virtual_lineage._cell_id_to_last_index

    @property
    def _ast_cache(self):
        return self._virtual_lineage._ast_cache

    @_ast_cache.setter
    def _ast_cache(self, v):
        self._virtual_lineage._ast_cache = v

    @property
    def _simulation_cell_hashes(self):
        return self._virtual_lineage._simulation_cell_hashes

    @_simulation_cell_hashes.setter
    def _simulation_cell_hashes(self, v):
        self._virtual_lineage._simulation_cell_hashes = v

    @property
    def _ast_cache_max_size(self):
        return self._virtual_lineage._ast_cache_max_size

    @_ast_cache_max_size.setter
    def _ast_cache_max_size(self, v):
        self._virtual_lineage._ast_cache_max_size = v

    # --- Class-level forwarders for moved methods (backward compat) -----
    #
    # Tests and ``UpstreamChecker`` reach into many former simulator helpers
    # via the class or instance.  Keep that surface intact by forwarding to
    # ``_virtual_lineage``.

    # Static-method forwarders (tests access these on the class).
    _validate_file_freshness = VirtualLineage._validate_file_freshness
    _stat_file_deps = VirtualLineage._stat_file_deps
    _iter_body_nodes = VirtualLineage._iter_body_nodes

    def _get_metadata_only(self, *args, **kwargs):
        return self._virtual_lineage._get_metadata_only(*args, **kwargs)
    def _get_cached_ast(self, *args, **kwargs):
        return self._virtual_lineage._get_cached_ast(*args, **kwargs)
    def _check_cell_file_deps(self, *args, **kwargs):
        return self._virtual_lineage._check_cell_file_deps(*args, **kwargs)
    def _scan_main_cache_for_changes(self, *args, **kwargs):
        return self._virtual_lineage._scan_main_cache_for_changes(*args, **kwargs)
    def _check_lightweight_hash_cache(self, *args, **kwargs):
        return self._virtual_lineage._check_lightweight_hash_cache(*args, **kwargs)
    def _restore_cached_state(self, *args, **kwargs):
        return self._virtual_lineage._restore_cached_state(*args, **kwargs)
    def _find_incremental_start(self, *args, **kwargs):
        return self._virtual_lineage._find_incremental_start(*args, **kwargs)
    def _collect_notebook_statements(self, *args, **kwargs):
        return self._virtual_lineage._collect_notebook_statements(*args, **kwargs)
    def _reapply_unsaved_extensions(self, *args, **kwargs):
        return self._virtual_lineage._reapply_unsaved_extensions(*args, **kwargs)
    def _propagate_loop_derived_vars(self, *args, **kwargs):
        return self._virtual_lineage._propagate_loop_derived_vars(*args, **kwargs)
    def _skipped_stmt_metric(self, *args, **kwargs):
        return self._virtual_lineage._skipped_stmt_metric(*args, **kwargs)
    def _collect_skipped_statement_metrics(self, *args, **kwargs):
        return self._virtual_lineage._collect_skipped_statement_metrics(*args, **kwargs)
    def _is_reinit_to_skip(self, *args, **kwargs):
        return self._virtual_lineage._is_reinit_to_skip(*args, **kwargs)
    def _filter_accumulator_reinits(self, *args, **kwargs):
        return self._virtual_lineage._filter_accumulator_reinits(*args, **kwargs)
    def _check_loop_derived_trust_override(self, *args, **kwargs):
        return self._virtual_lineage._check_loop_derived_trust_override(*args, **kwargs)
    def _build_loop_var_input_lineages(self, *args, **kwargs):
        return self._virtual_lineage._build_loop_var_input_lineages(*args, **kwargs)
    def _build_simulation_trace_codes(self, *args, **kwargs):
        return self._virtual_lineage._build_simulation_trace_codes(*args, **kwargs)
    def _simulate_one_node(self, *args, **kwargs):
        return self._virtual_lineage._simulate_one_node(*args, **kwargs)
    def _simulate_one_cell(self, *args, **kwargs):
        return self._virtual_lineage._simulate_one_cell(*args, **kwargs)
    def _simulate_cells_pass1(self, *args, **kwargs):
        return self._virtual_lineage._simulate_cells_pass1(*args, **kwargs)
    def _collect_loop_mutation_info(self, *args, **kwargs):
        return self._virtual_lineage._collect_loop_mutation_info(*args, **kwargs)
    def _apply_loop_mutation_lineages(self, *args, **kwargs):
        return self._virtual_lineage._apply_loop_mutation_lineages(*args, **kwargs)
    def _simulate_control_structure(self, *args, **kwargs):
        return self._virtual_lineage._simulate_control_structure(*args, **kwargs)
    def _resolve_input_lineage(self, *args, **kwargs):
        return self._virtual_lineage._resolve_input_lineage(*args, **kwargs)
    def _hash_module_with_deps(self, *args, **kwargs):
        return self._virtual_lineage._hash_module_with_deps(*args, **kwargs)
    def _compute_module_source_hash(self, *args, **kwargs):
        return self._virtual_lineage._compute_module_source_hash(*args, **kwargs)
    def _resolve_virtual_input_lineages(self, *args, **kwargs):
        return self._virtual_lineage._resolve_virtual_input_lineages(*args, **kwargs)
    def _apply_cache_hit_propagation(self, *args, **kwargs):
        return self._virtual_lineage._apply_cache_hit_propagation(*args, **kwargs)
    def _collect_historical_file_deps(self, *args, **kwargs):
        return self._virtual_lineage._collect_historical_file_deps(*args, **kwargs)
    def _try_virtual_cache_propagation(self, *args, **kwargs):
        return self._virtual_lineage._try_virtual_cache_propagation(*args, **kwargs)
    def _build_file_hash_component(self, *args, **kwargs):
        return self._virtual_lineage._build_file_hash_component(*args, **kwargs)
    def _compute_virtual_output_lineage(self, *args, **kwargs):
        return self._virtual_lineage._compute_virtual_output_lineage(*args, **kwargs)
    def _collect_session_file_deps(self, *args, **kwargs):
        return self._virtual_lineage._collect_session_file_deps(*args, **kwargs)
    def _propagate_import_lineage(self, *args, **kwargs):
        return self._virtual_lineage._propagate_import_lineage(*args, **kwargs)
    def _update_virtual_lineage(self, *args, **kwargs):
        return self._virtual_lineage._update_virtual_lineage(*args, **kwargs)
    def _check_file_deps_for_restore(self, *args, **kwargs):
        return self._virtual_lineage._check_file_deps_for_restore(*args, **kwargs)
    def _check_lineage_consistency(self, *args, **kwargs):
        return self._virtual_lineage._check_lineage_consistency(*args, **kwargs)
    def _restore_vars_from_cache(self, *args, **kwargs):
        return self._virtual_lineage._restore_vars_from_cache(*args, **kwargs)
    def _record_restored_cell_hash(self, *args, **kwargs):
        return self._virtual_lineage._record_restored_cell_hash(*args, **kwargs)
    def _try_virtual_restore(self, *args, **kwargs):
        return self._virtual_lineage._try_virtual_restore(*args, **kwargs)
    def _update_tracking_after_restore(self, *args, **kwargs):
        return self._virtual_lineage._update_tracking_after_restore(*args, **kwargs)
    def _eliminate_broken_vars_via_current_cell_probe(self, *args, **kwargs):
        return self._virtual_lineage._eliminate_broken_vars_via_current_cell_probe(*args, **kwargs)
    def _code_exists_in_notebook(self, *args, **kwargs):
        return self._virtual_lineage._code_exists_in_notebook(*args, **kwargs)
    def _find_directly_mismatched_vars(self, *args, **kwargs):
        return self._virtual_lineage._find_directly_mismatched_vars(*args, **kwargs)
    def _compute_tainted_vars_from_unsaved_edits(self, *args, **kwargs):
        return self._virtual_lineage._compute_tainted_vars_from_unsaved_edits(*args, **kwargs)
    def _is_valid_extension(self, *args, **kwargs):
        return self._virtual_lineage._is_valid_extension(*args, **kwargs)
    def _recurse_control_structure_mutations(self, *args, **kwargs):
        return self._virtual_lineage._recurse_control_structure_mutations(*args, **kwargs)
    def _find_loop_mutated_vars(self, *args, **kwargs):
        return self._virtual_lineage._find_loop_mutated_vars(*args, **kwargs)
    def _update_stale_file_deps(self, *args, **kwargs):
        return self._virtual_lineage._update_stale_file_deps(*args, **kwargs)

    # --- Method forwarders for moved Phase-2 classifier methods --------
    def _check_loop_var_inputs_changed(self, *args, **kwargs):
        return self._classifier._check_loop_var_inputs_changed(*args, **kwargs)
    def _collect_non_module_inputs(self, *args, **kwargs):
        return self._classifier._collect_non_module_inputs(*args, **kwargs)
    def _find_mismatched_data_inputs(self, *args, **kwargs):
        return self._classifier._find_mismatched_data_inputs(*args, **kwargs)
    def _check_code_matches_loop_trust(self, *args, **kwargs):
        return self._classifier._check_code_matches_loop_trust(*args, **kwargs)
    def _check_var_extension_valid(self, *args, **kwargs):
        return self._classifier._check_var_extension_valid(*args, **kwargs)
    def _handle_mismatch_code_matches(self, *args, **kwargs):
        return self._classifier._handle_mismatch_code_matches(*args, **kwargs)
    def _classify_one_broken_var(self, *args, **kwargs):
        return self._classifier._classify_one_broken_var(*args, **kwargs)
    def _handle_mismatch_prereqs(self, *args, **kwargs):
        return self._classifier._handle_mismatch_prereqs(*args, **kwargs)
    def _handle_lineage_mismatch(self, *args, **kwargs):
        return self._classifier._handle_lineage_mismatch(*args, **kwargs)
    def _classify_broken_vars(self, *args, **kwargs):
        return self._classifier._classify_broken_vars(*args, **kwargs)
    def _run_pass2_identify_broken_vars(self, *args, **kwargs):
        return self._classifier._run_pass2_identify_broken_vars(*args, **kwargs)
    def _check_tainted_input_valid(self, *args, **kwargs):
        return self._classifier._check_tainted_input_valid(*args, **kwargs)
    def _all_tainted_inputs_valid(self, *args, **kwargs):
        return self._classifier._all_tainted_inputs_valid(*args, **kwargs)
    def _resolve_tainted_stmt(self, *args, **kwargs):
        return self._classifier._resolve_tainted_stmt(*args, **kwargs)
    def _check_inp_lineage_skip(self, *args, **kwargs):
        return self._classifier._check_inp_lineage_skip(*args, **kwargs)
    def _should_add_input_to_needed(self, *args, **kwargs):
        return self._classifier._should_add_input_to_needed(*args, **kwargs)
    def _cascade_failed_restore_inputs(self, *args, **kwargs):
        return self._classifier._cascade_failed_restore_inputs(*args, **kwargs)
    def _backward_scan_pass(self, *args, **kwargs):
        return self._classifier._backward_scan_pass(*args, **kwargs)
    def _check_missing_required_inputs(self, *args, **kwargs):
        return self._classifier._check_missing_required_inputs(*args, **kwargs)

    def _build_reexecution_plan(
        self,
        simulation_trace: list,
        broken_vars: set[str],
        vars_tainted_by_upstream_mismatch: set[str],
        simulation_trace_codes: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        vars_derived_from_loops: set[str],
        vars_mutated_by_loops: set[str],
        upstream_has_modifications: bool,
        stmt_lookup_times: dict[str, float],
        notebook_cells: list[str],
    ) -> tuple[list[str], list[dict], float]:
        """Build the list of statements to re-execute and restored info.

        Returns (statements_to_reexecute, restored_statements_info, total_restore_time).
        """
        if self.debug:
            logger.debug("[UPSTREAM_DEBUG] Simulation trace contents:")
            for i, (stmt, outputs, _, _, _, _) in enumerate(simulation_trace):
                logger.debug("[UPSTREAM_DEBUG]   [%s] outputs=%s: %s...", i, outputs, stmt[:60])

        loop_derived_trust_overridden = self._virtual_lineage._check_loop_derived_trust_override(
            upstream_has_modifications, vars_mutated_by_loops, simulation_trace_codes,
        )

        stmts_to_run_indices, restored_statements_info, total_restore_time = self._classifier._backward_scan_pass(
            simulation_trace, broken_vars, vars_tainted_by_upstream_mismatch,
            virtual_lineage, virtual_modules, vars_derived_from_loops,
            loop_derived_trust_overridden, upstream_has_modifications,
            simulation_trace_codes, stmt_lookup_times,
        )

        skipped_metrics = self._virtual_lineage._collect_skipped_statement_metrics(
            simulation_trace, stmts_to_run_indices, restored_statements_info,
            virtual_modules, stmt_lookup_times,
        )
        restored_statements_info.extend(skipped_metrics)

        stmts_to_run_indices = self._schedule_loop_var_contexts(stmts_to_run_indices, simulation_trace)
        stmts_to_run_indices = self._virtual_lineage._filter_accumulator_reinits(stmts_to_run_indices, simulation_trace, vars_mutated_by_loops)
        stmts_to_run_indices = self._dedup_sorted_indices(stmts_to_run_indices)

        statements_to_reexecute: list[str] = []
        for idx in stmts_to_run_indices:
            stmt_code = simulation_trace[idx][0]
            statements_to_reexecute.append(stmt_code)
            if self.debug:
                logger.debug("[UPSTREAM] Scheduled for execution: %s", stmt_code[:40])

        restored_statements_info.reverse()

        vars_updated_by_trace: set[str] = set()
        for stmt_code in statements_to_reexecute:
            try:
                _, outputs = CodeAnalyzer.analyze_code_block(stmt_code)
                vars_updated_by_trace.update(outputs)
            except (SyntaxError, ValueError):
                logger.debug("Failed to analyze statement for variable outputs: %.40s", stmt_code)

        self._virtual_lineage._reapply_unsaved_extensions(
            broken_vars, vars_updated_by_trace, simulation_trace,
            notebook_cells, statements_to_reexecute,
        )

        return statements_to_reexecute, restored_statements_info, total_restore_time

    def _dedup_sorted_indices(self, stmts_to_run_indices: list[int]) -> list[int]:
        """Return *stmts_to_run_indices* sorted and deduplicated while preserving order."""
        stmts_to_run_indices.sort()
        seen: set[int] = set()
        unique: list[int] = []
        for idx in stmts_to_run_indices:
            if idx not in seen:
                seen.add(idx)
                unique.append(idx)
        return unique

    def _is_loop_var_assignment_for_context(
        self,
        i: int,
        stmt_code: str,
        outputs: set[str],
        simulation_trace: list,
        scheduled_contexts: set[str],
        iteration_context_pattern: re.Pattern,
    ) -> bool:
        """Return True if *stmt* is a loop-var assignment for a scheduled iteration context."""
        for j in range(i + 1, min(i + 4, len(simulation_trace))):
            next_stmt = simulation_trace[j][0]
            match = iteration_context_pattern.search(next_stmt)
            if match and match.group(1) in scheduled_contexts:
                if len(outputs) == 1:
                    var_name = list(outputs)[0]
                    if f"{var_name} = " in stmt_code or f"{var_name}=" in stmt_code:
                        return True
            elif match:
                break
        return False

    def _schedule_loop_var_contexts(
        self,
        stmts_to_run_indices: list[int],
        simulation_trace: list,
    ) -> list[int]:
        """Ensure loop variable assignments are scheduled alongside their iteration bodies.

        This prevents stale restorations from overwriting loop variable
        assignments when iteration bodies are re-executed.
        """
        iteration_context_pattern = re.compile(r'# __iteration_context__: ([a-f0-9]+)')
        stmts_set = set(stmts_to_run_indices)

        scheduled_contexts: set[str] = set()
        for idx in stmts_to_run_indices:
            stmt_code = simulation_trace[idx][0]
            match = iteration_context_pattern.search(stmt_code)
            if match:
                scheduled_contexts.add(match.group(1))

        if not scheduled_contexts:
            return stmts_to_run_indices

        additional_indices: list[int] = []
        for i, (stmt_code, outputs, _inputs, _, _, _) in enumerate(simulation_trace):
            if i in stmts_set or iteration_context_pattern.search(stmt_code):
                continue
            if self._is_loop_var_assignment_for_context(
                i, stmt_code, outputs, simulation_trace, scheduled_contexts, iteration_context_pattern,
            ):
                if self.debug:
                    logger.debug("[UPSTREAM] Adding loop var assignment for scheduled context: %s", stmt_code[:40])
                additional_indices.append(i)

        return stmts_to_run_indices + additional_indices

    def _simulate_and_find_changes(
        self,
        current_cell_idx: int,
        notebook_cells: list[str],
        required_inputs: set[str] | None = None,
        current_cell_outputs: set[str] | None = None
    ) -> tuple[list[str], list[ProcessResult]]:
        """Simulate notebook execution statement-by-statement.

        Returns:
            Tuple of (list of statement codes that need re-execution,
            list of dicts with info about restored statements).
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
            return [], [], 0.0

        return self._build_reexecution_plan(
            simulation_trace, broken_vars, vars_tainted, simulation_trace_codes,
            virtual_lineage, virtual_modules, vars_derived_from_loops,
            vars_mutated_by_loops, upstream_has_modifications,
            stmt_lookup_times, notebook_cells,
        )

