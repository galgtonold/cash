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

import ast
import hashlib
import logging
import os
import re
import time as time_module
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..exceptions import UpstreamStateError
from ..utils import resolve_file_dep_path
from ._protocols import CashInstanceProtocol, ShellProtocol, TrackingState
from .analysis import CodeAnalyzer
from .cache_key import CacheKeyContext, compute_cache_key
from .cache_status import CacheStatus
from .control_structures import extract_target_names, get_control_structure_type, is_control_structure
from .server_discovery import get_notebook_cells
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

from .simulator_types import (
    IncrementalStartResult as _IncrementalStartResult,
    SimulationCacheEntry as _SimulationCacheEntry,
    TraceEntry as _TraceEntry,
)


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

    def _check_loop_var_inputs_changed(
        self,
        var_name: str,
        input_lineages_for_var: dict[str, str],
        vars_derived_from_loops: set[str],
        loop_target_vars: set[str],
    ) -> bool:
        """Return True if any non-loop data input for *var_name* has changed lineage."""
        for inp_name, expected_lineage in input_lineages_for_var.items():
            if inp_name in vars_derived_from_loops:
                continue  # Skip other loop-derived vars (their lineages are inherently mismatched)
            if inp_name in loop_target_vars:
                continue  # Skip loop iteration vars (e.g., 'x' in 'for x in data')
            actual_inp_lineage = self.variable_lineage.get(inp_name)
            if actual_inp_lineage and actual_inp_lineage != expected_lineage:
                if self.debug:
                    logger.debug("[UPSTREAM_DEBUG] Loop input '%s' lineage changed: virtual=%s, actual=%s",
                          inp_name, expected_lineage[:8], actual_inp_lineage[:8])
                return True
        return False

    def _collect_non_module_inputs(
        self,
        stmt_inputs: list[str],
        virtual_modules: set[str],
    ) -> set[str]:
        """Return the subset of *stmt_inputs* that are data variables (not modules)."""
        data_inputs: set[str] = set()
        for inp in stmt_inputs:
            if inp in virtual_modules:
                continue
            val = self.shell.user_ns.get(inp)
            if val is not None and isinstance(val, types.ModuleType):
                continue
            data_inputs.add(inp)
        return data_inputs

    def _find_mismatched_data_inputs(
        self,
        var_name: str,
        data_inputs: set[str],
        vars_derived_from_loops: set[str],
        loop_target_vars: set[str],
        virtual_lineage: dict[str, str],
    ) -> set[str]:
        """Return data inputs whose virtual and actual lineages differ."""
        mismatched: set[str] = set()
        for inp in data_inputs:
            if inp == var_name:
                continue  # skip self-referential input (e.g., df['x'] = f(df))
            if inp in vars_derived_from_loops:
                continue  # loop-derived, expected mismatch
            if inp in loop_target_vars:
                continue  # loop iteration target, expected mismatch
            if inp in virtual_lineage and inp in self.variable_lineage and virtual_lineage[inp] != self.variable_lineage[inp]:
                mismatched.add(inp)
        return mismatched

    def _check_code_matches_loop_trust(
        self,
        var_name: str,
        last_stmt_for_var: str,
        vars_derived_from_loops: set[str],
        loop_target_vars: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        upstream_has_modifications: bool,
        loop_derived_trust_overridden: bool,
    ) -> bool:
        """Return True if var_name should be trusted in-memory when code matches trace.

        Called only when upstream_has_modifications is False and loop-derived
        check applies.  Checks that no non-loop data inputs have a lineage
        mismatch.
        """
        try:
            stmt_inputs, _ = CodeAnalyzer.analyze_code_block(last_stmt_for_var)
            data_inputs = self._collect_non_module_inputs(stmt_inputs, virtual_modules)
            mismatched_inputs = self._find_mismatched_data_inputs(
                var_name, data_inputs, vars_derived_from_loops, loop_target_vars, virtual_lineage,
            )
            if not mismatched_inputs:
                if self.debug:
                    logger.debug(
                        "[UPSTREAM_DEBUG]   -> Code matches, required input '%s' mismatch due to "
                        "loop-derived inputs %s. "
                        "Trusting in-memory value (upstream unchanged).",
                        var_name, data_inputs & vars_derived_from_loops,
                    )
                return True
        except (KeyError, ValueError, TypeError):
            logger.debug("[UPSTREAM] Failed to check loop-derived inputs for '%s'", var_name)
        return False

    def _check_var_extension_valid(
        self,
        var_name: str,
        actual_lineage: str,
        virtual_lineage: dict[str, str],
        upstream_has_modifications: bool,
        notebook_cells: list[str],
    ) -> bool:
        """Return True if the in-memory value is a valid downstream extension."""
        if var_name not in self.executed_cell_codes:
            return False
        mem_code = self.executed_cell_codes[var_name]
        if not self._virtual_lineage._is_valid_extension(mem_code, actual_lineage, virtual_lineage, required_dependency=var_name):
            return False
        if upstream_has_modifications:
            code_still_in_notebook = self._virtual_lineage._code_exists_in_notebook(mem_code, notebook_cells)
            if code_still_in_notebook:
                if self.debug:
                    logger.debug("[UPSTREAM_DEBUG]   -> Valid extension (code still exists in notebook), keeping")
                return True
            if self.debug:
                logger.debug("[UPSTREAM_DEBUG]   -> Extension code no longer exists in notebook (modified/deleted upstream). Rejecting.")
            return False
        if self.debug:
            logger.debug("[UPSTREAM_DEBUG]   -> Valid extension (no upstream modifications), keeping")
        logger.debug("[UPSTREAM] Variable '%s' is a valid extension of notebook state. Keeping.", var_name)
        return True

    def _handle_mismatch_code_matches(
        self,
        var_name: str,
        last_stmt_for_var: str | None,
        vars_derived_from_loops: set[str],
        loop_target_vars: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        upstream_has_modifications: bool,
        loop_derived_trust_overridden: bool,
        required_inputs: set[str] | None,
        broken_vars: set[str],
    ) -> bool:
        """Handle the code-matches-but-lineage-differs case.

        Returns True if the caller should stop processing this variable (either
        the variable was trusted or marked broken), False if processing should
        continue (code did not match).
        """
        if var_name not in self.executed_cell_codes:
            return False
        last_stmt_for_var_real = last_stmt_for_var
        if last_stmt_for_var_real is None:
            return False
        sim_code = _normalize_stmt(last_stmt_for_var_real)
        mem_code = _normalize_stmt(self.executed_cell_codes[var_name])
        if sim_code != mem_code:
            return False

        # Code matches simulation but lineage differs.
        if (required_inputs and var_name in required_inputs
                and vars_derived_from_loops and not upstream_has_modifications
                and not loop_derived_trust_overridden):
            if self._check_code_matches_loop_trust(
                var_name, last_stmt_for_var_real, vars_derived_from_loops, loop_target_vars,
                virtual_lineage, virtual_modules, upstream_has_modifications, loop_derived_trust_overridden,
            ):
                return True
            if self.debug:
                logger.debug("[UPSTREAM_DEBUG]   -> Code matches but '%s' is a REQUIRED INPUT with lineage mismatch. Marking as broken.", var_name)
            logger.debug("[UPSTREAM] Variable '%s' is a required input with lineage mismatch. Must re-execute.", var_name)
            broken_vars.add(var_name)
            return True
        # For non-required variables, trust the memory
        if self.debug:
            logger.debug("[UPSTREAM_DEBUG]   -> Lineage mismatch but Code Matches trace (%s). Assuming valid extension due to cache miss. Keeping.", var_name)
        logger.debug("[UPSTREAM] Variable '%s' mismatch but code matches trace. Keeping.", var_name)
        return True

    def _classify_one_broken_var(
        self,
        var_name: str,
        vars_derived_from_loops: set[str],
        upstream_has_modifications: bool,
        loop_derived_trust_overridden: bool,
        loop_var_input_lineages: dict[str, dict[str, str]],
        loop_target_vars: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        vars_with_stale_files: set[str],
        vars_mutated_by_loops: set[str],
        vars_tainted_by_upstream_mismatch: set[str],
        simulation_trace: list,
        required_inputs: set[str] | None,
        current_cell_outputs: set[str] | None,
        notebook_cells: list[str],
        broken_vars: set[str],
        simulation_trace_codes: set[str] | None = None,
    ) -> None:
        """Classify a single variable and add to *broken_vars* if needed."""
        # Skip loop-derived vars when upstream is unchanged AND producing code
        # is on disk (not overridden by unsaved edit). FAST MODE can't track
        # per-iteration lineage, so we trust in-memory state.
        if var_name in vars_derived_from_loops and not upstream_has_modifications and not loop_derived_trust_overridden:
            # Don't trust if the variable was overwritten by a downstream cell.
            # Check that executed_cell_codes for this var matches an upstream statement.
            overwritten_downstream = False
            exec_code = self.executed_cell_codes.get(var_name)
            if exec_code and simulation_trace_codes is not None:
                if exec_code not in simulation_trace_codes:
                    overwritten_downstream = True
                    if self.debug:
                        logger.debug(
                            "[UPSTREAM_DEBUG] NOT trusting loop-derived '%s' â€” "
                            "executed code '%.40s' not in upstream simulation",
                            var_name, exec_code,
                        )

            if not overwritten_downstream:
                input_lineages_for_var = loop_var_input_lineages.get(var_name, {})
                inputs_changed = self._check_loop_var_inputs_changed(
                    var_name, input_lineages_for_var, vars_derived_from_loops, loop_target_vars,
                )
                if not inputs_changed:
                    if self.debug:
                        source = "directly mutated by loop" if var_name in vars_mutated_by_loops else "transitively derived from loop mutation"
                        logger.debug("[UPSTREAM_DEBUG] Skipping mismatch check for '%s' - %s, trusting in-memory state (upstream unchanged, inputs consistent)", var_name, source)
                    return
                if self.debug:
                    source = "directly mutated by loop" if var_name in vars_mutated_by_loops else "transitively derived from loop mutation"
                    logger.debug("[UPSTREAM_DEBUG] NOT trusting '%s' (%s) â€” loop input lineage changed, will check lineage", var_name, source)

        actual_lineage = self.variable_lineage[var_name]
        if var_name not in virtual_lineage:
            if self.debug:
                logger.debug("[UPSTREAM_DEBUG] Variable '%s' is in memory but not in virtual state (downstream or external)", var_name)
            return

        final_virtual_hash = virtual_lineage[var_name]
        if actual_lineage == final_virtual_hash:
            if var_name in vars_tainted_by_upstream_mismatch:
                if self.debug:
                    logger.debug("[UPSTREAM_DEBUG] Lineage matches for '%s' but tainted by upstream mismatch. Marking broken.", var_name)
                broken_vars.add(var_name)
            return

        self._handle_lineage_mismatch(
            var_name, actual_lineage, final_virtual_hash,
            vars_derived_from_loops, upstream_has_modifications, loop_derived_trust_overridden,
            loop_target_vars, virtual_lineage, virtual_modules,
            vars_with_stale_files, simulation_trace, required_inputs, current_cell_outputs,
            notebook_cells, broken_vars,
        )

    def _handle_mismatch_prereqs(
        self,
        var_name: str,
        actual_lineage: str,
        final_virtual_hash: str,
        virtual_lineage: dict[str, str],
        vars_with_stale_files: set[str],
        upstream_has_modifications: bool,
        required_inputs: set[str] | None,
        current_cell_outputs: set[str] | None,
        notebook_cells: list[str],
        broken_vars: set[str],
    ) -> bool:
        """Check early-exit conditions for a lineage mismatch.

        Returns True if the caller should stop processing this variable
        (it was already handled â€” marked broken, kept, or lineage reset).
        """
        # Read-only input: reject downstream mutations (e.g., df['SMA']=...)
        if required_inputs and var_name in required_inputs and current_cell_outputs is not None and var_name not in current_cell_outputs:
            if self.debug:
                logger.debug("[UPSTREAM_DEBUG]   -> '%s' is a READ-ONLY required input "
                      "(not in current cell outputs). Rejecting downstream extension "
                      "to force restoration to upstream state.", var_name)
            broken_vars.add(var_name)
            return True

        if self._check_var_extension_valid(
            var_name, actual_lineage, virtual_lineage, upstream_has_modifications, notebook_cells,
        ):
            return True

        if var_name in vars_with_stale_files:
            if self.debug:
                logger.debug("[UPSTREAM_DEBUG]   -> Variable '%s' has stale file dependencies. Forcing re-execution.", var_name)
            logger.debug("[UPSTREAM] Variable '%s' has stale file dependencies. Must re-execute.", var_name)
            broken_vars.add(var_name)
            return True

        # Downstream advancement: if var is also a current-cell output reset lineage.
        if required_inputs and var_name in required_inputs and current_cell_outputs and var_name in current_cell_outputs:
            if self.debug:
                logger.debug("[UPSTREAM_DEBUG]   -> '%s' is also an OUTPUT of the current cell. "
                      "Lineage is ahead due to downstream advancement. "
                      "Resetting lineage from %s to virtual %s.",
                      var_name, actual_lineage[:8], final_virtual_hash[:8])
            self.lineage.reset_to(var_name, final_virtual_hash)
            return True

        return False

    def _handle_lineage_mismatch(
        self,
        var_name: str,
        actual_lineage: str,
        final_virtual_hash: str,
        vars_derived_from_loops: set[str],
        upstream_has_modifications: bool,
        loop_derived_trust_overridden: bool,
        loop_target_vars: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        vars_with_stale_files: set[str],
        simulation_trace: list,
        required_inputs: set[str] | None,
        current_cell_outputs: set[str] | None,
        notebook_cells: list[str],
        broken_vars: set[str],
    ) -> None:
        """Handle a confirmed lineage mismatch for *var_name*."""
        if self.debug:
            logger.debug("[UPSTREAM_DEBUG] Lineage mismatch for '%s': virtual=%s, actual=%s", var_name, final_virtual_hash[:8], actual_lineage[:8])

        if self._handle_mismatch_prereqs(
            var_name, actual_lineage, final_virtual_hash, virtual_lineage,
            vars_with_stale_files, upstream_has_modifications, required_inputs,
            current_cell_outputs, notebook_cells, broken_vars,
        ):
            return

        last_stmt_for_var = None
        for stmt, outputs, _, _, _, _ in reversed(simulation_trace):
            if var_name in outputs:
                last_stmt_for_var = stmt
                break

        if self._handle_mismatch_code_matches(
            var_name, last_stmt_for_var, vars_derived_from_loops, loop_target_vars,
            virtual_lineage, virtual_modules, upstream_has_modifications,
            loop_derived_trust_overridden, required_inputs, broken_vars,
        ):
            return

        if required_inputs and var_name in required_inputs:
            if self.debug:
                logger.debug("[UPSTREAM_DEBUG]   -> Required input mismatch and INVALID extension. Forcing strict restoration")
            logger.debug("[UPSTREAM] Variable '%s' is a required input mismatch. Forcing strict restoration.", var_name)
        logger.debug("[UPSTREAM] Variable '%s' is broken. Exp: %s, Act: %s", var_name, final_virtual_hash[:8], actual_lineage[:8])
        broken_vars.add(var_name)

    def _classify_broken_vars(
        self,
        vars_to_check: set[str],
        vars_derived_from_loops: set[str],
        upstream_has_modifications: bool,
        loop_derived_trust_overridden: bool,
        loop_var_input_lineages: dict[str, dict[str, str]],
        loop_target_vars: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        vars_with_stale_files: set[str],
        vars_mutated_by_loops: set[str],
        vars_tainted_by_upstream_mismatch: set[str],
        simulation_trace: list,
        required_inputs: set[str] | None,
        current_cell_outputs: set[str] | None,
        notebook_cells: list[str],
        broken_vars: set[str],
        simulation_trace_codes: set[str] | None = None,
    ) -> None:
        """Classify each variable in *vars_to_check* and populate *broken_vars*.

        Also checks required inputs that are missing from memory entirely.
        """
        for var_name in vars_to_check:
            self._classify_one_broken_var(
                var_name, vars_derived_from_loops, upstream_has_modifications,
                loop_derived_trust_overridden, loop_var_input_lineages, loop_target_vars,
                virtual_lineage, virtual_modules, vars_with_stale_files, vars_mutated_by_loops,
                vars_tainted_by_upstream_mismatch, simulation_trace, required_inputs,
                current_cell_outputs, notebook_cells, broken_vars,
                simulation_trace_codes=simulation_trace_codes,
            )

        # Only required inputs matter here; temporary intermediates can stay missing.
        self._check_missing_required_inputs(
            required_inputs, virtual_lineage, virtual_modules, broken_vars,
        )

    def _run_pass2_identify_broken_vars(
        self,
        simulation_trace: list,
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        vars_mutated_by_loops: set[str],
        vars_with_stale_files: set[str],
        vars_derived_from_loops: set[str],
        loop_target_vars: set[str],
        upstream_has_modifications: bool,
        required_inputs: set[str] | None,
        current_cell_outputs: set[str] | None,
        notebook_cells: list[str],
        current_cell_idx: int,
    ) -> tuple[set[str], set[str], set[str]]:
        """Pass 2: identify broken variables.

        Returns (broken_vars, simulation_trace_codes, vars_tainted_by_upstream_mismatch).
        """
        if self.debug:
            logger.debug("[UPSTREAM_DEBUG] Simulation complete. Virtual lineage keys: %s", list(virtual_lineage.keys()))
            logger.debug("[UPSTREAM_DEBUG] Actual variable_lineage keys: %s", list(self.variable_lineage.keys()))
            logger.debug("[UPSTREAM_DEBUG] Simulation trace has %s statements", len(simulation_trace))
            if vars_mutated_by_loops:
                logger.debug("[UPSTREAM_DEBUG] Variables mutated by loops (trusted): %s", vars_mutated_by_loops)

        vars_to_check: set[str] = set()
        if required_inputs:
            for var_name in required_inputs:
                if var_name in self.variable_lineage:
                    vars_to_check.add(var_name)

        simulation_trace_codes = self._virtual_lineage._build_simulation_trace_codes(simulation_trace)

        vars_tainted_by_upstream_mismatch: set[str] = set()
        if not upstream_has_modifications:
            vars_tainted_by_upstream_mismatch = self._virtual_lineage._compute_tainted_vars_from_unsaved_edits(
                virtual_lineage, simulation_trace, simulation_trace_codes,
                current_cell_idx, notebook_cells,
            )

        loop_derived_trust_overridden = self._virtual_lineage._check_loop_derived_trust_override(
            upstream_has_modifications, vars_mutated_by_loops, simulation_trace_codes,
        )

        loop_var_input_lineages = self._virtual_lineage._build_loop_var_input_lineages(
            simulation_trace, vars_derived_from_loops, virtual_lineage, virtual_modules,
        )

        broken_vars: set[str] = set()
        self._classify_broken_vars(
            vars_to_check, vars_derived_from_loops, upstream_has_modifications,
            loop_derived_trust_overridden, loop_var_input_lineages, loop_target_vars,
            virtual_lineage, virtual_modules, vars_with_stale_files, vars_mutated_by_loops,
            vars_tainted_by_upstream_mismatch, simulation_trace, required_inputs,
            current_cell_outputs, notebook_cells, broken_vars,
            simulation_trace_codes=simulation_trace_codes,
        )

        if self.debug:
            logger.debug("[UPSTREAM_DEBUG] Broken vars: %s", broken_vars)
            if not broken_vars:
                logger.debug("[UPSTREAM_DEBUG] No broken vars, nothing to re-execute")

        return broken_vars, simulation_trace_codes, vars_tainted_by_upstream_mismatch

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

        stmts_to_run_indices, restored_statements_info, total_restore_time = self._backward_scan_pass(
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

    def _check_tainted_input_valid(
        self,
        inp: str,
        virtual_lineage: dict[str, str],
        upstream_has_modifications: bool,
        simulation_trace_codes: set[str],
    ) -> bool:
        """Return True if *inp* is an unsaved-edit input that should be trusted.

        This is called when inp has a lineage mismatch to decide if we can
        trust the in-memory value (produced by unsaved code) rather than
        cascading into the old disk code.
        """
        if inp not in virtual_lineage or inp not in self.variable_lineage:
            return False
        if self.variable_lineage[inp] == virtual_lineage[inp]:
            return False
        if upstream_has_modifications:
            return False
        inp_producing_code = self.executed_cell_codes.get(inp)
        if inp_producing_code is None:
            return False
        normalized_inp_code = re.sub(r'# __iteration_context__:.*?\n', '', inp_producing_code).strip()
        return normalized_inp_code not in simulation_trace_codes

    def _all_tainted_inputs_valid(
        self,
        stmt_code: str,
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        upstream_has_modifications: bool,
        simulation_trace_codes: set[str],
    ) -> bool:
        """Return True if all inputs for a tainted statement are available and fresh."""
        stmt_inputs_check, _ = CodeAnalyzer.analyze_code_block(stmt_code)
        for inp in stmt_inputs_check:
            if inp in virtual_modules:
                if inp in self.shell.user_ns:
                    continue
                return False
            if inp in _BUILTIN_NAMES:
                continue
            if inp not in self.shell.user_ns:
                return False
            if self._check_tainted_input_valid(inp, virtual_lineage, upstream_has_modifications, simulation_trace_codes):
                if self.debug:
                    logger.debug("[UPSTREAM] Input '%s' has different lineage but produced "
                          "by unsaved edit (code not on disk). Trusting in-memory value.", inp)
                continue
            if inp in virtual_lineage and inp in self.variable_lineage and self.variable_lineage[inp] != virtual_lineage[inp]:
                if self.debug:
                    logger.debug("[UPSTREAM] Tainted stmt input '%s' has stale lineage "
                          "(actual=%s, virtual=%s). Cascading.",
                          inp, self.variable_lineage[inp][:8], virtual_lineage[inp][:8])
                return False
        return True

    def _resolve_tainted_stmt(
        self,
        i: int,
        stmt_code: str,
        outputs: set[str],
        needed_outputs_pre: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        upstream_has_modifications: bool,
        simulation_trace_codes: set[str],
        needed_vars: set[str],
        resolved_vars: set[str],
        stmts_to_run_indices: list[int],
    ) -> tuple[set[str], float, float, bool]:
        """Handle the force-reexecute branch for a tainted statement.

        Returns (restored_vars, restore_time, saved_time, handled) where
        *handled* is True when the statement was fully resolved (inputs valid),
        False when it needs cascading.
        """
        if self._all_tainted_inputs_valid(stmt_code, virtual_lineage, virtual_modules, upstream_has_modifications, simulation_trace_codes):
            stmts_to_run_indices.append(i)
            needed_vars.difference_update(outputs)
            resolved_vars.update(outputs - needed_outputs_pre)
            if self.debug:
                logger.debug("[UPSTREAM] Tainted stmt scheduled (inputs in memory): %s...", stmt_code[:60])
            return set(), 0.0, 0.0, True
        if self.debug:
            logger.debug("[UPSTREAM] Tainted stmt, inputs missing, cascading: %s...", stmt_code[:60])
        return set(), 0.0, 0.0, False

    def _check_inp_lineage_skip(
        self,
        inp: str,
        virtual_lineage: dict[str, str],
        upstream_has_modifications: bool,
        simulation_trace_codes: set[str],
    ) -> bool:
        """Return True if *inp* should be skipped based on lineage/unsaved-edit checks."""
        if inp not in self.variable_lineage or inp not in virtual_lineage:
            return False
        if self.variable_lineage[inp] == virtual_lineage[inp]:
            if inp in self.shell.user_ns:
                if self.debug:
                    logger.debug("[UPSTREAM] Input '%s' already valid in memory (lineage matches virtual). Skipping.", inp)
                return True
            return False
        # Lineage mismatch â€” check for unsaved edit
        if upstream_has_modifications or inp not in self.shell.user_ns:
            return False
        inp_prod_code = self.executed_cell_codes.get(inp)
        if inp_prod_code is None:
            return False
        norm_code = re.sub(r'# __iteration_context__:.*?\n', '', inp_prod_code).strip()
        if norm_code not in simulation_trace_codes:
            if self.debug:
                logger.debug("[UPSTREAM] Input '%s' lineage mismatch but produced by unsaved edit. Trusting in-memory.", inp)
            return True
        return False

    def _should_add_input_to_needed(
        self,
        inp: str,
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        vars_derived_from_loops: set[str],
        loop_derived_trust_overridden: bool,
        upstream_has_modifications: bool,
        simulation_trace_codes: set[str],
        needed_vars: set[str],
    ) -> bool:
        """Return True if *inp* should be added to *needed_vars* during cascade.

        Handles module, builtin, lineage-matching, unsaved-edit, and loop-derived
        special cases.  Side-effect: may add *inp* to *needed_vars* for modules.
        """
        if inp in virtual_modules:
            if inp not in self.shell.user_ns:
                needed_vars.add(inp)
                if self.debug:
                    logger.debug("[UPSTREAM] Module '%s' is in virtual_modules but NOT in memory. Scheduling re-import.", inp)
            return False  # handled (either added or skipped)
        if inp in _BUILTIN_NAMES:
            return False
        if self._check_inp_lineage_skip(inp, virtual_lineage, upstream_has_modifications, simulation_trace_codes):
            return False
        if inp in vars_derived_from_loops and not upstream_has_modifications and not loop_derived_trust_overridden:
            if inp in self.shell.user_ns:
                if self.debug:
                    logger.debug("[UPSTREAM] Input '%s' is loop-derived and code matches disk. Trusting in-memory.", inp)
                return False
            if self.debug:
                logger.debug("[UPSTREAM] Input '%s' is loop-derived but NOT in memory. Scheduling re-execution.", inp)
        return True

    def _cascade_failed_restore_inputs(
        self,
        i: int,
        stmt_code: str,
        outputs: set[str],
        needed_outputs: set[str],
        restored_vars: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        vars_derived_from_loops: set[str],
        loop_derived_trust_overridden: bool,
        upstream_has_modifications: bool,
        simulation_trace_codes: set[str],
        needed_vars: set[str],
        resolved_vars: set[str],
        stmts_to_run_indices: list[int],
    ) -> None:
        """Schedule *stmt* for re-execution and cascade its unresolved inputs."""
        stmts_to_run_indices.append(i)
        stmt_inputs, _ = CodeAnalyzer.analyze_code_block(stmt_code)
        for inp in stmt_inputs:
            if inp in resolved_vars or inp in needed_vars:
                continue
            if self._should_add_input_to_needed(
                inp, virtual_lineage, virtual_modules, vars_derived_from_loops,
                loop_derived_trust_overridden, upstream_has_modifications,
                simulation_trace_codes, needed_vars,
            ):
                needed_vars.add(inp)

        outputs_only = outputs - set(stmt_inputs)
        if outputs_only:
            removed = outputs_only & needed_vars
            if removed:
                needed_vars -= removed
                resolved_vars.update(removed)
                if self.debug:
                    logger.debug("[UPSTREAM] Scheduled stmt [%s] will produce %s. Removing from needed_vars.", i, removed)
        if self.debug:
            logger.debug("[UPSTREAM] Virtual Restore FAILED for: %s. Needed: %s, Restored: %s", stmt_code[:40], needed_outputs, restored_vars)

    def _backward_scan_pass(
        self,
        simulation_trace: list,
        broken_vars: set[str],
        vars_tainted_by_upstream_mismatch: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        vars_derived_from_loops: set[str],
        loop_derived_trust_overridden: bool,
        upstream_has_modifications: bool,
        simulation_trace_codes: set[str],
        stmt_lookup_times: dict[str, float],
    ) -> tuple[list[int], list[dict], float]:
        """Scan the simulation trace backwards to build the re-execution schedule.

        Returns (stmts_to_run_indices, restored_statements_info, total_restore_time).
        """
        resolved_vars: set[str] = set()
        needed_vars: set[str] = set(broken_vars)
        stmts_to_run_indices: list[int] = []
        restored_statements_info: list[dict] = []

        stmt_positions: dict[str, int] = {}
        for i, (stmt_code, _, _, _, _, _) in enumerate(simulation_trace):
            stmt_positions[stmt_code] = i
        total_restore_time = 0.0

        for i in range(len(simulation_trace) - 1, -1, -1):
            stmt_code, outputs, inputs, input_hashes, produced_lineages, _ = simulation_trace[i]

            is_needed = any(out in needed_vars for out in outputs)
            if not is_needed:
                continue

            needed_outputs_pre = outputs.intersection(needed_vars)
            force_reexecute = bool(needed_outputs_pre & vars_tainted_by_upstream_mismatch)

            if force_reexecute:
                _, restore_time, saved_time, handled = self._resolve_tainted_stmt(
                    i, stmt_code, outputs, needed_outputs_pre, virtual_lineage, virtual_modules,
                    upstream_has_modifications, simulation_trace_codes, needed_vars, resolved_vars,
                    stmts_to_run_indices,
                )
                if handled:
                    continue
                restored_vars: set[str] = set()
                restore_time = 0.0
                saved_time = 0.0
            else:
                restored_vars, restore_time, saved_time = self._virtual_lineage._try_virtual_restore(
                    stmt_code, outputs, inputs, input_hashes, virtual_modules, expected_lineages=produced_lineages,
                )
            total_restore_time += restore_time

            needed_outputs = outputs.intersection(needed_vars)
            if needed_outputs and needed_outputs.issubset(restored_vars):
                if self.debug:
                    logger.debug("[UPSTREAM] Virtual Restore SUCCESS for: %s", stmt_code[:40])
                lookup_time_for_stmt = stmt_lookup_times.get(stmt_code, 0.0)
                restored_statements_info.append({
                    'code': stmt_code,
                    'restored_vars': list(restored_vars),
                    'status': CacheStatus.RESTORED,
                    'is_upstream': True,
                    'source': 'DISK',
                    'saved_time': saved_time,
                    'total_time': restore_time + lookup_time_for_stmt,
                    'position': stmt_positions.get(stmt_code, 999999),
                })
                needed_vars.difference_update(restored_vars)
                resolved_vars.update(restored_vars)
            else:
                self._cascade_failed_restore_inputs(
                    i, stmt_code, outputs, needed_outputs, restored_vars,
                    virtual_lineage, virtual_modules, vars_derived_from_loops,
                    loop_derived_trust_overridden, upstream_has_modifications,
                    simulation_trace_codes, needed_vars, resolved_vars, stmts_to_run_indices,
                )

        return stmts_to_run_indices, restored_statements_info, total_restore_time

    def _check_missing_required_inputs(
        self,
        required_inputs: set[str] | None,
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        broken_vars: set[str],
    ) -> None:
        """Mark required inputs that exist in virtual lineage but are absent from memory."""
        utility_vars = {'ip', 'cash_magics', 'get_ipython', '__builtins__', 'In', 'Out'}
        for var_name in (required_inputs or []):
            if var_name not in virtual_lineage or var_name in self.variable_lineage:
                continue
            # Skip if this is a module â€” BUT only if it's actually in memory!
            # After a kernel restart, imported modules may be missing.
            if var_name in virtual_modules:
                if var_name in self.shell.user_ns:
                    if self.debug:
                        logger.debug("[UPSTREAM_DEBUG] Skipping missing module '%s' (already in memory)", var_name)
                    continue
                if self.debug:
                    logger.debug("[UPSTREAM_DEBUG] Module '%s' is not in memory. Marking as broken for re-import.", var_name)
                broken_vars.add(var_name)
                continue

            if var_name in utility_vars or var_name.startswith('_'):
                if self.debug:
                    logger.debug("[UPSTREAM_DEBUG] Skipping utility variable '%s'", var_name)
                continue

            if self.debug:
                logger.debug("[UPSTREAM_DEBUG] Required input '%s' should exist but is missing from memory. Virtual lineage: %s", var_name, virtual_lineage.get(var_name)[:8])
            logger.debug("[UPSTREAM] Variable '%s' should exist but is missing.", var_name)
            broken_vars.add(var_name)

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

        broken_vars, simulation_trace_codes, vars_tainted = self._run_pass2_identify_broken_vars(
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

