from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from .analysis import CodeAnalyzer

if TYPE_CHECKING:
    from cash.notebook.mismatch_classifier import MismatchClassifier
    from cash.notebook.virtual_lineage import VirtualLineage

logger = logging.getLogger(__name__)


class ReexecutionPlanner:
    """Phase 3 of NotebookSimulator: build the re-execution plan.

    Holds references to VirtualLineage and MismatchClassifier so the
    planner can call into helper methods that still live on those
    phases (e.g. _check_loop_derived_trust_override, _backward_scan_pass,
    _collect_skipped_statement_metrics, _filter_accumulator_reinits,
    _reapply_unsaved_extensions). Pure-phase invariants land in a later
    refactor.
    """

    def __init__(
        self,
        virtual_lineage: "VirtualLineage",
        classifier: "MismatchClassifier",
        debug: bool = False,
    ) -> None:
        self._virtual_lineage = virtual_lineage
        self._classifier = classifier
        self.debug = debug

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
