"""Code run in the kernel but not yet saved to the notebook.

The simulation reads the saved notebook; the kernel ran what the user typed.
:class:`UnsavedEdits` tells an unsaved edit that only extends the saved code
(kept) from one that changes what a name holds (tainted), and puts the
extensions back after an upstream re-run.
"""

from __future__ import annotations

import ast
import logging
from typing import TYPE_CHECKING

from cash.control_markers import strip_markers

from ...analysis.code_analyzer import CodeAnalyzer, clean_cell_source, parse_cell_source
from ..cache_key import (
    statement_source_hash,
)
from ._types import normalize_stmt

if TYPE_CHECKING:
    from .virtual_lineage import VirtualLineage


__all__ = ["UnsavedEdits"]

logger = logging.getLogger(__name__)


class UnsavedEdits:
    """The upstream check's handling of unsaved edits."""

    def __init__(self, virtual_lineage: VirtualLineage) -> None:
        self.virtual_lineage = virtual_lineage

    def compute_tainted_vars_from_unsaved_edits(
        self,
        virtual_lineage: dict[str, str],
        simulation_trace: list,
        simulation_trace_codes: set[str],
        current_cell_idx: int,
        notebook_cells: list[str],
    ) -> set[str]:
        """Identify variables transitively tainted by an unsaved upstream edit.

        When a user executes modified code without saving, the in-memory variable
        lineage diverges from the simulation's virtual lineage.  This method:

        1. Finds variables whose memory lineage differs from the virtual lineage
           AND whose producing code is traceable to disk (directly mismatched).
        2. Propagates that taint forward through the simulation trace so that
           downstream dependents are also flagged as stale.

        The directly-mismatched variables themselves are NOT included in the
        returned set — they are handled by the backward scan which has full
        unsaved-edit context.  Only their transitive dependents are returned.
        """
        directly_mismatched_vars = self._find_directly_mismatched_vars(
            virtual_lineage, simulation_trace_codes, current_cell_idx, notebook_cells
        )

        if not directly_mismatched_vars:
            return set()

        # Walk simulation trace forward: taint outputs of any statement whose
        # inputs include a directly-mismatched or already-tainted variable.
        # Directly-mismatched vars themselves stay out of the taint set so the
        # backward scan handles them with proper unsaved-edit trust logic.
        vars_tainted: set[str] = set()
        propagation_sources = set(directly_mismatched_vars)
        for entry in simulation_trace:
            if entry.inputs & propagation_sources:
                vars_tainted.update(entry.outputs - directly_mismatched_vars)
                propagation_sources.update(entry.outputs)

        if logger.isEnabledFor(logging.DEBUG) and vars_tainted:
            logger.debug(
                "[UPSTREAM_DEBUG] Transitive mismatch propagation (unsaved edit): "
                "root mismatches = %s, tainted dependents = %s",
                directly_mismatched_vars,
                vars_tainted,
            )
        return vars_tainted

    def _find_directly_mismatched_vars(
        self,
        virtual_lineage: dict[str, str],
        simulation_trace_codes: set[str],
        current_cell_idx: int,
        notebook_cells: list[str],
    ) -> set[str]:
        """Return variables whose in-memory lineage differs from virtual lineage.

        A variable is included when its producing code is on disk (traceable),
        so its mismatch represents an unsaved upstream edit rather than a
        downstream or external mutation.
        """
        directly_mismatched: set[str] = set()
        for vname in virtual_lineage:
            if vname not in self.virtual_lineage.tracking_state.variable_lineage:
                continue
            if virtual_lineage[vname] == self.virtual_lineage.tracking_state.variable_lineage[vname]:
                continue
            producing_code = self.virtual_lineage.tracking_state.executed_cell_codes.get(vname)
            if producing_code is None:
                continue
            normalized_prod = strip_markers(producing_code).strip()
            if normalized_prod in simulation_trace_codes:
                directly_mismatched.add(vname)
            elif vname in virtual_lineage:
                # Written last by this cell or one below it: ahead of the cells
                # above, not an edit to them. This cell counts: re-running one
                # that adds a column to a frame from above
                # (``docs['topic'] = ...``) found the frame ahead by its own
                # earlier write and rebuilt everything derived from it.
                is_downstream = any(
                    normalized_prod in notebook_cells[di] for di in range(current_cell_idx, len(notebook_cells))
                )
                if not is_downstream:
                    directly_mismatched.add(vname)
        return directly_mismatched

    def is_valid_extension(
        self, code: str, actual_lineage: str, virtual_lineage: dict[str, str], required_dependency: str | None = None
    ) -> bool:
        """
        Check if the 'actual_lineage' is a valid derivation from the 'virtual_lineage' state.
        This verifies if executing 'code' using 'virtual_lineage' inputs produces 'actual_lineage'.
        True implies the actual state is a valid extension (e.g. unsaved downstream cell) of the notebook state.

        If required_dependency is provided, ensures that 'code' actually takes that variable as input.
        This distinguishes valid extensions (x = x + 1) from conflicting redefinitions (x = 5).
        """
        try:
            inputs, outputs = CodeAnalyzer.analyze_code_block(code)

            if required_dependency and required_dependency not in inputs:
                return False

            input_lineages = []
            # Note: CodeAnalyzer inputs are a set. We sort for deterministic hashing order.
            # However, the lineage hash construction below sorts them anyway.
            sorted_inputs = sorted(inputs)
            for inp in sorted_inputs:
                if self.virtual_lineage._unbound_builtin(inp, virtual_lineage):
                    continue

                if inp in virtual_lineage:
                    input_lineages.append(virtual_lineage[inp])
                elif inp in self.virtual_lineage.tracking_state.variable_lineage:
                    # Fallback to memory if virtual missing (external var not in notebook)
                    input_lineages.append(self.virtual_lineage.tracking_state.variable_lineage[inp])
                else:
                    # Input missing entirely. Cannot verify.
                    return False

            source_hash = statement_source_hash(code)
            # Route through the shared func-inclusive projection (matches the
            # recorder in statement/lineage.py) so an unsaved edit that calls a
            # user-defined function is not spuriously rejected for lacking the
            # function-source component. A hand-rolled sha256(code)+input_lineages
            # omitted it, so any function-routed edit always projected != recorded
            # and was wrongly discarded. [layer 1]
            projected = self.virtual_lineage._compute_virtual_output_lineages(
                source_hash, input_lineages, "", inputs, outputs, code
            )

            return actual_lineage in projected.values()

        except (KeyError, TypeError, ValueError, SyntaxError):
            return False

    def code_exists_in_notebook(self, mem_code: str, notebook_cells: list[str]) -> bool:
        """Return True if the normalized form of *mem_code* appears as a top-level
        statement in any notebook cell.

        Used when upstream modifications are detected to verify that an
        extension-validated variable's producing code has not been deleted or
        replaced.  On any parse/IO failure, returns False (conservative).
        """
        try:
            normalized_mem_code = normalize_stmt(mem_code)
            for cell_code in notebook_cells:
                if not clean_cell_source(cell_code).strip():
                    continue
                try:
                    cell_tree = parse_cell_source(cell_code)
                    if cell_tree is None:
                        continue
                    for node in cell_tree.body:
                        try:
                            node_code = normalize_stmt(ast.unparse(node))
                            if node_code == normalized_mem_code:
                                return True
                        except (ValueError, TypeError):
                            logger.debug("[UPSTREAM] Failed to unparse node during notebook code search")
                except (SyntaxError, ValueError):
                    logger.debug("[UPSTREAM] Failed to parse cell during notebook code search")
        except (OSError, SyntaxError, ValueError, ImportError):
            logger.debug("[UPSTREAM] Notebook code search failed, assuming code is gone")
        return False

    def reapply_unsaved_extensions(
        self,
        broken_vars: set[str],
        vars_updated_by_trace: set[str],
        simulation_trace: list,
        notebook_cells: list[str],
        statements_to_reexecute: list[str],
    ) -> None:
        """Re-apply unsaved extension code for broken variables.

        If a broken variable's producing code is NOT in the notebook (unsaved
        extension), schedule it for re-execution — unless the trace already
        updated that variable.
        """
        all_notebook_stmts = self._collect_notebook_statements(notebook_cells)

        for var_name in broken_vars:
            if var_name in vars_updated_by_trace:
                continue

            if var_name in self.virtual_lineage.tracking_state.executed_cell_codes:
                mem_code = self.virtual_lineage.tracking_state.executed_cell_codes[var_name]

                is_in_trace = False
                for entry in simulation_trace:
                    if entry.stmt_code.strip() == mem_code.strip():
                        is_in_trace = True
                        break

                if not is_in_trace:
                    if mem_code.strip() in all_notebook_stmts:
                        logger.debug("[UPSTREAM] Skipping downstream statement for '%s'", var_name)
                        continue

                    if mem_code not in statements_to_reexecute:
                        logger.debug("[UPSTREAM] Re-applying unsaved extension for '%s'", var_name)
                        statements_to_reexecute.append(mem_code)

    def _collect_notebook_statements(self, notebook_cells: list[str]) -> set[str]:
        """Collect all normalized statement codes from notebook cells.

        Used to distinguish downstream statements from unsaved extensions.
        """
        all_notebook_stmts: set[str] = set()
        for cell_code in notebook_cells:
            try:
                if clean_cell_source(cell_code).strip():
                    tree = parse_cell_source(cell_code)
                    if tree is not None:
                        for node in tree.body:
                            try:
                                stmt_code = ast.unparse(node)
                                all_notebook_stmts.add(stmt_code.strip())
                            except (TypeError, ValueError):
                                logger.debug("Failed to unparse AST node in notebook cell")
            except (SyntaxError, ValueError):
                logger.debug("Failed to parse notebook cell for unsaved extension check")
        return all_notebook_stmts
