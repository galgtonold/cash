from __future__ import annotations

"""Phase 2 of the notebook simulator: classify broken / tainted variables.

Extracted from ``NotebookSimulator``. Holds references to a
:class:`VirtualLineage` instance (for cache-probing and helper methods like
``_check_loop_derived_trust_override``) and to the shared ``TrackingState``
dicts. Pure-phase invariants land in a later refactor.
"""

import ast
import logging
import re
import types

from .._protocols import TrackingState
from ..analysis import CodeAnalyzer
from ..cacheability import analyze_statement
from ..cache_status import CacheStatus
from ._types import RestoreCollector, apply_collected_mutations
from .virtual_lineage import VirtualLineage, _BUILTIN_NAMES, _normalize_stmt

__all__ = ["MismatchClassifier"]

logger = logging.getLogger(__name__)


class MismatchClassifier:
    """Phase 2 of NotebookSimulator: classify broken / tainted variables.

    Holds references to a :class:`VirtualLineage` instance (for cache-probing
    and helper methods like ``_check_loop_derived_trust_override``) and to the
    shared ``TrackingState`` dicts. Pure-phase invariants land in a later
    refactor.
    """

    def __init__(
        self,
        virtual_lineage: VirtualLineage,
        tracking_state: TrackingState,
        debug: bool = False,
    ) -> None:
        self._virtual_lineage = virtual_lineage
        self.debug = debug
        self.set_tracking_state(tracking_state)

        # Buffered TrackingState mutations; orchestrator drains after the phase.
        self._restores = RestoreCollector()

    def set_tracking_state(self, state: TrackingState) -> None:
        """Re-wire shared state refs (mirrors NotebookSimulator.set_tracking_state)."""
        self._tracking_state = state
        self.executed_cell_codes = state.executed_cell_codes
        self.executed_cell_hashes = state.executed_cell_hashes
        self.variable_lineage = state.variable_lineage
        self.lineage = state.lineage
        self.executed_file_deps = state.executed_file_deps
        self.vars_with_mutation_lineage = state.vars_with_mutation_lineage
        self.executed_input_lineages = state.executed_input_lineages

    # --- shell/cash convenience accessors (read-through to VirtualLineage) ---

    @property
    def shell(self):
        return self._virtual_lineage.shell

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
                # A reassignment accumulator (``total = total + b``) is produced by
                # a per-iteration body statement, so its recorded code carries the
                # CAS-86 ``# __iteration_context__:`` marker while the simulation
                # trace codes are stored stripped. Strip it here too (mirroring
                # _check_loop_derived_trust_override) or the marked code never
                # matches and the accumulator is falsely treated as overwritten
                # downstream, defeating the loop trust. [CAS-120]
                normalized_exec_code = re.sub(
                    r'# __iteration_context__:.*?\n', '', exec_code
                ).strip()
                if (
                    exec_code not in simulation_trace_codes
                    and normalized_exec_code not in simulation_trace_codes
                ):
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
        # CAS-165/166: a bare ``est.fit(X, y)`` receiver has a SELF-REFERENTIAL
        # key. CAS-138 adds it to the statement's OUTPUTS (so the fit bumps its
        # lineage) while it is also an INPUT (its pre-fit lineage pins the key).
        # On a warm isolated re-run the bumped lineage is "ahead" of the virtual
        # (constructor) lineage exactly like the downstream-advancement case, but
        # the receiver has no Store target so it is NOT in ``current_cell_outputs``
        # -> without this branch it hits the "read-only input: reject" path below,
        # whose reset-to-L0 is an incidental side effect of a full upstream
        # re-derivation. That side channel DESYNCS for a pandas / >8 MiB input, so
        # the fit perpetually MISSES and re-serialises the model every run. Reset
        # the receiver's lineage to the virtual (simulated-constructor) lineage:
        # CHEAP and deterministic, so the key is stable across warm re-runs (HIT),
        # yet a constructor EDIT changes the virtual lineage and still forces a
        # re-fit. Gated on ``not upstream_has_modifications`` so a real upstream /
        # constructor edit falls through to the value-refreshing re-derivation
        # (a lineage-only reset there would leave the STALE estimator object in
        # ``user_ns`` and serve a wrong result). Fit-only: ``partial_fit`` is
        # cumulative, so a lineage-only reset would double-count on a miss -- it
        # keeps the value-safe path.
        if (required_inputs and var_name in required_inputs
                and not upstream_has_modifications
                and self._is_estimator_fit_selfref(var_name)):
            if self.debug:
                logger.debug(
                    "[UPSTREAM_DEBUG]   -> '%s' is a bare estimator .fit() receiver "
                    "(self-referential CAS-138 key). Resetting lineage from %s to "
                    "virtual %s for a stable warm-re-run cache hit.",
                    var_name, actual_lineage[:8], final_virtual_hash[:8],
                )
            self._restores.record_lineage_reset(var_name=var_name, lineage_hash=final_virtual_hash)
            return True

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

        # Self-modifying no-lineage output of a single-unit (while / with) loop
        # in the current cell. Its recorded lineage is legitimately "ahead" — it
        # is the cell's OWN loop output, not downstream advancement — but unlike
        # a regular self-assign the runtime's value-based loop lineage does not
        # match the simulator's projection, so the collapse branch below would
        # reset the recorded lineage to the cell-entry base. For a no-lineage var
        # (int / list / set: no ``_cash_lineage_hash`` escape hatch) that makes
        # ``executed_input_lineages[var][var] == variable_lineage[var]`` and the
        # downstream stale-value guard then declines, so the loop re-accumulates
        # (or, with a control var pinned, never re-runs) on an isolated re-run.
        # Mark it broken directly so its producer restores the cell-entry base
        # and the loop recomputes from scratch — mirroring the for-loop path,
        # whose per-iteration capture keeps the guard's base distinct. Scoped to
        # input∩output, so it fires only when re-running the loop cell itself,
        # never when a downstream cell merely reads the var. [CAS-59]
        if (required_inputs and var_name in required_inputs
                and current_cell_outputs and var_name in current_cell_outputs
                and self._is_singleunit_loop_nolineage_selfmod(var_name)):
            if self.debug:
                logger.debug("[UPSTREAM_DEBUG]   -> '%s' is a no-lineage self-modifying "
                      "output of a single-unit (while/with) loop. Marking broken so the "
                      "loop recomputes from its cell-entry base on isolated re-run.", var_name)
            broken_vars.add(var_name)
            return True

        # Downstream advancement: if var is also a current-cell output reset lineage.
        if required_inputs and var_name in required_inputs and current_cell_outputs and var_name in current_cell_outputs:
            if self.debug:
                logger.debug("[UPSTREAM_DEBUG]   -> '%s' is also an OUTPUT of the current cell. "
                      "Lineage is ahead due to downstream advancement. "
                      "Resetting lineage from %s to virtual %s.",
                      var_name, actual_lineage[:8], final_virtual_hash[:8])
            self._restores.record_lineage_reset(var_name=var_name, lineage_hash=final_virtual_hash)
            # Caller (_classify_broken_vars) drains between iterations so the
            # reset is visible to subsequent classification iterations.
            return True

        return False

    def _is_singleunit_loop_nolineage_selfmod(self, var_name: str) -> bool:
        """True if *var_name* is a no-lineage var self-modified by a single-unit loop.

        Detects the CAS-59 shape: the variable's producing statement is a
        ``while`` or ``with`` block (executed as one opaque unit, unlike a ``for``
        loop's per-iteration replay) that writes the variable in place or
        re-binds it each pass — ``n += 1``, ``total += n``, ``acc.append(..)``,
        or a walrus in the condition (``while (n := n + 1) <= 5``) — AND the live
        value carries no ``_cash_lineage_hash``. Lineage-carrying receivers
        (DataFrame / Series) are excluded — they reset correctly through the
        value-lineage path and must keep it.

        The caller only reaches this for a var that is already both a required
        input and a current-cell output, so for a single-unit loop the var is
        genuinely self-referential across iterations. We confirm the loop writes
        it via ``all_mutated_vars`` (in-place mutation, incl. method receivers
        the output analysis misses) OR the static output set (Name re-bind /
        walrus target the mutation visitor misses).
        """
        live = self.shell.user_ns.get(var_name)
        if getattr(live, '_cash_lineage_hash', None) is not None:
            return False
        code = self.executed_cell_codes.get(var_name)
        if not code:
            return False
        try:
            tree = ast.parse(code.strip())
        except (SyntaxError, ValueError):
            return False
        if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.While, ast.With)):
            return False
        try:
            if var_name in analyze_statement(code, None).all_mutated_vars:
                return True
            _, outputs = CodeAnalyzer.analyze_code_block(code)
            return var_name in outputs
        except (SyntaxError, ValueError, TypeError):
            return False

    def _is_estimator_fit_selfref(self, var_name: str) -> bool:
        """True if *var_name* was last produced by a bare ``var_name.fit(...)`` on
        a live sklearn-style estimator.

        A bare ``clf.fit(X, y)`` is routed to CACHING with a self-referential key:
        CAS-138 adds ``clf`` to the statement's outputs (so the fit bumps its
        lineage) while ``clf`` is also an input (its pre-fit lineage pins the key),
        so ``executed_cell_codes['clf']`` records the bare-fit statement. The
        caller only reaches here on a lineage mismatch with the upstream
        UNMODIFIED, which for a fit-produced receiver can only happen when the
        current cell IS that bare fit -- a downstream reader would have the fit in
        its simulated upstream, so its virtual lineage would already match. That
        makes the recorded-code check a reliable "is the current cell a bare
        estimator fit" signal without threading a new current-cell parameter.

        Scoped to ``fit`` ONLY. ``fit`` overwrites the estimator (re-running is
        idempotent) so a lineage-only reset is safe; ``partial_fit`` is CUMULATIVE,
        so a lineage-only reset while the partially-fitted object survives in
        ``user_ns`` would double-count on a miss -- it keeps the value-safe
        full-re-derivation path. The estimator duck-type (a callable ``fit`` AND a
        callable ``get_params``) mirrors ``StatementProcessor._estimator_fit_receivers``
        and excludes ``list.append`` / a generic object that merely exposes ``fit``.
        """
        code = self.executed_cell_codes.get(var_name)
        if not code:
            return False
        try:
            tree = ast.parse(code.strip())
        except (SyntaxError, ValueError):
            return False
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
            return False
        call = tree.body[0].value
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            return False
        if call.func.attr != 'fit':
            return False
        # Receiver must be the bare name itself (``clf.fit`` -> base 'clf'); a
        # chained/attribute receiver (``obj.model.fit``) is not this var.
        if not isinstance(call.func.value, ast.Name) or call.func.value.id != var_name:
            return False
        v = self.shell.user_ns.get(var_name)
        if isinstance(v, types.ModuleType):
            return False
        return callable(getattr(v, 'fit', None)) and callable(getattr(v, 'get_params', None))

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
            # Drain between iterations: a lineage reset buffered for this var
            # must be visible when classifying the remaining vars.
            apply_collected_mutations(self._restores, self._tracking_state)

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
            # Skip genuine builtins, but NOT a user variable that shadows a
            # builtin name (``sum = 10``) — such a name IS tracked in
            # variable_lineage and must have its freshness checked.
            if inp in _BUILTIN_NAMES and inp not in self.variable_lineage:
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
        # A user variable shadowing a builtin name (``sum = 10``) is tracked in
        # variable_lineage — fall through to the real freshness checks so its
        # producer is scheduled when stale, instead of assuming it is a builtin
        # that is always available.
        if inp in _BUILTIN_NAMES and inp not in self.variable_lineage:
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
                # Drain so subsequent iterations of this reverse-trace loop see
                # the lineage / file-dep writes buffered by the restore — next
                # statements may depend on the just-restored variable's lineage.
                apply_collected_mutations(
                    self._virtual_lineage._restores, self._tracking_state,
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
            if var_name not in virtual_lineage:
                continue
            # Modules need their own freshness check: simulating an `import`
            # statement writes the module's lineage into ``variable_lineage``
            # via ``_propagate_import_lineage`` — but after a real kernel
            # restart the module object itself is not in ``user_ns``. The
            # generic ``var_name in self.variable_lineage`` short-circuit
            # below would otherwise hide the missing-module case and the
            # import would never get scheduled for upstream re-execute.
            if var_name in virtual_modules:
                if var_name in self.shell.user_ns:
                    if self.debug:
                        logger.debug("[UPSTREAM_DEBUG] Skipping missing module '%s' (already in memory)", var_name)
                    continue
                if self.debug:
                    logger.debug("[UPSTREAM_DEBUG] Module '%s' is not in memory. Marking as broken for re-import.", var_name)
                broken_vars.add(var_name)
                continue
            # Gate on the LIVE namespace, not the tracking dict. A ``del x`` (or
            # ``%reset``) removes ``x`` from ``user_ns`` but leaves
            # ``variable_lineage['x']`` behind; the old ``in self.variable_lineage``
            # short-circuit therefore hid the missing input and never scheduled the
            # producer to rebuild it. ``virtual_lineage`` is already
            # position-scoped by the simulator (a del/%reset ABOVE the target pops
            # the name; one BELOW is never simulated), so an input that survives to
            # here yet is absent from memory must be reconstructed.
            if var_name in self.shell.user_ns:
                continue

            if var_name in utility_vars or var_name.startswith('_'):
                if self.debug:
                    logger.debug("[UPSTREAM_DEBUG] Skipping utility variable '%s'", var_name)
                continue

            if self.debug:
                logger.debug("[UPSTREAM_DEBUG] Required input '%s' should exist but is missing from memory. Virtual lineage: %s", var_name, virtual_lineage.get(var_name)[:8])
            logger.debug("[UPSTREAM] Variable '%s' should exist but is missing.", var_name)
            broken_vars.add(var_name)
