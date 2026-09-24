"""Phase 2 of the notebook simulator: which names memory cannot be trusted for.

:meth:`MismatchClassifier.classify` compares a :class:`SimulationResult` with
the live lineages and returns a :class:`ClassificationResult`;
:meth:`MismatchClassifier.backward_scan_pass` then walks the trace back from
the broken names, restoring what the cache holds and scheduling the rest.
"""

from __future__ import annotations

import ast
import functools
import logging
import types
from dataclasses import dataclass, field

from cash.control_markers import strip_markers

from ...analysis.cacheability import analyze_statement
from ...analysis.cacheability_decision import is_lineage_exempt, receiver_is_identity_coupled
from ...analysis.code_analyzer import CodeAnalyzer
from ...analysis.namespace_effects import is_estimator
from ...value_types import BUILTIN_NAMES
from .._protocols import TrackingState
from .._trace import trace_event
from ..cache_key import statement_source_hash
from ..cache_status import CacheStatus
from ._types import (
    CellCheck,
    ClassificationResult,
    SimulationResult,
)
from .virtual_lineage import VirtualLineage, normalize_stmt

__all__ = ["MismatchClassifier"]

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=4096)
def import_only(stmt_code: str) -> bool:
    """Is *stmt_code* nothing but imports? Such a statement is re-run, never
    restored from the cache: a restored ``from helper import summary`` hands
    back the function object it bound when it was stored -- the RAM tier keeps
    functions by reference -- so after an edit to ``helper.py`` the pre-edit
    function came back, and everything keyed on it matched its pre-edit entry.
    Found as the intermittent ``test_a_from_import`` failure: it needs the
    import to be slow enough to have been stored, which a loaded machine made
    it, and which a helper doing real work at import time always is.
    """
    try:
        body = ast.parse(stmt_code).body
    except SyntaxError:
        return False
    return bool(body) and all(isinstance(node, ast.Import | ast.ImportFrom) for node in body)


@dataclass
class _BackwardScan:
    """Progress of one backward scan over the trace."""

    needed: set[str]
    """Names still to be produced by a statement above."""

    resolved: set[str] = field(default_factory=set)
    """Names already produced by a scheduled or restored statement."""

    run: list[int] = field(default_factory=list)
    """Trace indices scheduled to re-run, in the order found."""


class MismatchClassifier:
    """Phase 2 of NotebookSimulator: classify broken / tainted variables.

    Restores through, and asks trace questions of, the :class:`VirtualLineage`
    it is given. A lineage reset goes to ``TrackingState`` at once, so the
    names classified after it see it.
    """

    def __init__(
        self,
        virtual_lineage: VirtualLineage,
        tracking_state: TrackingState,
    ) -> None:
        self.virtual_lineage = virtual_lineage
        self.tracking_state = tracking_state

    # --- shell/cash convenience accessors (read-through to VirtualLineage) ---

    @property
    def shell(self):
        return self.virtual_lineage.shell

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
            actual_inp_lineage = self.tracking_state.variable_lineage.get(inp_name)
            if actual_inp_lineage and actual_inp_lineage != expected_lineage:
                logger.debug(
                    "[UPSTREAM_DEBUG] Loop input '%s' lineage changed: virtual=%s, actual=%s",
                    inp_name,
                    expected_lineage[:8],
                    actual_inp_lineage[:8],
                )
                return True
        return self._built_on_an_older_input(var_name, vars_derived_from_loops, loop_target_vars)

    def _built_on_an_older_input(
        self,
        var_name: str,
        vars_derived_from_loops: set[str],
        loop_target_vars: set[str],
    ) -> bool:
        """True when *var_name*, or a loop-derived value it was built from, was
        built on an input that has been rebuilt since.

        The input check above compares simulated lineages with live ones, and a
        loop-derived chain agrees with itself there even when nothing in it was
        re-run: after the sweep re-ran on edited data, ``best = pick(sweep)``
        and ``best_scores = f(best)`` still looked current.
        What each value was built from is recorded when it ran, so
        compare that with the live lineage instead. Walks the loop-derived
        inputs only; everything else gets the lineage checks.
        """
        # A function or class reads the globals it names when it RUNS, so it
        # holds no copy of them to go stale. `def draw_importance(ax)` records
        # `results` among its inputs; taking that as "built on an older results"
        # rebuilt the chart functions after the comparison cell re-ran, with
        # `results = {}` and not the loop that fills it: UpstreamStateError,
        # 'logreg', on a run order with no edit. A VALUE computed by
        # calling one is still walked through it: `best = score(1)` read `rows`.
        user_ns = self.shell.user_ns
        if isinstance(user_ns.get(var_name), (types.FunctionType, type)):
            return False
        seen: set[str] = set()
        todo = [var_name]
        while todo:
            name = todo.pop()
            if name in seen:
                continue
            seen.add(name)
            for inp, built_on in (self.tracking_state.executed_input_lineages.get(name) or {}).items():
                if inp == name or inp in loop_target_vars:
                    continue
                live = self.tracking_state.variable_lineage.get(inp)
                if live is not None and live != built_on:
                    trace_event(
                        "built_on_older_input",
                        var=var_name,
                        via=name,
                        input=inp,
                        built_on=str(built_on)[:12],
                        live=str(live)[:12],
                    )
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "[UPSTREAM_DEBUG] '%s' was built on an older '%s' (%s, now %s)",
                            name,
                            inp,
                            str(built_on)[:8],
                            live[:8],
                        )
                    return True
                if inp in vars_derived_from_loops:
                    todo.append(inp)
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
        sim: SimulationResult,
    ) -> set[str]:
        """Return data inputs whose virtual and actual lineages differ."""
        mismatched: set[str] = set()
        for inp in data_inputs:
            if inp == var_name:
                continue  # skip self-referential input (e.g., df['x'] = f(df))
            if inp in sim.vars_derived_from_loops:
                continue  # loop-derived, expected mismatch
            if inp in sim.loop_target_vars:
                continue  # loop iteration target, expected mismatch
            if (
                inp in sim.virtual_lineage
                and inp in self.tracking_state.variable_lineage
                and sim.virtual_lineage[inp] != self.tracking_state.variable_lineage[inp]
            ):
                mismatched.add(inp)
        return mismatched

    def _check_code_matches_loop_trust(self, var_name: str, last_stmt_for_var: str, sim: SimulationResult) -> bool:
        """Return True if var_name should be trusted in-memory when code matches trace.

        Called only when upstream_has_modifications is False and loop-derived
        check applies.  Checks that no non-loop data inputs have a lineage
        mismatch.
        """
        try:
            stmt_inputs, _ = CodeAnalyzer.analyze_code_block(last_stmt_for_var)
            data_inputs = self._collect_non_module_inputs(stmt_inputs, sim.virtual_modules)
            mismatched_inputs = self._find_mismatched_data_inputs(var_name, data_inputs, sim)
            if not mismatched_inputs:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "[UPSTREAM_DEBUG]   -> Code matches, required input '%s' mismatch due to "
                        "loop-derived inputs %s. "
                        "Trusting in-memory value (upstream unchanged).",
                        var_name,
                        data_inputs & sim.vars_derived_from_loops,
                    )
                return True
        except (KeyError, ValueError, TypeError):
            logger.debug("[UPSTREAM] Failed to check loop-derived inputs for '%s'", var_name)
        return False

    def _check_var_extension_valid(
        self,
        var_name: str,
        actual_lineage: str,
        sim: SimulationResult,
        notebook_cells: list[str],
    ) -> bool:
        """Return True if the in-memory value is a valid downstream extension."""
        if var_name not in self.tracking_state.executed_cell_codes:
            return False
        mem_code = self.tracking_state.executed_cell_codes[var_name]
        if not self.virtual_lineage.is_valid_extension(
            mem_code, actual_lineage, sim.virtual_lineage, required_dependency=var_name
        ):
            return False
        if sim.upstream_has_modifications:
            code_still_in_notebook = self.virtual_lineage.code_exists_in_notebook(mem_code, notebook_cells)
            if code_still_in_notebook:
                logger.debug("[UPSTREAM_DEBUG]   -> Valid extension (code still exists in notebook), keeping")
                return True
            logger.debug(
                "[UPSTREAM_DEBUG]   -> Extension code no longer exists in notebook (modified/deleted upstream). Rejecting."
            )
            return False
        logger.debug("[UPSTREAM_DEBUG]   -> Valid extension (no upstream modifications), keeping")
        logger.debug("[UPSTREAM] Variable '%s' is a valid extension of notebook state. Keeping.", var_name)
        return True

    def _handle_mismatch_code_matches(
        self,
        var_name: str,
        last_stmt_for_var: str | None,
        sim: SimulationResult,
        check: CellCheck,
        result: ClassificationResult,
    ) -> bool:
        """Handle the code-matches-but-lineage-differs case.

        Returns True if the caller should stop processing this variable (either
        the variable was trusted or marked broken), False if processing should
        continue (code did not match).
        """
        if var_name not in self.tracking_state.executed_cell_codes:
            return False
        last_stmt_for_var_real = last_stmt_for_var
        if last_stmt_for_var_real is None:
            return False
        sim_code = normalize_stmt(last_stmt_for_var_real)
        mem_code = normalize_stmt(self.tracking_state.executed_cell_codes[var_name])
        if sim_code != mem_code:
            return False

        # Code matches simulation but lineage differs.
        required_inputs = check.required_inputs
        if (
            required_inputs
            and var_name in required_inputs
            and sim.vars_derived_from_loops
            and not sim.upstream_has_modifications
            and not result.loop_derived_trust_overridden
        ):
            if self._check_code_matches_loop_trust(var_name, last_stmt_for_var_real, sim):
                return True
            logger.debug(
                "[UPSTREAM_DEBUG]   -> Code matches but '%s' is a REQUIRED INPUT with lineage mismatch. Marking as broken.",
                var_name,
            )
            logger.debug(
                "[UPSTREAM] Variable '%s' is a required input with lineage mismatch. Must re-execute.", var_name
            )
            result.broken_vars.add(var_name)
            return True
        # For non-required variables, trust the memory
        logger.debug(
            "[UPSTREAM_DEBUG]   -> Lineage mismatch but Code Matches trace (%s). Assuming valid extension due to cache miss. Keeping.",
            var_name,
        )
        logger.debug("[UPSTREAM] Variable '%s' mismatch but code matches trace. Keeping.", var_name)
        return True

    def _classify_one_broken_var(
        self,
        var_name: str,
        sim: SimulationResult,
        check: CellCheck,
        result: ClassificationResult,
        loop_var_input_lineages: dict[str, dict[str, str]],
    ) -> None:
        """Classify a single variable and add it to ``result.broken_vars`` if needed."""
        vars_derived_from_loops = sim.vars_derived_from_loops
        vars_mutated_by_loops = sim.vars_mutated_by_loops
        simulation_trace_codes = result.trace_codes
        # Skip loop-derived vars when upstream is unchanged AND producing code
        # is on disk (not overridden by unsaved edit). FAST MODE can't track
        # per-iteration lineage, so we trust in-memory state.
        # ...unless a file behind it changed: nothing upstream shows that, and
        # trusting memory then serves the value read from the old file.
        if (
            var_name in vars_derived_from_loops
            and not sim.upstream_has_modifications
            and not result.loop_derived_trust_overridden
            and var_name not in sim.vars_with_stale_files
        ):
            # Don't trust if the variable was overwritten by a downstream cell.
            # Check that executed_cell_codes for this var matches an upstream statement.
            overwritten_downstream = False
            exec_code = self.tracking_state.executed_cell_codes.get(var_name)
            if exec_code and simulation_trace_codes is not None:
                # A reassignment accumulator (``total = total + b``) is produced by
                # a per-iteration body statement, so its recorded code carries the
                # ``# __iteration_context__:`` marker while the simulation
                # trace codes are stored stripped. Strip it here too (mirroring
                # check_loop_derived_trust_override) or the marked code never
                # matches and the accumulator is falsely treated as overwritten
                # downstream, defeating the loop trust.
                normalized_exec_code = strip_markers(exec_code).strip()
                if exec_code not in simulation_trace_codes and normalized_exec_code not in simulation_trace_codes:
                    overwritten_downstream = True
                    logger.debug(
                        "[UPSTREAM_DEBUG] NOT trusting loop-derived '%s' — "
                        "executed code '%.40s' not in upstream simulation",
                        var_name,
                        exec_code,
                    )

            if not overwritten_downstream:
                input_lineages_for_var = loop_var_input_lineages.get(var_name, {})
                inputs_changed = self._check_loop_var_inputs_changed(
                    var_name,
                    input_lineages_for_var,
                    vars_derived_from_loops,
                    sim.loop_target_vars,
                )
                if not inputs_changed:
                    if logger.isEnabledFor(logging.DEBUG):
                        source = (
                            "directly mutated by loop"
                            if var_name in vars_mutated_by_loops
                            else "transitively derived from loop mutation"
                        )
                        logger.debug(
                            "[UPSTREAM_DEBUG] Skipping mismatch check for '%s' - %s, trusting in-memory state (upstream unchanged, inputs consistent)",
                            var_name,
                            source,
                        )
                    return
                if logger.isEnabledFor(logging.DEBUG):
                    source = (
                        "directly mutated by loop"
                        if var_name in vars_mutated_by_loops
                        else "transitively derived from loop mutation"
                    )
                    logger.debug(
                        "[UPSTREAM_DEBUG] NOT trusting '%s' (%s) — loop input lineage changed, will check lineage",
                        var_name,
                        source,
                    )

        actual_lineage = self.tracking_state.variable_lineage[var_name]
        virtual_lineage = sim.virtual_lineage
        if var_name not in virtual_lineage:
            logger.debug(
                "[UPSTREAM_DEBUG] Variable '%s' is in memory but not in virtual state (downstream or external)",
                var_name,
            )
            return

        final_virtual_hash = virtual_lineage[var_name]
        if actual_lineage == final_virtual_hash:
            if var_name in result.tainted_vars:
                logger.debug(
                    "[UPSTREAM_DEBUG] Lineage matches for '%s' but tainted by upstream mismatch. Marking broken.",
                    var_name,
                )
                result.broken_vars.add(var_name)
            return

        self._handle_lineage_mismatch(var_name, actual_lineage, final_virtual_hash, sim, check, result)

    def _is_saved_figure(self, var_name: str) -> bool:
        """Whether *var_name* holds a live figure whose last change was a bare
        ``var_name.savefig(...)``."""
        try:
            if not receiver_is_identity_coupled(self.shell.user_ns.get(var_name)):
                return False
            body = ast.parse((self.tracking_state.executed_cell_codes.get(var_name) or "").strip()).body
        except (SyntaxError, ValueError, TypeError, AttributeError):
            return False
        if len(body) != 1 or not isinstance(body[0], ast.Expr) or not isinstance(body[0].value, ast.Call):
            return False
        func = body[0].value.func
        return (
            isinstance(func, ast.Attribute)
            and func.attr == "savefig"
            and isinstance(func.value, ast.Name)
            and func.value.id == var_name
        )

    def _handle_mismatch_prereqs(
        self,
        var_name: str,
        actual_lineage: str,
        final_virtual_hash: str,
        sim: SimulationResult,
        check: CellCheck,
        broken_vars: set[str],
    ) -> bool:
        """Check early-exit conditions for a lineage mismatch.

        Returns True if the caller should stop processing this variable
        (it was already handled — marked broken, kept, or lineage reset).
        """
        required_inputs = check.required_inputs
        current_cell_outputs = check.current_cell_outputs
        upstream_has_modifications = sim.upstream_has_modifications
        # a bare ``est.fit(X, y)`` receiver has a SELF-REFERENTIAL
        # key. cash adds it to the statement's OUTPUTS (so the fit bumps its
        # lineage) while it is also an INPUT (its pre-fit lineage pins the key).
        # On a warm isolated re-run the bumped lineage is "ahead" of the virtual
        # (constructor) lineage exactly like the downstream-advancement case, but
        # the receiver has no Store target so it is NOT in ``current_cell_outputs``
        # -> without this branch it hits the "read-only input: reject" path below,
        # whose reset-to-L0 is an incidental side effect of a full upstream
        # re-derivation. That side channel DESYNCS for a pandas / sampled-file input, so
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
        if (
            required_inputs
            and var_name in required_inputs
            and not upstream_has_modifications
            and self._is_estimator_fit_selfref(var_name)
        ):
            logger.debug(
                "[UPSTREAM_DEBUG]   -> '%s' is a bare estimator .fit() receiver "
                "(self-referential key). Resetting lineage from %s to "
                "virtual %s for a stable warm-re-run cache hit.",
                var_name,
                actual_lineage[:8],
                final_virtual_hash[:8],
            )
            self.tracking_state.lineage.reset_to(var_name, final_virtual_hash)
            return True

        # A figure this cell saved (``fig.savefig(...)``) is ahead of its
        # simulated lineage for the same reason: the save counts as a change so
        # an edit to the plotted data still redraws and resaves, and the
        # simulation stops before the cell doing it. Taken as a downstream
        # change, the figure was rebuilt every time the cell ran again. Saving
        # draws nothing, so the live figure is current. A draw (``ax.plot``)
        # still rebuilds: re-running it would add artists.
        if (
            required_inputs
            and var_name in required_inputs
            and not upstream_has_modifications
            and self._is_saved_figure(var_name)
        ):
            self.tracking_state.lineage.reset_to(var_name, final_virtual_hash)
            return True

        # Read-only input: reject downstream mutations (e.g., df['SMA']=...)
        if (
            required_inputs
            and var_name in required_inputs
            and current_cell_outputs is not None
            and var_name not in current_cell_outputs
        ):
            logger.debug(
                "[UPSTREAM_DEBUG]   -> '%s' is a READ-ONLY required input "
                "(not in current cell outputs). Rejecting downstream extension "
                "to force restoration to upstream state.",
                var_name,
            )
            broken_vars.add(var_name)
            return True

        if self._check_var_extension_valid(var_name, actual_lineage, sim, check.notebook_cells):
            return True

        if var_name in sim.vars_with_stale_files:
            logger.debug(
                "[UPSTREAM_DEBUG]   -> Variable '%s' has stale file dependencies. Forcing re-execution.", var_name
            )
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
        # never when a downstream cell merely reads the var.
        if (
            required_inputs
            and var_name in required_inputs
            and current_cell_outputs
            and var_name in current_cell_outputs
            and self._is_singleunit_loop_nolineage_selfmod(var_name)
        ):
            logger.debug(
                "[UPSTREAM_DEBUG]   -> '%s' is a no-lineage self-modifying "
                "output of a single-unit (while/with) loop. Marking broken so the "
                "loop recomputes from its cell-entry base on isolated re-run.",
                var_name,
            )
            broken_vars.add(var_name)
            return True

        # Downstream advancement: if var is also a current-cell output reset lineage.
        if (
            required_inputs
            and var_name in required_inputs
            and current_cell_outputs
            and var_name in current_cell_outputs
        ):
            # ...but only when the cell's own last run explains the gap. After an
            # upstream edit the live value may be the cell's output built on the
            # OLD upstream frame: resetting its lineage to the new virtual one
            # kept that value, and ``docs['n_chars'] = ...`` printed the rows an
            # edited filter had removed.
            if upstream_has_modifications and not self._current_cell_reproduces(
                var_name, actual_lineage, sim, check.cell_code
            ):
                logger.debug(
                    "[UPSTREAM_DEBUG]   -> '%s' is written by the current cell, "
                    "but re-running it on the edited upstream state does not give "
                    "its live lineage. Marking broken.",
                    var_name,
                )
                broken_vars.add(var_name)
                return True
            logger.debug(
                "[UPSTREAM_DEBUG]   -> '%s' is also an OUTPUT of the current cell. "
                "Lineage is ahead due to downstream advancement. "
                "Resetting lineage from %s to virtual %s.",
                var_name,
                actual_lineage[:8],
                final_virtual_hash[:8],
            )
            self.tracking_state.lineage.reset_to(var_name, final_virtual_hash)
            return True

        return False

    def _current_cell_reproduces(
        self,
        var_name: str,
        actual_lineage: str,
        sim: SimulationResult,
        code: str | None,
    ) -> bool:
        """True when running the current cell (*code*) on the simulated
        cell-entry state gives *var_name* its live lineage -- the gap is the
        cell's own earlier run and nothing upstream. Simulated on copies;
        nothing is kept."""
        if not code:
            return True
        scratch = SimulationResult(virtual_lineage=dict(sim.virtual_lineage), virtual_modules=set(sim.virtual_modules))
        try:
            self.virtual_lineage.simulate_one_cell(scratch, -1, code)
        except Exception:  # noqa: BLE001 - cannot tell: treat as not reproduced
            logger.debug("[UPSTREAM] could not re-simulate the current cell for '%s'", var_name)
            return False
        return scratch.virtual_lineage.get(var_name) == actual_lineage

    def _is_singleunit_loop_nolineage_selfmod(self, var_name: str) -> bool:
        """True if *var_name* is a no-lineage var self-modified by a single-unit loop.

        Detects the shape: the variable's producing statement is a
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
        if getattr(live, "_cash_lineage_hash", None) is not None:
            return False
        code = self.tracking_state.executed_cell_codes.get(var_name)
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
        cash adds ``clf`` to the statement's outputs (so the fit bumps its
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
        full-re-derivation path. The estimator is the runtime's
        (``is_estimator``).
        """
        code = self.tracking_state.executed_cell_codes.get(var_name)
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
        if call.func.attr != "fit":
            return False
        # Receiver must be the bare name itself (``clf.fit`` -> base 'clf'); a
        # chained/attribute receiver (``obj.model.fit``) is not this var.
        if not isinstance(call.func.value, ast.Name) or call.func.value.id != var_name:
            return False
        return is_estimator(self.shell.user_ns.get(var_name))

    def _handle_lineage_mismatch(
        self,
        var_name: str,
        actual_lineage: str,
        final_virtual_hash: str,
        sim: SimulationResult,
        check: CellCheck,
        result: ClassificationResult,
    ) -> None:
        """Handle a confirmed lineage mismatch for *var_name*."""
        logger.debug(
            "[UPSTREAM_DEBUG] Lineage mismatch for '%s': virtual=%s, actual=%s",
            var_name,
            final_virtual_hash[:8],
            actual_lineage[:8],
        )

        if self._handle_mismatch_prereqs(var_name, actual_lineage, final_virtual_hash, sim, check, result.broken_vars):
            return

        last_stmt_for_var = None
        for entry in reversed(sim.trace):
            if var_name in entry.outputs:
                last_stmt_for_var = entry.stmt_code
                break

        if self._handle_mismatch_code_matches(var_name, last_stmt_for_var, sim, check, result):
            return

        if check.required_inputs and var_name in check.required_inputs:
            logger.debug(
                "[UPSTREAM_DEBUG]   -> Required input mismatch and INVALID extension. Forcing strict restoration"
            )
            logger.debug("[UPSTREAM] Variable '%s' is a required input mismatch. Forcing strict restoration.", var_name)
        logger.debug(
            "[UPSTREAM] Variable '%s' is broken. Exp: %s, Act: %s", var_name, final_virtual_hash[:8], actual_lineage[:8]
        )
        result.broken_vars.add(var_name)

    def classify(self, sim: SimulationResult, check: CellCheck) -> ClassificationResult:
        """Pass 2: the names *check*'s cell reads that memory cannot be trusted for."""
        virtual_lineage = sim.virtual_lineage
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[UPSTREAM_DEBUG] Simulation complete. Virtual lineage keys: %s", list(virtual_lineage.keys()))
            logger.debug(
                "[UPSTREAM_DEBUG] Actual variable_lineage keys: %s", list(self.tracking_state.variable_lineage.keys())
            )
            logger.debug("[UPSTREAM_DEBUG] Simulation trace has %s statements", len(sim.trace))
            if sim.vars_mutated_by_loops:
                logger.debug("[UPSTREAM_DEBUG] Variables mutated by loops (trusted): %s", sim.vars_mutated_by_loops)

        required_inputs = check.required_inputs
        vars_to_check = {name for name in required_inputs or () if name in self.tracking_state.variable_lineage}

        trace_codes = self.virtual_lineage.build_simulation_trace_codes(sim.trace)
        tainted: set[str] = set()
        if not sim.upstream_has_modifications:
            tainted = self.virtual_lineage.compute_tainted_vars_from_unsaved_edits(
                virtual_lineage,
                sim.trace,
                trace_codes,
                check.current_cell_idx,
                check.notebook_cells,
            )
        result = ClassificationResult(
            broken_vars=set(),
            tainted_vars=tainted,
            trace_codes=trace_codes,
            loop_derived_trust_overridden=self.virtual_lineage.check_loop_derived_trust_override(
                sim.upstream_has_modifications,
                sim.vars_mutated_by_loops,
                trace_codes,
            ),
        )
        loop_var_input_lineages = self.virtual_lineage.build_loop_var_input_lineages(
            sim.trace,
            sim.vars_derived_from_loops,
            virtual_lineage,
            sim.virtual_modules,
        )

        for var_name in vars_to_check:
            # A lineage reset it makes is seen by the names classified after it.
            self._classify_one_broken_var(var_name, sim, check, result, loop_var_input_lineages)

        # Only required inputs matter here; temporary intermediates can stay missing.
        self._check_missing_required_inputs(required_inputs, virtual_lineage, sim.virtual_modules, result.broken_vars)
        self._repair_upstream_rerun_bindings(required_inputs, sim.trace, result.broken_vars)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[UPSTREAM_DEBUG] Broken vars: %s", result.broken_vars)
            if not result.broken_vars:
                logger.debug("[UPSTREAM_DEBUG] No broken vars, nothing to re-execute")
        return result

    def _check_tainted_input_valid(self, inp: str, sim: SimulationResult, result: ClassificationResult) -> bool:
        """Return True if *inp* is an unsaved-edit input that should be trusted.

        This is called when inp has a lineage mismatch to decide if we can
        trust the in-memory value (produced by unsaved code) rather than
        cascading into the old disk code.
        """
        virtual_lineage = sim.virtual_lineage
        if inp not in virtual_lineage or inp not in self.tracking_state.variable_lineage:
            return False
        if self.tracking_state.variable_lineage[inp] == virtual_lineage[inp]:
            return False
        if sim.upstream_has_modifications:
            return False
        inp_producing_code = self.tracking_state.executed_cell_codes.get(inp)
        if inp_producing_code is None:
            return False
        normalized_inp_code = strip_markers(inp_producing_code).strip()
        return normalized_inp_code not in result.trace_codes and self._ran_the_notebook_version(inp, result.trace_codes)

    def _all_tainted_inputs_valid(self, stmt_code: str, sim: SimulationResult, result: ClassificationResult) -> bool:
        """Return True if all inputs for a tainted statement are available and fresh."""
        virtual_lineage = sim.virtual_lineage
        stmt_inputs_check, _ = CodeAnalyzer.analyze_code_block(stmt_code)
        for inp in stmt_inputs_check:
            if inp in sim.virtual_modules:
                if inp in self.shell.user_ns:
                    continue
                return False
            # Skip genuine builtins, but NOT a user variable that shadows a
            # builtin name (``sum = 10``) — such a name IS tracked in
            # variable_lineage and must have its freshness checked.
            if inp in BUILTIN_NAMES and inp not in self.tracking_state.variable_lineage:
                continue
            if inp not in self.shell.user_ns:
                return False
            if self._check_tainted_input_valid(inp, sim, result):
                logger.debug(
                    "[UPSTREAM] Input '%s' has different lineage but produced "
                    "by unsaved edit (code not on disk). Trusting in-memory value.",
                    inp,
                )
                continue
            if (
                inp in virtual_lineage
                and inp in self.tracking_state.variable_lineage
                and self.tracking_state.variable_lineage[inp] != virtual_lineage[inp]
            ):
                logger.debug(
                    "[UPSTREAM] Tainted stmt input '%s' has stale lineage (actual=%s, virtual=%s). Cascading.",
                    inp,
                    self.tracking_state.variable_lineage[inp][:8],
                    virtual_lineage[inp][:8],
                )
                return False
        return True

    def _resolve_tainted_stmt(
        self,
        i: int,
        stmt_code: str,
        outputs: set[str],
        needed_outputs_pre: set[str],
        sim: SimulationResult,
        result: ClassificationResult,
        scan: _BackwardScan,
    ) -> bool:
        """Schedule a tainted statement whose inputs are all in memory.

        True when it was scheduled; False when its inputs need cascading.
        """
        if self._all_tainted_inputs_valid(stmt_code, sim, result):
            scan.run.append(i)
            scan.needed.difference_update(outputs)
            scan.resolved.update(outputs - needed_outputs_pre)
            logger.debug("[UPSTREAM] Tainted stmt scheduled (inputs in memory): %s...", stmt_code[:60])
            return True
        logger.debug("[UPSTREAM] Tainted stmt, inputs missing, cascading: %s...", stmt_code[:60])
        return False

    def _ran_the_notebook_version(self, inp: str, simulation_trace_codes: set[str]) -> bool:
        """Whether the kernel has run, for *inp*, a statement the notebook holds now.

        A value built by code that is not in the notebook is trusted as an
        unsaved edit: the user changed a cell, ran it and has not saved. Then
        the saved version ran before it, so it is in the variable's history.
        The other way round it is not: an edit that was saved but never run
        leaves the old value in memory, and trusting it served it. A user edited
        ``models = {...}`` and ran a cell that needed only the edited function,
        then one whose back-test loop reads ``models``: the loop re-ran on the
        dict the old statement had built (silent; old and new code agreed).
        """
        history = self.tracking_state.executed_cell_hashes.get(inp)
        if not history:
            return False
        return any(statement_source_hash(code) in history for code in simulation_trace_codes)

    def _check_inp_lineage_skip(self, inp: str, sim: SimulationResult, result: ClassificationResult) -> bool:
        """Return True if *inp* should be skipped based on lineage/unsaved-edit checks."""
        virtual_lineage = sim.virtual_lineage
        if inp not in self.tracking_state.variable_lineage or inp not in virtual_lineage:
            return False
        if self.tracking_state.variable_lineage[inp] == virtual_lineage[inp]:
            if inp in self.shell.user_ns:
                logger.debug("[UPSTREAM] Input '%s' already valid in memory (lineage matches virtual). Skipping.", inp)
                return True
            return False
        # Lineage mismatch — check for unsaved edit
        if sim.upstream_has_modifications or inp not in self.shell.user_ns:
            return False
        inp_prod_code = self.tracking_state.executed_cell_codes.get(inp)
        if inp_prod_code is None:
            return False
        norm_code = strip_markers(inp_prod_code).strip()
        if norm_code not in result.trace_codes and self._ran_the_notebook_version(inp, result.trace_codes):
            logger.debug(
                "[UPSTREAM] Input '%s' lineage mismatch but produced by unsaved edit. Trusting in-memory.", inp
            )
            return True
        return False

    def _should_add_input_to_needed(
        self,
        inp: str,
        sim: SimulationResult,
        result: ClassificationResult,
        needed_vars: set[str],
    ) -> bool:
        """Return True if *inp* should be added to *needed_vars* during cascade.

        Handles module, builtin, lineage-matching, unsaved-edit, and loop-derived
        special cases.  Side-effect: may add *inp* to *needed_vars* for modules.
        """
        if inp in sim.virtual_modules:
            if inp not in self.shell.user_ns:
                needed_vars.add(inp)
                logger.debug(
                    "[UPSTREAM] Module '%s' is in virtual_modules but NOT in memory. Scheduling re-import.", inp
                )
            return False  # handled (either added or skipped)
        # A user variable shadowing a builtin name (``sum = 10``) is tracked in
        # variable_lineage — fall through to the real freshness checks so its
        # producer is scheduled when stale, instead of assuming it is a builtin
        # that is always available.
        if inp in BUILTIN_NAMES and inp not in self.tracking_state.variable_lineage:
            return False
        if self._check_inp_lineage_skip(inp, sim, result):
            return False
        if (
            inp in sim.vars_derived_from_loops
            and not sim.upstream_has_modifications
            and not result.loop_derived_trust_overridden
        ):
            if inp in self.shell.user_ns and not self._built_on_an_older_input(inp, sim.vars_derived_from_loops, set()):
                logger.debug("[UPSTREAM] Input '%s' is loop-derived and code matches disk. Trusting in-memory.", inp)
                return False
            logger.debug("[UPSTREAM] Input '%s' is loop-derived but NOT in memory. Scheduling re-execution.", inp)
        return True

    def _cascade_failed_restore_inputs(
        self,
        i: int,
        stmt_code: str,
        outputs: set[str],
        needed_outputs: set[str],
        restored_vars: set[str],
        sim: SimulationResult,
        result: ClassificationResult,
        scan: _BackwardScan,
    ) -> None:
        """Schedule *stmt* for re-execution and cascade its unresolved inputs."""
        scan.run.append(i)
        stmt_inputs, _ = CodeAnalyzer.analyze_code_block(stmt_code)
        # A callee's globals are inputs too, once the statement runs; the ones
        # missing from the kernel must be rebuilt first (absent_callee_globals).
        callee_names = self.virtual_lineage.absent_callee_globals(
            set(stmt_inputs), sim.virtual_lineage, sim.virtual_modules
        )
        for inp in [*stmt_inputs, *sorted(callee_names - set(stmt_inputs))]:
            if inp in scan.resolved or inp in scan.needed:
                continue
            if self._should_add_input_to_needed(inp, sim, result, scan.needed):
                scan.needed.add(inp)

        outputs_only = outputs - set(stmt_inputs)
        if outputs_only:
            removed = outputs_only & scan.needed
            if removed:
                scan.needed -= removed
                scan.resolved.update(removed)
                logger.debug("[UPSTREAM] Scheduled stmt [%s] will produce %s. Removing from needed_vars.", i, removed)
        logger.debug(
            "[UPSTREAM] Virtual Restore FAILED for: %s. Needed: %s, Restored: %s",
            stmt_code[:40],
            needed_outputs,
            restored_vars,
        )

    def backward_scan_pass(
        self, sim: SimulationResult, result: ClassificationResult
    ) -> tuple[list[int], list[dict], float]:
        """Scan the simulation trace backwards to build the re-execution schedule.

        Returns (stmts_to_run_indices, restored_statements_info, total_restore_time).
        """
        trace = sim.trace
        scan = _BackwardScan(needed=set(result.broken_vars))
        restored_statements_info: list[dict] = []

        stmt_positions = {entry.stmt_code: i for i, entry in enumerate(trace)}
        total_restore_time = 0.0

        for i in range(len(trace) - 1, -1, -1):
            entry = trace[i]
            stmt_code, outputs = entry.stmt_code, entry.outputs

            if not any(out in scan.needed for out in outputs):
                continue

            needed_outputs_pre = outputs.intersection(scan.needed)
            restored_vars: set[str] = set()
            restore_time = 0.0
            saved_time = 0.0
            if needed_outputs_pre & result.tainted_vars:
                if self._resolve_tainted_stmt(i, stmt_code, outputs, needed_outputs_pre, sim, result, scan):
                    continue
            elif not import_only(stmt_code):  # an import is re-run, never restored: see `import_only`
                restored_vars, restore_time, saved_time = self.virtual_lineage.try_virtual_restore(
                    stmt_code,
                    outputs,
                    entry.inputs,
                    entry.input_hashes,
                    sim.virtual_modules,
                    expected_lineages=entry.produced_lineages,
                )
            total_restore_time += restore_time

            needed_outputs = outputs.intersection(scan.needed)
            if needed_outputs and needed_outputs.issubset(restored_vars):
                logger.debug("[UPSTREAM] Virtual Restore SUCCESS for: %s", stmt_code[:40])
                restored_statements_info.append(
                    {
                        "code": stmt_code,
                        "restored_vars": list(restored_vars),
                        "status": CacheStatus.RESTORED,
                        "is_upstream": True,
                        "source": "DISK",
                        "saved_time": saved_time,
                        "total_time": restore_time + sim.stmt_lookup_times.get(stmt_code, 0.0),
                        "position": stmt_positions.get(stmt_code, 999999),
                    }
                )
                scan.needed.difference_update(restored_vars)
                scan.resolved.update(restored_vars)
            else:
                self._cascade_failed_restore_inputs(
                    i, stmt_code, outputs, needed_outputs, restored_vars, sim, result, scan
                )

        return scan.run, restored_statements_info, total_restore_time

    def _needs_lineage_repair(self, var_name: str, utility_vars: set[str]) -> bool:
        """True when *var_name* is in memory but was bound before cash was listening.

        A name bound in the ``%cash_on`` cell by a statement that could have
        read something (``df = pd.read_parquet(...)``) has no runtime lineage.
        ``NotebookSimulator._adopt_untracked_names`` gives one only to bindings
        that provably read nothing, because adopting a lineage for a load
        leaves nothing that knows it came from a file; it lists the others in
        ``TrackingState.rerun_bindings``. Without a repair, every statement
        reading such a name was refused as "Input variable missing lineage",
        run after run, until some other cell happened to trigger one.

        Scheduling the binding to re-run IS the repair: under tracking it gets
        its lineage and its file dependencies and is keyed like any other
        statement, so after a restart the re-run is a restore -- what Restart
        & Run All would do with it.

        Only for those names, and once each. The first version repaired ANY
        in-memory name without a lineage, and that includes a from-import
        whose lineage the module invalidator dropped after an edit precisely
        so its readers recompute: re-running the import put a lineage back
        that keyed them like before the edit, and they were served the old
        value (``test_a_statement_depends_on_the_symbols_it_reads``, caught by
        the integration sweep).
        """
        pending = self.tracking_state.rerun_bindings
        if var_name not in pending or var_name in self.tracking_state.variable_lineage:
            return False
        if var_name in utility_vars or var_name.startswith("_"):
            return False

        if is_lineage_exempt(var_name, self.shell.user_ns.get(var_name)):
            return False
        pending.discard(var_name)
        return True

    def _repair_upstream_rerun_bindings(
        self,
        required_inputs: set[str] | None,
        simulation_trace: list,
        broken_vars: set[str],
    ) -> None:
        """Re-run every pending re-binding the current cell's inputs were built from.

        ``_check_missing_required_inputs`` only sees the names the cell READS.
        After a helper edit the invalidator drops the lineage of everything
        built from the module, which can be a chain: a loop filling ``blocks``
        with ``hm.summary(...)``, then ``tbl = pd.DataFrame(blocks.values())``.
        Repairing only ``tbl`` re-ran its statement on the stale ``blocks``,
        and the table came out pre-edit.
        So walk the simulation trace upstream from the required inputs, and
        repair every name in ``TrackingState.rerun_bindings`` on the way.
        """
        pending = self.tracking_state.rerun_bindings
        if not pending or not required_inputs:
            return
        producers: dict[str, list[set[str]]] = {}
        for entry in simulation_trace or ():
            for out in entry.outputs or ():
                producers.setdefault(out, []).append(set(entry.inputs or ()))
        utility_vars = {"ip", "cash_magics", "get_ipython", "__builtins__", "In", "Out"}
        seen: set[str] = set()
        todo = list(required_inputs)
        repaired: set[str] = set()
        while todo:
            name = todo.pop()
            if name in seen:
                continue
            seen.add(name)
            for inputs in producers.get(name, ()):
                todo.extend(inputs - seen)
            if name in pending and name in self.shell.user_ns and self._needs_lineage_repair(name, utility_vars):
                logger.debug(
                    "[UPSTREAM] '%s' was built from a reloaded module; re-running its binding under tracking.", name
                )
                repaired.add(name)
        if not repaired:
            return
        # Everything between the cell's inputs and a repaired name was built
        # from it and is stale too: `tbl = sorted(blocks.values())` kept its
        # lineage, so re-running only `blocks` left `tbl` pre-edit.
        memo: dict[str, bool] = {}

        def built_from_repaired(name: str, path: frozenset = frozenset()) -> bool:
            if name in memo:
                return memo[name]
            if name in repaired:
                memo[name] = True
                return True
            if name in path:
                return False
            hit = any(built_from_repaired(inp, path | {name}) for inputs in producers.get(name, ()) for inp in inputs)
            memo[name] = hit
            return hit

        for name in seen:
            if name in self.shell.user_ns and built_from_repaired(name):
                broken_vars.add(name)

    def _check_missing_required_inputs(
        self,
        required_inputs: set[str] | None,
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
        broken_vars: set[str],
    ) -> None:
        """Mark required inputs that exist in virtual lineage but are absent from memory."""
        utility_vars = {"ip", "cash_magics", "get_ipython", "__builtins__", "In", "Out"}
        for var_name in required_inputs or []:
            if var_name not in virtual_lineage:
                continue
            # Modules need their own freshness check: simulating an `import`
            # statement writes the module's lineage into ``variable_lineage``
            # via ``_propagate_import_lineage`` — but after a real kernel
            # restart the module object itself is not in ``user_ns``. The
            # generic ``var_name in self.tracking_state.variable_lineage`` short-circuit
            # below would otherwise hide the missing-module case and the
            # import would never get scheduled for upstream re-execute.
            if var_name in virtual_modules:
                if var_name in self.shell.user_ns:
                    logger.debug("[UPSTREAM_DEBUG] Skipping missing module '%s' (already in memory)", var_name)
                    continue
                logger.debug(
                    "[UPSTREAM_DEBUG] Module '%s' is not in memory. Marking as broken for re-import.", var_name
                )
                broken_vars.add(var_name)
                continue
            # Gate on the LIVE namespace, not the tracking dict. A ``del x`` (or
            # ``%reset``) removes ``x`` from ``user_ns`` but leaves
            # ``variable_lineage['x']`` behind; the old ``in self.tracking_state.variable_lineage``
            # short-circuit therefore hid the missing input and never scheduled the
            # producer to rebuild it. ``virtual_lineage`` is already
            # position-scoped by the simulator (a del/%reset ABOVE the target pops
            # the name; one BELOW is never simulated), so an input that survives to
            # here yet is absent from memory must be reconstructed.
            if var_name in self.shell.user_ns:
                if not self._needs_lineage_repair(var_name, utility_vars):
                    continue
                logger.debug(
                    "[UPSTREAM] Variable '%s' is in memory without a lineage; re-running its binding under tracking.",
                    var_name,
                )
                broken_vars.add(var_name)
                continue

            if var_name in utility_vars or var_name.startswith("_"):
                logger.debug("[UPSTREAM_DEBUG] Skipping utility variable '%s'", var_name)
                continue

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[UPSTREAM_DEBUG] Required input '%s' should exist but is missing from memory. Virtual lineage: %s",
                    var_name,
                    virtual_lineage.get(var_name)[:8],
                )
            logger.debug("[UPSTREAM] Variable '%s' should exist but is missing.", var_name)
            broken_vars.add(var_name)
