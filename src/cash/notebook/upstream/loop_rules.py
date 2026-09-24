"""What the upstream check decides about loops and the accumulators they fill.

A loop is simulated as one unit, so the rules for re-running one -- which
accumulator inits must run with it, which loops read data that changed, when
loop output is trusted -- live in :class:`LoopRules`, used by the simulator,
the classifier and the planner alike.
"""

from __future__ import annotations

import hashlib
import logging
import re
import types
from typing import TYPE_CHECKING

from cash.control_markers import iteration_digest, strip_markers

if TYPE_CHECKING:
    from .virtual_lineage import VirtualLineage


__all__ = ["LoopRules"]

logger = logging.getLogger(__name__)


class LoopRules:
    """Loop and accumulator rules of the upstream check."""

    # Control-structure wrapper prefixes. Delimited (space or colon) so a plain
    # identifier that merely begins with a keyword (``elsewhere = ...``,
    # ``exception = ...``) is NOT mistaken for a wrapper and wrongly skipped.
    _CTRL_PREFIXES = (
        "for ",
        "while ",
        "async for ",
        "if ",
        "elif ",
        "else:",
        "with ",
        "async with ",
        "try:",
        "except ",
        "except:",
        "finally:",
    )

    def __init__(self, virtual_lineage: VirtualLineage) -> None:
        self.virtual_lineage = virtual_lineage

    def loop_accumulators_with_external_init(
        self,
        vars_mutated_by_loops: set[str],
        simulation_trace: list,
        loop_target_vars: set[str],
    ) -> set[str]:
        """Loop accumulators whose value ALSO derives from an external input.

        A reassignment accumulator (``result = result + 1``) enters the loop-trust
        set so that a no-change re-run is not re-executed (which would re-drain a
        one-shot iterable). But when the accumulator ALSO has an external
        dependency via a *non-loop* producing statement — typically an
        initializer like ``result = np.zeros(N)`` — editing that external input
        (``N``) and re-running the edited cell before the reader leaves
        ``upstream_has_modifications`` False, and every current-state input
        lineage is consistent, so the trust would serve a stale value. Only the
        accumulator's transitive lineage betrays the staleness. Such accumulators
        are excluded from the loop-trust set entirely, so they fall back to the
        normal lineage-mismatch path (baseline behaviour) that correctly
        re-executes; a constant-init accumulator (``total = 0``) has no external
        dependency and keeps the new trust.

        Detection scans the trace for a *non-self-referential*, *non-control*
        statement producing the accumulator (its init) that reads a data variable
        which is not a loop target, not a builtin and not a module.
        """
        tainted: set[str] = set()
        for acc in vars_mutated_by_loops:
            for entry in simulation_trace:
                stmt_code, outputs, inputs = entry.stmt_code, entry.outputs, entry.inputs
                if acc not in outputs or acc in inputs:
                    continue  # not a producer, or self-referential (loop body)
                if stmt_code.lstrip().startswith(self._CTRL_PREFIXES):
                    continue  # loop/control wrapper; iterable feeds via the loop
                for inp in inputs:
                    if inp in loop_target_vars or self.virtual_lineage._unbound_builtin(inp):
                        continue
                    val = self.virtual_lineage.shell.user_ns.get(inp)
                    if val is not None and isinstance(val, types.ModuleType):
                        continue
                    tainted.add(acc)
                    break
                if acc in tainted:
                    break
        return tainted

    def loops_reading_changed_data(
        self,
        vars_mutated_by_loops: set[str],
        simulation_trace: list,
        loop_target_vars: set[str],
        vars_derived_from_loops: set[str],
    ) -> set[str]:
        """Loop-built variables whose loop reads data that has changed since it ran.

        Loop trust assumes the loop's inputs are what they were: with no code
        edit upstream, a loop-built value whose lineage disagrees with the
        simulation is trusted, because the two engines fold a loop differently.
        A new file in a folder the notebook globs is not a code edit. The
        frame read from it changed, the loop over it (``for k in grid:
        rows.append(score(raw, k))``) did not re-run, and everything derived
        from it -- the tuned parameter picked from ``rows`` -- was trusted and
        served from the old data.

        So compare each data input of a loop producing an accumulator, as the
        simulation has it at that point, with the lineage it had when the
        loop last ran (``TrackingState.control_outcomes``, recorded on entry).
        Inputs that are themselves loop-built or loop targets are skipped:
        their lineages disagree by construction.
        """
        changed: set[str] = set()
        outcomes = self.virtual_lineage.tracking_state.control_outcomes
        for entry in simulation_trace:
            stmt_code, outputs, inputs, input_hashes = (
                entry.stmt_code,
                entry.outputs,
                entry.inputs,
                entry.input_hashes or {},
            )
            accs = outputs & vars_mutated_by_loops
            if not accs or not stmt_code.lstrip().startswith(self._CTRL_PREFIXES):
                continue
            recorded = outcomes.get(hashlib.sha256(stmt_code.encode("utf-8")).hexdigest())
            if recorded is None:
                continue
            for inp in inputs:
                if (
                    inp in outputs
                    or inp in loop_target_vars
                    or inp in vars_derived_from_loops
                    or self.virtual_lineage._unbound_builtin(inp, input_hashes)
                ):
                    continue
                if isinstance(self.virtual_lineage.shell.user_ns.get(inp), types.ModuleType):
                    continue
                now, then = input_hashes.get(inp), recorded[0].get(inp)
                if now is not None and then is not None and now != then:
                    changed |= accs
                    break
        return changed

    def filter_accumulator_reinits(
        self,
        stmts_to_run_indices: list[int],
        simulation_trace: list,
        vars_mutated_by_loops: set[str],
    ) -> list[int]:
        """Remove accumulator initialization statements that would reset existing state.

        When adding new items to a cached loop, the backward scan may schedule
        the initialization (e.g. ``ticker_stats = {}``) for execution. If the
        accumulator already exists in memory with data, re-running the init would
        wipe accumulated state. Returns a filtered copy of *stmts_to_run_indices*.
        """
        scheduled_iteration_outputs: dict[str, list] = {}
        for idx in stmts_to_run_indices:
            stmt_code, outputs = simulation_trace[idx].stmt_code, simulation_trace[idx].outputs
            if iteration_digest(stmt_code) is not None:
                for out in outputs:
                    scheduled_iteration_outputs.setdefault(out, []).append(idx)

        fully_rerun_mutated = self._loop_vars_fully_rescheduled(
            stmts_to_run_indices,
            simulation_trace,
            vars_mutated_by_loops,
        )

        # A fully re-run loop replays its in-place mutations (.append / [k]=)
        # onto whatever the accumulator currently holds.  If the empty-container
        # init was never scheduled (the backward scan often schedules only the
        # loop body, treating the accumulator output as already satisfied), the
        # replay doubles the accumulated value.  Schedule the missing init so it
        # runs alongside the loop.  (Pure incremental extension keeps the init
        # unscheduled and is handled by the removal pass below.)
        stmts_to_run_indices = self._schedule_missing_accumulator_inits(
            stmts_to_run_indices,
            simulation_trace,
            fully_rerun_mutated,
        )

        indices_to_remove: set[int] = set()
        for idx in stmts_to_run_indices:
            if self._is_reinit_to_skip(
                idx,
                simulation_trace,
                scheduled_iteration_outputs,
                vars_mutated_by_loops,
                fully_rerun_mutated,
            ):
                indices_to_remove.add(idx)

        if indices_to_remove:
            return [idx for idx in stmts_to_run_indices if idx not in indices_to_remove]
        return stmts_to_run_indices

    def _is_reinit_to_skip(
        self,
        idx: int,
        simulation_trace: list,
        scheduled_iteration_outputs: dict[str, list],
        vars_mutated_by_loops: set[str],
        fully_rerun_mutated: set[str],
    ) -> bool:
        """Return True if the statement at *idx* is an accumulator init that should be skipped.

        Skips when the statement initialises to an empty container (e.g. ``x = {}``)
        but the accumulator already has data in memory, to avoid wiping state.
        """
        stmt_code, outputs = simulation_trace[idx].stmt_code, simulation_trace[idx].outputs
        if iteration_digest(stmt_code) is not None:
            return False
        if len(outputs) != 1:
            return False
        out_var = list(outputs)[0]
        if out_var in fully_rerun_mutated:
            # When the loop that mutates out_var is itself fully re-executed, the
            # init must run alongside it, else the accumulation doubles; skipping
            # is only safe for pure incremental extension of a cached loop.
            return False
        is_loop_updated = out_var in scheduled_iteration_outputs or out_var in vars_mutated_by_loops
        if not is_loop_updated:
            return False
        stripped = stmt_code.strip()
        empty_init_pattern = re.compile(
            rf"^{re.escape(out_var)}\s*=\s*(\{{\}}|\[\]|set\(\)|dict\(\)|list\(\)|frozenset\(\))$"
        )
        if not empty_init_pattern.match(stripped):
            return False
        if out_var not in self.virtual_lineage.shell.user_ns:
            return False
        existing_val = self.virtual_lineage.shell.user_ns[out_var]
        try:
            is_non_empty = bool(existing_val)
        except (ValueError, TypeError):
            is_non_empty = False
        if is_non_empty:
            logger.debug(
                "[UPSTREAM] Skipping accumulator init '%.40s' - already has %d items in memory",
                stmt_code,
                len(existing_val),
            )
            return True
        return False

    def _loop_vars_fully_rescheduled(
        self,
        stmts_to_run_indices: list[int],
        simulation_trace: list,
        vars_mutated_by_loops: set[str],
    ) -> set[str]:
        """Loop-mutated vars whose mutation is scheduled OUTSIDE a cached iteration-context body (=> full re-run)."""
        if not vars_mutated_by_loops:
            return set()
        patterns = {
            mv: re.compile(rf"\b{re.escape(mv)}\s*(?:\.\s*\w+\s*\(|\[[^\]]*\]\s*=(?!=))")
            for mv in vars_mutated_by_loops
        }
        fully_rerun_mutated: set[str] = set()
        for idx in stmts_to_run_indices:
            stmt_code, outputs = simulation_trace[idx].stmt_code, simulation_trace[idx].outputs
            if iteration_digest(stmt_code) is not None:
                continue
            for mv, pat in patterns.items():
                # Only a statement that WRITES it: `def draw_roc` iterating
                # `results.items()` matched the text, so the init was scheduled
                # for a loop that was not, and `results` was re-run empty.
                if mv not in fully_rerun_mutated and mv in outputs and pat.search(stmt_code):
                    fully_rerun_mutated.add(mv)
        return fully_rerun_mutated

    def _schedule_missing_accumulator_inits(
        self,
        stmts_to_run_indices: list[int],
        simulation_trace: list,
        fully_rerun_mutated: set[str],
    ) -> list[int]:
        """Schedule empty-container inits for fully-re-run loop accumulators.

        For each var in *fully_rerun_mutated*, if its ``x = []`` / ``x = {}`` /
        ``x = set()`` init appears in the trace but is not already scheduled,
        add it. This prevents the loop's in-place mutations from replaying onto
        a stale value (doubling). Only single-output empty-container inits are
        added, so non-init assignments are never pulled in.
        """
        if not fully_rerun_mutated:
            return stmts_to_run_indices
        scheduled = set(stmts_to_run_indices)
        empty_init_patterns = {
            mv: re.compile(rf"^{re.escape(mv)}\s*=\s*(\{{\}}|\[\]|set\(\)|dict\(\)|list\(\)|frozenset\(\))$")
            for mv in fully_rerun_mutated
        }
        additional: list[int] = []
        for idx, entry in enumerate(simulation_trace):
            if idx in scheduled:
                continue
            stmt_code, outputs = entry.stmt_code, entry.outputs
            if len(outputs) != 1:
                continue
            out_var = next(iter(outputs))
            pat = empty_init_patterns.get(out_var)
            if pat is not None and pat.match(stmt_code.strip()):
                additional.append(idx)
                scheduled.add(idx)
        if additional:
            return stmts_to_run_indices + additional
        return stmts_to_run_indices

    def check_loop_derived_trust_override(
        self,
        upstream_has_modifications: bool,
        vars_mutated_by_loops: set[str],
        simulation_trace_codes: set[str],
    ) -> bool:
        """Return True if loop-derived variable trust should be overridden.

        This happens when a loop-mutated variable was produced by code that is
        NOT in the current simulation trace (unsaved edit or stale execution).
        """
        if upstream_has_modifications or not vars_mutated_by_loops:
            return False
        for mv in vars_mutated_by_loops:
            if mv not in self.virtual_lineage.tracking_state.executed_cell_codes:
                continue
            exec_code = strip_markers(self.virtual_lineage.tracking_state.executed_cell_codes[mv]).strip()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[UPSTREAM_DEBUG] Checking loop trust for '%s': exec_code=%s", mv, repr(exec_code[:60]))
                matching = [sc for sc in simulation_trace_codes if exec_code in sc or sc in exec_code]
                logger.debug(
                    "[UPSTREAM_DEBUG]   Partial matches in simulation_trace_codes: %s", [repr(m[:60]) for m in matching]
                )
            if exec_code and exec_code not in simulation_trace_codes:
                logger.debug(
                    "[UPSTREAM_DEBUG] Loop-mutated var '%s' was produced by code "
                    "not found on disk (unsaved edit or stale execution). Distrusting ALL loop-derived vars.",
                    mv,
                )
                return True
        return False

    def build_loop_var_input_lineages(
        self,
        simulation_trace: list,
        vars_derived_from_loops: set[str],
        virtual_lineage: dict[str, str],
        virtual_modules: set[str],
    ) -> dict[str, dict[str, str]]:
        """Return a mapping of loop-derived variable → its data-input virtual lineages.

        Used to detect when loop inputs change (e.g., N=10→20) even when the
        producing code is unchanged on disk.
        """
        loop_var_input_lineages: dict[str, dict[str, str]] = {}
        for entry in simulation_trace:
            for out in entry.outputs:
                if out in vars_derived_from_loops:
                    data_input_lineages: dict[str, str] = {}
                    for inp in entry.inputs:
                        if inp in virtual_modules:
                            continue
                        if inp in virtual_lineage:
                            data_input_lineages[inp] = virtual_lineage[inp]
                    loop_var_input_lineages[out] = data_input_lineages
        return loop_var_input_lineages
