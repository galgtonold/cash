"""Simulating a control structure (a loop, ``if``, ``with`` or ``try``) as one unit.

:class:`ControlSimulation` gives a control structure one key and one lineage
update for everything it writes, adds the names its body changes in place,
splits a loop the runtime recorded a split for, and adopts the outcome the
runtime recorded for the structure when its inputs and files are unchanged.
"""

from __future__ import annotations

import ast
import builtins
import logging

from ...analysis.ast_util import called_names
from ...analysis.cacheability import statement_writes_files
from ...analysis.code_analyzer import CodeAnalyzer
from ...analysis.mutation_effects import control_structure_mutations, is_module_name
from ...source_norm import exact_source_digest
from ..cache_key import statement_source_hash
from ..callee_reach import module_state_names
from ..control_structures import extract_target_names, get_control_structure_type
from ..lineage_formula import output_lineage
from ..loop_split import split_nodes
from ..statement.file_deps import compute_file_hash_component
from ..tracking_state import TrackingState
from ._types import SimulationResult, TraceEntry
from .cache_probe import CacheProbe, ControlOutcome
from .statement_lineage import StatementLineage

__all__ = ["ControlSimulation"]

logger = logging.getLogger(__name__)


class ControlSimulation:
    """Simulates one control structure into a :class:`SimulationResult`."""

    def __init__(self, tracking_state: TrackingState, probe: CacheProbe, statements: StatementLineage) -> None:
        self.tracking_state = tracking_state
        self.probe = probe
        self.statements = statements

    def collect_mutations(
        self,
        node: ast.AST,
        loop_target_vars: set[str],
        vars_mutated_by_loops: set[str],
        virtual_modules: set[str],
    ) -> set[str]:
        """Collect control-body mutation info and return the mutated vars for this node.

        Updates *loop_target_vars* (for ``ast.For``) and *vars_mutated_by_loops*
        in place. Covers ALL control structures, not just loops: a var mutated in
        place inside an ``if`` / ``with`` / ``try`` body (``if cond:
        items.append(x)``) is not reported as a static output by
        ``CodeAnalyzer``, so without this its virtual lineage would stay stale and
        a downstream cell reading it would serve a pre-mutation value.
        Treated like a loop mutation so it is trusted in memory and its lineage is
        bumped, matching the runtime's ``update_lineage_after_execution``.
        """
        if isinstance(node, ast.For):
            loop_target_vars.update(extract_target_names(node.target))
        if not isinstance(node, (ast.For, ast.While, ast.If, ast.With, ast.AsyncWith, ast.Try)):
            return set()
        user_ns = self.statements.shell.user_ns
        mutated_vars = control_structure_mutations(
            node,
            self.statements.is_unbound_builtin,
            lambda name: is_module_name(name, user_ns, virtual_modules),
        )
        vars_mutated_by_loops.update(mutated_vars)
        # A local module the structure sets state on moves on too, as the
        # runtime's ``update_lineage_after_execution`` moves it; but it is
        # not trusted in memory as a loop's accumulator is: a reload drops
        # its state, which only its producers put back.
        return mutated_vars | module_state_names(ast.unparse(node), user_ns, structure=True)

    def _bump_mutated(
        self,
        mutated_vars: set[str],
        outputs: set[str],
        inputs: set[str],
        stmt_code: str,
        input_hashes: dict[str, str],
        virtual_lineage: dict[str, str],
    ) -> set[str]:
        """Update virtual lineage for loop-mutated vars and return the extra output set."""
        extra_outputs: set[str] = set()
        if not mutated_vars:
            return extra_outputs
        source_hash = statement_source_hash(stmt_code)
        for mv in mutated_vars:
            if mv not in outputs and mv in inputs:
                new_lineage = output_lineage(source_hash, input_hashes.values())
                virtual_lineage[mv] = new_lineage
                extra_outputs.add(mv)
                logger.debug(
                    "[UPSTREAM_DEBUG] Loop-mutated var '%s' virtual lineage updated to %s...",
                    mv,
                    new_lineage[:12],
                )
        return extra_outputs

    def simulate(self, node: ast.AST, sim: SimulationResult) -> None:
        """
        Simulate execution of a control structure as a single unit.

        The entire control structure is treated as one statement for
        simulation purposes: ``stmt_code = ast.unparse(node)``, one key, one
        lineage update for everything the structure writes.

        **This does NOT always match the runtime.** The simulator models every
        loop as one unit; ``ControlStructureProcessor`` only executes one as a
        unit when ``single_unit_policy.should_run_as_single_unit`` says
        so (roughly ``n >= 125`` for a one-statement body). Below that the
        runtime decomposes per-iteration and writes per-iteration entries,
        while this method still models the whole loop.

        That divergence is deliberate and load-bearing, not an oversight to
        "fix" by decomposing here:

        * What the simulator owes its callers is the loop's **effect on
          lineage** -- which variables it writes and what they now depend on --
          so downstream consumers invalidate correctly. Modelling the whole
          loop gets that right for both dispatch modes.
        * Reproducing per-iteration keys would mean re-deriving each
          iteration's ``__iteration_context__`` discriminator without running
          the loop, which requires the iteration VALUES the simulator does not
          have.

        The practical consequence, worth knowing before reasoning about loop
        cache keys: for a decomposed loop the key computed here corresponds to
        no entry the runtime ever writes, so it simply misses. What keeps an
        unrelated upstream edit from re-planning such a loop is the
        outcome the runtime recorded for it -- see
        ``TrackingState.control_outcomes`` in ``_simulate_unit``.
        """
        # A loop with a recorded split verdict is modelled as TWO statements.
        #
        # This is the mechanism, not a parity nicety: the re-execution planner
        # runs the statements simulated here, so splitting this model is what
        # actually makes the runtime execute a head and a tail. Splitting only
        # in the runtime leaves the planner re-running the whole loop against
        # entries written for halves -- a silent stale value, and the cause of
        # three reverted attempts. See ``notebook/loop_split.py``.
        #
        # Unless the loop's last run left an outcome that still holds. A
        # verdict applies from the NEXT run, so a loop that learned it ran
        # whole, and so did one a direct re-run split (the runtime records
        # its outcome under the whole loop's source either way). Its halves
        # have no outcome of their own until the planner runs them as two
        # statements, so modelled as halves after a restart, what the loop
        # built got lineages no entry was written with, and a restart plus
        # one run of the last cell replayed the loop and everything above
        # it -- whenever the first run's timing had recorded a verdict. The
        # record holding means the loop's inputs, files and callees are what
        # they were, so it has nothing to re-run and its outputs are the
        # recorded ones; any change falls through to the split.
        split_k = self.probe.loop_split(node)
        if split_k is not None and not self._recorded_outcome_holds(node, sim):
            try:
                halves = split_nodes(node, split_k)
            except ValueError:  # for/else -- not splittable
                halves = ()
            if halves:
                logger.debug("[UPSTREAM_DEBUG] loop split at k=%d -> simulating head and tail separately", split_k)
                for half in halves:
                    self._simulate_unit(half, sim)
                return

        self._simulate_unit(node, sim)

    def _simulate_unit(self, node: ast.AST, sim: SimulationResult) -> None:
        """Simulate ONE control structure as a single statement.

        Split out of :meth:`simulate` so a split loop can
        run it twice -- the second half seeing the virtual lineage the first
        produced, exactly as two source-level statements would.
        """
        stmt_code = ast.unparse(node)
        virtual_lineage = sim.virtual_lineage

        inputs, input_hashes = self._input_hashes(stmt_code, virtual_lineage)

        outputs, lookup_time, files_stale, _ = self.statements.apply(stmt_code, virtual_lineage, sim.virtual_modules)

        mutated_vars = self.collect_mutations(
            node, sim.loop_target_vars, sim.vars_mutated_by_loops, sim.virtual_modules
        )

        # CRITICAL FIX: Update virtual lineage for variables mutated inside loops.
        # CodeAnalyzer doesn't detect loop-mutated vars (like `groups` in
        # `for k, v in data: groups.setdefault(k, []).append(v)`) as outputs,
        # so their virtual lineage stays stale.  We compute a new lineage hash
        # that depends on the loop's code and input lineages, ensuring downstream
        # consumers (like `sums = {k: sum(v) for k, v in groups.items()}`) get
        # a different cache key when the loop's inputs change.
        extra_outputs = self._bump_mutated(mutated_vars, outputs, inputs, stmt_code, input_hashes, virtual_lineage)

        all_outputs = outputs | extra_outputs

        # What the runtime left behind when it last ran this very structure
        # (see TrackingState.control_outcomes): the files it read, and the
        # lineages it produced.
        recorded = self._recorded_outcome(stmt_code, virtual_lineage)
        if recorded is not None:
            if compute_file_hash_component(recorded[2]) != recorded[3]:
                # A file behind its outputs moved: the one change the entry
                # lineages cannot show, because a `Path` does not change when
                # the file it names does. Say so, or the loop trust keeps the
                # stale value.
                #
                # Checked BEFORE the input-lineage comparison and regardless of
                # how it comes out, because the two answer different questions.
                # "Have the files this structure read changed?" is decided by
                # the files alone; the entry lineages have nothing to say about
                # it either way. Requiring a match first made the check
                # unreachable whenever a name the loop reads is bound in the
                # same cell as `%cash_on`:
                #
                #     import cash
                #     %cash_on
                #     DATA = Path(...)
                #
                # Such a name has no runtime lineage (cash was not yet
                # listening when that cell started) while the simulation, which
                # reads that cell from the file, has one -- so `entry` lacked a
                # key `input_hashes` carried, equality was false forever, and
                # neither branch ran. (The gap itself
                # is closed separately, in `_entry_lineages`; this check no
                # longer depends on it either way.)
                #
                # Marking the outputs stale can only cause a re-run, never a
                # restore, so running it on a mismatch is the safe direction of
                # the one it was already taking on a match.
                files_stale = True
                sim.vars_with_stale_files.update(all_outputs | set(recorded[1]))
            elif recorded[0] == input_hashes:
                # The runtime ran this very structure with these very inputs
                # and the files it read are where it left them: what it left
                # behind is the answer, not a formula it never used.
                virtual_lineage.update(recorded[1])
                all_outputs = all_outputs | set(recorded[1])

        if logger.isEnabledFor(logging.DEBUG):
            cs_type = get_control_structure_type(node) if node else "unknown"
            logger.debug(
                "[UPSTREAM_DEBUG] Simulating %s as single unit: %s... Outputs: %s", cs_type, stmt_code[:60], all_outputs
            )

        if all_outputs:
            produced_lineages = {out: virtual_lineage[out] for out in all_outputs if out in virtual_lineage}
            sim.trace.append(TraceEntry(stmt_code, all_outputs, inputs, input_hashes, produced_lineages, files_stale))
            if lookup_time > 0:
                sim.stmt_lookup_times[stmt_code] = lookup_time
        elif self._may_write_files(node, stmt_code):
            # ``if PACK.exists(): shutil.rmtree(PACK)`` binds nothing, so it had
            # no trace entry, and a replay after a restart re-ran the cell's
            # ``PACK.mkdir()`` without it.
            # The same rule simple statements follow in simulate_one_node.
            sim.trace.append(TraceEntry(stmt_code, set(), inputs, input_hashes, {}, files_stale))

    def _input_hashes(self, stmt_code: str, virtual_lineage: dict[str, str]) -> tuple[set[str], dict[str, str]]:
        """What *stmt_code* reads, and the lineage each of those names has here."""
        inputs, _ = CodeAnalyzer.analyze_code_block(stmt_code)
        input_hashes = {}
        for inp in inputs:
            if inp in virtual_lineage:
                input_hashes[inp] = virtual_lineage[inp]
            elif inp in self.tracking_state.variable_lineage:
                input_hashes[inp] = self.tracking_state.variable_lineage[inp]
        return inputs, input_hashes

    def _recorded_outcome(self, stmt_code: str, virtual_lineage: dict[str, str]) -> ControlOutcome | None:
        """The outcome the runtime recorded for *stmt_code*: this session's,
        else an earlier kernel's that may still be trusted."""
        recorded = self.tracking_state.control_outcomes.get(exact_source_digest(stmt_code))
        if recorded is None:
            variable_lineage = self.tracking_state.variable_lineage
            recorded = self.probe.control_outcome(
                stmt_code, lambda name: virtual_lineage.get(name) or variable_lineage.get(name) or "ABSENT"
            )
        return recorded

    def _recorded_outcome_holds(self, node: ast.AST, sim: SimulationResult) -> bool:
        """Whether *node*'s recorded outcome would be taken as its result here:
        read with these input lineages, and every file behind it unchanged --
        the two checks ``_simulate_unit`` makes before trusting it."""
        stmt_code = ast.unparse(node)
        try:
            _, input_hashes = self._input_hashes(stmt_code, sim.virtual_lineage)
            recorded = self._recorded_outcome(stmt_code, sim.virtual_lineage)
            return (
                recorded is not None
                and recorded[0] == input_hashes
                and compute_file_hash_component(recorded[2]) == recorded[3]
            )
        except Exception:  # any doubt keeps the split
            logger.debug("[UPSTREAM_DEBUG] could not check a loop's recorded outcome", exc_info=True)
            return False

    @staticmethod
    def _may_write_files(node: ast.AST, stmt_code: str) -> bool:
        """A write in the text, or a call to something that might be a
        user function that writes (the planner decides which)."""

        if statement_writes_files(stmt_code):
            return True
        return any(not hasattr(builtins, name) for name in called_names(node))
