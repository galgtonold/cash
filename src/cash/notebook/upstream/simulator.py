"""Notebook simulator: pure-AST + cache-probing replay of upstream cells.

Kept apart from ``UpstreamChecker`` so the simulation can be tested on its
own, with a ``TrackingState`` and no orchestrator, backend or shell.

The simulator never executes user code via the IPython kernel. It simulates
statement-by-statement using AST analysis and the cache backend, producing a
plan of statements to re-execute and a list of restored statements. The
orchestrator (``UpstreamChecker``) takes that plan and runs it via the real
``process_statement_callback``.
"""

from __future__ import annotations

import ast
import builtins
import logging
import os
import sys
import types
from collections.abc import Callable
from typing import Any

from cash.control_markers import strip_markers

from ...analysis.ast_util import resolve_callee
from ...analysis.mutation_effects import CellEffects
from ...tracking.function_tracker import FunctionTracker, is_local_module
from .._protocols import CashInstanceProtocol, ShellProtocol
from .._trace import is_tracing, trace_event
from ..cache_status import CacheStatus
from ..tracking_state import TrackingState
from ._types import CellCheck, ClassificationResult, ReexecutionPlan, SimulationCache, SimulationResult
from .mismatch_classifier import MismatchClassifier
from .read_scope import ReadScope
from .reexecution_planner import ReexecutionPlanner
from .stale_values import StaleValueGuard
from .virtual_lineage import VirtualLineage, loop_derived_vars

__all__ = ["NotebookSimulator"]


logger = logging.getLogger(__name__)


class NotebookSimulator:
    """Replays upstream cells via AST simulation and cache probing.

    Owned by :class:`UpstreamChecker`, with which it shares the
    ``TrackingState``. :meth:`simulate_upstream` runs the three phases --
    :class:`VirtualLineage`, :class:`MismatchClassifier`,
    :class:`ReexecutionPlanner` -- in order.
    :meth:`simulate_cell` and :meth:`restore_statement` do one cell or one
    statement the same way.
    """

    def __init__(
        self,
        shell: ShellProtocol,
        cash_instance: CashInstanceProtocol | None,
        tracking_state: TrackingState,
        compute_hash_fn: Callable[[Any], str] | None = None,
        function_tracker: FunctionTracker | None = None,
    ) -> None:
        self.shell = shell
        self.cash_instance = cash_instance
        self.compute_hash_fn = compute_hash_fn

        #: The checker's, shared with the statement processor.
        self.tracking_state = tracking_state
        #: Set by ``reset_caches`` (``%cash_on``): adopt untracked names once.
        self._adopt_untracked_pending = False
        #: The previous simulation's per-cell snapshots, where the next one starts.
        self.cache = SimulationCache()

        # Phase-1 simulator. Shares ``shell``/``cash_instance``/tracking-state
        # references with us so writes are visible on both sides.
        self.virtual_lineage = VirtualLineage(
            shell=shell,
            cash_instance=cash_instance,
            tracking_state=tracking_state,
            compute_hash_fn=compute_hash_fn,
            function_tracker=function_tracker,
            cache=self.cache,
        )

        # Phase-2 classifier. Shares tracking-state references and routes
        # back to ``virtual_lineage`` for cache-probing helpers.
        self.classifier = MismatchClassifier(
            virtual_lineage=self.virtual_lineage,
            tracking_state=tracking_state,
        )

        # Phase-3 planner. Routes into VL + Classifier for helpers that
        # still live on those phases.
        self.planner = ReexecutionPlanner(
            virtual_lineage=self.virtual_lineage,
            classifier=self.classifier,
        )
        #: Inputs stale in memory though their lineage matches.
        self.stale_values = StaleValueGuard(shell, tracking_state, self.virtual_lineage, compute_hash_fn)
        #: The files the checked cell depends on.
        self.read_scope = ReadScope(shell, tracking_state, self.virtual_lineage)

    def reset_caches(self) -> None:
        """Forget the previous simulation.

        Called only by ``%cash_on``, so it also arms the one adoption of
        untracked names -- see ``_adopt_untracked_names``.
        """
        self.virtual_lineage.reset_caches()
        self._adopt_untracked_pending = True

    def _track_modules_bound_before_cash_on(self) -> None:
        """Track the local modules the namespace reached before cash was listening.

        Cash starts tracking a project module when a cell it PROCESSES imports
        it, and the cell that turns cash on is not one of them. So
        `import cash; %cash_on; import helpers as hm` -- the layout the
        quickstart recommends -- left `helpers` untracked: an edit to it
        reloaded nothing, and its source reached no cache key, so a value built
        from it was served pre-edit, even after Restart & Run All.
        Done before pass 1, so this very simulation already
        keys the module's readers on its source.
        """
        ft = self.virtual_lineage.function_tracker
        user_ns = self.shell.user_ns
        if ft is None or not user_ns:
            return

        names: set[str] = set()
        for value in list(user_ns.values()):
            if isinstance(value, types.ModuleType):
                names.add(value.__name__)
            else:
                owner = getattr(value, "__module__", None)
                if isinstance(owner, str) and owner != "__main__":
                    names.add(owner)
        for mod_name in names:
            module = sys.modules.get(mod_name)
            if module is None or mod_name in ft.tracked_modules:
                continue
            # cash itself is "local" in a development checkout, and `cash` is
            # bound in every notebook: without this it watched its own source.
            if mod_name == "cash" or mod_name.startswith("cash."):
                continue
            try:
                if is_local_module(module):
                    ft.track_module(mod_name)
            except (AttributeError, OSError, TypeError, ValueError):
                logger.debug("Could not track '%s' at %%cash_on", mod_name)

    def _adopt_untracked_names(self, virtual_lineage: dict[str, str], simulation_trace: list) -> None:
        """Give names bound before cash was listening the simulation's lineage.

        A name bound in the ``%cash_on`` cell (``DATA = Path(...)``, ``N = 3``)
        has no runtime lineage: cash was not listening when that cell started.
        The first statement reading it was refused as "Input variable missing
        lineage" -- in the quickstart's own layout, that is the cell that loads
        the data. The simulation reads the cell out
        of the .ipynb and has a lineage for it like any other, and it is the
        one the simulation keys the reader with, so adopting it is also what
        lets the runtime store under the key a restart will look up.

        Once per ``%cash_on``, at the first check after it, and only there:
        before anything is tracked no lineage can have been dropped ON PURPOSE
        (the module invalidator drops a from-import's lineage to force the
        next reader to miss), so there is nothing for this to undo. Names the
        runtime already has a lineage for are left alone; import-bound names
        already had theirs propagated and are not touched either.

        Only for a name whose binding statement cannot have read anything
        (``_binds_without_reading``). Nothing under tracking saw that statement
        run, so nothing recorded a file it read: adopt ``RAW = DATA.read_text()``
        and a changed file leaves every cell below serving the old text, where
        Restart & Run All would not
        (``test_a_value_read_in_the_cash_on_cell_still_follows_its_file``,
        which is how the first draft of this was caught). Such a name keeps the
        old behaviour -- refused once, then repaired under tracking.
        """
        if not self._adopt_untracked_pending:
            return
        self._adopt_untracked_pending = False
        user_ns = self.shell.user_ns
        if not user_ns:
            return
        runtime = self.tracking_state.variable_lineage
        imported = self.virtual_lineage.propagated_imports
        binder: dict[str, str] = {}
        for entry in simulation_trace or ():
            for out in entry.outputs or ():
                binder[out] = entry.stmt_code
        adopted = []
        untracked = self.tracking_state.rerun_bindings
        untracked.clear()
        for name, lineage_hash in virtual_lineage.items():
            if not lineage_hash or name in runtime or name in imported or name.startswith("_") or name not in user_ns:
                continue
            code = binder.get(name)
            if code is None or not _binds_without_reading(code, user_ns):
                # Re-run under tracking instead, by the first cell that needs
                # it -- see TrackingState.rerun_bindings.
                untracked.add(name)
                continue
            self.tracking_state.lineage.record(name, lineage_hash, value=user_ns[name])
            adopted.append(name)
        if adopted and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[UPSTREAM_DEBUG] Adopted simulated lineage for names bound before %%cash_on: %s", sorted(adopted)
            )

    # --- One statement or cell at a time, as a check does it ---

    def simulate_cell(
        self,
        cell_code: str,
        virtual_lineage: dict[str, str] | None = None,
        virtual_modules: set[str] | None = None,
    ) -> SimulationResult:
        """Simulate *cell_code* on its own, from *virtual_lineage*.

        Probes the cache and records what it learns for the live session (an
        imported module's lineage) exactly as a check does for a cell above.
        """
        sim = SimulationResult(virtual_lineage=dict(virtual_lineage or {}), virtual_modules=set(virtual_modules or ()))
        self.virtual_lineage.simulate_one_cell(sim, -1, cell_code)
        return sim

    def restore_statement(
        self,
        stmt_code: str,
        outputs: set[str],
        inputs: set[str],
        input_hashes: dict[str, str],
        virtual_modules: set[str] | None = None,
        expected_lineages: dict[str, str] | None = None,
    ) -> set[str]:
        """Restore *stmt_code*'s outputs from the cache entry its simulated
        inputs key; the names restored (none when the entry is missing, stale
        or for other lineages)."""
        restored, _restore_time, _saved_time = self.virtual_lineage.restorer.try_virtual_restore(
            stmt_code, outputs, inputs, input_hashes, virtual_modules, expected_lineages
        )
        return restored

    def record_replayed_file_deps(self, rerecorded: set[str]) -> None:
        self.virtual_lineage.record_replayed_file_deps(rerecorded)

    def simulate_upstream(
        self,
        current_cell_idx: int,
        notebook_cells: list[str],
        required_inputs: set[str] | None = None,
        effects: CellEffects | None = None,
        cell_code: str | None = None,
    ) -> ReexecutionPlan:
        """What must re-run, and what can be restored, before cell
        *current_cell_idx* runs on the state memory holds.

        *effects* is what the current cell writes (see
        :func:`~cash.analysis.mutation_effects.cell_effects`); None when it is
        not known, which is not the same as a cell that writes nothing.
        *cell_code* is the source being run, when it may differ from the saved
        cell.
        """
        check = CellCheck(current_cell_idx, notebook_cells, required_inputs, effects, cell_code)
        effects = effects or CellEffects()
        trace_event(
            "simulate_enter",
            cell_idx=current_cell_idx,
            reassigned=set(effects.reassigned),
            mutated=set(effects.mutated),
            required_inputs=required_inputs or set(),
            selfref=set(effects.selfref),
            method_receivers=set(effects.method_receivers),
        )
        if self._adopt_untracked_pending:
            self._track_modules_bound_before_cash_on()

        sim = self.virtual_lineage.simulate(current_cell_idx, notebook_cells)

        # Hand the simulation's view of every name to the runtime about to
        # execute this cell. A control structure records the lineages of what
        # it read, and for a name bound in the `%cash_on` cell the runtime has
        # none to record -- see TrackingState.simulated_lineage. Taken here,
        # after pass 1 and before the cell runs, so it describes the state the
        # cell is about to start from.
        self.tracking_state.simulated_lineage = dict(sim.virtual_lineage)
        self._adopt_untracked_names(sim.virtual_lineage, sim.trace)

        self._settle_loop_trust(sim)

        result = self.classifier.classify(sim, check)
        broken_vars = result.broken_vars
        trace_event("broken_after_pass2", broken=broken_vars, tainted=result.tainted_vars)
        if is_tracing():
            # Every variable the two engines disagree on, relevant or not. In a
            # plain top-to-bottom run there must be none: each one is a spurious
            # "changed" waiting for a cell that reads it.
            recorded = self.tracking_state.variable_lineage
            virtual_lineage = sim.virtual_lineage
            trace_event(
                "lineage_disagreement",
                cell_idx=current_cell_idx,
                vars={
                    v: [str(virtual_lineage[v])[:12], str(recorded[v])[:12]]
                    for v in sorted(virtual_lineage.keys() & recorded.keys())
                    if virtual_lineage[v] != recorded[v]
                },
            )

        self.stale_values.mark_stale_value_inputs_broken(
            required_inputs,
            effects,
            broken_vars,
            notebook_cells=notebook_cells,
            current_cell_idx=current_cell_idx,
            virtual_lineage=sim.virtual_lineage,
        )
        trace_event("broken_after_guard", broken=broken_vars)

        # Read-only consumable inputs (drained queue / exhausted generator) are
        # invisible to the guard above, which only examines self-WRITTEN vars.
        # Same ``broken_vars`` set, so the planner handles both identically.
        result.consumable_broken_vars = self.stale_values.mark_consumed_unrestorable_inputs_broken(
            required_inputs,
            broken_vars,
            notebook_cells=notebook_cells,
            current_cell_idx=current_cell_idx,
        )
        trace_event("broken_after_consumables", broken=broken_vars)

        # A ``# @cash: no-cache`` statement opts the whole statement out of cash,
        # so its self-modified vars must behave like an uncached Jupyter cell:
        # re-running ACCUMULATES (advances), never resets. Pass 2 still flags such
        # a var stale (its runtime lineage advanced past the simulation's), which
        # would re-execute its producer and reset it -- so drop no-cache-written
        # vars from broken_vars here (the self-write-set exclusion only
        # covered the stale-value guard, not the pass-2 lineage mismatch).
        if effects.nocache:
            removed = broken_vars & effects.nocache
            if removed:
                broken_vars -= removed
                trace_event("broken_drop_nocache", dropped=removed, broken=broken_vars)

        # Scope the writer-scheduling to files THIS cell's reconstruction reads
        #: a writer whose output no relevant consumer reads is
        # an unrelated / terminal side-effect that must never be re-fired here.
        relevant_read_paths, relevant_read_paths_known = self.read_scope.relevant_read_paths(
            required_inputs,
            sim.trace,
            notebook_cells,
            current_cell_idx,
        )

        # File writes have no variable edge, so an edited/new upstream writer
        # statement leaves broken_vars empty while the on-disk state a reader
        # depends on is stale. The plan must still be built so
        # the planner can schedule the writer.
        has_stale_file_writers = bool(
            self.planner.file_writers.find_stale_file_writer_indices(
                sim.trace,
                virtual_lineage=sim.virtual_lineage,
                relevant_read_paths=relevant_read_paths,
                relevant_read_paths_known=relevant_read_paths_known,
            )
        )

        if broken_vars:
            # A current-cell statement that is a cache hit restores what it
            # reads as well as what it writes: a broken ``df`` that the cell's
            # first ``df[...] = f(df)`` restores needs nothing upstream.
            self.virtual_lineage.restorer.eliminate_broken_vars_via_current_cell_probe(
                broken_vars,
                notebook_cells,
                current_cell_idx,
                sim.virtual_lineage,
                sim.virtual_modules,
            )
            if not broken_vars:
                logger.debug("[UPSTREAM] All broken vars resolved by current cell cache hits — skipping upstream")

        if not broken_vars and not has_stale_file_writers:
            return ReexecutionPlan([], [], 0.0)

        plan = self.planner.plan(
            sim,
            result,
            notebook_cells,
            relevant_read_paths=relevant_read_paths,
            relevant_read_paths_known=relevant_read_paths_known,
        )
        return plan

    # --- After the repair ran ---

    def resync_after_replay(self, records_before: dict[str, tuple]) -> None:
        """Bring the simulation's snapshots in line with what the replay recorded.

        After upstream statements run or are restored, ``variable_lineage``
        holds the authoritative lineage of each. A snapshot may hold a
        simulated one that differs (a control structure simulated as one
        unit), and without the sync the next check sees a mismatch and
        repairs again.
        """
        rerecorded = self._rerecorded_since(records_before)
        self._sync_simulation_cache_lineages(rerecorded)
        # The snapshots of the cells replayed here may not know the files
        # behind what the replay restored (see record_replayed_file_deps).
        self.record_replayed_file_deps(rerecorded)

    def lineage_records(self) -> dict[str, tuple]:
        """Each variable's recorded lineage and input-lineage map, as held now.

        The map object is kept (not copied): recording a variable replaces it,
        so ``is`` tells a re-recording apart even when the lineage came out the
        same.
        """
        return {
            v: (h, self.tracking_state.executed_input_lineages.get(v))
            for v, h in self.tracking_state.variable_lineage.items()
        }

    def _rerecorded_since(self, before: dict[str, tuple]) -> set[str]:
        """Variables this upstream pass recorded again (re-executed or restored)."""
        changed = set()
        for v, h in self.tracking_state.variable_lineage.items():
            old = before.get(v)
            if old is None or old[0] != h or old[1] is not self.tracking_state.executed_input_lineages.get(v):
                changed.add(v)
        return changed

    def _should_sync_cache_var(
        self,
        var_name: str,
        cumulative_stmt_codes: set[str],
        cached_vl: dict[str, str],
        idx: int,
    ) -> bool:
        """Return True if *var_name*'s cached lineage should be synced at cache index *idx*.

        A variable is synced only when its current runtime lineage was produced
        by code within cells 0..idx.  Variables produced by later cells are
        excluded to avoid contaminating earlier cache entries.
        """
        if var_name not in self.tracking_state.variable_lineage:
            return False
        if cached_vl[var_name] == self.tracking_state.variable_lineage[var_name]:
            return False  # Already matches, nothing to sync
        producing_code = self.tracking_state.executed_cell_codes.get(var_name)
        if producing_code is None:
            return True
        normalized_code = strip_markers(producing_code).strip()
        if normalized_code not in cumulative_stmt_codes:
            logger.debug(
                "[UPSTREAM_DEBUG] Skipping sync for '%s' in cache entry %d: producing code not in cells 0..%d",
                var_name,
                idx,
                idx,
            )
            return False
        return True

    def plan_cell_run(
        self,
        nodes: list,
        raw_cell: str,
        occurrence_counts: dict[str, int],
    ) -> dict[int, dict] | None:
        """Which of a run of assignments in the cell being run need not run.

        A cell rebuilding ``sales`` through a dozen steps
        writes only the last version to disk (``_written_later_in_cell``), and
        after a restart Run All re-ran every step to get back to it. Here the
        run is simulated the way the upstream repair simulates a cell above,
        and the same backward scan finds the latest versions it can restore;
        what they cover need not run.

        Returns ``{index in nodes: metric}`` for each statement that need not
        run -- restored, or skipped because what it built is current or
        overwritten -- or ``None`` to run them all. Every statement not in the
        result runs as it would have, in order, after the restores.
        """
        try:
            vl = self.virtual_lineage
            planner = self.planner
            classifier = self.classifier
            sim = SimulationResult(virtual_lineage=dict(self.tracking_state.variable_lineage))
            trace = sim.trace
            counts = dict(occurrence_counts)
            for node in nodes:
                before = len(trace)
                vl.simulate_one_node(sim, 0, node, counts, {}, raw_cell=raw_cell)
                if len(trace) != before + 1:
                    return None
            if any(entry.files_stale for entry in trace):
                return None  # a file it reads changed: run it
            final: dict[str, str] = {}
            for entry in trace:
                final.update(entry.produced_lineages)
            if set(final) != set().union(*(entry.outputs for entry in trace)):
                return None
            broken = {
                name
                for name, lineage in final.items()
                if name not in self.shell.user_ns or self.tracking_state.variable_lineage.get(name) != lineage
            }
            restored_by_index: dict[int, dict] = {}
            run: list[int] = []
            if broken:
                run, restored, _ = classifier.backward_scan_pass(
                    sim,
                    ClassificationResult(
                        broken_vars=broken,
                        tainted_vars=set(),
                        trace_codes={entry.stmt_code for entry in trace},
                    ),
                )
                while True:
                    size = len(run)
                    # Stricter than the repair's own pass: a statement that runs
                    # reads the version its run made before it, so that version's
                    # producer runs too. The live value may be a LATER version the
                    # scan restored -- ``is_big = sales['a'] > ...`` ran on the
                    # final ``sales`` otherwise.
                    run = sorted(
                        set(run)
                        | {
                            p
                            for i in run
                            for v in (trace[i].inputs or ())
                            if (p := planner.latest_producer(trace, v, before=i)) is not None
                        }
                    )
                    run = planner.complete_later_producers(run, trace)
                    if len(run) == size:
                        break
                for info in restored:
                    position = info.get("position")
                    if isinstance(position, int):
                        info["is_upstream"] = False
                        restored_by_index[position] = info
            run_set = set(run)
            planned: dict[int, dict] = {}
            for i, entry in enumerate(trace):
                if i in run_set:
                    continue
                planned[i] = restored_by_index.get(i) or {
                    "code": entry.stmt_code,
                    "status": CacheStatus.SKIPPED,
                    "is_upstream": False,
                    "saved_time": 0.0,
                    "total_time": 0.0,
                }
            if broken and not restored_by_index:
                return None  # nothing on disk to jump to: run as usual
            return planned
        except Exception:  # a plan that cannot be made is the ordinary run
            logger.debug("[UPSTREAM] cell run plan failed", exc_info=True)
            return None

    def _sync_simulation_cache_lineages(self, rerecorded: set[str]) -> None:
        """Sync simulation cache virtual lineages with actual runtime lineages.

        Only the *rerecorded* variables -- the ones this upstream pass just
        re-executed or restored -- are synced. Their runtime lineage is fresh.
        Any other variable's runtime lineage is only as fresh as its last run:
        after an upstream edit, a sibling the pass did not need (``a = f(x)``
        when only ``b = g(x)`` was asked for) still holds the value computed
        from the old ``x``, and its snapshot is the only place that knows.
        Syncing it laundered the stale value into a match, and the next cell
        that read it was served the old result.

        After upstream statements are executed/restored/skipped, ``variable_lineage``
        holds the authoritative lineage for each variable.  The simulation cache
        may store stale ``virtual_lineage`` values from an earlier run where
        forward propagation failed (e.g., the fallback lineage computed
        differently than the runtime lineage because a control structure was
        simulated as a single unit, or ``inspect.getsource`` returned different
        results).

        This method patches every cached ``virtual_lineage`` snapshot so that
        variables get their lineage updated to the authoritative value — but
        **only if the runtime lineage was produced by code within cells 0..idx**.
        Variables whose runtime lineage was produced by a *later* cell (beyond
        idx) are NOT synced.  This prevents downstream mutations from
        contaminating earlier cache entries.

        For example, if cell 2 produces ``df`` via ``df.sort_values(...)`` and
        cell 5 mutates it via ``df['SMA'] = ...``, after cell 5 executes the
        runtime lineage for ``df`` reflects the SMA mutation.  Without the
        scoping fix, syncing would update cell 2's cached virtual_lineage for
        ``df`` to the SMA-mutated lineage.  Then when cell 4 (a display cell)
        runs, reusing cache for cells 0-2 yields a virtual lineage that
        already matches the mutated actual lineage → no restoration → bug.

        With scoping, we check ``executed_cell_codes['df']`` to see which
        statement last produced ``df``'s runtime lineage.  If that statement
        is ``df['SMA'] = ...`` (from cell 5), it won't be found in cells
        0..2's trace segments, so cell 2's cache entry is NOT synced for
        ``df``.
        """
        if not len(self.cache):
            return

        updated = False
        # For each cache entry at index idx, collect ALL statement codes that
        # appear in the trace segments of cells 0..idx.  We only sync a
        # variable's lineage if the code that produced the current runtime
        # lineage (from executed_cell_codes) is among these statements.
        cumulative_stmt_codes = set()
        #: ``{var: (old, new)}`` synced so far; later entries' recorded inputs
        #: follow (below).
        moved: dict[str, tuple[str, str]] = {}
        for idx in range(len(self.cache)):
            entry = self.cache.entry(idx)
            if entry is None:
                continue
            cell_trace = entry.trace_segment
            for trace_entry in cell_trace:
                cumulative_stmt_codes.add(trace_entry.stmt_code)
                # A statement below a synced one read the value it now names.
                # Left behind, a loop there compared its recorded inputs with
                # the old lineage and read as reading changed data on every run
                # after a repair: ``results = {}`` and everything built on it
                # re-ran each time.
                input_hashes = trace_entry.input_hashes
                if moved and isinstance(input_hashes, dict):
                    for var_name, (old, new) in moved.items():
                        if input_hashes.get(var_name) == old:
                            input_hashes[var_name] = new

            cached_vl = entry.virtual_lineage
            for var_name in list(cached_vl.keys()):
                if var_name not in rerecorded:
                    continue
                if not self._should_sync_cache_var(var_name, cumulative_stmt_codes, cached_vl, idx):
                    continue
                # Safe to sync: the runtime lineage was produced by code within
                # cells 0..idx, so this is a valid forward-propagation correction.
                if cached_vl[var_name] != self.tracking_state.variable_lineage[var_name]:
                    moved[var_name] = (cached_vl[var_name], self.tracking_state.variable_lineage[var_name])
                cached_vl[var_name] = self.tracking_state.variable_lineage[var_name]
                updated = True

        if updated and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[UPSTREAM_DEBUG] Synced simulation cache lineages with runtime state (scoped to producing code)"
            )

    def _settle_loop_trust(self, sim: SimulationResult) -> None:
        """Decide which loop outputs memory is trusted for (``vars_mutated_by_loops``
        and ``vars_derived_from_loops`` of *sim*)."""
        # A reassignment accumulator that ALSO derives from an external input via
        # a non-loop producing statement (``result = np.zeros(N)``) must not join
        # the loop-trust set: editing that input and re-running the edited cell
        # first makes upstream_has_modifications False and every current-state
        # input lineage consistent, so the trust would serve a stale value. Drop
        # such accumulators so they follow the baseline lineage-mismatch path
        # (which re-executes correctly); a constant-init accumulator keeps the
        # new trust so one-shot iterables are not re-drained.
        externally_tainted = self.virtual_lineage.loop_rules.loop_accumulators_with_external_init(
            sim.vars_mutated_by_loops, sim.trace, sim.loop_target_vars
        )
        if externally_tainted:
            sim.vars_mutated_by_loops = sim.vars_mutated_by_loops - externally_tainted
            logger.debug(
                "[UPSTREAM_DEBUG] Dropped externally-dependent loop accumulators from loop-trust set: %s",
                externally_tainted,
            )

        sim.vars_derived_from_loops = loop_derived_vars(sim.vars_mutated_by_loops, sim.trace)

        # A loop whose data changed underneath it (a new file, not a code
        # edit) loses the trust, and so does everything built from it.
        changed_loops = self.virtual_lineage.loop_rules.loops_reading_changed_data(
            sim.vars_mutated_by_loops,
            sim.trace,
            sim.loop_target_vars,
            sim.vars_derived_from_loops,
        )
        if changed_loops:
            untrusted = loop_derived_vars(changed_loops, sim.trace)
            sim.vars_mutated_by_loops = sim.vars_mutated_by_loops - untrusted
            sim.vars_derived_from_loops = sim.vars_derived_from_loops - untrusted
            trace_event("loop_trust_dropped", vars=untrusted)

        if sim.loop_target_vars:
            logger.debug("[UPSTREAM_DEBUG] Loop target variables (iteration vars): %s", sim.loop_target_vars)


#: Builtins a ``%cash_on``-cell binding may call and still count as reading
#: nothing. Classes are judged separately (``_binds_without_reading``).
_PURE_BUILTINS = frozenset(
    {
        "len",
        "range",
        "min",
        "max",
        "abs",
        "round",
        "sorted",
        "sum",
        "zip",
        "enumerate",
        "reversed",
        "repr",
        "hash",
        "isinstance",
        "getattr",
    }
)

#: ``os.path`` functions that only compute a string -- ``DATA =
#: os.path.join(ROOT, "data")`` is as common in a setup cell as ``Path(...)``.
_PURE_PATH_FUNCS = frozenset(
    {
        "join",
        "dirname",
        "basename",
        "split",
        "splitext",
        "normpath",
        "abspath",
        "expanduser",
    }
)


def _binds_without_reading(code: str, user_ns: dict) -> bool:
    """True when *code* provably reads nothing outside the notebook.

    Every call must be a standard-library or builtin CLASS (``Path(...)``,
    ``datetime.date(...)``) or one of ``_PURE_BUILTINS``. A method call
    (``DATA.read_text()``), a third-party or user function, or anything that
    does not resolve fails -- deliberately: one ``load(DATA)`` wrongly adopted
    pins a stale value, while one wrongly refused only costs the first
    reader's cache.
    """

    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    stdlib = getattr(sys, "stdlib_module_names", frozenset())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = resolve_callee(node.func, user_ns, builtins_fallback=True)
        if callee is None:
            return False
        if isinstance(callee, type):
            root = (getattr(callee, "__module__", "") or "").split(".")[0]
            if root == "builtins" or root in stdlib:
                continue
            return False
        name = getattr(callee, "__name__", None)
        if name in _PURE_BUILTINS and getattr(builtins, name, None) is callee:
            continue
        if name in _PURE_PATH_FUNCS and getattr(os.path, name, None) is callee:
            continue
        return False
    return True
