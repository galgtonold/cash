from __future__ import annotations

import ast
import builtins
import logging
import sys
import textwrap
from typing import TYPE_CHECKING

from cash.control_markers import iteration_digest

from ...analysis.cacheability import statement_writes_files
from ...analysis.code_analyzer import CodeAnalyzer
from ...analysis.mutations import consumed_input_names
from .._trace import trace_event
from ..cache_key import statement_source_hash
from ..stateful_carriers import carrier_kind_from_producer, stateful_carrier_kind
from ._types import ClassificationResult, ReexecutionPlan, SimulationResult, latest_producer
from .carrier_fills import fills_carrier
from .figure_writes import FigureWriteGuard
from .file_writers import FileWriterScheduler
from .mismatch_classifier import import_only

if TYPE_CHECKING:
    from .._protocols import ShellProtocol
    from ..tracking_state import TrackingState
    from .cache_probe import CacheProbe
    from .cache_restore import CacheRestorer
    from .loop_rules import LoopRules
    from .mismatch_classifier import MismatchClassifier
    from .simulated_callables import SimulatedCallables
    from .unsaved_edits import UnsavedEdits

logger = logging.getLogger(__name__)


def _is_definition(code: str) -> bool:
    """Is *code* a single top-level ``def`` / ``async def`` / ``class``?"""
    stripped = code.lstrip()
    if not stripped.startswith(("def ", "async def ", "class ", "@")):
        return False
    try:
        body = ast.parse(textwrap.dedent(code)).body
    except (SyntaxError, ValueError):
        return False
    return len(body) == 1 and isinstance(body[0], (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))


def _imported_roots(code: str) -> set[str]:
    """The top-level modules an import-only statement imports."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            roots.add(node.module.split(".")[0])
    return roots


class ReexecutionPlanner:
    """Phase 3 of NotebookSimulator: build the re-execution plan.

    :meth:`plan` turns a :class:`SimulationResult` and a
    :class:`ClassificationResult` into a :class:`ReexecutionPlan`, starting
    from the classifier's backward scan.
    """

    def __init__(
        self,
        shell: ShellProtocol,
        tracking_state: TrackingState,
        *,
        classifier: MismatchClassifier,
        restorer: CacheRestorer,
        loop_rules: LoopRules,
        unsaved_edits: UnsavedEdits,
        callables: SimulatedCallables,
        probe: CacheProbe,
    ) -> None:
        self.shell = shell
        self.tracking_state = tracking_state
        self.classifier = classifier
        self.restorer = restorer
        self.loop_rules = loop_rules
        self.unsaved_edits = unsaved_edits
        self.callables = callables
        self.probe = probe
        #: The file-writer pass, with the memos it keeps.
        self.file_writers = FileWriterScheduler(shell, tracking_state, probe)
        #: Refuses figure saves whose figure would be written blank.
        self.figure_writes = FigureWriteGuard(shell)

    @staticmethod
    def _drop_scheduled_from_restored(simulation_trace, stmts_to_run_indices, restored):
        """A statement scheduled to run is not also reported as restored.

        Several passes promote a statement the backward scan restored to a
        re-run and leave its restore entry behind; the badge then listed it
        twice, ``^CACHED: models = {}`` above ``^CACHED: models = {}``. Done
        once here, by code as the per-pass filters do, so no pass can leave one
        behind.
        """
        scheduled = {simulation_trace[i].stmt_code for i in stmts_to_run_indices}
        if not scheduled:
            return restored
        return [info for info in restored if info.get("code") not in scheduled]

    def plan(
        self,
        sim: SimulationResult,
        result: ClassificationResult,
        notebook_cells: list[str],
        relevant_read_paths: set[str] | None = None,
        relevant_read_paths_known: bool = True,
    ) -> ReexecutionPlan:
        """Pass 3: the statements to re-run, and those restored instead, so the
        names *result* found broken hold what a from-the-top run gives."""
        simulation_trace = sim.trace
        broken_vars = result.broken_vars
        virtual_lineage = sim.virtual_lineage
        virtual_modules = sim.virtual_modules
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[UPSTREAM_DEBUG] Simulation trace contents:")
            for i, entry in enumerate(simulation_trace):
                logger.debug("[UPSTREAM_DEBUG]   [%s] outputs=%s: %s...", i, entry.outputs, entry.stmt_code[:60])

        # ``(trace index, paths)`` of the writers this plan leaves out of date.
        stale_exports: list[tuple[int, list[str]]] = []
        stmts_to_run_indices, restored_statements_info, total_restore_time = self.classifier.backward_scan_pass(
            sim, result
        )

        stmts_to_run_indices = self._schedule_consumable_producer_touches(
            stmts_to_run_indices,
            simulation_trace,
            result.consumable_broken_vars,
        )

        stmts_to_run_indices = self._complete_shadowed_var_producers(
            stmts_to_run_indices,
            simulation_trace,
            virtual_lineage,
        )

        stmts_to_run_indices = self._schedule_conditional_producer_inits(
            stmts_to_run_indices,
            simulation_trace,
            broken_vars,
        )

        stmts_to_run_indices = self._complete_inputs_produced_before(
            stmts_to_run_indices,
            simulation_trace,
        )

        stmts_to_run_indices, restored_statements_info = self.file_writers.schedule(
            stmts_to_run_indices,
            simulation_trace,
            restored_statements_info,
            broken_vars,
            virtual_lineage,
            relevant_read_paths=relevant_read_paths,
            relevant_read_paths_known=relevant_read_paths_known,
            stale_exports=stale_exports,
        )

        stmts_to_run_indices, restored_statements_info = self._complete_stateful_carrier_history(
            stmts_to_run_indices,
            simulation_trace,
            restored_statements_info,
        )

        stmts_to_run_indices, restored_statements_info = self.figure_writes.guard_global_writes(
            stmts_to_run_indices,
            simulation_trace,
            restored_statements_info,
        )

        stmts_to_run_indices, restored_statements_info = self.figure_writes.guard_unfilled_writes(
            stmts_to_run_indices,
            simulation_trace,
            restored_statements_info,
        )

        stmts_to_run_indices = sorted(set(stmts_to_run_indices) | set(self.unrun_process_writers(simulation_trace)))

        # Again, now that the file-write passes are done: they PROMOTE restored
        # statements to re-execution (a restore validated before a scheduled
        # write), and a promoted statement's inputs were never cascaded. After
        # a restart ``counts = build_counts(events)`` restored, was promoted
        # behind ``OUT.mkdir()``, and ran without its ``def`` -- a NameError.
        # Inputs back, later producers forward, until neither adds anything: a
        # producer brought in for an input may itself be followed by writes.
        while True:
            before = len(stmts_to_run_indices)
            stmts_to_run_indices = self._complete_inputs_produced_before(
                stmts_to_run_indices,
                simulation_trace,
                virtual_lineage,
                virtual_modules,
            )
            stmts_to_run_indices = self.complete_later_producers(
                stmts_to_run_indices,
                simulation_trace,
            )
            stmts_to_run_indices = self._complete_import_path_setup(
                stmts_to_run_indices,
                simulation_trace,
            )
            if len(stmts_to_run_indices) == before:
                break

        restored_statements_info = self._drop_scheduled_from_restored(
            simulation_trace,
            stmts_to_run_indices,
            restored_statements_info,
        )
        skipped_metrics = self.restorer.collect_skipped_statement_metrics(
            simulation_trace,
            stmts_to_run_indices,
            restored_statements_info,
            virtual_modules,
            sim.stmt_lookup_times,
        )
        restored_statements_info.extend(skipped_metrics)
        restored_statements_info = self.file_writers.note_stale_exports(
            simulation_trace, stmts_to_run_indices, restored_statements_info, stale_exports
        )

        stmts_to_run_indices = self._schedule_loop_var_contexts(stmts_to_run_indices, simulation_trace)
        stmts_to_run_indices = self.loop_rules.filter_accumulator_reinits(
            stmts_to_run_indices, simulation_trace, sim.vars_mutated_by_loops
        )
        stmts_to_run_indices = self._dedup_sorted_indices(stmts_to_run_indices)
        restored_statements_info = self._drop_scheduled_from_restored(
            simulation_trace,
            stmts_to_run_indices,
            restored_statements_info,
        )

        statements_to_reexecute: list[str] = []
        for idx in stmts_to_run_indices:
            stmt_code = simulation_trace[idx].stmt_code
            statements_to_reexecute.append(stmt_code)
            trace_event("schedule_reexec", stmt=stmt_code[:80])
            logger.debug("[UPSTREAM] Scheduled for execution: %s", stmt_code[:40])

        restored_statements_info.reverse()

        vars_updated_by_trace: set[str] = set()
        for stmt_code in statements_to_reexecute:
            try:
                _, outputs = CodeAnalyzer.analyze_code_block(stmt_code)
                vars_updated_by_trace.update(outputs)
            except (SyntaxError, ValueError):
                logger.debug("Failed to analyze statement for variable outputs: %.40s", stmt_code)

        self.unsaved_edits.reapply_unsaved_extensions(
            broken_vars,
            vars_updated_by_trace,
            simulation_trace,
            notebook_cells,
            statements_to_reexecute,
        )

        return ReexecutionPlan(statements_to_reexecute, restored_statements_info, total_restore_time)

    def _schedule_consumable_producer_touches(
        self,
        stmts_to_run_indices: list[int],
        simulation_trace: list,
        consumable_broken_vars: set[str],
    ) -> list[int]:
        """Schedule every upstream statement that FILLS a broken consumable.

        The backward scan schedules the producers of a broken var — statements
        with the var among their trace ``outputs``. That is not enough to rebuild
        a consumable, because the statements that *fill* it usually do not own it
        as an output::

            q = Queue()                  outputs={'q'}   <- scheduled
            for i in range(3):
                q.put(i)                 outputs={'i'}   <- NOT scheduled

        ``put`` is not in ``MUTATING_METHODS`` (so the static detector does not
        attribute it), and the runtime's broad-precise mutation observation skips
        control-structure bodies (the simulation treats a loop as one unit), so
        the loop's mutation of ``q`` is invisible from both sides. Re-running only
        ``q = Queue()`` hands the consumer a fresh EMPTY queue — turning
        ``got=[]`` into ``got=[]`` again. (The same notebook with a TOP-LEVEL
        ``q.put(1)`` already worked: there the runtime observes the receiver and
        the trace does carry ``q`` as an output.)

        So for a var broken by the consumable channel specifically, also schedule
        any upstream statement that *draws on or feeds* it — reusing the same
        consumption analysis that scoped the channel. ``q.put(i)`` and a
        first-half consumer (``first3 = [next(it) for _ in range(3)]``) both
        qualify; a reporting read (``n = q.qsize()``) does not. Indices are
        merged and sorted, so ``q = Queue()`` still runs before the loop refills
        it.

        Scoped to ``consumable_broken_vars`` — vars this run's consumable probe
        actually flagged as diverged — so no other broken var's plan changes.
        """
        if not consumable_broken_vars:
            return stmts_to_run_indices
        scheduled = set(stmts_to_run_indices)
        for i, entry in enumerate(simulation_trace):
            stmt_code, _outputs, inputs = entry.stmt_code, entry.outputs, entry.inputs
            touched = consumable_broken_vars & set(inputs or ())
            if not touched or i in scheduled:
                continue
            try:
                consumed = consumed_input_names(ast.parse(textwrap.dedent(stmt_code)))
            except (SyntaxError, ValueError, TypeError):
                continue
            if touched & consumed:
                scheduled.add(i)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "[UPSTREAM] Consumable-chain completion: scheduling [%s] which fills/draws %s: %.60s",
                        i,
                        sorted(touched & consumed),
                        stmt_code,
                    )
        return sorted(scheduled)

    def _complete_shadowed_var_producers(
        self,
        stmts_to_run_indices: list[int],
        simulation_trace: list,
        virtual_lineage: dict[str, str],
    ) -> list[int]:
        """Re-schedule all producers of a shadowed variable consumed at a stale version.

        A variable produced by more than one upstream statement (shadowed —
        e.g. ``x`` reassigned across several cells) lives in a single namespace
        slot, which holds only its FINAL version. The backward scan resolves an
        input by name against that final version, so when a scheduled statement
        actually consumed an EARLIER version (``y = x + 100`` reading the cell-1
        ``x`` while a later cell rebinds ``x``), only the consumer is scheduled
        and it re-executes against the wrong value.

        Detect that precisely: a scheduled statement whose recorded input lineage
        for a shadowed variable differs from that variable's final lineage
        consumed a non-final version. To re-materialise the namespace faithfully
        in trace order, schedule (a) the producer whose output lineage equals the
        CONSUMED version (so the consumer sees the right value) and (b) the
        producer whose output lineage equals the FINAL version (so the namespace
        ends in the correct state). Matching on lineage — not "all producers" —
        is what keeps a fully-overwritten earlier definition (e.g. a dead
        ``x = 123`` superseded by a later ``x = {...}``) out of the schedule.
        Gated on the version mismatch, so the common "consume the final version"
        case (and every non-shadowed variable) is untouched.
        """
        producers: dict[str, list[int]] = {}
        for i, entry in enumerate(simulation_trace):
            for v in entry.outputs:
                producers.setdefault(v, []).append(i)
        shadowed = {v for v, idxs in producers.items() if len(idxs) > 1}
        if not shadowed:
            return stmts_to_run_indices

        scheduled = set(stmts_to_run_indices)
        changed = True
        while changed:
            changed = False
            for i in list(scheduled):
                entry = simulation_trace[i]
                if _is_definition(entry.stmt_code):
                    # A def or class reads its body's globals when called, at
                    # whatever version they then are: defining it consumes
                    # none. Taken as a consumer, `def plot_region` re-ran the
                    # whole back-test cell the summary cell had just rebuilt.
                    continue
                inputs, input_hashes = entry.inputs, (entry.input_hashes or {})
                for v in inputs:
                    if v not in shadowed:
                        continue
                    consumed = input_hashes.get(v)
                    final = virtual_lineage.get(v)
                    if consumed is None or final is None or consumed == final:
                        continue  # consumed the final version (or unknown) — no shadow hazard
                    wanted = {consumed, final}
                    for p in producers[v]:
                        if p in scheduled:
                            continue
                        produced = (simulation_trace[p].produced_lineages or {}).get(v)
                        if produced in wanted:
                            scheduled.add(p)
                            changed = True
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(
                                    "[UPSTREAM] Shadow-completion: scheduling producer [%s] of "
                                    "shadowed '%s' (produced %s; consumed %s, final %s)",
                                    p,
                                    v,
                                    str(produced)[:8],
                                    str(consumed)[:8],
                                    str(final)[:8],
                                )
        return sorted(scheduled)

    def _schedule_conditional_producer_inits(
        self,
        stmts_to_run_indices: list[int],
        simulation_trace: list,
        broken_vars: set[str],
    ) -> list[int]:
        """Schedule the unconditional initializer behind a conditional rebind.

        Pattern: ``x = 'default'`` followed by ``if flag: x = 'overridden'``.
        When ``flag`` flips and the conditional rebind is scheduled (it lists
        ``x`` among its outputs), the backward scan resolves ``x`` against that
        rebind and stops — it never schedules the earlier ``x = 'default'``
        initializer. Re-running only the conditional, whose branch no longer
        fires, leaves ``x`` at its stale prior value.

        Detect precisely: a SCHEDULED statement that outputs a broken var
        *conditionally only* — the var is in its ``outputs`` but NOT in its
        ``top_level_assigned_names`` (so it never assigns it unconditionally).
        For each such var, schedule the NEAREST EARLIER trace statement that
        DOES assign it unconditionally (in both ``outputs`` and
        ``top_level_assigned_names``) — the guaranteed init — so the namespace
        is correctly re-seeded before the (possibly skipped) conditional runs.

        Narrowing that keeps this off the cost-model floor-skip path: the
        trigger fires ONLY when a single scheduled statement *both* rebinds the
        var conditionally *and* lacks an unconditional assignment of it. A plain
        upstream ``scale = 17`` assigns its var unconditionally, so it is never a
        conditional rebind and never pulls in a spurious earlier init. We only
        reach back for an init when the scheduled statement genuinely cannot
        guarantee the var's value on its own.
        """
        if not broken_vars:
            return stmts_to_run_indices

        # Precompute per-statement unconditional-assignment names lazily.
        top_level_cache: dict[int, set[str]] = {}

        def _top_level(idx: int) -> set[str]:
            if idx not in top_level_cache:
                try:
                    top_level_cache[idx] = CodeAnalyzer.top_level_assigned_names(simulation_trace[idx].stmt_code)
                except (SyntaxError, ValueError):
                    top_level_cache[idx] = set()
            return top_level_cache[idx]

        scheduled = set(stmts_to_run_indices)
        additions: set[int] = set()

        for i in list(scheduled):
            entry = simulation_trace[i]
            outputs = entry.outputs
            cond_only_vars = {v for v in outputs if v in broken_vars and v not in _top_level(i)}
            if not cond_only_vars:
                continue
            for v in cond_only_vars:
                # Already covered by an earlier unconditional producer in the plan?
                if any(p < i and v in simulation_trace[p].outputs and v in _top_level(p) for p in scheduled):
                    continue
                # Find the NEAREST earlier guaranteed init of v.
                for p in range(i - 1, -1, -1):
                    if v in simulation_trace[p].outputs and v in _top_level(p):
                        additions.add(p)
                        logger.debug(
                            "[UPSTREAM] Conditional-init: scheduling unconditional "
                            "initializer [%s] of '%s' behind conditional rebind [%s]",
                            p,
                            v,
                            i,
                        )
                        break

        if not additions:
            return stmts_to_run_indices
        return sorted(scheduled | additions)

    def _complete_inputs_produced_before(
        self,
        stmts_to_run_indices: list[int],
        simulation_trace: list,
        virtual_lineage: dict[str, str] | None = None,
        virtual_modules: set[str] | None = None,
    ) -> list[int]:
        """Give every scheduled statement a producer, before it, of each input
        that is not live.

        The backward scan resolves an input by NAME, so a name bound twice
        resolves to its last producer. That is right while the name is live or
        the last producer runs first -- but ``import glob`` in cell 2 and again
        in cell 3, after a restart, resolved to cell 3's import, which runs
        AFTER cell 2's ``files = glob.glob(...)``: the replay raised
        ``NameError: glob`` and every jump downstream was refused until the
        imports were merged. The shadow pass above
        cannot see it: both imports bind the same lineage.

        With *virtual_lineage*, the globals the statement's callees read count
        as inputs too (``absent_callee_globals``).
        """
        user_ns = self.shell.user_ns
        live_lineage = self.tracking_state.variable_lineage
        scheduled = set(stmts_to_run_indices)
        pending = sorted(scheduled)
        while pending:
            i = pending.pop(0)
            inputs = set(simulation_trace[i].inputs or ())
            if virtual_lineage is not None:
                inputs |= self.callables.absent_callee_globals(inputs, virtual_lineage, virtual_modules or set())
            for v in sorted(inputs):
                if v not in user_ns and hasattr(builtins, v):
                    continue
                if v in user_ns and not self._live_is_behind_producer(simulation_trace, v, i, live_lineage):
                    continue
                # The LATEST producer before the statement must run, not just
                # any earlier one: a scheduled `sales['timestamp'] = ...` does
                # not stand in for the `sales['refund'] = ...` between it and
                # its reader. Accepting it rebuilt a cleaning cell without the
                # refund write after a restart, and the summary below showed 0
                # refunds for three stores, silently.
                p = latest_producer(simulation_trace, v, before=i)
                if p is not None and p not in scheduled:
                    scheduled.add(p)
                    pending.append(p)
                    trace_event(
                        "input_producer_completion",
                        stmt=simulation_trace[p].stmt_code[:80],
                        var=v,
                        consumer=simulation_trace[i].stmt_code[:80],
                    )
        return sorted(scheduled)

    def _live_is_behind_producer(self, simulation_trace: list, var: str, before: int, live_lineage: dict) -> bool:
        """Is live *var* NOT what its latest producer before *before* makes?

        Live used to be enough. A user ran the export cell (``results["f1"] =
        ...`` in place), re-ran the sweep cell (``results`` rebound, no f1),
        then the chart cell: the plan re-ran ``best = results.sort_values(
        ['f1', ...])`` on the live table without the f1 write above it, and
        raised ``UpstreamStateError: 'f1'``.

        Behind only on evidence: the live lineage is what an EARLIER producer
        made. A live value matching no producer (a loop's iterations, an edit
        the file does not have yet) keeps the old answer, which is what a
        per-iteration loop's statements need.
        """
        p = latest_producer(simulation_trace, var, before=before)
        live = live_lineage.get(var)
        if p is None or live is None:
            return False
        produced = (simulation_trace[p].produced_lineages or {}).get(var)
        if produced is None or produced == live:
            return False
        return any(
            (simulation_trace[q].produced_lineages or {}).get(var) == live
            for q in range(p)
            if var in simulation_trace[q].outputs
        )

    def unrun_process_writers(self, simulation_trace: list) -> list[int]:
        """The statements above the cell that changed the process's
        environment or working directory when they ran
        (``callee_reach.process_state_writes``) and have not run in this
        kernel.

        A restart puts both back as the shell started the kernel, and they
        are no variable a reader names: ``x = mylib.mode()`` reads
        ``os.environ["MODE"]`` inside ``mode``, ``mylib.read("d.txt")`` reads
        a path relative to the working directory. Run alone after a restart,
        the cell got the shell's values where Restart & Run All gives the
        notebook's. They run again, in notebook order, with their inputs.

        Only a statement known to have run (in this kernel or, kept on
        record, an earlier one): a setting edited and not run yet reaches no
        reader, as without a restart.
        """
        state = self.tracking_state
        if not simulation_trace or (not state.process_state_writers and self.probe.cash_instance is None):
            return []
        found: list[int] = []
        for idx, entry in enumerate(simulation_trace):
            source_hash = statement_source_hash(entry.stmt_code)
            if source_hash in state.process_state_ran:
                continue
            kinds = state.process_state_writers.get(source_hash)
            if kinds is None and self.probe.cash_instance is not None:
                kinds = self.probe.process_state(source_hash)
                if kinds:
                    state.process_state_writers[source_hash] = kinds
            if kinds:
                found.append(idx)
                trace_event("process_state_writer", stmt=entry.stmt_code[:80], kinds=sorted(kinds))
        return found

    def complete_later_producers(self, stmts_to_run_indices: list[int], simulation_trace: list) -> list[int]:
        """Every statement after a re-run producer of a variable that also
        writes it re-runs too.

        Re-running ``results = {}`` leaves ``results`` as it binds it: empty,
        unless the ``for`` loop below that fills it runs as well. The plan
        re-ran the init and the functions reading ``results``, not the loop,
        and the report raised ``UpstreamStateError: 'logreg'`` with the dict
        left empty. The backward completion cannot see it:
        it asks for the producer BEFORE a reader, and the init is one.

        Only a statement that READS the variable as it writes it continues
        what the earlier one built. One that only binds the name starts over
        and owes it nothing: a re-run ``for r in sorted(obs.run.unique())``
        pulled in a chart loop with a ``for r`` of its own, whose ``ax`` pulled
        in the UMAP loop ``for ax, col in zip(axes, ...)`` -- which then drew the
        new labels over an embedding nothing had rebuilt.
        """
        scheduled = set(stmts_to_run_indices)
        pending = sorted(scheduled)
        while pending:
            i = pending.pop(0)
            for v in simulation_trace[i].outputs:
                for p in range(i + 1, len(simulation_trace)):
                    if p in scheduled or v not in simulation_trace[p].outputs or v not in simulation_trace[p].inputs:
                        continue
                    scheduled.add(p)
                    pending.append(p)
                    trace_event(
                        "later_producer_completion",
                        stmt=simulation_trace[p].stmt_code[:80],
                        var=v,
                        after=simulation_trace[i].stmt_code[:80],
                    )
        return sorted(scheduled)

    @staticmethod
    def _complete_import_path_setup(stmts_to_run_indices: list[int], simulation_trace: list) -> list[int]:
        """Before a re-run import of a module that is not loaded, the notebook's
        earlier changes to ``sys.path``.

        ``sys.path.insert(0, lib)`` is a setting of ``sys``, replayed for a
        statement that reads ``sys``. An import does not read it, yet finding a
        module not loaded yet depends on it: after a restart, a jump below
        ``import bt`` re-ran the import without the insert above it and stopped
        on ``No module named 'bt'``.
        """

        scheduled = set(stmts_to_run_indices)
        added: set[int] = set()
        for i in sorted(scheduled):
            code = simulation_trace[i].stmt_code
            if not import_only(code) or all(m in sys.modules for m in _imported_roots(code)):
                continue
            for j in range(i):
                if j in scheduled or j in added:
                    continue
                entry_code = simulation_trace[j].stmt_code
                if "sys" in simulation_trace[j].outputs and "path" in entry_code:
                    added.add(j)
                    trace_event("import_path_setup", stmt=entry_code[:80], before=code[:80])
        return sorted(scheduled | added)

    def _complete_stateful_carrier_history(
        self,
        stmts_to_run_indices: list[int],
        simulation_trace: list,
        restored_statements_info: list[dict],
    ) -> tuple[list[int], list[dict]]:
        """Never re-execute a SUBSET of a stateful carrier's history.

        The backward scan resolves a scheduled statement's inputs by *value
        lineage*. A **stateful carrier** (see ``stateful_carriers``) breaks that
        model: the consumer depends on the carrier's hidden internal state, and
        no value-level edge records it. So the scan schedules the consumer and
        leaves the statements that ESTABLISHED the state behind, producing a
        result matching no execution of the notebook::

            [7] OMIT  rng = np.random.default_rng(7)     <- establishes position
            [8] RUN   steps = rng.standard_normal(n)     <- redraws from an
                                                            ALREADY-ADVANCED rng

            [9]  RUN   fig, ax = plt.subplots()          <- fresh BLANK figure
            [10] OMIT  ax.bar(names, totals)             <- fills it
            [11] OMIT  ax.set_title('Totals')
            [12] RUN   fig.savefig(path)                 <- writes the BLANK one
                                                            over the good chart

        Both are the same defect, so both take the same repair: for a scheduled
        statement consuming carrier ``v``, re-execute ``v``'s whole establishing
        history — its producer ``p``, plus every statement between ``p`` and the
        consumer that mutates ANY name ``p`` produced. The sibling-name clause is
        what catches ``ax.bar`` for a ``fig.savefig``: the write's receiver is
        ``fig``, the fill targets ``ax``, and only their CO-PRODUCTION by
        ``fig, ax = plt.subplots()`` ties them together.

        Gated on ``stateful_carrier_kind`` — a deliberately small, measured table.
        Widening it is not free: a hit FORCES a producer re-execution, and doing
        that for the consumable types (generators/queues), which ``consumables.py``
        already covers behind a divergence probe, re-initialised cross-cell
        accumulators and regressed 12 integration tests.

        The cross-cell-accumulator hazard does not apply here, for the
        reason the consumable channel gives: re-executing a producer chain IN
        TRACE ORDER is exactly what ``run_all`` does. The danger there is
        re-deriving an object while re-executing only PART of what fills it —
        precisely what this pass exists to prevent.
        """
        user_ns = self._user_ns()
        if user_ns is None:
            return stmts_to_run_indices, restored_statements_info

        scheduled = set(stmts_to_run_indices)
        added: set[int] = set()

        # Fixed point: a newly scheduled producer is itself a consumer whose own
        # carrier inputs must be completed. Bounded by the trace length.
        changed = True
        while changed:
            changed = False
            for i in sorted(scheduled):
                for v in simulation_trace[i].inputs:
                    try:
                        kind = stateful_carrier_kind(user_ns.get(v))
                    except (TypeError, ValueError, AttributeError, RecursionError):
                        kind = None
                    p = latest_producer(simulation_trace, v, i)
                    if kind is None and v not in user_ns and p is not None:
                        # No live object after a restart: recognise the carrier
                        # by the code that makes it.
                        kind = carrier_kind_from_producer(simulation_trace[p].stmt_code)
                    if kind is None or p is None:
                        continue
                    pending = set()
                    if p not in scheduled:
                        pending.add(p)
                    sibling_names = set(simulation_trace[p].outputs)
                    for j in range(p + 1, i):
                        if j in scheduled or j in pending:
                            continue
                        # Outputs catch the FLAT fill (``ax.bar(...)``). A loop
                        # is one trace entry whose outputs never mention the
                        # carrier, so the body has to be inspected directly or
                        # the figure is rebuilt from a subset of its history
                        # .
                        entry = simulation_trace[j]
                        if not fills_carrier(entry, sibling_names):
                            continue
                        # A second savefig is not a fill; the write gates decide
                        # whether it re-fires.
                        if set(entry.outputs) & sibling_names or not statement_writes_files(entry.stmt_code):
                            pending.add(j)
                    if not pending:
                        continue
                    # A fill we FORCE must be able to run. The loop above
                    # selects fills by co-produced NAME (`ax`), which says
                    # nothing about the data they read: `ax.plot(sub[...])`
                    # reads `sub`, `sub` is not a carrier, and the backward scan
                    # never treated it as a broken var, so nothing pulls its
                    # producer in. Scheduling the reader without the writer is
                    # a blocking report -- `NameError: name 'sub' is
                    # not defined`, surfaced as an UpstreamStateError on a
                    # completely unrelated cell, five cells blocked at once.
                    #
                    # It took eleven attempts to reproduce because `fig`, `ax`
                    # and `sub` come from one cell and are normally all present
                    # or all absent; both extremes are harmless (no live `fig`
                    # and this pass never fires, live `sub` and the fill just
                    # runs). Only a state that separates them reaches it.
                    pending |= self._producers_of_read_names(
                        simulation_trace,
                        pending,
                        scheduled,
                    )
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "[UPSTREAM] Carrier-history completion for '%s' (%s) consumed "
                            "by [%s]: scheduling %s so the carrier is not rebuilt from a "
                            "subset of its history",
                            v,
                            kind,
                            i,
                            sorted(pending),
                        )
                    for idx in sorted(pending):
                        trace_event(
                            "carrier_history_completion",
                            var=v,
                            kind=kind,
                            consumer=i,
                            stmt=simulation_trace[idx].stmt_code[:80],
                        )
                    scheduled |= pending
                    added |= pending
                    changed = True

        if added:
            # A statement promoted to re-execution must not ALSO appear as
            # restored: its plan-time restore was decided against the carrier's
            # stale state, and re-execution in trace order supersedes it.
            added_codes = {simulation_trace[i].stmt_code for i in added}
            restored_statements_info = [
                info for info in restored_statements_info if info.get("code") not in added_codes
            ]
        return sorted(scheduled), restored_statements_info

    def _producers_of_read_names(
        self,
        simulation_trace: list,
        pending: set[int],
        scheduled: set[int],
    ) -> set[int]:
        """Producers of the data the *pending* statements read, transitively.

        Only reached for statements the carrier-history pass has already decided
        to force, so the blast radius is the closure of what those statements
        need — not the notebook. A statement already scheduled, or already
        pending, is left alone; a name with no producer in the trace (a module,
        a builtin, something bound outside this notebook) contributes nothing.

        Transitive on purpose: ``ax.plot(sub[...])`` needs ``sub = mm[...]``,
        which needs ``mm = monthly_margin(tx)``. Stopping at one level would
        move the same NameError one statement up.

        **A producer that writes files is never added.**
        ``FileWriterScheduler.schedule`` runs BEFORE this pass, so anything
        added here has already bypassed its scope and repeatability gates —
        re-firing a ``to_csv(..., mode='a')`` would duplicate a line on disk,
        which a kernel restart cannot undo. Skipping it can leave the fill
        unrunnable, and that is the right trade: the failure is loud, names the
        missing variable and is fixed by running the cell, whereas a duplicated
        append is silent and permanent.
        """
        user_ns = self._user_ns()
        recorded = self.tracking_state.variable_lineage

        extra: set[int] = set()
        frontier = list(pending)
        while frontier:
            j = frontier.pop()
            expected = simulation_trace[j].input_hashes or {}  # input lineages at j
            for name in simulation_trace[j].inputs:
                # A value that is live and still what this statement read needs
                # no producer: the fill can run as it is. Without this, the
                # wider fill rule pulled `imp.plot.barh(..., ax=ax)` in and then
                # re-fitted the whole model chain behind a perfectly good `imp`.
                if name in user_ns and name in expected and recorded.get(name) == expected[name]:
                    continue
                producer = latest_producer(simulation_trace, name, j)
                if producer is None:
                    continue
                if producer in scheduled or producer in pending or producer in extra:
                    continue
                if statement_writes_files(simulation_trace[producer].stmt_code):
                    logger.debug(
                        "[UPSTREAM] Not scheduling file-writing producer [%s] "
                        "for '%s': re-firing it could duplicate on-disk output",
                        producer,
                        name,
                    )
                    continue
                extra.add(producer)
                frontier.append(producer)
        return extra

    def _user_ns(self) -> dict:
        return self.shell.user_ns

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
    ) -> bool:
        """Return True if *stmt* is a loop-var assignment for a scheduled iteration context."""
        for j in range(i + 1, min(i + 4, len(simulation_trace))):
            next_stmt = simulation_trace[j].stmt_code
            digest = iteration_digest(next_stmt)
            if digest is not None and digest in scheduled_contexts:
                if len(outputs) == 1:
                    var_name = list(outputs)[0]
                    if f"{var_name} = " in stmt_code or f"{var_name}=" in stmt_code:
                        return True
            elif digest is not None:
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
        stmts_set = set(stmts_to_run_indices)

        scheduled_contexts: set[str] = set()
        for idx in stmts_to_run_indices:
            digest = iteration_digest(simulation_trace[idx].stmt_code)
            if digest is not None:
                scheduled_contexts.add(digest)

        if not scheduled_contexts:
            return stmts_to_run_indices

        additional_indices: list[int] = []
        for i, entry in enumerate(simulation_trace):
            stmt_code, outputs = entry.stmt_code, entry.outputs
            if i in stmts_set or iteration_digest(stmt_code) is not None:
                continue
            if self._is_loop_var_assignment_for_context(
                i,
                stmt_code,
                outputs,
                simulation_trace,
                scheduled_contexts,
            ):
                logger.debug("[UPSTREAM] Adding loop var assignment for scheduled context: %s", stmt_code[:40])
                additional_indices.append(i)

        return stmts_to_run_indices + additional_indices
