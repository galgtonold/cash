from __future__ import annotations

import ast
import logging
import re
import textwrap
from typing import TYPE_CHECKING

from ..analysis import CodeAnalyzer
from ..cacheability import consumed_input_names
from .._trace import trace_event

if TYPE_CHECKING:
    from .mismatch_classifier import MismatchClassifier
    from .virtual_lineage import VirtualLineage

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
        consumable_broken_vars: set[str] | None = None,
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

        stmts_to_run_indices = self._schedule_consumable_producer_touches(
            stmts_to_run_indices, simulation_trace, consumable_broken_vars or set(),
        )

        stmts_to_run_indices = self._complete_shadowed_var_producers(
            stmts_to_run_indices, simulation_trace, virtual_lineage,
        )

        stmts_to_run_indices = self._schedule_conditional_producer_inits(
            stmts_to_run_indices, simulation_trace, broken_vars,
        )

        stmts_to_run_indices, restored_statements_info = self._schedule_file_write_statements(
            stmts_to_run_indices, simulation_trace, restored_statements_info,
            broken_vars, virtual_lineage,
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
            trace_event("schedule_reexec", stmt=stmt_code[:80])
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
            stmt_code, _outputs, inputs = entry[0], entry[1], entry[2]
            touched = consumable_broken_vars & set(inputs or ())
            if not touched or i in scheduled:
                continue
            try:
                consumed = consumed_input_names(ast.parse(textwrap.dedent(stmt_code)))
            except (SyntaxError, ValueError, TypeError):
                continue
            if touched & consumed:
                scheduled.add(i)
                if self.debug:
                    logger.debug(
                        "[UPSTREAM] Consumable-chain completion: scheduling [%s] "
                        "which fills/draws %s: %.60s",
                        i, sorted(touched & consumed), stmt_code,
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
            for v in entry[1]:  # outputs
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
                inputs, input_hashes = entry[2], (entry[3] or {})
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
                        produced = (simulation_trace[p][4] or {}).get(v)
                        if produced in wanted:
                            scheduled.add(p)
                            changed = True
                            if self.debug:
                                logger.debug(
                                    "[UPSTREAM] Shadow-completion: scheduling producer [%s] of "
                                    "shadowed '%s' (produced %s; consumed %s, final %s)",
                                    p, v, str(produced)[:8], str(consumed)[:8], str(final)[:8],
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
                    top_level_cache[idx] = CodeAnalyzer.top_level_assigned_names(
                        simulation_trace[idx][0]
                    )
                except (SyntaxError, ValueError):
                    top_level_cache[idx] = set()
            return top_level_cache[idx]

        scheduled = set(stmts_to_run_indices)
        additions: set[int] = set()

        for i in list(scheduled):
            entry = simulation_trace[i]
            outputs = entry[1]
            cond_only_vars = {
                v for v in outputs
                if v in broken_vars and v not in _top_level(i)
            }
            if not cond_only_vars:
                continue
            for v in cond_only_vars:
                # Already covered by an earlier unconditional producer in the plan?
                if any(
                    p < i and v in simulation_trace[p][1] and v in _top_level(p)
                    for p in scheduled
                ):
                    continue
                # Find the NEAREST earlier guaranteed init of v.
                for p in range(i - 1, -1, -1):
                    if v in simulation_trace[p][1] and v in _top_level(p):
                        additions.add(p)
                        if self.debug:
                            logger.debug(
                                "[UPSTREAM] Conditional-init: scheduling unconditional "
                                "initializer [%s] of '%s' behind conditional rebind [%s]",
                                p, v, i,
                            )
                        break

        if not additions:
            return stmts_to_run_indices
        return sorted(scheduled | additions)

    # Cheap textual pre-filter before running the full AST side-effect
    # analysis on a trace statement. Superset of the names in the write
    # detection tables (open modes, pandas to_*, save/savefig, json/pickle
    # dump, csv.writer, os/shutil mutations, pathlib write_text/bytes).
    _WRITE_MARKERS = (
        'open(', 'write', 'to_', 'save', 'dump', 'os.', 'shutil.',
    )

    def _schedule_file_write_statements(
        self,
        stmts_to_run_indices: list[int],
        simulation_trace: list,
        restored_statements_info: list[dict],
        broken_vars: set[str] | None = None,
        virtual_lineage: dict | None = None,
    ) -> tuple[list[int], list[dict]]:
        """Schedule upstream file-WRITING statements whose effect is stale (CAS-81/82).

        File writes have no variable edge, so the backward scan never
        schedules them: editing a writer cell and re-running only the reader
        served the stale pre-edit file (CAS-81); and when a writer cell's
        sibling statements DID re-run, the side-effect-only write statement
        was still skipped and the reader's freshness stayed decided against
        the pre-run file state (CAS-82).

        A writer statement is scheduled when

        * its code was never executed in this session (edited or new — the
          runtime records every executed file-writing statement's code in
          ``executed_write_stmt_codes``), or
        * any of its inputs is produced by an already-scheduled statement
          (its payload changed).

        Restored statements positioned after the first scheduled writer whose
        variables carry file dependencies are PROMOTED to re-execution: their
        plan-time restore was validated against the pre-write file state, and
        re-executing them in trace order (after the writer) overwrites the
        stale restore with a fresh read.
        """
        scheduled = set(stmts_to_run_indices)
        scheduled_outputs: set[str] = set()
        for idx in scheduled:
            scheduled_outputs.update(simulation_trace[idx][1])
        # A writer whose input is BROKEN is also stale, even when nothing in
        # the variable plan demands that input (the write may be the only
        # consumer — e.g. an edited payload that exists just to be dumped).
        changed_inputs = scheduled_outputs | set(broken_vars or ())

        tracking = getattr(self._classifier, '_tracking_state', None)
        runtime_lineage = getattr(tracking, 'variable_lineage', None) or {}

        def _input_changed(name: str) -> bool:
            if name in changed_inputs:
                return True
            if virtual_lineage is None:
                return False
            virt = virtual_lineage.get(name)
            run = runtime_lineage.get(name)
            return virt is not None and run is not None and virt != run

        writer_indices = self._find_stale_file_writer_indices(
            simulation_trace, scheduled_outputs=changed_inputs, skip=scheduled,
            virtual_lineage=virtual_lineage,
        )
        if not writer_indices:
            return stmts_to_run_indices, restored_statements_info

        scheduled.update(writer_indices)
        first_writer = min(writer_indices)

        # Re-materialise a scheduled writer's changed inputs: their producers
        # may be missing from the plan when the write is the only consumer.
        pending = list(writer_indices)
        while pending:
            w = pending.pop()
            for v in set(simulation_trace[w][2]):
                if not _input_changed(v):
                    continue
                for prod in range(w - 1, -1, -1):
                    if v in simulation_trace[prod][1]:
                        if prod not in scheduled:
                            scheduled.add(prod)
                            pending.append(prod)
                            if self.debug:
                                logger.debug(
                                    "[UPSTREAM] Scheduling producer [%s] of writer "
                                    "input '%s'", prod, v,
                                )
                        break

        # Every statement AFTER a scheduled writer whose variables carry file
        # dependencies must re-execute: its cached value (whether restored at
        # plan time or simply sitting untouched in memory) was validated
        # against the PRE-write file state. Re-execution happens in trace
        # order, after the writer, so its freshness is decided against the
        # freshly written file.
        tracking = getattr(self._classifier, '_tracking_state', None)
        executed_file_deps = getattr(tracking, 'executed_file_deps', None) or {}
        promoted: set[int] = set()
        for i in range(first_writer + 1, len(simulation_trace)):
            if i in scheduled:
                continue
            outputs = simulation_trace[i][1]
            if any(executed_file_deps.get(v) for v in outputs):
                scheduled.add(i)
                promoted.add(i)
                if self.debug:
                    logger.debug(
                        "[UPSTREAM] Promoting file-reader [%s] to re-exec (writer "
                        "scheduled at [%s]): %s",
                        i, first_writer, simulation_trace[i][0][:40],
                    )

        if promoted:
            promoted_codes = {simulation_trace[i][0] for i in promoted}
            restored_statements_info = [
                info for info in restored_statements_info
                if info.get('code') not in promoted_codes
            ]

        return sorted(scheduled), restored_statements_info

    def _find_stale_file_writer_indices(
        self,
        simulation_trace: list,
        scheduled_outputs: frozenset | set = frozenset(),
        skip: frozenset | set = frozenset(),
        virtual_lineage: dict | None = None,
    ) -> list[int]:
        """Trace indices of file-WRITING statements whose effect is stale.

        A writer is stale when its code was never executed in this session
        (edited/new — the runtime records executed file-writing statements'
        code in ``executed_write_stmt_codes``) or when one of its inputs is
        produced by an already-scheduled statement. Also used by the
        simulator to gate its no-broken-vars early return, so the common
        path stays cheap: a textual marker pre-filter runs before any AST
        analysis.
        """
        tracking = getattr(self._classifier, '_tracking_state', None)
        executed_writes = getattr(tracking, 'executed_write_stmt_codes', None)
        if executed_writes is None:
            return []
        runtime_lineage = getattr(tracking, 'variable_lineage', None) or {}

        from ..cacheability import statement_writes_files

        def _input_lineage_drifted(name: str) -> bool:
            # The writer's payload changed even though nothing in the variable
            # plan demands it: the simulated (current-code) lineage differs
            # from the lineage last seen at runtime.
            if virtual_lineage is None:
                return False
            virt = virtual_lineage.get(name)
            run = runtime_lineage.get(name)
            return virt is not None and run is not None and virt != run

        writer_indices: list[int] = []
        for i, entry in enumerate(simulation_trace):
            if i in skip:
                continue
            stmt_code, _outputs, inputs = entry[0], entry[1], entry[2]
            if not statement_writes_files(stmt_code):
                continue
            changed = stmt_code not in executed_writes
            inputs_changed = bool(set(inputs) & scheduled_outputs) or any(
                _input_lineage_drifted(v) for v in inputs
            )
            if changed or inputs_changed:
                writer_indices.append(i)
                if self.debug:
                    logger.debug(
                        "[UPSTREAM] File-writer scheduling: [%s] %s (changed=%s, "
                        "inputs_changed=%s)", i, stmt_code[:40], changed, inputs_changed,
                    )
        return writer_indices

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
