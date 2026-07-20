from __future__ import annotations

import ast
import logging
import os
import re
import textwrap
import warnings
from typing import TYPE_CHECKING

from ...exceptions import CashWarning
from ...utils import resolve_file_dep_path
from ..analysis import CodeAnalyzer
from ..cacheability import consumed_input_names, statement_saves_current_pyplot_figure
from ..cache_key import write_provenance_key
from ..file_dep_snapshot import file_dep_is_fresh
from .._trace import trace_event
from .stateful_carriers import stateful_carrier_kind

if TYPE_CHECKING:
    from .mismatch_classifier import MismatchClassifier
    from .virtual_lineage import VirtualLineage

logger = logging.getLogger(__name__)

# A pyplot call that REGISTERS a new current figure in the process-global Gcf
# registry -- what ``plt.gcf()`` (and therefore ``plt.savefig()``) resolves to.
# Used to find the current-figure producer for a module-level save when the
# producer bound no name for the carrier-output check to catch (``plt.figure()``).
_PYPLOT_FIGURE_MAKER = re.compile(
    r'\b(?:plt|pyplot)\s*\.\s*(?:subplots|subplot_mosaic|figure|subplot|axes)\b'
)


_CONTROL_NODES = (
    ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.AsyncWith, ast.Try,
)


def _control_body_touches(code: str, sibling_names: set[str]) -> bool:
    """True when a CONTROL STRUCTURE's body calls a method on one of *sibling_names*.

    The simulation treats a loop / ``if`` / ``with`` as ONE trace entry and does
    not surface the mutations performed inside its body, so the statement's
    recorded outputs never mention ``ax`` for::

        for c in summary.columns:
            ax.plot(range(len(summary)), summary[c].values, label=c)

    :meth:`ReexecutionPlanner._complete_stateful_carrier_history` keys on exactly
    those outputs, so the loop was left out of the plan while
    ``fig, ax = plt.subplots()`` and ``fig.savefig(path)`` were scheduled -- the
    figure was rebuilt EMPTY and the blank PNG was written over the good chart
    (CAS-213). That method's docstring already describes this failure for the
    flat ``ax.bar(...)`` form; only the loop shape escaped, because the flat one
    IS visible in the outputs.

    Deliberately restricted to control structures: the flat form is already
    covered by the outputs check, and widening this to plain statements would
    also promote pure reads (``ax.get_title()``), risking exactly the
    over-scheduling regressions that method is documented to be narrow about.
    """
    if not sibling_names:
        return False
    try:
        tree = ast.parse(textwrap.dedent(code))
    except (SyntaxError, ValueError, TypeError):
        return False
    for node in ast.walk(tree):
        if not isinstance(node, _CONTROL_NODES):
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and isinstance(sub.func.value, ast.Name)
                    and sub.func.value.id in sibling_names):
                return True
    return False


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
        relevant_read_paths: set[str] | None = None,
        relevant_read_paths_known: bool = True,
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
            relevant_read_paths=relevant_read_paths,
            relevant_read_paths_known=relevant_read_paths_known,
        )

        stmts_to_run_indices, restored_statements_info = self._complete_stateful_carrier_history(
            stmts_to_run_indices, simulation_trace, restored_statements_info,
        )

        stmts_to_run_indices, restored_statements_info = self._guard_global_figure_writes(
            stmts_to_run_indices, simulation_trace, restored_statements_info,
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

    def _latest_producer(self, simulation_trace: list, var: str, before: int) -> int | None:
        """Index of the LAST statement before *before* that outputs *var*."""
        for p in range(before - 1, -1, -1):
            if var in simulation_trace[p][1]:
                return p
        return None

    def _complete_stateful_carrier_history(
        self,
        stmts_to_run_indices: list[int],
        simulation_trace: list,
        restored_statements_info: list[dict],
    ) -> tuple[list[int], list[dict]]:
        """Never re-execute a SUBSET of a stateful carrier's history (CAS-175/178).

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

        The CAS-75 cross-cell-accumulator hazard does not apply here, for the
        reason the consumable channel gives: re-executing a producer chain IN
        TRACE ORDER is exactly what ``run_all`` does. The danger there is
        re-deriving an object while re-executing only PART of what fills it —
        precisely what this pass exists to prevent.
        """
        user_ns = getattr(getattr(self._virtual_lineage, 'shell', None), 'user_ns', None)
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
                for v in simulation_trace[i][2]:  # inputs
                    try:
                        kind = stateful_carrier_kind(user_ns.get(v))
                    except (TypeError, ValueError, AttributeError, RecursionError):
                        kind = None
                    if kind is None:
                        continue
                    p = self._latest_producer(simulation_trace, v, i)
                    if p is None:
                        continue
                    pending = set()
                    if p not in scheduled:
                        pending.add(p)
                    sibling_names = set(simulation_trace[p][1])
                    for j in range(p + 1, i):
                        if j in scheduled or j in pending:
                            continue
                        # Outputs catch the FLAT fill (``ax.bar(...)``). A loop
                        # is one trace entry whose outputs never mention the
                        # carrier, so the body has to be inspected directly or
                        # the figure is rebuilt from a subset of its history
                        # (CAS-213).
                        if (set(simulation_trace[j][1]) & sibling_names
                                or _control_body_touches(
                                    simulation_trace[j][0], sibling_names)):
                            pending.add(j)
                    if not pending:
                        continue
                    if self.debug:
                        logger.debug(
                            "[UPSTREAM] Carrier-history completion for '%s' (%s) consumed "
                            "by [%s]: scheduling %s so the carrier is not rebuilt from a "
                            "subset of its history",
                            v, kind, i, sorted(pending),
                        )
                    for idx in sorted(pending):
                        trace_event(
                            "carrier_history_completion", var=v, kind=kind,
                            consumer=i, stmt=simulation_trace[idx][0][:80],
                        )
                    scheduled |= pending
                    added |= pending
                    changed = True

        if added:
            # A statement promoted to re-execution must not ALSO appear as
            # restored: its plan-time restore was decided against the carrier's
            # stale state, and re-execution in trace order supersedes it.
            added_codes = {simulation_trace[i][0] for i in added}
            restored_statements_info = [
                info for info in restored_statements_info
                if info.get('code') not in added_codes
            ]
        return sorted(scheduled), restored_statements_info

    def _latest_current_figure_producer(
        self, simulation_trace: list, before: int, user_ns,
    ) -> int | None:
        """Trace index of the statement that most recently registered the current figure.

        Models what ``plt.gcf()`` (hence ``plt.savefig()``) would resolve to:
        scanning backward from *before*, the nearest statement that either binds a
        matplotlib Figure/Axes carrier (``fig, ax = plt.subplots()``) or textually
        calls a pyplot figure-registering function (``plt.figure()`` bound to no
        name). ``None`` when no figure producer precedes the write in the trace.
        """
        for p in range(before - 1, -1, -1):
            if user_ns is not None:
                for out in simulation_trace[p][1]:  # outputs
                    try:
                        if stateful_carrier_kind(user_ns.get(out)) in (
                            'matplotlib Figure', 'matplotlib Axes',
                        ):
                            return p
                    except (TypeError, ValueError, AttributeError, RecursionError):
                        pass
            if _PYPLOT_FIGURE_MAKER.search(simulation_trace[p][0]):
                return p
        return None

    def _guard_global_figure_writes(
        self,
        stmts_to_run_indices: list[int],
        simulation_trace: list,
        restored_statements_info: list[dict],
    ) -> tuple[list[int], list[dict]]:
        """Refuse to re-run a ``plt.savefig()`` orphaned from its figure (CAS-187).

        ``plt.savefig(path)`` writes pyplot's CURRENT figure through the
        process-global ``Gcf`` registry; its only variable input is the module
        ``plt``. Unlike the receiver-bound ``fig.savefig(path)`` that
        ``_complete_stateful_carrier_history`` defends, there is NO value-level
        edge for the planner to follow to the figure. So if the plan schedules
        such a write while the statement that registered the current figure
        (``fig, ax = plt.subplots()`` / ``plt.figure()``) is NOT scheduled,
        re-running the write calls ``plt.gcf()``, which INVENTS a blank default
        figure and flushes it over the user's chart -- a silent on-disk wrong
        answer with no in-notebook signal. (Measured: a 960x540 chart becomes a
        640x480 blank -- exactly matplotlib's default figure geometry.)

        We cannot follow the Gcf edge, so -- in the spirit of CAS-172 -- we bound
        the CONSEQUENCE instead of trying to enumerate what might skip the
        producer: drop the orphaned write from the plan and warn. The user
        re-runs the cell; the good chart on disk is left untouched. Governing
        principle: cash must never write a figure the user did not draw -- either
        the real one, or a loud refusal.

        Fires ONLY for a module-level ``plt.savefig`` whose most-recent preceding
        figure producer is absent from the plan. On the healthy path (first run,
        or any plan that also rebuilds the figure) the producer is scheduled and
        this is a no-op, so ``fig.savefig`` and the CAS-144/175 tests are
        untouched. A missed re-save is acceptable; a blank PNG on disk is not.
        """
        user_ns = getattr(getattr(self._virtual_lineage, 'shell', None), 'user_ns', None)
        scheduled = set(stmts_to_run_indices)
        refused: set[int] = set()

        for w in sorted(scheduled):
            code = simulation_trace[w][0]
            if not statement_saves_current_pyplot_figure(code, user_ns):
                continue
            producer = self._latest_current_figure_producer(simulation_trace, w, user_ns)
            if producer is not None and producer in scheduled:
                continue  # the figure is being (re)built coherently -- allow
            refused.add(w)
            self._warn_orphaned_figure_write(code, producer, w)

        if not refused:
            return stmts_to_run_indices, restored_statements_info

        remaining = [i for i in stmts_to_run_indices if i not in refused]
        # A refused write must not linger in the restored set either -- it is not
        # being run at all this pass.
        refused_codes = {simulation_trace[i][0] for i in refused}
        restored_statements_info = [
            info for info in restored_statements_info
            if info.get('code') not in refused_codes
        ]
        return remaining, restored_statements_info

    def _warn_orphaned_figure_write(self, code: str, producer: int | None, w: int) -> None:
        """Emit the CAS-187 refusal as a CashWarning (and a trace/debug record)."""
        warnings.warn(
            "cash refused to re-run a plt.savefig() during upstream reconstruction: "
            "the statement that drew the current figure is not being re-run, so the "
            "save would flush a blank/wrong figure over your chart on disk. Re-run "
            "the plot cell to regenerate it. (plt.savefig() has no variable link to "
            "its figure; use fig.savefig(path) to let cash track it. CAS-187.)",
            CashWarning,
            stacklevel=2,
        )
        trace_event("refuse_orphaned_figure_write", stmt=code[:80], producer=producer)
        if self.debug:
            logger.debug(
                "[UPSTREAM] CAS-187 refusing orphaned plt.savefig at [%s] (figure "
                "producer %s not scheduled): %.60s", w, producer, code,
            )

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
        relevant_read_paths: set[str] | None = None,
        relevant_read_paths_known: bool = True,
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
        # Live kernel namespace, to tell "provably unchanged" apart from
        # "absent". A writer input that is gone from user_ns (kernel restart,
        # ``del``, or an isolated re-run whose producer never ran this session)
        # has no runtime lineage -- but neither does an unchanged one, so the
        # lineage comparison below reads absent-as-unchanged and never schedules
        # the input's producer. That left ``df.to_csv(path)`` scheduled WITHOUT
        # its producer, so it ran against a missing ``df`` and raised NameError
        # post-restart, poisoning the whole notebook (CAS-153).
        user_ns = getattr(getattr(self._virtual_lineage, 'shell', None), 'user_ns', None)

        def _input_changed(name: str) -> bool:
            if name in changed_inputs:
                return True
            if user_ns is not None and name not in user_ns:
                # Genuinely absent from the live namespace: schedule its producer
                # so the writer runs against a re-materialised (cache-restored or
                # recomputed) input rather than crashing -- exactly what run_all
                # already does for the same notebook.
                return True
            if virtual_lineage is None:
                return False
            virt = virtual_lineage.get(name)
            run = runtime_lineage.get(name)
            return virt is not None and run is not None and virt != run

        writer_indices = self._find_stale_file_writer_indices(
            simulation_trace, scheduled_outputs=changed_inputs, skip=scheduled,
            virtual_lineage=virtual_lineage,
            relevant_read_paths=relevant_read_paths,
            relevant_read_paths_known=relevant_read_paths_known,
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
        relevant_read_paths: set[str] | None = None,
        relevant_read_paths_known: bool = True,
    ) -> list[int]:
        """Trace indices of file-WRITING statements whose effect is stale.

        A writer is stale when its code was never executed in this session
        (edited/new — the runtime records executed file-writing statements'
        code in ``executed_write_stmt_codes``) or when one of its inputs is
        produced by an already-scheduled statement. Also used by the
        simulator to gate its no-broken-vars early return, so the common
        path stays cheap: a textual marker pre-filter runs before any AST
        analysis.

        Scoped to the current cell (CAS-193/196/200): a writer whose resolvable
        output path is read by NO consumer relevant to this reconstruction
        (``relevant_read_paths``) is an unrelated / terminal side-effect and is
        never scheduled — re-firing it can only repeat an external write (a
        non-idempotent ``mode='a'`` append corrupts the file) without helping
        reconstruct any value the current cell needs. The gate applies only when
        the read set is fully known and the writer's own path resolves; every
        uncertain case falls through to the prior (conservative) behaviour.
        """
        tracking = getattr(self._classifier, '_tracking_state', None)
        executed_writes = getattr(tracking, 'executed_write_stmt_codes', None)
        if executed_writes is None:
            return []
        runtime_lineage = getattr(tracking, 'variable_lineage', None) or {}

        from ..cacheability import (
            REPEATABILITY_ACCUMULATING,
            statement_write_repeatability,
            statement_writes_files,
        )

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
            # Scope gate: skip a writer whose output file no relevant consumer
            # reads (CAS-193/196/200). Its write runs when the user runs its own
            # cell; reconstruction of an unrelated cell must never re-fire it.
            if self._writer_output_unread(
                stmt_code, relevant_read_paths, relevant_read_paths_known,
            ):
                if self.debug:
                    logger.debug(
                        "[UPSTREAM] File-writer output read by no relevant "
                        "consumer; not scheduling (scope): %s", stmt_code[:60],
                    )
                continue
            # Repeatability gate (CAS-210). The scope gate above keys on
            # RELEVANCE -- "does a relevant consumer read this file?" -- not on
            # whether repeating the write is safe. Those coincide in the cases
            # it was built for, which is why it reads as a correctness
            # guarantee; they come apart for an APPEND whose file is also read.
            # There the gate steps aside and the re-fire silently duplicates the
            # payload on disk, damage a kernel restart cannot undo.
            #
            # Not re-firing is also what the user's own kernel does: the write
            # runs when they run its own cell, and they did not run it here.
            #
            # ONLY a PROVABLE append is refused. CAS-210 also suggests refusing
            # UNKNOWN repeatability, and that was measured rather than reasoned
            # about: the two failure modes are not symmetric. Refusing a write
            # that SHOULD re-fire leaves the reader on stale data every time,
            # silently, on a common path -- the measurement broke two CAS-82
            # tests exactly that way. Leaving an unprovable append re-firing
            # costs a duplicated line in an uncommon shape. Narrow is the right
            # side of that trade; see the measurement recorded on CAS-210.
            if statement_write_repeatability(stmt_code) == REPEATABILITY_ACCUMULATING:
                if self.debug:
                    logger.debug(
                        "[UPSTREAM] File-writer is a non-idempotent append; not "
                        "re-firing (repeatability): %s", stmt_code[:60],
                    )
                continue
            changed = stmt_code not in executed_writes
            inputs_changed = bool(set(inputs) & scheduled_outputs) or any(
                _input_lineage_drifted(v) for v in inputs
            )
            # ``changed`` fires for every writer after a kernel restart because
            # ``executed_write_stmt_codes`` is session-scoped and starts empty.
            # Before re-firing such a writer (which would re-run its
            # non-idempotent side effect and re-derive stale data), check its
            # persisted provenance: if the payload is unchanged AND the output
            # file is still fresh on disk, the effect is already applied — do
            # NOT schedule it (CAS-153 round-3). Only short-circuit the pure
            # ``changed`` path; a genuine input change (``inputs_changed``) must
            # always re-run.
            if changed and not inputs_changed and self._writer_output_already_fresh(
                stmt_code, inputs, virtual_lineage, runtime_lineage,
            ):
                changed = False
                if self.debug:
                    logger.debug(
                        "[UPSTREAM] File-writer effect already fresh on disk; "
                        "not re-firing: %s", stmt_code[:60],
                    )
            if changed or inputs_changed:
                writer_indices.append(i)
                if self.debug:
                    logger.debug(
                        "[UPSTREAM] File-writer scheduling: [%s] %s (changed=%s, "
                        "inputs_changed=%s)", i, stmt_code[:40], changed, inputs_changed,
                    )
        return writer_indices

    @staticmethod
    def _normalize_path_forms(path: str) -> set[str]:
        """Comparable forms of a file path: resolved-abspath (normcased) + basename.

        Both sides of the writer-output vs relevant-reads comparison are reduced
        to this set so a cwd-relative write (``'audit.log'``) and an absolute read
        of the same file compare equal, while a basename match keeps the gate
        biased toward NOT suppressing (a false "read" only foregoes a suppression;
        a false "unread" would wrongly drop a needed write).
        """
        forms: set[str] = set()
        try:
            resolved = resolve_file_dep_path(path)
        except (OSError, ValueError, TypeError):
            resolved = None
        for candidate in (resolved, path):
            if not candidate:
                continue
            try:
                forms.add(os.path.normcase(os.path.abspath(candidate)))
                forms.add(os.path.normcase(os.path.basename(candidate)))
            except (OSError, ValueError, TypeError):
                continue
        return forms

    def _writer_output_unread(
        self,
        stmt_code: str,
        relevant_read_paths: set[str] | None,
        relevant_read_paths_known: bool,
    ) -> bool:
        """True when a writer's output file is read by no relevant consumer.

        The scope gate for CAS-193/196/200. Conservative in every uncertain
        case — returns ``False`` (do not suppress) unless the read set is fully
        known AND the writer's output path(s) all resolve statically AND none of
        them match any relevant read path. Then the writer is an unrelated /
        terminal side-effect for this cell and must not be re-fired.
        """
        if not relevant_read_paths_known or relevant_read_paths is None:
            return False
        from ..cacheability import statement_written_paths
        user_ns = getattr(getattr(self._virtual_lineage, 'shell', None), 'user_ns', None)
        written = statement_written_paths(stmt_code, namespace=user_ns)
        if not written:
            return False  # unresolvable target -> stay conservative
        read_forms: set[str] = set()
        for rp in relevant_read_paths:
            read_forms |= self._normalize_path_forms(rp)
        for wp in written:
            if self._normalize_path_forms(wp) & read_forms:
                return False  # this output IS read by a relevant consumer
        return True

    def _writer_output_already_fresh(
        self,
        stmt_code: str,
        inputs,
        virtual_lineage: dict | None,
        runtime_lineage: dict,
    ) -> bool:
        """True when a writer's effect is already on disk and provably current.

        Consulted only for a writer that looks ``changed`` purely because its
        code was never seen THIS session (the post-restart case). Returns True —
        meaning "do not re-fire" — only when the persisted provenance for this
        exact writer source is present, EVERY recorded output path is still fresh
        (:func:`file_dep_is_fresh`), AND every recorded input lineage still
        matches the writer's current (simulated / runtime) lineage.

        Conservative in every uncertain case: no backend, missing provenance, an
        unreadable / stale output file, or a drifted input lineage all return
        False, so the writer is scheduled exactly as before (CAS-153 round-3).
        """
        cash = getattr(self._virtual_lineage, 'cash_instance', None)
        backend = getattr(cash, 'backend', None) if cash is not None else None
        if backend is None or not hasattr(backend, 'get_metadata'):
            return False
        try:
            record = backend.get_metadata(write_provenance_key(stmt_code))
        except (OSError, TypeError, ValueError, AttributeError):
            return False
        if not record or not record.get('write_provenance'):
            return False
        paths = record.get('paths') or []
        file_deps = record.get('file_deps') or {}
        if not paths or not file_deps:
            return False
        for path in paths:
            stored = file_deps.get(path)
            if not stored:
                return False
            fresh, _reason = file_dep_is_fresh(path, stored)
            if not fresh:
                return False
        # The output on disk is only the writer's CURRENT output if its inputs
        # still carry the lineage they had when it was written. A drift means
        # the file was produced from a now-stale payload.
        stored_lineages = record.get('input_lineages') or {}
        for var, stored_lineage in stored_lineages.items():
            current = (virtual_lineage or {}).get(var)
            if current is None:
                current = runtime_lineage.get(var)
            if current != stored_lineage:
                return False
        return True

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
