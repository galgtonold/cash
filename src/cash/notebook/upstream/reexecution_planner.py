from __future__ import annotations

import ast
import builtins
import logging
import os
import re
import stat
import sys
import textwrap
import types
from typing import TYPE_CHECKING

from cash.control_markers import iteration_digest

from ...analysis.ast_util import called_names
from ...analysis.cacheability import (
    REPEATABILITY_ACCUMULATING,
    REPEATABILITY_REPLACING,
    consumed_input_names,
    resolve_literal_path,
    statement_calls_user_writer,
    statement_saves_current_pyplot_figure,
    statement_write_repeatability,
    statement_writes_files,
    statement_written_paths,
)
from ...analysis.code_analyzer import CodeAnalyzer
from ...diagnostics import warn_diagnostic
from ...exceptions import CashWarning
from ...tracking.file_dep_snapshot import snapshot_is_fresh
from ...utils import resolve_file_dep_path
from .._trace import trace_event
from ..cache_key import called_function_globals, write_provenance_key
from ..cache_status import CacheStatus
from ..carrier_history import carrier_history_fingerprint
from ..stateful_carriers import carrier_kind_from_producer, stateful_carrier_kind
from .mismatch_classifier import import_only
from .virtual_lineage import key_lineages

if TYPE_CHECKING:
    from .mismatch_classifier import MismatchClassifier
    from .virtual_lineage import VirtualLineage

logger = logging.getLogger(__name__)

# A pyplot call that REGISTERS a new current figure in the process-global Gcf
# registry -- what ``plt.gcf()`` (and therefore ``plt.savefig()``) resolves to.
# Used to find the current-figure producer for a module-level save when the
# producer bound no name for the carrier-output check to catch (``plt.figure()``).
_PYPLOT_FIGURE_MAKER = re.compile(r"\b(?:plt|pyplot)\s*\.\s*(?:subplots|subplot_mosaic|figure|subplot|axes)\b")


_CONTROL_NODES = (
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.If,
    ast.With,
    ast.AsyncWith,
    ast.Try,
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
    . That method's docstring already describes this failure for the
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
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and isinstance(sub.func.value, ast.Name)
                and sub.func.value.id in sibling_names
            ):
                return True
    return False


def _literal_path_bindings(simulation_trace: list | None) -> dict[str, str]:
    """``{name: path}`` for names the notebook binds to one literal path.

    ``OUT = Path('report')``, ``EXPORTS = BASE / 'exports'``: what a writer's
    ``OUT / 'chart.png'`` means when the kernel does not hold ``OUT`` yet.
    A name bound more than once, or by anything else, is left out.
    """

    bound: dict[str, str | None] = {}
    for entry in simulation_trace or ():
        outputs = entry[1]
        if not outputs:
            continue
        value = None
        try:
            node = ast.parse(entry[0]).body
        except (SyntaxError, ValueError, TypeError):
            node = []
        if (
            len(node) == 1
            and isinstance(node[0], ast.Assign)
            and len(node[0].targets) == 1
            and isinstance(node[0].targets[0], ast.Name)
        ):
            known = {k: v for k, v in bound.items() if v is not None}
            value = resolve_literal_path(node[0].value, known)
        for name in outputs:
            bound[name] = value if name not in bound or bound[name] == value else None
    return {name: path for name, path in bound.items() if path is not None}


def _only_defines(code: str) -> bool:
    """True when *code* only defines functions or classes."""
    try:
        body = ast.parse(textwrap.dedent(code)).body
    except (SyntaxError, ValueError, TypeError):
        return False
    return bool(body) and all(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in body)


def _root_name(node: ast.AST) -> str | None:
    """``axes[0]`` / ``ax.twinx()`` / ``*axs`` -> the name they are reached from."""
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _passes_carrier_to_a_call(code: str, sibling_names: set[str]) -> bool:
    """True when the statement calls into one of *sibling_names* or hands it to a call.

    Two ways to draw on a figure that its recorded outputs do not show:

    * a call on a PART of it -- ``axes[1].set_xlabel(...)``,
      ``ax.xaxis.set_major_formatter(...)``: the receiver is reached from the
      carrier, but is not a bare name;
    * a call on SOMETHING ELSE that receives it -- ``tot.plot(ax=axes[0])``,
      ``imp.plot.barh(..., ax=ax)``, ``sns.barplot(data=df, ax=ax)``,
      ``draw_panel(ax)``. The outputs name the receiver (``tot``, ``imp``),
      never ``ax``.

    Missing either re-drew the figure without it: round 21's blank chart
    (r21s2), and again in the replay acceptance corpus. A call that merely
    reads the axes is re-run too: within the carrier's own history, one
    statement too many is harmless; one too few writes a wrong file.
    """
    if not sibling_names:
        return False
    try:
        tree = ast.parse(textwrap.dedent(code))
    except (SyntaxError, ValueError, TypeError):
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _root_name(node.func) in sibling_names:
            return True
        for arg in [*node.args, *(kw.value for kw in node.keywords)]:
            if _root_name(arg) in sibling_names:
                return True
    return False


def _fills_carrier(entry, sibling_names: set[str]) -> bool:
    """Is trace *entry* part of the history of a carrier co-produced as *sibling_names*?

    One predicate for both the pass that completes a carrier's history and the
    guard that refuses a write whose history is incomplete -- when they
    disagreed, the guard let a blank chart through that the pass never saw.
    """
    code = entry[0]
    return bool(
        set(entry[1]) & sibling_names
        or _control_body_touches(code, sibling_names)
        or _passes_carrier_to_a_call(code, sibling_names)
    )


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

    Holds references to VirtualLineage and MismatchClassifier so the
    planner can call into helper methods that still live on those
    phases (e.g. check_loop_derived_trust_override, backward_scan_pass,
    collect_skipped_statement_metrics, filter_accumulator_reinits,
    reapply_unsaved_extensions). Pure-phase invariants land in a later
    refactor.
    """

    def __init__(
        self,
        virtual_lineage: "VirtualLineage",
        classifier: "MismatchClassifier",
        debug: bool = False,
    ) -> None:
        self.virtual_lineage = virtual_lineage
        self.classifier = classifier
        self.debug = debug
        #: ``(trace index, paths)`` of the writers the last plan left out of date.
        self.stale_exports: list[tuple[int, list[str]]] = []

    @staticmethod
    def _drop_scheduled_from_restored(simulation_trace, stmts_to_run_indices, restored):
        """A statement scheduled to run is not also reported as restored.

        Several passes promote a statement the backward scan restored to a
        re-run and leave its restore entry behind; the badge then listed it
        twice, ``^CACHED: models = {}`` above ``^CACHED: models = {}`` (round
        25, r25s5 and r25s1). Done once here, by code as the per-pass filters
        do, so no pass can leave one behind.
        """
        scheduled = {simulation_trace[i][0] for i in stmts_to_run_indices}
        if not scheduled:
            return restored
        return [info for info in restored if info.get("code") not in scheduled]

    def build_reexecution_plan(
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

        self.stale_exports = []
        loop_derived_trust_overridden = self.virtual_lineage.check_loop_derived_trust_override(
            upstream_has_modifications,
            vars_mutated_by_loops,
            simulation_trace_codes,
        )

        stmts_to_run_indices, restored_statements_info, total_restore_time = self.classifier.backward_scan_pass(
            simulation_trace,
            broken_vars,
            vars_tainted_by_upstream_mismatch,
            virtual_lineage,
            virtual_modules,
            vars_derived_from_loops,
            loop_derived_trust_overridden,
            upstream_has_modifications,
            simulation_trace_codes,
            stmt_lookup_times,
        )

        stmts_to_run_indices = self._schedule_consumable_producer_touches(
            stmts_to_run_indices,
            simulation_trace,
            consumable_broken_vars or set(),
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

        stmts_to_run_indices, restored_statements_info = self._schedule_file_write_statements(
            stmts_to_run_indices,
            simulation_trace,
            restored_statements_info,
            broken_vars,
            virtual_lineage,
            relevant_read_paths=relevant_read_paths,
            relevant_read_paths_known=relevant_read_paths_known,
        )

        stmts_to_run_indices, restored_statements_info = self._complete_stateful_carrier_history(
            stmts_to_run_indices,
            simulation_trace,
            restored_statements_info,
        )

        stmts_to_run_indices, restored_statements_info = self._guard_global_figure_writes(
            stmts_to_run_indices,
            simulation_trace,
            restored_statements_info,
        )

        stmts_to_run_indices, restored_statements_info = self._guard_unfilled_figure_writes(
            stmts_to_run_indices,
            simulation_trace,
            restored_statements_info,
        )

        # Again, now that the file-write passes are done: they PROMOTE restored
        # statements to re-execution (a restore validated before a scheduled
        # write), and a promoted statement's inputs were never cascaded. After
        # a restart ``counts = build_counts(events)`` restored, was promoted
        # behind ``OUT.mkdir()``, and ran without its ``def`` -- a NameError
        # (replay corpus, churn).
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
        skipped_metrics = self.virtual_lineage.collect_skipped_statement_metrics(
            simulation_trace,
            stmts_to_run_indices,
            restored_statements_info,
            virtual_modules,
            stmt_lookup_times,
        )
        restored_statements_info.extend(skipped_metrics)
        restored_statements_info = self._note_stale_exports(
            simulation_trace, stmts_to_run_indices, restored_statements_info
        )

        stmts_to_run_indices = self._schedule_loop_var_contexts(stmts_to_run_indices, simulation_trace)
        stmts_to_run_indices = self.virtual_lineage.filter_accumulator_reinits(
            stmts_to_run_indices, simulation_trace, vars_mutated_by_loops
        )
        stmts_to_run_indices = self._dedup_sorted_indices(stmts_to_run_indices)
        restored_statements_info = self._drop_scheduled_from_restored(
            simulation_trace,
            stmts_to_run_indices,
            restored_statements_info,
        )

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

        self.virtual_lineage.reapply_unsaved_extensions(
            broken_vars,
            vars_updated_by_trace,
            simulation_trace,
            notebook_cells,
            statements_to_reexecute,
        )

        return statements_to_reexecute, restored_statements_info, total_restore_time

    def _note_stale_exports(
        self, simulation_trace: list, stmts_to_run_indices: list[int], restored_statements_info: list[dict]
    ) -> list[dict]:
        """Mark the writers this repair left out of date (see
        :meth:`find_stale_file_writer_indices`) as such, in place of the
        "already current" skipped row they would otherwise get."""
        stale = {
            i: paths for i, paths in getattr(self, "stale_exports", None) or () if i not in set(stmts_to_run_indices)
        }
        if not stale:
            return restored_statements_info
        kept = [
            m
            for m in restored_statements_info
            if m.get("position") not in stale or str(m.get("status")) != str(CacheStatus.SKIPPED)
        ]
        for i, paths in sorted(stale.items()):
            trace_event("stale_export", stmt=simulation_trace[i][0][:80], paths=paths)
            kept.append(
                {
                    "code": simulation_trace[i][0],
                    "status": CacheStatus.SKIPPED,
                    "saved_time": 0.0,
                    "is_upstream": True,
                    "source": "Skipped",
                    "position": i,
                    "has_cache": False,
                    "stale_export": True,
                    "written_paths": paths,
                }
            )
        return kept

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
                if _is_definition(entry[0]):
                    # A def or class reads its body's globals when called, at
                    # whatever version they then are: defining it consumes
                    # none. Taken as a consumer, `def plot_region` re-ran the
                    # whole back-test cell the summary cell had just rebuilt
                    # (round 25, r25s5).
                    continue
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
                    top_level_cache[idx] = CodeAnalyzer.top_level_assigned_names(simulation_trace[idx][0])
                except (SyntaxError, ValueError):
                    top_level_cache[idx] = set()
            return top_level_cache[idx]

        scheduled = set(stmts_to_run_indices)
        additions: set[int] = set()

        for i in list(scheduled):
            entry = simulation_trace[i]
            outputs = entry[1]
            cond_only_vars = {v for v in outputs if v in broken_vars and v not in _top_level(i)}
            if not cond_only_vars:
                continue
            for v in cond_only_vars:
                # Already covered by an earlier unconditional producer in the plan?
                if any(p < i and v in simulation_trace[p][1] and v in _top_level(p) for p in scheduled):
                    continue
                # Find the NEAREST earlier guaranteed init of v.
                for p in range(i - 1, -1, -1):
                    if v in simulation_trace[p][1] and v in _top_level(p):
                        additions.add(p)
                        if self.debug:
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
        imports were merged (round 22, r22s3 and r22s4). The shadow pass above
        cannot see it: both imports bind the same lineage.

        With *virtual_lineage*, the globals the statement's callees read count
        as inputs too (``absent_callee_globals``).
        """
        user_ns = self.virtual_lineage.shell.user_ns
        live_lineage = getattr(self.virtual_lineage, "variable_lineage", None) or {}
        scheduled = set(stmts_to_run_indices)
        pending = sorted(scheduled)
        while pending:
            i = pending.pop(0)
            inputs = set(simulation_trace[i][2] or ())
            if virtual_lineage is not None:
                inputs |= self.virtual_lineage.absent_callee_globals(inputs, virtual_lineage, virtual_modules or set())
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
                # refunds for three stores, silently (round 25, r25s2).
                p = self.latest_producer(simulation_trace, v, before=i)
                if p is not None and p not in scheduled:
                    scheduled.add(p)
                    pending.append(p)
                    trace_event(
                        "input_producer_completion",
                        stmt=simulation_trace[p][0][:80],
                        var=v,
                        consumer=simulation_trace[i][0][:80],
                    )
        return sorted(scheduled)

    def _live_is_behind_producer(self, simulation_trace: list, var: str, before: int, live_lineage: dict) -> bool:
        """Is live *var* NOT what its latest producer before *before* makes?

        Live used to be enough. r25s3 ran the export cell (``results["f1"] =
        ...`` in place), re-ran the sweep cell (``results`` rebound, no f1),
        then the chart cell: the plan re-ran ``best = results.sort_values(
        ['f1', ...])`` on the live table without the f1 write above it, and
        raised ``UpstreamStateError: 'f1'`` (round 25).

        Behind only on evidence: the live lineage is what an EARLIER producer
        made. A live value matching no producer (a loop's iterations, an edit
        the file does not have yet) keeps the old answer, which is what a
        per-iteration loop's statements need.
        """
        p = self.latest_producer(simulation_trace, var, before=before)
        live = live_lineage.get(var)
        if p is None or live is None:
            return False
        produced = (simulation_trace[p][4] or {}).get(var)
        if produced is None or produced == live:
            return False
        return any((simulation_trace[q][4] or {}).get(var) == live for q in range(p) if var in simulation_trace[q][1])

    def complete_later_producers(self, stmts_to_run_indices: list[int], simulation_trace: list) -> list[int]:
        """Every statement after a re-run producer of a variable that also
        writes it re-runs too.

        Re-running ``results = {}`` leaves ``results`` as it binds it: empty,
        unless the ``for`` loop below that fills it runs as well. The plan
        re-ran the init and the functions reading ``results``, not the loop,
        and the report raised ``UpstreamStateError: 'logreg'`` with the dict
        left empty (round 25, r25s1). The backward completion cannot see it:
        it asks for the producer BEFORE a reader, and the init is one.

        Only a statement that READS the variable as it writes it continues
        what the earlier one built. One that only binds the name starts over
        and owes it nothing: a re-run ``for r in sorted(obs.run.unique())``
        pulled in a chart loop with a ``for r`` of its own, whose ``ax`` pulled
        in the UMAP loop ``for ax, col in zip(axes, ...)`` -- which then drew the
        new labels over an embedding nothing had rebuilt (round 29, r29s4).
        """
        scheduled = set(stmts_to_run_indices)
        pending = sorted(scheduled)
        while pending:
            i = pending.pop(0)
            for v in simulation_trace[i][1]:
                for p in range(i + 1, len(simulation_trace)):
                    if p in scheduled or v not in simulation_trace[p][1] or v not in simulation_trace[p][2]:
                        continue
                    scheduled.add(p)
                    pending.append(p)
                    trace_event(
                        "later_producer_completion",
                        stmt=simulation_trace[p][0][:80],
                        var=v,
                        after=simulation_trace[i][0][:80],
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
        on ``No module named 'bt'`` (round 29, r29s3, 2/2).
        """

        scheduled = set(stmts_to_run_indices)
        added: set[int] = set()
        for i in sorted(scheduled):
            code = simulation_trace[i][0]
            if not import_only(code) or all(m in sys.modules for m in _imported_roots(code)):
                continue
            for j in range(i):
                if j in scheduled or j in added:
                    continue
                entry_code = simulation_trace[j][0]
                if "sys" in simulation_trace[j][1] and "path" in entry_code:
                    added.add(j)
                    trace_event("import_path_setup", stmt=entry_code[:80], before=code[:80])
        return sorted(scheduled | added)

    def latest_producer(self, simulation_trace: list, var: str, before: int) -> int | None:
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
        user_ns = getattr(getattr(self.virtual_lineage, "shell", None), "user_ns", None)
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
                    p = self.latest_producer(simulation_trace, v, i)
                    if kind is None and v not in user_ns and p is not None:
                        # No live object after a restart: recognise the carrier
                        # by the code that makes it.
                        kind = carrier_kind_from_producer(simulation_trace[p][0])
                    if kind is None or p is None:
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
                        # .
                        entry = simulation_trace[j]
                        if not _fills_carrier(entry, sibling_names):
                            continue
                        # A second savefig is not a fill; the write gates decide
                        # whether it re-fires.
                        if set(entry[1]) & sibling_names or not statement_writes_files(entry[0]):
                            pending.add(j)
                    if not pending:
                        continue
                    # A fill we FORCE must be able to run. The loop above
                    # selects fills by co-produced NAME (`ax`), which says
                    # nothing about the data they read: `ax.plot(sub[...])`
                    # reads `sub`, `sub` is not a carrier, and the backward scan
                    # never treated it as a broken var, so nothing pulls its
                    # producer in. Scheduling the reader without the writer is
                    # the round-14 BLOCKING report -- `NameError: name 'sub' is
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
                    if self.debug:
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
                            stmt=simulation_trace[idx][0][:80],
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
        ``_schedule_file_write_statements`` runs BEFORE this pass, so anything
        added here has already bypassed its scope and repeatability gates —
        re-firing a ``to_csv(..., mode='a')`` would duplicate a line on disk,
        which a kernel restart cannot undo. Skipping it can leave the fill
        unrunnable, and that is the right trade: the failure is loud, names the
        missing variable and is fixed by running the cell, whereas a duplicated
        append is silent and permanent.
        """
        user_ns = getattr(getattr(self.virtual_lineage, "shell", None), "user_ns", None) or {}
        recorded = getattr(self.virtual_lineage, "variable_lineage", None) or {}

        extra: set[int] = set()
        frontier = list(pending)
        while frontier:
            j = frontier.pop()
            expected = simulation_trace[j][3] or {}  # input lineages at j
            for name in simulation_trace[j][2]:  # inputs
                # A value that is live and still what this statement read needs
                # no producer: the fill can run as it is. Without this, the
                # wider fill rule pulled `imp.plot.barh(..., ax=ax)` in and then
                # re-fitted the whole model chain behind a perfectly good `imp`
                # (replay acceptance corpus, round 21).
                if name in user_ns and name in expected and recorded.get(name) == expected[name]:
                    continue
                producer = self.latest_producer(simulation_trace, name, j)
                if producer is None:
                    continue
                if producer in scheduled or producer in pending or producer in extra:
                    continue
                if statement_writes_files(simulation_trace[producer][0]):
                    if self.debug:
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

    def _latest_current_figure_producer(
        self,
        simulation_trace: list,
        before: int,
        user_ns,
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
                            "matplotlib Figure",
                            "matplotlib Axes",
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
        """Refuse to re-run a ``plt.savefig()`` orphaned from its figure.

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

        We cannot follow the Gcf edge, so -- in the same spirit -- we bound
        the CONSEQUENCE instead of trying to enumerate what might skip the
        producer: drop the orphaned write from the plan and warn. The user
        re-runs the cell; the good chart on disk is left untouched. Governing
        principle: cash must never write a figure the user did not draw -- either
        the real one, or a loud refusal.

        Fires ONLY for a module-level ``plt.savefig`` whose most-recent preceding
        figure producer is absent from the plan. On the healthy path (first run,
        or any plan that also rebuilds the figure) the producer is scheduled and
        this is a no-op, so ``fig.savefig`` and the tests are
        untouched. A missed re-save is acceptable; a blank PNG on disk is not.
        """
        user_ns = getattr(getattr(self.virtual_lineage, "shell", None), "user_ns", None)
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
        restored_statements_info = [info for info in restored_statements_info if info.get("code") not in refused_codes]
        return remaining, restored_statements_info

    @staticmethod
    def _receiver_bound_figure_write(code: str) -> str | None:
        """Receiver name for a ``fig.savefig(...)``, or ``None``.

        The module-level ``plt.savefig()`` twin lives in ``cacheability`` and is
        handled by :meth:`_guard_global_figure_writes`; this is the form that
        one deliberately does not flag.
        """
        if "savefig" not in code:
            return None
        try:
            tree = ast.parse(textwrap.dedent(code))
        except (SyntaxError, ValueError):
            return None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "savefig"
                and isinstance(node.func.value, ast.Name)
            ):
                return node.func.value.id
        return None

    def _guard_unfilled_figure_writes(
        self,
        stmts_to_run_indices: list[int],
        simulation_trace: list,
        restored_statements_info: list[dict],
    ) -> tuple[list[int], list[dict]]:
        """Refuse a ``fig.savefig()`` whose figure is rebuilt but never drawn.

        ``statement_saves_current_pyplot_figure`` says of the receiver-bound
        form: "NOT flagged: it is defended by the carrier-history pass (its
        input ``fig`` is a tracked carrier)". That defence has a hole, and it
        opens exactly where it matters most.

        :meth:`_complete_stateful_carrier_history` classifies a carrier from the
        LIVE object -- ``stateful_carrier_kind(user_ns.get(v))`` -- and
        ``stateful_carrier_kind(None)`` is ``None``. After a kernel restart
        ``fig`` is not in ``user_ns``, so the pass silently does nothing, and a
        plan may schedule ``fig, ax = plt.subplots(...)`` and
        ``fig.savefig(path)`` while leaving ``ax.plot(...)`` behind. The write
        then flushes a freshly-created, EMPTY figure over the user's chart.

        Measured end to end, asking for an unrelated downstream cell after a
        restart: the real chart (1502 purple pixels, 1974 saturated) became 0
        and 0 at the user's own 800x400 geometry -- an empty axes frame, no
        error, no badge, and the good bytes gone. A restart cannot undo it.

        The remedy is the one the sibling guard already applies, and its
        governing principle is the same: cash must never write a figure the
        user did not draw -- either the real one, or a loud refusal. Refusing
        costs a re-save the user can redo by running the cell; writing costs
        them the chart.

        Deliberately structural, not liveness-based -- reading the trace rather
        than the namespace is the whole point, since the namespace is what is
        missing after a restart. Fires only when the producer IS scheduled (so a
        blank figure is genuinely about to be built) and some statement between
        it and the write that touches a co-produced name is NOT. When the
        carrier is live, the carrier-history pass owns the case and this is a
        no-op.
        """
        user_ns = getattr(getattr(self.virtual_lineage, "shell", None), "user_ns", None)
        scheduled = set(stmts_to_run_indices)
        refused: set[int] = set()

        for w in sorted(scheduled):
            code = simulation_trace[w][0]
            receiver = self._receiver_bound_figure_write(code)
            if receiver is None:
                continue
            # `plt.savefig(...)` matches the same shape but belongs to
            # `_guard_global_figure_writes`, which ran first. Letting both own it
            # would refuse a legitimate write twice and, worse, treat the MODULE
            # `plt` as a figure whose "fills" are every statement that touched
            # it. That detector falls back to the conventional alias, so it is
            # still right after a restart, when `plt` is not in the namespace.
            if statement_saves_current_pyplot_figure(code, user_ns):
                continue
            if user_ns is not None:
                try:
                    if stateful_carrier_kind(user_ns.get(receiver)) is not None:
                        continue  # live carrier: the history pass has it
                except (TypeError, ValueError, AttributeError, RecursionError):
                    pass
            producer = self.latest_producer(simulation_trace, receiver, w)
            if producer is None or producer not in scheduled:
                continue  # the figure is not being rebuilt here
            sibling_names = set(simulation_trace[producer][1])
            fills = [
                j
                for j in range(producer + 1, w)
                if _fills_carrier(simulation_trace[j], sibling_names)
                and not (
                    statement_writes_files(simulation_trace[j][0]) and not set(simulation_trace[j][1]) & sibling_names
                )
            ]
            if all(j in scheduled for j in fills):
                continue  # rebuilt coherently -- allow
            refused.add(w)
            self._warn_orphaned_figure_write(code, producer, w)

        if not refused:
            return stmts_to_run_indices, restored_statements_info

        remaining = [i for i in stmts_to_run_indices if i not in refused]
        refused_codes = {simulation_trace[i][0] for i in refused}
        restored_statements_info = [info for info in restored_statements_info if info.get("code") not in refused_codes]
        return remaining, restored_statements_info

    def _warn_orphaned_figure_write(self, code: str, producer: int | None, w: int) -> None:
        """Emit the refusal as a CashWarning (and a trace/debug record)."""
        warn_diagnostic(
            CashWarning,
            "NOTEBOOK-SAVEFIG-SKIP",
            "cash refused to re-run a plt.savefig() during upstream "
            "reconstruction: the statement that drew the current figure is not "
            "being re-run, so the save would flush a blank figure over your "
            "chart on disk. The file is untouched.",
            "re-run the cell that draws the plot if the image should be "
            "rewritten; to stop this arising at all, save through the figure "
            "object -- fig.savefig(path) -- rather than through pyplot.",
        )
        trace_event("refuse_orphaned_figure_write", stmt=code[:80], producer=producer)
        if self.debug:
            logger.debug(
                "[UPSTREAM] refusing orphaned plt.savefig at [%s] (figure producer %s not scheduled): %.60s",
                w,
                producer,
                code,
            )

    # Cheap textual pre-filter before running the full AST side-effect
    # analysis on a trace statement. Superset of the names in the write
    # detection tables (open modes, pandas to_*, save/savefig, json/pickle
    # dump, csv.writer, os/shutil mutations, pathlib write_text/bytes).
    _WRITE_MARKERS = (
        "open(",
        "write",
        "to_",
        "save",
        "dump",
        "os.",
        "shutil.",
    )

    def _user_ns(self):
        return getattr(getattr(self.virtual_lineage, "shell", None), "user_ns", None)

    @staticmethod
    def _called_names(code: str) -> set[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set()
        return set(called_names(tree))

    def _trace_defs(self, simulation_trace: list | None) -> dict:
        """``{name: trace entry}`` of the last ``def`` binding each name.

        After a kernel restart a helper is not defined yet, so its body can
        only be read from the notebook -- the ``def`` statement in the trace,
        whose inputs are the globals the body reads."""
        memo_key = (id(simulation_trace), len(simulation_trace or ()))
        memo = self.__dict__.get("_trace_defs_memo")
        if memo is not None and memo[0] == memo_key:
            return memo[1]
        defs: dict = {}
        self.__dict__["_trace_defs_memo"] = (memo_key, defs)
        for entry in simulation_trace or ():
            code = entry[0].lstrip()
            if not code.startswith(("def ", "async def ", "@")):
                continue
            try:
                node = ast.parse(entry[0]).body[0]
            except (SyntaxError, IndexError):
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs[node.name] = entry
        return defs

    def _unbound_helpers(self, names, defs: dict) -> list:
        """The ``def`` entries for *names* that are not live functions."""
        user_ns = self._user_ns() or {}
        return [defs[n] for n in names if n in defs and not isinstance(user_ns.get(n), types.FunctionType)]

    def _is_file_writer(self, stmt_code: str, simulation_trace: list | None = None) -> bool:
        """A statement that writes files -- in its own text, or through a user
        function it calls (``save_png(kind, path)``, whose ``savefig`` sits in
        the helper). A replay that re-ran a cell's inline writes but not its
        helper's left the report folder half old, half new (round 23)."""

        if _only_defines(stmt_code):
            # ``def save_page(...)`` writes nothing when it runs; its callers do,
            # and they are writers above. Taken for one, it had no provenance to
            # vouch for it after a restart, was re-fired, and pulled every write
            # of its cell along (round 23, r23s2).
            return False
        if statement_writes_files(stmt_code):
            return True
        if statement_calls_user_writer(stmt_code, self._user_ns()) is not None:
            return True
        defs = self._trace_defs(simulation_trace)
        seen: set[str] = set()
        pending = self._unbound_helpers(self._called_names(stmt_code), defs)
        while pending:
            entry = pending.pop()
            code = entry[0]
            if code in seen:
                continue
            seen.add(code)
            if statement_writes_files(code) and statement_write_repeatability(code) != REPEATABILITY_ACCUMULATING:
                return True
            pending.extend(self._unbound_helpers(self._called_names(code), defs))
        return False

    def _writer_inputs(self, inputs, simulation_trace: list | None = None) -> set[str]:
        """A writer's inputs plus the notebook globals its callees read: the
        helper that plots ``scores`` depends on ``scores`` though the call
        site never names it."""

        user_ns = self._user_ns()
        names = set(inputs)
        if user_ns:
            names |= called_function_globals(names, user_ns)
        defs = self._trace_defs(simulation_trace)
        pending = self._unbound_helpers(names, defs)
        seen: set[str] = set()
        while pending:
            entry = pending.pop()
            if entry[0] in seen:
                continue
            seen.add(entry[0])
            new = set(entry[2]) - names
            names |= new
            pending.extend(self._unbound_helpers(new, defs))
        return names

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
        """Schedule upstream file-WRITING statements whose effect is stale.

        File writes have no variable edge, so the backward scan never
        schedules them: editing a writer cell and re-running only the reader
        served the stale pre-edit file; and when a writer cell's
        sibling statements DID re-run, the side-effect-only write statement
        was still skipped and the reader's freshness stayed decided against
        the pre-run file state.

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

        tracking = getattr(self.classifier, "tracking_state", None)
        runtime_lineage = getattr(tracking, "variable_lineage", None) or {}
        # Live kernel namespace, to tell "provably unchanged" apart from
        # "absent". A writer input that is gone from user_ns (kernel restart,
        # ``del``, or an isolated re-run whose producer never ran this session)
        # has no runtime lineage -- but neither does an unchanged one, so the
        # lineage comparison below reads absent-as-unchanged and never schedules
        # the input's producer. That left ``df.to_csv(path)`` scheduled WITHOUT
        # its producer, so it ran against a missing ``df`` and raised NameError
        # post-restart, poisoning the whole notebook.
        user_ns = getattr(getattr(self.virtual_lineage, "shell", None), "user_ns", None)

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

        self.stale_exports = []
        writer_indices = self.find_stale_file_writer_indices(
            simulation_trace,
            scheduled_outputs=changed_inputs,
            skip=scheduled,
            virtual_lineage=virtual_lineage,
            relevant_read_paths=relevant_read_paths,
            relevant_read_paths_known=relevant_read_paths_known,
            stale_exports=self.stale_exports,
        )
        if not writer_indices:
            return stmts_to_run_indices, restored_statements_info

        writer_indices = self._whole_cell_writers(simulation_trace, writer_indices, scheduled)
        scheduled.update(writer_indices)
        first_writer = min(writer_indices)

        def _rebound_after(name: str, w: int) -> int | None:
            for j in range(len(simulation_trace) - 1, w, -1):
                if name in simulation_trace[j][1]:
                    return j
            return None

        # Re-materialise a scheduled writer's changed inputs: their producers
        # may be missing from the plan when the write is the only consumer.
        #
        # An input the trace binds AGAIN after the writer is changed too, for
        # this writer: the live value is the later binding. A chart cell that
        # reuses `fig, axes = plt.subplots(...)` for a second figure re-ran the
        # first figure's `save(fig, ...)` and its draws on the second figure's
        # axes (r22s3 session). Its producer before the writer re-runs, and so
        # does the last one, so the name ends bound as the cell leaves it.
        pending = list(writer_indices)
        while pending:
            w = pending.pop()
            for v in self._writer_inputs(simulation_trace[w][2], simulation_trace):
                later = _rebound_after(v, w)
                if later is None and not _input_changed(v):
                    continue
                for prod in range(w - 1, -1, -1):
                    if v in simulation_trace[prod][1]:
                        if prod not in scheduled:
                            scheduled.add(prod)
                            pending.append(prod)
                            if self.debug:
                                logger.debug(
                                    "[UPSTREAM] Scheduling producer [%s] of writer input '%s'",
                                    prod,
                                    v,
                                )
                        break
                if later is not None and later not in scheduled:
                    scheduled.add(later)
                    pending.append(later)

        # Every statement AFTER a scheduled writer whose variables carry file
        # dependencies must re-execute: its cached value (whether restored at
        # plan time or simply sitting untouched in memory) was validated
        # against the PRE-write file state. Re-execution happens in trace
        # order, after the writer, so its freshness is decided against the
        # freshly written file.
        tracking = getattr(self.classifier, "tracking_state", None)
        executed_file_deps = getattr(tracking, "executed_file_deps", None) or {}
        # Only what depends on a file a scheduled writer WRITES. "Any file
        # dependency" promoted nearly everything after the writer, because
        # recorded dependencies include inherited ones: saving a cleaned copy of
        # the data re-ran the backtest and the forecast behind it for a cell that
        # read only the copy (round 21, r21s2's doubled backtest; replay corpus).
        # A writer whose path does not resolve keeps the broad rule.
        written_forms = self._written_path_forms(simulation_trace, writer_indices)

        def _reads_written(outputs) -> bool:
            deps = set()
            for v in outputs:
                dep = executed_file_deps.get(v)
                if dep:
                    deps.update(dep.keys() if hasattr(dep, "keys") else dep)
            if not deps:
                return False
            if written_forms is None:
                return True
            return any(self._normalize_path_forms(d) & written_forms for d in deps)

        promoted: set[int] = set()
        promoted_outputs: set[str] = set()
        for i in range(first_writer + 1, len(simulation_trace)):
            if i in scheduled:
                continue
            outputs, inputs = simulation_trace[i][1], simulation_trace[i][2]
            # ...and, in trace order, whatever consumes a re-read value.
            if _reads_written(outputs) or (written_forms is not None and set(inputs) & promoted_outputs):
                scheduled.add(i)
                promoted.add(i)
                promoted_outputs.update(outputs)
                if self.debug:
                    logger.debug(
                        "[UPSTREAM] Promoting file-reader [%s] to re-exec (writer scheduled at [%s]): %s",
                        i,
                        first_writer,
                        simulation_trace[i][0][:40],
                    )

        if promoted:
            promoted_codes = {simulation_trace[i][0] for i in promoted}
            restored_statements_info = [
                info for info in restored_statements_info if info.get("code") not in promoted_codes
            ]

        return sorted(scheduled), restored_statements_info

    def _whole_cell_writers(self, simulation_trace: list, writer_indices, scheduled) -> list[int]:
        """*writer_indices* plus every other file writer in the same cells.

        A cell's writes are replayed together or not at all. Re-running only
        the stale ones left states no run order produces: a report folder
        whose grid came from the new models and whose ROC chart from the old,
        or -- when ``shutil.rmtree`` re-ran and the loop that refills the
        folder did not -- charts that were simply gone (round 23, r23s1, three
        ways). Running the cell writes every one of its files; so does this.

        Only writes that provably REPLACE their file are pulled in: they land
        the same bytes when repeated. A write that is not stale itself and may
        append -- ``os.write(fd, ...)`` on a descriptor opened elsewhere -- is
        left to run when its own cell runs; re-firing it here duplicated a
        counter line on every replay (CAS-176 probe). The exception is a cell
        whose replay already re-fires such a write -- a ``shutil.rmtree`` --
        where everything but a provable append follows it, or ``PACK.mkdir()``
        stays behind and the next write finds no folder.
        """

        cells = {getattr(simulation_trace[w], "cell", -1) for w in writer_indices} - {-1}
        if not cells:
            return list(writer_indices)
        # Cells whose replay already re-fires a write that is not provably
        # replacing (``rmtree``, ``mkdir``): the rest of their writes follow it.
        destructive = {
            getattr(simulation_trace[w], "cell", -1)
            for w in writer_indices
            if statement_write_repeatability(simulation_trace[w][0]) != REPEATABILITY_REPLACING
        }
        extra = []
        chosen = set(writer_indices)
        for j, entry in enumerate(simulation_trace):
            if j in chosen or j in scheduled or getattr(entry, "cell", -1) not in cells:
                continue
            code = entry[0]
            if not self._is_file_writer(code, simulation_trace):
                continue
            verdict = statement_write_repeatability(code)
            if verdict == REPEATABILITY_ACCUMULATING:
                continue
            if verdict != REPEATABILITY_REPLACING and getattr(entry, "cell", -1) not in destructive:
                continue
            extra.append(j)
            if self.debug:
                logger.debug(
                    "[UPSTREAM] Scheduling same-cell file-writer [%s] with its cell's stale writers: %s",
                    j,
                    code[:60],
                )
        return sorted(chosen | set(extra))

    def find_stale_file_writer_indices(
        self,
        simulation_trace: list,
        scheduled_outputs: frozenset | set = frozenset(),
        skip: frozenset | set = frozenset(),
        virtual_lineage: dict | None = None,
        relevant_read_paths: set[str] | None = None,
        relevant_read_paths_known: bool = True,
        stale_exports: list | None = None,
    ) -> list[int]:
        """Trace indices of file-WRITING statements whose effect is stale.

        A writer is stale when its code was never executed in this session
        (edited/new — the runtime records executed file-writing statements'
        code in ``executed_write_stmt_codes``) or when one of its inputs is
        produced by an already-scheduled statement. Also used by the
        simulator to gate its no-broken-vars early return, so the common
        path stays cheap: a textual marker pre-filter runs before any AST
        analysis.

        Scoped to the current cell: a writer whose resolvable
        output path is read by NO consumer relevant to this reconstruction
        (``relevant_read_paths``) is an unrelated / terminal side-effect and is
        never scheduled — re-firing it can only repeat an external write (a
        non-idempotent ``mode='a'`` append corrupts the file) without helping
        reconstruct any value the current cell needs. The gate applies only when
        the read set is fully known and the writer's own path resolves; every
        uncertain case falls through to the prior (conservative) behaviour.

        A writer the scope gate leaves alone although what it writes has
        changed -- an input's lineage drifted from the one it was written
        with -- is appended to *stale_exports* as ``(index, paths)``: the
        file on disk is now out of date, and the badge must not call it
        current (round 28, r28s3 and r28s5).
        """
        tracking = getattr(self.classifier, "tracking_state", None)
        executed_writes = getattr(tracking, "executed_write_stmt_codes", None)
        if executed_writes is None:
            return []
        runtime_lineage = getattr(tracking, "variable_lineage", None) or {}

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
            if not self._is_file_writer(stmt_code, simulation_trace):
                continue
            inputs = self._writer_inputs(inputs, simulation_trace)
            # Scope gate: skip a writer whose output file no relevant consumer
            # reads. Its write runs when the user runs its own
            # cell; reconstruction of an unrelated cell must never re-fire it.
            unread = self._writer_output_unread(
                stmt_code,
                relevant_read_paths,
                relevant_read_paths_known,
                simulation_trace,
            )
            trace_event(
                "writer_considered",
                stmt=stmt_code[:80],
                unread=unread,
                read_paths_known=relevant_read_paths_known,
                read_paths=sorted(relevant_read_paths or ())[:20],
            )
            if unread:
                if stale_exports is not None:
                    self._note_if_stale(
                        stale_exports, i, stmt_code, inputs, virtual_lineage, runtime_lineage, simulation_trace
                    )
                if self.debug:
                    logger.debug(
                        "[UPSTREAM] File-writer output read by no relevant consumer; not scheduling (scope): %s",
                        stmt_code[:60],
                    )
                continue
            # Repeatability gate. The scope gate above keys on
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
            # ONLY a PROVABLE append is refused. The evidence also suggests refusing
            # UNKNOWN repeatability, and that was measured rather than reasoned
            # about: the two failure modes are not symmetric. Refusing a write
            # that SHOULD re-fire leaves the reader on stale data every time,
            # silently, on a common path -- the measurement broke two
            # tests exactly that way. Leaving an unprovable append re-firing
            # costs a duplicated line in an uncommon shape. Narrow is the right
            # side of that trade; see the recorded measurement.
            if statement_write_repeatability(stmt_code) == REPEATABILITY_ACCUMULATING:
                if self.debug:
                    logger.debug(
                        "[UPSTREAM] File-writer is a non-idempotent append; not re-firing (repeatability): %s",
                        stmt_code[:60],
                    )
                continue
            changed = stmt_code not in executed_writes
            scheduled_inputs = set(inputs) & scheduled_outputs
            drifted = any(_input_lineage_drifted(v) for v in inputs)
            inputs_changed = bool(scheduled_inputs) or drifted
            # ``changed`` fires for every writer after a kernel restart because
            # ``executed_write_stmt_codes`` is session-scoped and starts empty.
            # Before re-firing such a writer (which would re-run its
            # non-idempotent side effect and re-derive stale data), check its
            # persisted provenance: if the payload is unchanged AND the output
            # file is still fresh on disk, the effect is already applied — do
            # NOT schedule it (round-3).
            #
            # An input whose producer is scheduled is answered the same way. A
            # producer is scheduled to REBUILD a value as often as to change it
            # -- after a restart, everything the cell needs is -- and the
            # provenance tells the two apart: the lineage the input had when
            # the file was written against the lineage the simulation gives it
            # now, which an upstream edit changes. Without this every writer
            # above a restarted cell re-fired, with everything it reads
            # (round 23, r23s3: a 263 s sweep). A lineage that drifted from the
            # runtime's (an unsaved edit) always re-runs.
            if (
                (changed or scheduled_inputs)
                and not drifted
                and self._writer_output_already_fresh(
                    stmt_code,
                    inputs,
                    virtual_lineage,
                    runtime_lineage,
                    must_cover=scheduled_inputs,
                    simulation_trace=simulation_trace,
                    index=i,
                )
            ):
                changed = inputs_changed = False
                if self.debug:
                    logger.debug(
                        "[UPSTREAM] File-writer effect already fresh on disk; not re-firing: %s",
                        stmt_code[:60],
                    )
            trace_event(
                "writer_decided",
                stmt=stmt_code[:80],
                refire=changed or inputs_changed,
                changed=changed,
                scheduled_inputs=sorted(scheduled_inputs),
                drifted=drifted,
            )
            if changed or inputs_changed:
                writer_indices.append(i)
                if self.debug:
                    logger.debug(
                        "[UPSTREAM] File-writer scheduling: [%s] %s (changed=%s, inputs_changed=%s)",
                        i,
                        stmt_code[:40],
                        changed,
                        inputs_changed,
                    )
        return writer_indices

    def _note_if_stale(
        self,
        stale_exports: list,
        i: int,
        stmt_code: str,
        inputs,
        virtual_lineage: dict | None,
        runtime_lineage: dict,
        simulation_trace: list,
    ) -> None:
        """Add writer *i* to *stale_exports* when what it recorded as it wrote
        says its data has changed since: an input's lineage, or for a figure
        what was drawn into it. Not the drift of its inputs at the end of the
        simulation: that missed a chart drawn through ``for ax in axes`` --
        ``fig`` itself never changes -- and every chart after a restart, which
        has no runtime lineage to drift from (round 29, r29s4 and r29s5). A
        folder has no content to be out of date (r29s2: ``os.makedirs``)."""
        reason = self._writer_not_fresh_because(
            stmt_code, inputs, virtual_lineage, runtime_lineage, simulation_trace=simulation_trace, index=i
        )
        if reason not in self._DATA_CHANGED:
            return
        paths = sorted(p for p in (self._writer_paths(stmt_code, simulation_trace) or ()) if not os.path.isdir(p))
        if paths:
            stale_exports.append((i, paths))

    def _written_path_forms(self, simulation_trace: list, writer_indices) -> set[str] | None:
        """Comparable forms of every path the writers write, or ``None`` if any
        writer's output does not resolve."""
        forms: set[str] = set()
        for w in writer_indices:
            written = self._writer_paths(simulation_trace[w][0], simulation_trace)
            if not written:
                return None
            for p in written:
                forms |= self._normalize_path_forms(p)
        return forms

    def _writer_paths(self, stmt_code: str, simulation_trace: list | None = None) -> set[str] | None:
        """The paths a writer writes: from its code, else from the record it
        left when it last ran.

        After a kernel restart ``OUT`` in ``OUT / 'table.csv'`` is not bound
        yet, so the code alone no longer resolves -- and every writer looked
        like it might feed the cell being run. A name the notebook binds to a
        literal path (``OUT = Path('report')``) resolves from *simulation_trace*;
        a folder removed by ``shutil.rmtree(OUT)`` leaves no record to fall back on.
        """
        user_ns = getattr(getattr(self.virtual_lineage, "shell", None), "user_ns", None)
        namespace = {**_literal_path_bindings(simulation_trace), **(user_ns or {})}
        written = statement_written_paths(stmt_code, namespace=namespace)
        if written:
            return written
        cash = getattr(self.virtual_lineage, "cash_instance", None)
        backend = getattr(cash, "backend", None) if cash is not None else None
        if backend is None:
            return None
        try:
            record = backend.get_metadata(write_provenance_key(stmt_code))
        except (OSError, TypeError, ValueError, AttributeError):
            return None
        if not record or not record.get("write_provenance") or not record.get("paths"):
            return None
        return set(record["paths"])

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

    def _read_path_index(self, relevant_read_paths) -> tuple[set[str], list[str], list[str]]:
        """``(comparable forms, folders listed, places)`` of the paths read, once
        per set of read paths rather than once per writer: each is a resolve and
        an ``isdir``, and 12 writers over 1,312 read files made 44,000 of them
        before one restarted cell (r23s2). The simulator and the planner ask
        with the same set in one check, so it is kept across passes too; r24s4's
        10,000 documents were indexed twice per cell, 1.8 s, and each resolved
        twice."""
        cached = getattr(self, "_read_index", None)
        if cached is not None and cached[0] is relevant_read_paths and cached[1] == len(relevant_read_paths):
            return cached[2]
        read_forms: set[str] = set()
        read_dirs: list[str] = []
        read_places: list[str] = []
        for rp in relevant_read_paths:
            # One stat answers both "is it there" and "is it a folder" for a
            # path that has not moved; only a missing one takes the relocation
            # fallbacks and a second look.
            try:
                is_dir = stat.S_ISDIR(os.stat(rp).st_mode)
                resolved = rp
            except (OSError, ValueError, TypeError):
                try:
                    resolved = resolve_file_dep_path(rp) or rp
                except (OSError, ValueError, TypeError):
                    resolved = rp
                is_dir = resolved != rp and os.path.isdir(resolved)
            for candidate in {resolved, rp}:
                try:
                    read_forms.add(os.path.normcase(os.path.abspath(candidate)))
                    read_forms.add(os.path.normcase(os.path.basename(candidate)))
                except (OSError, ValueError, TypeError):
                    continue
            read_places.append(os.path.normcase(os.path.abspath(resolved)))
            if is_dir:
                # A listed / globbed folder: whatever is written inside it is
                # read by the next listing.
                read_dirs.append(os.path.normcase(os.path.abspath(resolved)) + os.sep)
        index = (read_forms, read_dirs, read_places)
        self._read_index = (relevant_read_paths, len(relevant_read_paths), index)
        return index

    def _writer_output_unread(
        self,
        stmt_code: str,
        relevant_read_paths: set[str] | None,
        relevant_read_paths_known: bool,
        simulation_trace: list | None = None,
    ) -> bool:
        """True when a writer's output file is read by no relevant consumer.

        The scope gate. Conservative in every uncertain
        case — returns ``False`` (do not suppress) unless the read set is fully
        known AND the writer's output path(s) all resolve statically AND none of
        them match any relevant read path. Then the writer is an unrelated /
        terminal side-effect for this cell and must not be re-fired.
        """
        if not relevant_read_paths_known or relevant_read_paths is None:
            return False
        written = self._writer_paths(stmt_code, simulation_trace)
        if not written:
            return False  # unresolvable target -> stay conservative
        read_forms, read_dirs, read_places = self._read_path_index(relevant_read_paths)
        for wp in written:
            if self._normalize_path_forms(wp) & read_forms:
                return False  # this output IS read by a relevant consumer
            where = os.path.normcase(os.path.abspath(resolve_file_dep_path(wp) or wp))
            if any(where.startswith(d) for d in read_dirs):
                return False
            # A folder made or removed (``OUT.mkdir()``): read by whatever
            # reads a file inside it.
            if any(place.startswith(where + os.sep) for place in read_places):
                return False
        return True

    def _writer_output_already_fresh(
        self,
        stmt_code: str,
        inputs,
        virtual_lineage: dict | None,
        runtime_lineage: dict,
        must_cover: set[str] | frozenset[str] = frozenset(),
        simulation_trace: list | None = None,
        index: int | None = None,
    ) -> bool:
        """True when a writer's effect is already on disk and provably current.

        *must_cover*: inputs the record has to have a lineage for -- ones
        whose producer is being re-run, which it can vouch for only if it
        knows what they were.

        *simulation_trace* / *index*: where the writer is in the simulation. Its
        inputs are compared with the lineages they have THERE, which is what
        the runtime recorded; the end of the simulation is later, and a
        ``plt.close(fig)`` or a second chart after the write had moved ``fig``
        on by then. A figure is compared by its drawing history instead of its
        lineage (``carrier_history``).

        Consulted only for a writer that looks ``changed`` purely because its
        code was never seen THIS session (the post-restart case). Returns True —
        meaning "do not re-fire" — only when the persisted provenance for this
        exact writer source is present, EVERY recorded output path is still fresh
        (:func:`snapshot_is_fresh`), AND every recorded input lineage still
        matches the writer's current (simulated / runtime) lineage.

        Conservative in every uncertain case: no backend, missing provenance, an
        unreadable / stale output file, or a drifted input lineage all return
        False, so the writer is scheduled exactly as before (round-3).
        """
        return (
            self._writer_not_fresh_because(
                stmt_code, inputs, virtual_lineage, runtime_lineage, must_cover, simulation_trace, index
            )
            is None
        )

    #: Reasons `_writer_not_fresh_because` gives that mean the file on disk was
    #: written from data that has since changed -- the rest mean only that
    #: nothing vouches for it.
    _DATA_CHANGED = frozenset({"input changed", "figure drawn differently"})

    def _writer_not_fresh_because(
        self,
        stmt_code: str,
        inputs,
        virtual_lineage: dict | None,
        runtime_lineage: dict,
        must_cover: set[str] | frozenset[str] = frozenset(),
        simulation_trace: list | None = None,
        index: int | None = None,
    ) -> str | None:
        """Why a writer's file is not provably current, or ``None`` when it is.
        See :meth:`_writer_output_already_fresh`."""

        def stale(reason: str, **detail) -> str:
            trace_event("writer_not_fresh", stmt=stmt_code[:80], reason=reason, **detail)
            return reason

        cash = getattr(self.virtual_lineage, "cash_instance", None)
        backend = getattr(cash, "backend", None) if cash is not None else None
        if backend is None:
            return "no backend"
        try:
            record = backend.get_metadata(write_provenance_key(stmt_code))
        except (OSError, TypeError, ValueError, AttributeError):
            return "no provenance"
        if not record or not record.get("write_provenance"):
            return stale("no provenance")
        paths = record.get("paths") or []
        file_deps = record.get("file_deps") or {}
        if not paths or not file_deps:
            return stale("no files recorded")
        missing = [path for path in paths if not file_deps.get(path)]
        if missing:
            return stale("file not recorded", path=missing[0])
        fresh, changed = snapshot_is_fresh({path: file_deps[path] for path in paths})
        if not fresh:
            return stale("file changed", path=changed.path)
        # The output on disk is only the writer's CURRENT output if its inputs
        # still carry the lineage they had when it was written. A drift means
        # the file was produced from a now-stale payload.
        stored_lineages = record.get("input_lineages") or {}
        if not set(must_cover) <= set(stored_lineages):
            return stale("input not recorded", inputs=sorted(set(must_cover) - set(stored_lineages)))
        entry = simulation_trace[index] if simulation_trace is not None and index is not None else None
        histories = record.get("carrier_histories") or {}
        for var, stored_lineage in stored_lineages.items():
            if var in histories:
                if entry is None or histories[var] != self._carrier_history_at(simulation_trace, index, var):
                    return stale("figure drawn differently", var=var)
                continue
            current = None
            if entry is not None:
                # After the writer ran: what the runtime recorded.
                current = entry[4].get(var) if var in entry[1] else key_lineages(entry[3]).get(var)
            if current is None:
                current = (virtual_lineage or {}).get(var)
            if current is None:
                current = runtime_lineage.get(var)
            if current != stored_lineage:
                return stale("input changed", var=var)
        return None

    @staticmethod
    def _carrier_history_at(simulation_trace: list, index: int, carrier: str) -> str | None:
        """The history fingerprint of figure *carrier* at the writer at *index*,
        from the statements of the writer's cell that come before it -- the same
        span the runtime took (``StatementProcessor._carrier_histories``)."""
        cell = getattr(simulation_trace[index], "cell", -1)
        if cell == -1:
            return None
        first = index
        while first > 0 and getattr(simulation_trace[first - 1], "cell", -1) == cell:
            first -= 1
        return carrier_history_fingerprint(
            [(entry[0], key_lineages(entry[3])) for entry in simulation_trace[first:index]],
            carrier,
        )

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
            next_stmt = simulation_trace[j][0]
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
            digest = iteration_digest(simulation_trace[idx][0])
            if digest is not None:
                scheduled_contexts.add(digest)

        if not scheduled_contexts:
            return stmts_to_run_indices

        additional_indices: list[int] = []
        for i, (stmt_code, outputs, _inputs, _, _, _) in enumerate(simulation_trace):
            if i in stmts_set or iteration_digest(stmt_code) is not None:
                continue
            if self._is_loop_var_assignment_for_context(
                i,
                stmt_code,
                outputs,
                simulation_trace,
                scheduled_contexts,
            ):
                if self.debug:
                    logger.debug("[UPSTREAM] Adding loop var assignment for scheduled context: %s", stmt_code[:40])
                additional_indices.append(i)

        return stmts_to_run_indices + additional_indices
