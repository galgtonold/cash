"""Refusing to re-run a figure save that would write a figure nobody drew.

A plan may re-run a ``savefig`` while leaving out what drew the figure: a
``plt.savefig()`` whose current figure is not rebuilt, or a
``fig.savefig()`` whose figure is rebuilt empty after a restart. Either
writes a blank image over the user's chart. :class:`FigureWriteGuard` drops
such a save from the plan and warns; the user re-runs the cell to rewrite it.
"""

from __future__ import annotations

import logging

from ...analysis.cacheability import statement_writes_files
from ...analysis.namespace_effects import statement_saves_current_pyplot_figure
from ...diagnostics import warn_diagnostic
from ...exceptions import CashWarning
from .._protocols import ShellProtocol
from .._trace import trace_event
from ..stateful_carriers import stateful_carrier_kind
from ._types import latest_producer
from .carrier_fills import fills_carrier
from .library_rules import figure_save_receiver, is_figure_carrier, makes_current_figure

__all__ = ["FigureWriteGuard"]

logger = logging.getLogger(__name__)


class FigureWriteGuard:
    """Drops from a plan the figure saves whose figure would be blank."""

    def __init__(self, shell: ShellProtocol) -> None:
        self.shell = shell

    def _current_figure_producer(
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
            if user_ns is not None and any(is_figure_carrier(user_ns.get(out)) for out in simulation_trace[p].outputs):
                return p
            if makes_current_figure(simulation_trace[p].stmt_code):
                return p
        return None

    def guard_global_writes(
        self,
        stmts_to_run_indices: list[int],
        simulation_trace: list,
        restored_statements_info: list[dict],
    ) -> tuple[list[int], list[dict]]:
        """Refuse to re-run a ``plt.savefig()`` orphaned from its figure.

        ``plt.savefig(path)`` writes pyplot's CURRENT figure through the
        process-global ``Gcf`` registry; its only variable input is the module
        ``plt``. Unlike the receiver-bound ``fig.savefig(path)`` that
        ``ReexecutionPlanner._complete_stateful_carrier_history`` defends, there is NO value-level
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
        user_ns = self.shell.user_ns
        scheduled = set(stmts_to_run_indices)
        refused: set[int] = set()

        for w in sorted(scheduled):
            code = simulation_trace[w].stmt_code
            if not statement_saves_current_pyplot_figure(code, user_ns):
                continue
            producer = self._current_figure_producer(simulation_trace, w, user_ns)
            if producer is not None and producer in scheduled:
                continue  # the figure is being (re)built coherently -- allow
            refused.add(w)
            self._warn_orphaned_write(code, producer, w)

        if not refused:
            return stmts_to_run_indices, restored_statements_info

        remaining = [i for i in stmts_to_run_indices if i not in refused]
        # A refused write must not linger in the restored set either -- it is not
        # being run at all this pass.
        refused_codes = {simulation_trace[i].stmt_code for i in refused}
        restored_statements_info = [info for info in restored_statements_info if info.get("code") not in refused_codes]
        return remaining, restored_statements_info

    def guard_unfilled_writes(
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

        :meth:`ReexecutionPlanner._complete_stateful_carrier_history` classifies a carrier from the
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
        user_ns = self.shell.user_ns
        scheduled = set(stmts_to_run_indices)
        refused: set[int] = set()

        for w in sorted(scheduled):
            code = simulation_trace[w].stmt_code
            receiver = figure_save_receiver(code)
            if receiver is None:
                continue
            # `plt.savefig(...)` matches the same shape but belongs to
            # `guard_global_writes`, which ran first. Letting both own it
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
            producer = latest_producer(simulation_trace, receiver, w)
            if producer is None or producer not in scheduled:
                continue  # the figure is not being rebuilt here
            sibling_names = set(simulation_trace[producer].outputs)
            fills = [
                j
                for j in range(producer + 1, w)
                if fills_carrier(simulation_trace[j], sibling_names)
                and not (
                    statement_writes_files(simulation_trace[j].stmt_code)
                    and not set(simulation_trace[j].outputs) & sibling_names
                )
            ]
            if all(j in scheduled for j in fills):
                continue  # rebuilt coherently -- allow
            refused.add(w)
            self._warn_orphaned_write(code, producer, w)

        if not refused:
            return stmts_to_run_indices, restored_statements_info

        remaining = [i for i in stmts_to_run_indices if i not in refused]
        refused_codes = {simulation_trace[i].stmt_code for i in refused}
        restored_statements_info = [info for info in restored_statements_info if info.get("code") not in refused_codes]
        return remaining, restored_statements_info

    def _warn_orphaned_write(self, code: str, producer: int | None, w: int) -> None:
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
        logger.debug(
            "[UPSTREAM] refusing orphaned plt.savefig at [%s] (figure producer %s not scheduled): %.60s",
            w,
            producer,
            code,
        )
