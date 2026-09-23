"""What rebuilding a variable after a restart would re-run, and the
end-of-cell pass that writes the costly ones to disk."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cash.notebook._protocols import CashInstanceProtocol, ShellProtocol, TrackingState

logger = logging.getLogger(__name__)

_LOG_PROCESSOR = "[PROCESSOR]"

__all__ = ["RebuildCostLedger"]


class RebuildCostLedger:
    """Tracks, per variable, the entries not on disk it was computed through.

    A statement is persisted by its own compute time, so a cheap statement
    over a costly input stays in RAM. This ledger remembers the cost behind
    each variable so the end of a cell can persist the final values that
    would be costly to rebuild.
    """

    def __init__(
        self, shell: ShellProtocol, tracking_state: TrackingState, cash_instance: CashInstanceProtocol | None
    ) -> None:
        self.shell = shell
        self.tracking_state = tracking_state
        self.cash_instance = cash_instance
        # What rebuilding a variable after a restart would re-run: the entries
        # not on disk it was computed through, ``{cache key: seconds}``; and the
        # key that last produced each variable in this cell (``end_cell_persistence``).
        self._unsaved_ancestry: dict[str, dict[str, float]] = {}
        self._cell_last_key: dict[str, str] = {}
        # One per control structure running: the entries not on disk its body
        # produced, and the names they were produced for.
        self._structure_costs: list[tuple[dict[str, float], set[str]]] = []
        self._collapsed = 0  # ancestries summed into one (``_capped``)

    def begin_cell(self) -> None:
        """Forget which key last produced each variable, before a cell runs."""
        self._cell_last_key = {}

    #: Ancestry entries kept per variable; past this they are summed into one,
    #: which may count a shared ancestor twice -- too much persisted, never too little.
    MAX_ANCESTRY = 256

    def note(self, cache_key: str, inputs, outputs, seconds: float, on_disk: bool) -> None:
        """Record what rebuilding *outputs* after a restart would re-run.

        Nothing, when their entry is on disk. Otherwise this statement and the
        entries not on disk its inputs came through -- by key, so an ancestor
        reached twice (``vs_plan = wk_store.merge(plan)``, both from ``sales``)
        is counted once.
        """
        ancestry: dict[str, float] = {}
        if not on_disk:
            for name in inputs:
                ancestry.update(self._unsaved_ancestry.get(name, {}))
            ancestry[cache_key] = seconds
            ancestry = self._capped(ancestry)
        for name in outputs:
            self._unsaved_ancestry[name] = ancestry
            self._cell_last_key[name] = cache_key
        for spent, names in self._structure_costs:
            spent.update(ancestry)
            names.update(outputs)

    def _capped(self, ancestry: dict[str, float]) -> dict[str, float]:
        if len(ancestry) <= self.MAX_ANCESTRY:
            return ancestry
        self._collapsed += 1
        return {f"collapsed:{self._collapsed}": sum(ancestry.values())}

    def begin_structure(self) -> None:
        """A control structure starts: collect what its body leaves unsaved."""
        self._structure_costs.append(({}, set()))

    def end_structure(self, reads, changed, success: bool) -> None:
        """A control structure ended. A name it *changed* that no body statement
        produced -- ``parts`` in ``for f in files: ... parts.append(d)``, which
        the loop owns -- costs what the whole body left unsaved to rebuild, and
        what the structure read."""
        spent, names = self._structure_costs.pop() if self._structure_costs else ({}, set())
        if not success:
            return
        ancestry = dict(spent)
        for name in reads:
            ancestry.update(self._unsaved_ancestry.get(name, {}))
        ancestry = self._capped(ancestry)
        for name in set(changed) - names:
            self._unsaved_ancestry[name] = ancestry
            # No entry holds its final value: the one that bound it (``parts =
            # []``) holds what it was before the loop.
            self._cell_last_key.pop(name, None)
        for outer, outer_names in self._structure_costs:
            outer.update(ancestry)
            outer_names.update(changed)

    #: Unsaved compute behind a cheap statement's inputs past which its final
    #: value gets an entry anyway (see :meth:`final_over_costly_inputs`).
    COSTLY_INPUTS_S = 0.1

    def final_over_costly_inputs(self, inputs, outputs, *, in_loop: bool, written_later: frozenset[str]) -> bool:
        """Whether a statement too cheap to cache leaves a final value over
        inputs that would be costly to rebuild.

        ``is_refund = sales["qty"] < 0`` takes a millisecond, over a ``sales``
        that took seconds and is not on disk. With no entry, the end-of-cell
        pass had nothing to persist, and after a restart the cell rebuilt
        ``sales`` to get ``is_refund`` back. An intermediate
        -- a name the cell writes again (*written_later*) -- still gets none.
        """
        try:
            if in_loop:
                # Inside a loop iteration nothing is final: the next iteration
                # overwrites it, and the inputs' unsaved cost only grows as the
                # loop runs, so EVERY iteration qualified. A 631-iteration
                # loop wrote an entry per iteration for a ~0.1 ms statement --
                # each a full snapshot of the 2.4 MB frame it changes, 1.5 GiB
                # in all -- and a re-run copied every one back: 0.05 s plain,
                # 11-23 s cached. What the loop leaves is judged when it ends.
                return False
            if not outputs or set(outputs) & set(written_later):
                return False
            cost = sum(sum(self._unsaved_ancestry.get(name, {}).values()) for name in inputs)
            return cost >= self.COSTLY_INPUTS_S
        except Exception:  # noqa: BLE001 - the floor is the safe answer
            return False

    def end_cell_persistence(self) -> None:
        """Write to disk what this cell left that would be costly to rebuild.

        A statement is persisted by its own compute time, so a cheap statement
        over a costly input stays in RAM, and after a restart the next cell that
        needs it rebuilds the whole chain behind it.
        Here each variable's final value, as the cell leaves it, is
        judged by what rebuilding it would cost -- the entries not on disk it
        came through -- by the same cost-model rule. The final value only: the
        ten versions ``sales`` goes through in one cell are not worth ten copies.

        Every final value, not only one a cell below reads: running this cell
        again after a restart restores its last versions rather than rebuilding
        them (``UpstreamChecker.plan_cell_run``), and ``is_refund`` beside the
        final ``sales`` is one of them. Still only in a
        notebook, where a restart re-runs cells by their source.
        """
        backend = getattr(self.cash_instance, "backend", None) if self.cash_instance else None
        last, self._cell_last_key = self._cell_last_key, {}
        later = self.tracking_state.read_by_later_cells
        self.tracking_state.read_by_later_cells = None
        if backend is None or later is None:
            return
        persist = backend.persist_from_memory
        costs: dict[str, float] = {}
        for name, key in last.items():
            if name not in self.shell.user_ns:
                continue
            if self.tracking_state.variable_sources.get(name) != key:
                continue
            cost = sum(self._unsaved_ancestry.get(name, {}).values())
            if cost > 0:
                costs[key] = max(costs.get(key, 0.0), cost)
        for key, cost in costs.items():
            try:
                written = persist(key, cost)
            except Exception:  # noqa: BLE001 - persisting ahead of need must never break a cell
                logger.debug("%s end-of-cell persistence failed for %s", _LOG_PROCESSOR, key, exc_info=True)
                continue
            if written:
                for name, k in last.items():
                    if k == key:
                        self._unsaved_ancestry[name] = {}
