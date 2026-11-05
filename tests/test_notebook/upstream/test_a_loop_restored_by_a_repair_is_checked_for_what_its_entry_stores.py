"""A loop run as one unit is restored by a repair, checked for the lineages
its entry stores.

The loop stores the list it changes in place with the lineage its statement
gives it, then moves that lineage on (``update_mutated_variable_lineages``).
The runtime records both: the moved one as the loop's outcome, which the
simulation takes, and the stored one, which a restore checks the entry for.
Checked for the moved one, the entry never matched, and the repair replayed
the loop with what it reads.
"""

from __future__ import annotations

import ast

from cash.notebook.tracking_state import outcome_key
from tests._cell_driver import run_cash_cell

XS = "xs = list(range(3000))"
# Some 30 ms: a statement under the cost model's floor writes no entry.
LOOP = "acc = []\nfor x in xs:\n    acc.append(sum(range(1000)) + x)"
POP = "last = acc.pop()"
CELLS = [XS, LOOP, POP]


def _statuses(magics) -> dict[str, str]:
    return {s["code"].splitlines()[0]: str(s["status"]) for s in magics._last_cell_metrics["statements"]}


def test_the_outcome_records_the_lineage_the_entry_stores(cash_magics, cash_instance):
    run_cash_cell(cash_magics, XS, cells=CELLS)
    run_cash_cell(cash_magics, LOOP, cells=CELLS)
    loop = ast.unparse(ast.parse(LOOP).body[1])
    outcome = cash_magics.tracking_state.control_outcomes[outcome_key(loop, 0)]
    left, stored = outcome[1], outcome[4]

    assert stored["acc"] != left["acc"]
    assert cash_magics.tracking_state.variable_lineage["acc"] == left["acc"]
    (entry,) = [meta for meta in cash_instance.backend.list_entries() if meta.get("code") == loop]
    assert entry["output_lineages"]["acc"] == stored["acc"]


def test_running_the_pop_again_restores_the_loop_from_its_entry(cash_magics):
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)

    run_cash_cell(cash_magics, POP, cells=CELLS)

    statuses = _statuses(cash_magics)
    assert "RESTORED" in statuses["for x in xs:"], statuses
    assert "COMPUTED" not in statuses.get("acc = []", ""), statuses
    ns = cash_magics.shell.user_ns
    assert len(ns["acc"]) == 2999 and ns["last"] == 499500 + 2999
    # Left with the lineage the simulation gave it: the next repair finds
    # the entry the same way.
    run_cash_cell(cash_magics, POP, cells=CELLS)
    assert "RESTORED" in _statuses(cash_magics)["for x in xs:"]
    assert len(ns["acc"]) == 2999 and ns["last"] == 499500 + 2999
