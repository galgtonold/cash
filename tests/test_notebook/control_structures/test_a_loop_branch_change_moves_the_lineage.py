"""A change an ``if`` branch makes inside a loop is seen by what reads it.

An ``if`` in a loop body updates the lineage of what its branch may change
only when the branch runs, and the loop leaves a variable only a branch can
change with its lineage when no branch ran. These pin that a change a branch
did make is never missed: by a later statement of the same loop, by a
statement after the loop, and by the next run of the loop when which passes
take the branch changes.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


SETUP = (
    "class Box:\n"
    "    def __init__(self):\n"
    "        self.items = []\n"
    "    def push(self, x):\n"
    "        self.items.append(x)\n"
    "    def __len__(self):\n"
    "        return len(self.items)\n"
)


@pytest.mark.parametrize(
    ("change", "name"),
    [
        ("box.push(k)", "box"),  # a bare method call
        ("acc.add(k)", "acc"),
        ("d.update({k: 1})", "d"),
        ("d[k] = 1", "d"),
    ],
)
def test_a_later_statement_of_the_loop_sees_the_change(cash_magics, change, name):
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, SETUP)
    loop = (
        "seen = []\n"
        "for k in range(4):\n"
        "    if k == trig:\n"
        f"        {change}\n"
        f"    n = len({name})\n"
        "    seen.append(n)"
    )
    for trig, expected in ((1, [0, 1, 1, 1]), (2, [0, 0, 1, 1]), (-1, [0, 0, 0, 0]), (1, [0, 1, 1, 1])):
        run_cash_cell(cash_magics, f"box = Box()\nacc = set()\nd = {{}}\ntrig = {trig}")
        run_cash_cell(cash_magics, loop)
        assert ns["seen"] == expected, (change, trig, ns["seen"])


def test_a_statement_after_the_loop_sees_a_branch_that_ran(cash_magics):
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, "data = [0] * 5\ntrig = -1")
    loop = "for k in range(5):\n    if k == trig:\n        data[k] = 100"
    run_cash_cell(cash_magics, loop)
    run_cash_cell(cash_magics, "total = sum(data)")
    assert ns["total"] == 0
    run_cash_cell(cash_magics, "trig = 2")
    run_cash_cell(cash_magics, loop)
    run_cash_cell(cash_magics, "total = sum(data)")
    assert ns["total"] == 100
    run_cash_cell(cash_magics, "trig = -1")
    run_cash_cell(cash_magics, loop)
    run_cash_cell(cash_magics, "total = sum(data)")
    assert ns["total"] == 100


def test_a_variable_no_branch_changed_keeps_its_lineage(cash_magics):
    lineage = cash_magics.tracking_state.variable_lineage
    run_cash_cell(cash_magics, "data = [0] * 5\nother = []\ntrig = -1")
    before = lineage["data"]
    run_cash_cell(cash_magics, "for k in range(5):\n    if k == trig:\n        data[k] = 100\n    other.append(k)")
    assert lineage["data"] == before
    assert cash_magics.shell.user_ns["other"] == [0, 1, 2, 3, 4]
    run_cash_cell(cash_magics, "trig = 4")
    run_cash_cell(cash_magics, "for k in range(5):\n    if k == trig:\n        data[k] = 100\n    other.append(k)")
    assert lineage["data"] != before


def test_a_branch_first_taken_in_the_rest_run_as_one_unit_moves_the_lineage(cash_magics):
    # A loop over a generator runs its first passes one by one and the rest
    # as one unit, which reports no branch it ran: the branch first fires
    # there, after the passes run one by one.
    ns = cash_magics.shell.user_ns
    lineage = cash_magics.tracking_state.variable_lineage
    summary = "summary = summarize(errors)"
    run_cash_cell(
        cash_magics,
        "import time\n"
        "def summarize(v):\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    return (len(v), sum(v))\n"
        "lines = (n for n in range(400))\n"
        "errors = []",
    )
    run_cash_cell(cash_magics, summary)
    assert ns["summary"] == (0, 0)
    before = lineage["errors"]
    run_cash_cell(cash_magics, "for n in lines:\n    if n > 300:\n        errors.append(n)")
    assert lineage["errors"] != before
    run_cash_cell(cash_magics, summary)
    assert ns["summary"] == (99, 34650)
