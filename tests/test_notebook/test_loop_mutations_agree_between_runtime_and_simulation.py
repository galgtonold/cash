"""The runtime and the simulation agree on what a control structure mutated.

After a loop runs, the runtime gives every variable the loop changed in place
a new lineage (``update_lineage_after_execution``); the simulation of the same
cell after a restart must pick the same variables, or the two key the
statements below the loop apart. They used to have one collector each, and
the two differed on a nested loop's ``else``, on a nested loop's own target,
and on builtin names. Both now call ``control_structure_mutations``; this
checks the runtime's call during a real cell run against the simulation's for
the same loop and state.
"""

from __future__ import annotations

import ast
from unittest.mock import patch

import pytest

from cash.notebook.control_structures import helpers
from cash.notebook.upstream import NotebookSimulator
from tests._cell_driver import run_cash_cell

SHAPES = {
    "a nested loop's else": (
        "acc = []\nlog = []",
        "for i in range(3):\n    for j in range(2):\n        pass\n    else:\n        acc.append(i)\n"
        "    while False:\n        pass\n    else:\n        log.append(i)\n",
        {"acc", "log"},
    ),
    "a nested loop's own target": (
        "rows = [[], []]\nout = []",
        "for i in range(2):\n    for row in rows:\n        row.append(i)\n    out.append(i)\n",
        {"out"},
    ),
    "a builtin name the user bound": (
        "list = []\nseen = []",
        "for i in range(3):\n    list.append(i)\n    seen.append(len(seen))\n",
        {"list", "seen"},
    ),
    "a branch in the loop": (
        "evens = {}\nodds = []",
        "for i in range(4):\n    if i % 2:\n        odds.append(i)\n    else:\n        evens[i] = i\n",
        {"evens", "odds"},
    ),
}


@pytest.mark.parametrize("seed, loop, expected", list(SHAPES.values()), ids=list(SHAPES))
def test_runtime_and_simulation_pick_the_same_mutated_vars(
    cash_magics, cash_instance, mock_shell, seed, loop, expected
):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    run_cash_cell(cash_magics, seed)
    node = ast.parse(loop).body[0]

    runtime: list[set[str]] = []
    collect = helpers.control_structure_mutations

    def spy(n, is_builtin):
        found = collect(n, is_builtin)
        if ast.unparse(n) == ast.unparse(node):
            runtime.append(found)
        return found

    with patch.object(helpers, "control_structure_mutations", spy):
        run_cash_cell(cash_magics, loop)
    assert runtime, "the runtime never collected the loop's mutations"

    simulation = NotebookSimulator(mock_shell, cash_instance, cash_magics.tracking_state).virtual_lineage
    simulated = simulation._collect_loop_mutation_info(node, set(), set())

    assert runtime[-1] == simulated == expected
