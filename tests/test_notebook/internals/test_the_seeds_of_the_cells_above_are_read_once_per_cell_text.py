"""What the cells above seed is read once per cell text, not before each cell.

Once a statement has seeded a global stream, each cell asks every cell above
it whether it seeds that stream, as a restart and Run All would see it. It
unparsed each top-level statement of each cell above, and walked each one's
tree for a call to a notebook function that seeds, before every cell: in a
174-cell notebook, 15,000 cells read per Run All, a second of it.

Counted in unparses and tree walks, not timed.
"""

from __future__ import annotations

import ast
from unittest.mock import patch

from tests._cell_driver import run_cash_cell


def _reads_for_the_last_cell(magics, n: int, tag: str) -> int:
    cells = [
        "import numpy as np",
        "def reseed(s):\n    np.random.seed(s)",
        "np.random.seed(0)\nbase = 1",
        "reseed(1)\nother = 2",
    ]
    cells += [f"{tag}{i} = base + {i}\nprint({tag}{i})" for i in range(n)]
    for cell in cells[:-1]:
        run_cash_cell(magics, cell, cells=cells)
    unparse, children = ast.unparse, ast.iter_child_nodes
    calls = 0

    def counting_unparse(*args, **kwargs):
        nonlocal calls
        calls += 1
        return unparse(*args, **kwargs)

    def counting_children(*args, **kwargs):
        nonlocal calls
        calls += 1
        return children(*args, **kwargs)

    with patch.object(ast, "unparse", counting_unparse), patch.object(ast, "iter_child_nodes", counting_children):
        run_cash_cell(magics, cells[-1], cells=cells)
    return calls


def test_the_last_cell_reads_the_cells_above_no_more_in_a_long_notebook(cash_magics):
    short = _reads_for_the_last_cell(cash_magics, 10, "short")
    long = _reads_for_the_last_cell(cash_magics, 60, "long")

    assert short > 0, "nothing was read; the cell never reached the checks"
    assert long <= short + 20, f"{long} reads at 60 cells against {short} at 10"
