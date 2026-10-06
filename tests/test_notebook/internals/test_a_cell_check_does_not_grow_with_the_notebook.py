"""Checking a cell costs the same in a long notebook as in a short one.

Every cell's upstream check re-read every cell above it -- the statements
each one writes, which of them write files, where each statement stands --
from scratch, though none of it changed. Run All on a 300-cell notebook spent
80 ms a cell on that against 25 ms at 40 cells, and grew quadratically. What
depends on the cell sources alone is now read once per notebook version.

Counted in parses, not timed: a parse per cell above is the cost that grew.
"""

from __future__ import annotations

import ast
from unittest.mock import patch

from tests._cell_driver import run_cash_cell


def _parses_for_the_last_cell(magics, n: int, tag: str) -> int:
    cells = ["base = 1"] + [f"{tag}{i} = base + {i}" for i in range(n)]
    for cell in cells[:-1]:
        run_cash_cell(magics, cell, cells=cells)
    real = ast.parse
    calls = 0

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    with patch.object(ast, "parse", counting):
        run_cash_cell(magics, cells[-1], cells=cells)
    return calls


def test_the_last_cell_parses_no_more_in_a_long_notebook(cash_magics):
    short = _parses_for_the_last_cell(cash_magics, 10, "short")
    long = _parses_for_the_last_cell(cash_magics, 60, "long")

    assert short > 0, "the check parsed nothing; the cell never reached it"
    assert long <= short + 5, f"{long} parses at 60 cells against {short} at 10"
