"""The upstream check looks into each class a cell above reads once, not
once per cell.

Before every cell, the check asks each cell above what it reaches
(``reached_user_code``, for the environment it reads). A cell reading a
pandas frame meets ``DataFrame``, and telling whether a cell gave that class
a method read all of its ~500 members, for every such cell above, on every
cell run: 66-80 ms before each cell of a 300-cell notebook. Inside one check
no code of the notebook runs, so a class is looked into once there
(``callee_reach.one_walk``); the next check looks again.

Counted in member reads, not timed.
"""

from __future__ import annotations

import pytest

from cash.notebook import callee_reach
from tests._cell_driver import run_cash_cell

pd = pytest.importorskip("pandas")


def _member_reads(monkeypatch) -> list:
    reads: list = []
    real = callee_reach._class_member_functions

    def counting(member):
        reads.append(member)
        return real(member)

    monkeypatch.setattr(callee_reach, "_class_member_functions", counting)
    return reads


def test_one_walk_reads_a_library_class_once_for_many_statements(monkeypatch):
    ns = {"__name__": "__main__", "df": pd.DataFrame({"a": [1, 2]})}
    members = len(vars(pd.DataFrame))
    reads = _member_reads(monkeypatch)

    # Control: outside a walk, each statement reads the class again.
    for i in range(3):
        callee_reach.reached_user_code(f"w{i} = df.shape[0] + {i}", ns)
    assert len(reads) >= 3 * members, "the counter never saw the class being read"

    reads.clear()
    with callee_reach.one_walk():
        for i in range(40):
            assert callee_reach.reached_user_code(f"w{i} = df.shape[0] + {i}", ns) == callee_reach._EMPTY
    assert len(reads) <= members, f"{len(reads)} member reads for 40 statements; one class has {members}"


def _member_reads_for_the_last_cell(magics, monkeypatch, n: int, tag: str) -> int:
    cells = ["import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})"]
    cells += [f"{tag}{i} = df.shape[0] + {i}" for i in range(n)]
    for cell in cells[:-1]:
        run_cash_cell(magics, cell, cells=cells)
    reads = _member_reads(monkeypatch)
    run_cash_cell(magics, cells[-1], cells=cells)
    monkeypatch.undo()
    return len(reads)


def test_the_last_cells_check_reads_the_class_no_more_in_a_long_notebook(cash_magics, monkeypatch):
    short = _member_reads_for_the_last_cell(cash_magics, monkeypatch, 5, "short")
    long = _member_reads_for_the_last_cell(cash_magics, monkeypatch, 40, "long")

    assert short > 0, "the check never looked into DataFrame; the cells never reached it"
    assert long <= short + 50, f"{long} class member reads at 40 cells against {short} at 5"
