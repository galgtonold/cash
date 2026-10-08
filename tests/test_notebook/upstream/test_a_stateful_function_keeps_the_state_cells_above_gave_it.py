"""A function that keeps state between calls keeps what the cells above did.

A factory closure (``log = make_log()``), a ``nonlocal`` counter or a mutable
default argument, called once in each of several cells: plain Jupyter carries
the state from cell to cell. Cash re-ran the line that made it before every cell
calling it, so each cell started from fresh state on a first Run All. Only a
cell whose own or a later cell's calls are in the state (an isolated re-run)
has it made fresh.
"""

import json

import pytest

from tests._cell_driver import run_cash_cell

DEFS = """def make_log():
    seen = []
    def log(msg):
        seen.append(msg)
        return len(seen)
    return log
def make_counter():
    n = 0
    def nxt():
        nonlocal n
        n += 1
        return n
    return nxt
def take(item, acc=[]):
    acc.append(item)
    return list(acc)"""
CELLS = [
    DEFS,
    "log = make_log()\nnext_id = make_counter()",
    "a = log('loaded')\nrun1 = next_id()\nt1 = take('x')",
    "b = log('cleaned')\nrun2 = next_id()\nt2 = take('y')",
    "c = log('fitted')\nrun3 = next_id()\nt3 = take('z')",
]


@pytest.fixture(autouse=True)
def _saved_notebook(monkeypatch, tmp_path):
    """The kernel's notebook is found (what the cells define is read from it);
    ``run_cash_cell`` hands over its cells."""
    path = tmp_path / "nb.ipynb"
    cells = [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": c} for c in CELLS]
    path.write_text(json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}), encoding="utf-8")
    monkeypatch.setattr("cash.notebook.upstream.checker.get_notebook_path", lambda: str(path))


def _run_all(magics):
    magics.cash_on("")
    for cell in CELLS:
        run_cash_cell(magics, cell, cells=CELLS)


def test_each_cell_continues_from_the_cells_above(cash_magics, mock_shell):
    _run_all(cash_magics)
    ns = mock_shell.user_ns
    assert (ns["a"], ns["b"], ns["c"]) == (1, 2, 3)
    assert (ns["run1"], ns["run2"], ns["run3"]) == (1, 2, 3)
    assert ns["t3"] == ["x", "y", "z"]


def test_an_isolated_re_run_still_starts_from_fresh_state(cash_magics, mock_shell):
    """Control: the reset the re-run needs is kept. Re-running the first
    calling cell after the cells below it ran starts it from fresh state, as
    on the first run."""
    _run_all(cash_magics)
    run_cash_cell(cash_magics, CELLS[2], cells=CELLS)
    ns = mock_shell.user_ns
    assert (ns["a"], ns["run1"], ns["t1"]) == (1, 1, ["x"])
    run_cash_cell(cash_magics, CELLS[2], cells=CELLS)
    assert (ns["a"], ns["run1"], ns["t1"]) == (1, 1, ["x"])
