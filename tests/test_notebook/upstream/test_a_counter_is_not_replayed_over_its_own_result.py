"""``n = n + 1 if 'n' in globals() else 1`` reads the name it binds. With
no cell above binding ``n``, the check of a cell below replayed it over the
``n`` its own run had bound, which takes ``n`` a step on: the counter looked
changed, and each cell below reading ``n`` ran it again. A Run All in plain
Python runs it once."""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

INIT = "P = 'data.txt'\nopen(P, 'w').write('0')\ndef bump(v):\n    open(P, 'w').write(str(v))"
COUNT = "n = n + 1 if 'n' in globals() else 1"
NOTEBOOKS = {
    # (cells, n after one Run All)
    "a reader of n": (["x = 1", COUNT, "m = n * 2"], 1),
    "the writer in its own cell": ([INIT, COUNT, "bump(n)", "t = open(P).read()"], 1),
    "a second producer below": ([INIT, COUNT, "n = n + 1\nbump(n)", "t = open(P).read()"], 2),
    "a file named after n": ([COUNT + "\nopen(f'd_{n}.txt', 'w').write('x' * n)", "t = open(f'd_{n}.txt').read()"], 1),
}


@pytest.mark.parametrize("notebook", list(NOTEBOOKS))
def test_each_run_all_runs_the_counter_once(cash_magics, tmp_path, monkeypatch, notebook):
    monkeypatch.chdir(tmp_path)
    cells, once = NOTEBOOKS[notebook]
    for runs in (1, 2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert cash_magics.shell.user_ns["n"] == runs * once


def test_a_counter_reading_another_name_runs_again_when_it_moves(cash_magics, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cells = ["k = 1", "n = n + k if 'n' in globals() else 0", "m = n"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    cells[0] = "k = 5"
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["m"] == cash_magics.shell.user_ns["n"] == 5
