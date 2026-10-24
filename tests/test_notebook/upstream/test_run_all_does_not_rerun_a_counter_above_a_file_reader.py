"""``n = n + 1 if 'n' in globals() else 1`` above a writer of ``n`` to a
file: replaying it over the ``n`` its own run bound takes ``n`` a step on,
so the simulation saw the writer's input drift and re-ran the counter and
the writer while it checked each reader below. A first Run All in a fresh
kernel ended with ``n == 3`` where plain Python gives 1."""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

INIT = "P = 'data.txt'\nopen(P, 'w').write('0')"
BUMPS = {
    "in the cell": "n = n + 1 if 'n' in globals() else 1\nopen(P, 'w').write(str(n))",
    "through a function": "n = n + 1 if 'n' in globals() else 1\nbump(n)",
}
DEF = "\ndef bump(v):\n    open(P, 'w').write(str(v))"
READERS = {
    "statement": "t = open(P).read()",
    "loop": "t = ''\nfor name in [P]:\n    t += open(name).read()",
}


@pytest.mark.parametrize("reader", list(READERS))
@pytest.mark.parametrize("bump", list(BUMPS))
def test_a_first_run_all_runs_the_counter_once(cash_magics, tmp_path, monkeypatch, bump, reader):
    monkeypatch.chdir(tmp_path)
    cells = [INIT + DEF, BUMPS[bump], READERS[reader], "w = [t, 'r']"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    ns = cash_magics.shell.user_ns
    assert (ns["n"], ns["t"], ns["w"]) == (1, "1", ["1", "r"])


def test_an_edited_counter_is_run_again_for_the_reader(cash_magics, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cells = [INIT + DEF, BUMPS["in the cell"], READERS["statement"]]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    cells[1] = cells[1].replace("n + 1", "n + 10")
    run_cash_cell(cash_magics, cells[2], cells=cells)
    ns = cash_magics.shell.user_ns
    assert ns["t"] == str(ns["n"]) and ns["n"] == 11
