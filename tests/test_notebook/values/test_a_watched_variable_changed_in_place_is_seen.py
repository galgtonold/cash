"""A variable the watchers vouch for is still seen changing, however it
changes.

With the C extension, a variable read by several statements is watched and
its digest is not read again while it provably did not change
(`cash.value_watch`). A change made in place, through another name, inside
a function, or to the class its objects belong to must still reach every
statement that reads it: the statement reading it re-runs and gives the
new answer, never the one stored before the change.
"""

from __future__ import annotations

import sys
import types

import pytest

from cash import value_watch
from tests._cell_driver import run_cash_cell

SETUP = [
    "from _cwatch_nb_classes import Pt",
    "data = {'rows': [{'v': i} for i in range(200)], 'meta': {'n': 1}, 'pts': [Pt(i) for i in range(20)]}",
    "total = sum(r['v'] for r in data['rows']) + data['meta']['n'] + sum(p.x for p in data['pts'])",
    "again = sum(r['v'] for r in data['rows'])",
    "def touch(d):\n    return len(d)",
]
POKE = "poked = touch(data)"
READ = "result = sum(r['v'] for r in data['rows']) + data['meta']['n'] + sum(p.x for p in data['pts'])"

CHANGES = {
    "in place": ["data['rows'].append({'v': 1000})"],
    "through an alias": ["rows = data['rows']", "rows[0]['v'] = 500"],
    "inside a function": ["def bump(d):\n    d['meta']['n'] += 7\n    d['rows'][1]['v'] += 3", "bump(data)"],
    "an object's attribute": ["data['pts'][3].x = 40"],
    "through a method": ["data['pts'][4].move(10)"],
    "the class of the objects": ["Pt.move_all(data['pts'], 2)"],
    "by code holding it elsewhere": ["Pt.hold(data['meta'])", "Pt.bump_held(5)"],
}


def _expected(ns: dict) -> int:
    data = ns["data"]
    return sum(r["v"] for r in data["rows"]) + data["meta"]["n"] + sum(p.x for p in data["pts"])


@pytest.fixture
def watched_notebook(cash_magics, monkeypatch):
    """cash_magics with the fast path watching values of any size, and a
    class the cells' objects belong to that pickle can find by name."""
    if not value_watch.enabled():
        pytest.skip(f"the fast path is off: {value_watch.REASON}")
    cw = value_watch._cw
    old = cw.stats()["min_bytes"]
    cw.configure(min_bytes=0)
    mod = types.ModuleType("_cwatch_nb_classes")
    exec(
        "class Pt:\n"
        "    def __init__(self, x):\n        self.x = x\n"
        "    def move(self, d):\n        self.x += d\n"
        "    @classmethod\n"
        "    def move_all(cls, pts, d):\n        for p in pts:\n            p.x += d\n"
        "    @staticmethod\n"
        "    def hold(d):\n        HELD.append(d)\n"
        "    @staticmethod\n"
        "    def bump_held(n):\n        HELD[-1]['n'] += n\n"
        "HELD = []\n",
        mod.__dict__,
    )
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    yield cash_magics
    cw.configure(min_bytes=old)


@pytest.mark.parametrize("how", list(CHANGES))
def test_a_change_reaches_the_statement_that_reads_the_variable(watched_notebook, how):
    magics = watched_notebook
    ns = magics.shell.user_ns
    # The change cells join the notebook only after the first reads: written
    # above READ from the start, cash would run them as READ's upstream.
    unchanged = [*SETUP, POKE, READ]
    cells = [*SETUP, POKE, *CHANGES[how], READ]
    hits = value_watch.stats()["hits"]
    for cell in SETUP:
        run_cash_cell(magics, cell, cells=unchanged)
    for _ in range(3):  # code that could have changed it ran: read again
        run_cash_cell(magics, POKE, cells=unchanged)
        run_cash_cell(magics, READ, cells=unchanged)
    assert value_watch.stats()["hits"] > hits, "the variable was never vouched for"
    before = ns["result"]
    assert before == _expected(ns)
    for cell in CHANGES[how]:
        run_cash_cell(magics, cell, cells=cells)
    run_cash_cell(magics, READ, cells=cells)
    assert ns["result"] == _expected(ns) != before


def test_a_rebound_variable_is_read_again(watched_notebook):
    magics = watched_notebook
    ns = magics.shell.user_ns
    cells = ["xs = [str(i) for i in range(300)]", "n = len(''.join(xs))"]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(magics, cell, cells=cells)
    for i in range(5):
        cell = f"xs = [str(i) + 'x' * {i} for i in range(300)]"
        run_cash_cell(magics, cell, cells=[cell, cells[1]])
        run_cash_cell(magics, cells[1], cells=[cell, cells[1]])
        assert ns["n"] == len("".join(ns["xs"]))
