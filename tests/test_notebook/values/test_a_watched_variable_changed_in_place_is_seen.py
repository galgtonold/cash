"""A variable the watchers vouch for is still seen changing, however it
changes.

With the C extension, a variable read by several statements is watched and
its digest is not read again while it provably did not change
(`cash.value_watch`). A change made in place, through another name, inside
a function, or to the class its objects belong to must still reach every
statement that reads it: the statement reading it re-runs and gives the
new answer, never the one stored before the change.

Every statement is stored, however fast: under the store floor the reading
statement re-ran anyway and nothing was checked, until a loaded CI runner
made it slow enough to store and the change made by code holding the value
elsewhere came back stale.
"""

from __future__ import annotations

import sys

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


CLASSES = """
class Pt:
    def __init__(self, x):
        self.x = x
    def move(self, d):
        self.x += d
    @classmethod
    def move_all(cls, pts, d):
        for p in pts:
            p.x += d
    @staticmethod
    def hold(d):
        HELD.append(d)
    @staticmethod
    def bump_held(n):
        HELD[-1]['n'] += n

HELD = []
"""


@pytest.fixture
def watched_notebook(cash_magics, cash_instance, tmp_path, monkeypatch):
    """cash_magics with the fast path watching values of any size, every
    statement stored, and the cells' class in a module of the user's own,
    as a notebook's helpers are."""
    if not value_watch.enabled():
        pytest.skip(f"the fast path is off: {value_watch.REASON}")
    cw = value_watch._cw
    old = cw.stats()["min_bytes"]
    cw.configure(min_bytes=0)
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    (tmp_path / "_cwatch_nb_classes.py").write_text(CLASSES, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "_cwatch_nb_classes", raising=False)
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
