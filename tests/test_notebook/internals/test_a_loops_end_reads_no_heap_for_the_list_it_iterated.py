"""A loop's end reads no heap for the list it iterated.

A loop over a list of dicts building a list of tuples ends with a share
check of what it bound. The list it iterated joins the group as their
holder, and its reference count shows references the check does not
account for: cash's own frames and its record of the loop's header. The
climb up ``gc.get_referrers`` from it read the whole heap twice per check,
four times per loop end. Each of those references is a local of a running
function or an item of a container one is bound to, so the referrers are
told by count. Pins the work, not the behaviour: a refactor may expect this
to fail.
"""

from __future__ import annotations

import sys

import pytest

import cash.notebook.shared_objects as shared_objects
from tests._cell_driver import run_cash_cell

SESSIONS = (
    "import random\n"
    "r = random.Random(0)\n"
    "sessions = [{'id': k, 'actions': [{'name': 'a_%d' % r.randrange(8), 'at': r.randrange(99)}\n"
    "                                  for _ in range(40)]} for k in range(30)]"
)
INTERVALS = (
    "time_intervals = []\n"
    "for session in sessions:\n"
    "    actions = session['actions']\n"
    "    for i in range(len(actions) - 1):\n"
    "        action = actions[i]\n"
    "        time_intervals.append((action['name'], actions[i + 1]['at'] - action['at']))"
)
PAIRS = "pairs = []\nfor session in sessions:\n    for a in session['actions'][:5]:\n        pairs.append((session['id'], a['at']))"
CHANGE = "sessions[0]['actions'][0]['at'] = 500"
TOTAL = "total = sum(t for _, t in time_intervals) + sum(at for _, at in pairs)"


@pytest.fixture
def climb_reads(monkeypatch):
    reads = []
    real = shared_objects._referrers

    def counted(objs, stop):
        if sys._getframe(1).f_code.co_name == "_climb_level":
            reads.append(len(objs))
        return real(objs, stop)

    monkeypatch.setattr(shared_objects, "_referrers", counted)
    return reads


def _plain(cells: list[str]) -> int:
    ns: dict = {}
    for cell in cells:
        exec(cell, ns)
    return ns["total"]


def test_the_loops_end_climbs_by_count(cash_magics, climb_reads):
    cells = [SESSIONS, INTERVALS, PAIRS, CHANGE, INTERVALS + "  # again", PAIRS + "  # again", TOTAL]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for code in cells:
        run_cash_cell(cash_magics, code, cells=cells)
    assert cash_magics.shell.user_ns["total"] == _plain(cells)
    if shared_objects._STRONG_LOCALS and shared_objects._LOCALS_READABLE:
        assert climb_reads == []  # 16 reads before
    else:
        # A local bound to the list itself is not counted there (borrowed
        # on 3.14, read only as a copy before 3.13): the heap is read.
        assert climb_reads
