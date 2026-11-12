"""A name a loop binds anew on each pass does not move what it was taken from.

``for session in sessions: actions = session['actions']`` leaves ``actions``
bound to the last session's list, the same list the loop before it left
there. In a loop run whole (its ``continue`` keeps it from being split),
taken as a list the loop changed in place, it moved ``sessions`` on,
and every cell after it rebuilt ``sessions`` from its parse (21 s, never
stored), on every run. A plain kernel runs neither the parse nor the other
loop again. A loop that does change the list it took still moves
``sessions``, so a cell reading it sees the change.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

PARSE = """import os, time, random
def parse_log(n, log):
    fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
    os.write(fd, b'p')
    os.close(fd)
    time.sleep(SLEEP)
    r = random.Random(0)
    return [{'id': i, 'actions': [{'name': 'a_%d' % r.randrange(8)} for _ in range(6)]} for i in range(n)]"""

COUNTS = """import numpy as np
trans_count = np.zeros((8, 8), dtype=int)
def act_to_id(action):
    return int(action["name"].split("_")[1])
for session in sessions:
    actions = session['actions']
    for i in range(len(actions) - 2):
        trans_count[act_to_id(actions[i]), act_to_id(actions[i + 1])] += 1"""

STREAKS = """secc_sess = 0
for session in sessions:
    actions = session['actions']
    strik_number = 1
    for i in range(len(actions) - 1):
        if actions[i]['name'] == actions[i + 1]['name']:
            strik_number += 1
            if strik_number == 3:
                secc_sess += 1
                continue
        else:
            strik_number = 1"""


def _cells(log) -> list[str]:
    return [
        PARSE.replace("SLEEP", repr(ABOVE_PERSISTENCE_FLOOR_S)),
        f"sessions = parse_log(300, {str(log)!r})  # @cash:no-cache",
        COUNTS,
        STREAKS,
        "total = int(trans_count.sum()) + secc_sess",
    ]


def _plain(cells: list[str]) -> int:
    ns: dict = {}
    for cell in cells:
        exec(cell, ns)
    return ns["total"]


def test_running_the_cells_below_alone_leaves_the_parse(cash_magics, tmp_path):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    log = tmp_path / "parse.log"
    cells = _cells(log)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert log.read_bytes() == b"p", "control: one parse"

    for index in (4, 3, 4, 2, 4):
        run_cash_cell(cash_magics, cells[index], cells=cells)

    assert log.read_bytes() == b"p"
    assert cash_magics.shell.user_ns["total"] == _plain(_cells(tmp_path / "plain.log"))


@pytest.mark.parametrize(
    "body, moved",
    [("total = len(actions)", False), ("actions.append(0)", True), ("grow(actions)", True)],
    ids=["only read", "changed by its text", "handed to a function"],
)
def test_a_loop_changing_the_list_it_took_still_moves_its_holder(cash_magics, tmp_path, body, moved):
    """Guard: a loop that changes the list it took, by its text or through
    a function it hands it to, still moves ``sessions`` on; one that only
    reads it does not."""
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cells = [
        "def grow(xs):\n    xs.append(0)\nsessions = [{'actions': [1, 2]}, {'actions': [3]}]",
        "for session in sessions:\n    actions = session['actions']\n    actions.sort()",
        f"for session in sessions:\n    actions = session['actions']\n    {body}",
    ]
    for cell in cells[:2]:
        run_cash_cell(cash_magics, cell, cells=cells)
    lineage = cash_magics.tracking_state.variable_lineage["sessions"]

    run_cash_cell(cash_magics, cells[2], cells=cells)

    assert (cash_magics.tracking_state.variable_lineage["sessions"] != lineage) is moved
