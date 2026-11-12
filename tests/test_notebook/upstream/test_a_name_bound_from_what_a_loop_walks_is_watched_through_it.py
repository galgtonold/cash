"""A name a loop binds from what it walks is watched through what it walks.

``for session in sessions: actions = session['actions']`` binds ``actions``
before anything in the loop reads it. Handed to code whose effect is
unknown (``act_to_id(actions[i])`` reads ``action["name"].split(...)``), the
name itself was watched: before the loop it showed whatever list an earlier
loop left there, after it the last session's list, so the loop was taken to
change ``actions`` on every run. That moved ``sessions`` on through the two
names sharing a list, and every cell below rebuilt ``sessions`` from its
parse. A plain kernel runs the parse once.

What a call can change through such a name is what holds the values the
loop binds it to: ``sessions`` is watched instead, and a loop that does
change the sessions' lists through it (``random.shuffle(actions)``, a
helper whose text cannot be read) moves ``sessions`` on, so a cell reading
it is not served a value the edited loop no longer makes.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

#: Passes enough for the loop to run as one unit.
N_SESSIONS = 3000

PARSE = """import os, time, random
def parse_log(n, log):
    fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
    os.write(fd, b'p')
    os.close(fd)
    time.sleep(SLEEP)
    r = random.Random(0)
    return [{'actions': [{'name': 'a_%d' % r.randrange(8), 'at': r.randrange(99)} for _ in range(6)]} for _ in range(n)]"""

TIMES = """gaps = []
for session in sessions:
    actions = sorted(session['actions'], key=lambda x: x['at'])
    gaps.append(actions[-1]['at'] - actions[0]['at'])"""

COUNTS = """import numpy as np
trans_count = np.zeros((8, 8), dtype=int)
def act_to_id(action):
    return int(action["name"].split("_")[1])
for session in sessions:
    actions = session['actions']
    for i in range(len(actions) - 2):
        trans_count[act_to_id(actions[i]), act_to_id(actions[i + 1])] += 1"""


def _cells(log) -> list[str]:
    return [
        PARSE.replace("SLEEP", repr(ABOVE_PERSISTENCE_FLOOR_S)),
        f"sessions = parse_log({N_SESSIONS}, {str(log)!r})  # @cash:no-cache",
        TIMES,
        COUNTS,
        "total = int(trans_count.sum()) + sum(gaps)",
    ]


def _plain(cells: list[str]) -> dict:
    ns: dict = {}
    for cell in cells:
        exec(cell, ns)
    return ns


def test_the_loops_below_the_parse_leave_it_alone(cash_magics, tmp_path):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    log = tmp_path / "parse.log"
    cells = _cells(log)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert log.read_bytes() == b"p", "control: one parse"

    for index in (3, 4, 2, 3, 4, 3, 4):
        run_cash_cell(cash_magics, cells[index], cells=cells)

    assert log.read_bytes() == b"p"
    assert cash_magics.shell.user_ns["total"] == _plain(_cells(tmp_path / "plain.log"))["total"]


SETUP = (
    "import heapq, random, time\n"
    "exec(compile('def hidden(xs):\\n    xs.reverse()', 'nofile', 'exec'))\n"
    "random.seed(1)\n"
    "sessions = [{'actions': [5, 3, 1, 4, 2]} for _ in range(N)]"
)
LOOK = f"def look(s):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return list(s[0]['actions'])\nfirst = look(sessions)"


@pytest.mark.parametrize("n", [5, N_SESSIONS], ids=["passes run one by one", "run as one unit"])
@pytest.mark.parametrize(
    "change",
    ["hidden(actions)", "heapq.heapify(actions)", "random.shuffle(actions)", "actions.reverse()"],
    ids=["a helper whose text cannot be read", "a library function", "a random draw", "the loop's own text"],
)
def test_a_loop_changing_the_lists_it_binds_is_seen_below(cash_magics, n, change):
    """What a cell below reads matches restart and Run All, before the loop
    is edited to change nothing, after, and once it changes them again."""
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    seen = []
    for body in (change, "count = len(actions)", change):
        cells = [
            SETUP.replace("N", str(n)),
            f"for session in sessions:\n    actions = session['actions']\n    {body}",
            LOOK,
        ]
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        seen.append((cash_magics.shell.user_ns["first"], _plain(cells)["first"]))

    assert [mine for mine, _ in seen] == [plain for _, plain in seen]
