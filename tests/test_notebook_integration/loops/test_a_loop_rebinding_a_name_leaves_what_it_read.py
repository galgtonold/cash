"""Loops that bind a name anew on each pass leave the list they read alone.

Two loops over ``sessions`` each run ``actions = session['actions']``,
leaving ``actions`` on the last session's list. The second loop, which has a
``continue`` and so runs whole rather than split, found it on
the same list as before and took it for a list changed in place, moving
``sessions`` on; the cell after it then rebuilt ``sessions`` from its parse
on every run, and after a restart too (never stored, it runs again rather
than being restored). A plain kernel runs neither the parse nor the other
loop again.
"""

import pytest

pytestmark = [pytest.mark.timeout(300)]

PARSE = (
    "import random, time\n"
    "def parse_log(n):\n"
    "    time.sleep(1.5)\n"
    "    r = random.Random(0)\n"
    "    return [{'id': i, 'actions': [{'name': 'a_%d' % r.randrange(8)} for _ in range(6)]} for i in range(n)]"
)
SESSIONS = "sessions = parse_log(400)  # @cash:no-cache"
COUNTS = (
    "import numpy as np\n"
    "trans_count = np.zeros((8, 8), dtype=int)\n"
    "for session in sessions:\n"
    "    actions = session['actions']\n"
    "    for i in range(len(actions) - 2):\n"
    "        trans_count[int(actions[i]['name'][2:]), int(actions[i + 1]['name'][2:])] += 1"
)
STREAKS = (
    "secc_sess = 0\n"
    "for session in sessions:\n"
    "    actions = session['actions']\n"
    "    run = 1\n"
    "    for i in range(len(actions) - 1):\n"
    "        if actions[i]['name'] == actions[i + 1]['name']:\n"
    "            run += 1\n"
    "            if run == 3:\n"
    "                secc_sess += 1\n"
    "                continue\n"
    "        else:\n"
    "            run = 1"
)
TOTAL = "print(int(trans_count.sum()) + secc_sess)"
STATEMENTS = "[s['code'] for s in get_ipython().magics_manager.registry['CashMagics']._last_cell_metrics['statements']]"


def _plain() -> str:
    ns: dict = {}
    for cell in (PARSE, SESSIONS, COUNTS, STREAKS, TOTAL.replace("print(", "_total = (")):
        exec(cell.replace("time.sleep(1.5)", "pass"), ns)
    return str(ns["_total"])


@pytest.mark.fresh_kernel
def test_the_cells_below_the_parse_leave_it_alone(nb_runner):
    nb_runner.create_notebook([PARSE, SESSIONS, COUNTS, STREAKS, TOTAL])
    nb_runner.start_kernel()
    nb_runner.run_all()
    plain = _plain()
    assert nb_runner.get_output(5).strip().splitlines()[-1] == plain

    for cell in (3, 4, 5, 4, 5):
        nb_runner.run_cell(cell)
        assert "parse_log(400)" not in nb_runner.peek(STATEMENTS)
    assert nb_runner.get_output(5).strip().splitlines()[-1] == plain

    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.get_output(5).strip().splitlines()[-1] == plain
    nb_runner.run_cell(4)
    nb_runner.run_cell(5)
    assert "parse_log(400)" not in nb_runner.peek(STATEMENTS)
    assert nb_runner.get_output(5).strip().splitlines()[-1] == plain
