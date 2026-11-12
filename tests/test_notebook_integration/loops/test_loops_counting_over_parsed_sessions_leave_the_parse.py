"""Loops counting over parsed sessions leave the parse alone, after a restart too.

Three loops over ``sessions`` each open with ``actions = ...``; the last
hands the actions to a function of the user's that reads ``.split`` of each
name. Watched as itself, ``actions`` showed another list after the loop
than before it, on every run: ``sessions`` moved on, and the next cell
rebuilt it from its parse (never stored) and ran the loops over it again.
A plain kernel runs neither the parse nor the other loops again.
"""

import ast

import pytest

pytestmark = [pytest.mark.timeout(600)]

PARSE = (
    "import random, time\n"
    "def parse_log(n):\n"
    "    time.sleep(0.5)\n"
    "    r = random.Random(0)\n"
    "    return [{'actions': [{'name': 'a_%d' % r.randrange(8), 'at': r.randrange(99)} for _ in range(6)]}\n"
    "            for _ in range(n)]"
)
SESSIONS = "sessions = parse_log(3000)  # @cash:no-cache"
STREAKS = (
    "streaks = 0\n"
    "for session in sessions:\n"
    "    actions = session['actions']\n"
    "    run = 1\n"
    "    for i in range(len(actions) - 1):\n"
    "        if actions[i]['name'] == actions[i + 1]['name']:\n"
    "            run += 1\n"
    "            if run == 3:\n"
    "                streaks += 1\n"
    "                continue\n"
    "        else:\n"
    "            run = 1"
)
GAPS = (
    "gaps = []\n"
    "for session in sessions:\n"
    "    actions = sorted(session['actions'], key=lambda x: x['at'])\n"
    "    gaps.append(actions[-1]['at'] - actions[0]['at'])"
)
COUNTS = (
    "import numpy as np\n"
    "trans_count = np.zeros((8, 8), dtype=int)\n"
    "def act_to_id(action):\n"
    "    return int(action['name'].split('_')[1])\n"
    "for session in sessions:\n"
    "    actions = session['actions']\n"
    "    for i in range(len(actions) - 2):\n"
    "        trans_count[act_to_id(actions[i]), act_to_id(actions[i + 1])] += 1"
)
SHOW = "print(int(trans_count.sum()) + streaks + sum(gaps))"
CELLS = [PARSE, SESSIONS, STREAKS, GAPS, COUNTS, SHOW]
RAN = (
    "[s['code'] for s in (getattr(get_ipython().magics_manager.registry['CashMagics'], '_last_cell_metrics', None) "
    "or {}).get('statements', []) if 'COMPUTED' in str(s['status'])]"
)


def _plain() -> str:
    ns: dict = {}
    for cell in CELLS[:-1]:
        exec(cell.replace("time.sleep(0.5)", "pass"), ns)
    return str(int(ns["trans_count"].sum()) + ns["streaks"] + sum(ns["gaps"]))


#: What tells each cell's statements apart, by its number.
MARKS = {
    1: ("def parse_log",),
    2: ("parse_log(3000)",),
    3: ("streaks = 0", "run = 1"),
    4: ("gaps = []", "sorted(session"),
    5: ("trans_count = np.zeros", "act_to_id(actions"),
}


def _ran_above(nb_runner, cell: int) -> list[str]:
    """The statements of other cells the last run of *cell* ran."""
    ran = ast.literal_eval(nb_runner.peek(RAN))
    marks = [mark for number, marks in MARKS.items() if number != cell for mark in marks]
    return [code for code in ran if any(mark in code for mark in marks)]


@pytest.mark.fresh_kernel
def test_the_loops_below_the_parse_rebuild_nothing_above_them(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    plain = _plain()
    assert nb_runner.get_output(6).strip().splitlines()[-1] == plain

    for cell in (5, 6, 4, 5, 6, 3, 6):
        nb_runner.run_cell(cell)
        assert _ran_above(nb_runner, cell) == [], cell
    assert nb_runner.get_output(6).strip().splitlines()[-1] == plain

    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    for cell in (5, 6, 4, 6):
        nb_runner.run_cell(cell)
        assert _ran_above(nb_runner, cell) == [], cell
    assert nb_runner.get_output(6).strip().splitlines()[-1] == plain
