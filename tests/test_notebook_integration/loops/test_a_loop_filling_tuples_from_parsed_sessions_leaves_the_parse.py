"""A loop filling tuples from parsed sessions leaves the parse alone.

``time_intervals`` is filled with tuples of numbers taken from each
session's actions, and a heatmap cell reads the transition counts another
loop makes. On a slow machine the share checks run out of time, and a check
cut short links each output it could not clear to every value its
statement read. ``trans_count`` and ``time_intervals`` hold nothing any
other value holds, which their reference counts show; but cash's own record
of the loop's outputs, a dict of an array and numbers, is one the collector
stops tracking, so ``trans_count`` showed a holder it could not name, was
linked to the sessions, and each run of a loop moved them on: the next cell
rebuilt everything up to the parse. A plain kernel runs neither the parse
nor the loops again.

When ``time_intervals`` does hold a session's own dict, a change through
it is a change to ``sessions``, and the counts below must see it.
"""

import ast

import pytest

pytestmark = [pytest.mark.timeout(600)]

PARSE = (
    "import random, time\n"
    "def parse_log(n):\n"
    "    time.sleep(0.5)\n"
    "    r = random.Random(0)\n"
    "    return [{'id': k, 'actions': [{'name': 'a_%d' % r.randrange(8), 'at': r.randrange(99)}\n"
    "                                  for _ in range(3000)]}\n"
    "            for k in range(4)]"
)
SESSIONS = "sessions = parse_log(4)  # @cash:no-cache"
INTERVALS = (
    "time_intervals = []\n"
    "for session in sessions:\n"
    "    actions = session['actions']\n"
    "    for i in range(len(actions) - 1):\n"
    "        action = actions[i]\n"
    "        time_intervals.append((action['name'], actions[i + 1]['at'] - action['at']))\n"
    "    latest, n = actions, len(time_intervals)"
)
COUNTS = (
    "import numpy as np\n"
    "trans_count = np.zeros((8, 8), dtype=int)\n"
    "for session in sessions:\n"
    "    actions = session['actions']\n"
    "    for i in range(len(actions) - 1):\n"
    "        trans_count[int(actions[i]['name'][2:]), int(actions[i + 1]['name'][2:])] += 1"
)
HEATMAP = "print(int((trans_count * np.arange(64).reshape(8, 8)).sum()), len(time_intervals))"
#: A slow machine: every share check runs out of time.
NO_BUDGET = (
    "exec('from cash.notebook.statement import processor as p\\n"
    "p.SHARE_CHECK_FLOOR_S = 0.0\\np.SHARE_CHECK_FACTOR = 0.0\\np.SHARE_FALLBACK_FLOOR_S = 0.0', {})"
)
RAN = (
    "[s['code'] for s in (getattr(get_ipython().magics_manager.registry['CashMagics'], '_last_cell_metrics', None) "
    "or {}).get('statements', []) if 'COMPUTED' in str(s['status'])]"
)
MARKS = {
    1: ("def parse_log",),
    2: ("parse_log(4)",),
    3: ("time_intervals = []", "time_intervals.append"),
    4: ("trans_count = np.zeros", "trans_count["),
}


def _plain(cells: list[str]) -> str:
    ns: dict = {}
    for cell in cells[:-1]:
        exec(cell.replace("time.sleep(0.5)", "pass"), ns)
    weighted = int((ns["trans_count"] * ns["np"].arange(64).reshape(8, 8)).sum())
    return f"{weighted} {len(ns['time_intervals'])}"


def _ran_above(nb_runner, cell: int) -> list[str]:
    """The statements of the cells above *cell* its last run ran."""
    ran = ast.literal_eval(nb_runner.peek(RAN))
    others = [mark for number, marks in MARKS.items() if number < cell for mark in marks]
    return [code for code in ran if any(mark in code for mark in others)]


def _start(nb_runner, cells: list[str]) -> None:
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.peek(NO_BUDGET)
    nb_runner.run_all()


@pytest.mark.fresh_kernel
def test_the_heatmap_after_the_intervals_loop_again_runs_nothing_above(nb_runner):
    cells = [PARSE, SESSIONS, INTERVALS, COUNTS, HEATMAP]
    _start(nb_runner, cells)
    plain = _plain(cells)
    assert nb_runner.get_output(5).strip().splitlines()[-1] == plain

    for cell in (3, 5, 3, 5):
        nb_runner.run_cell(cell)
        assert _ran_above(nb_runner, cell) == [], cell
    assert nb_runner.get_output(5).strip().splitlines()[-1] == plain


@pytest.mark.fresh_kernel
def test_intervals_holding_a_session_carry_a_change_to_the_heatmap(nb_runner):
    """Guard: ``time_intervals`` holds the last session's own dict, and a
    change through it reaches the counts the heatmap reads."""
    intervals = INTERVALS + "\ntime_intervals.append(session)"
    change = "time_intervals[-1]['actions'][0]['name'] = 'a_7'\ntime_intervals[-1]['actions'][1]['name'] = 'a_7'"
    cells = [PARSE, SESSIONS, intervals, change, COUNTS, HEATMAP]
    _start(nb_runner, cells)
    plain = _plain(cells)
    assert nb_runner.get_output(6).strip().splitlines()[-1] == plain

    for cell in (3, 4, 5, 6):
        nb_runner.run_cell(cell)
    assert nb_runner.get_output(6).strip().splitlines()[-1] == plain
