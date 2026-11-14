"""A loop over a list of dicts building tuples shows what plain Python shows.

At a loop's end the share check tells what holds the list it iterated from
reference counts, where each reference is one it can see, and reads the
heap otherwise. A session held where no count sees it -- a closure, a
global list of a helper module -- keeps its share: a change made through it
reaches the cells below, as in plain Python.
"""

import pytest

pytestmark = [pytest.mark.timeout(300)]

SESSIONS = (
    "import random\n"
    "r = random.Random(0)\n"
    "sessions = [{'id': k, 'actions': [{'name': 'a_%d' % r.randrange(8), 'at': r.randrange(99)}\n"
    "                                  for _ in range(50)]} for k in range(40)]"
)
INTERVALS = (
    "time_intervals = []\n"
    "for session in sessions:\n"
    "    actions = session['actions']\n"
    "    for i in range(len(actions) - 1):\n"
    "        action = actions[i]\n"
    "        time_intervals.append((action['name'], actions[i + 1]['at'] - action['at']))"
)
#: Where a session is held where no count sees it, and the change made
#: through that holder.
HOLDERS = {
    "closure": ("def make(s):\n    return lambda: s\npeek = make(sessions[0])", "peek()['actions'][0]['at'] = 500"),
    "module": (
        "import types, sys\nholder = types.ModuleType('holder')\nholder.KEPT = [sessions[1]]\nsys.modules['holder'] = holder",
        "sys.modules['holder'].KEPT[0]['actions'][0]['at'] = 700",
    ),
}
TOTAL = "print(sum(t for _, t in time_intervals), sum(a['at'] for s in sessions for a in s['actions']))"


def _plain(cells: list[str]) -> str:
    ns: dict = {}
    for cell in cells[:-1]:
        exec(cell, ns)
    return f"{sum(t for _, t in ns['time_intervals'])} {sum(a['at'] for s in ns['sessions'] for a in s['actions'])}"


@pytest.mark.parametrize("kind", sorted(HOLDERS))
def test_the_loops_outputs_equal_plain_and_a_hidden_holders_change_reaches_below(nb_runner, kind):
    hold, change = HOLDERS[kind]
    cells = [SESSIONS, INTERVALS, hold, change, INTERVALS + "  # again", TOTAL]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    plain = _plain(cells)
    assert nb_runner.get_output(6).strip().splitlines()[-1] == plain

    for cell in (5, 6, 4, 5, 6):
        nb_runner.run_cell(cell)
    assert nb_runner.get_output(6).strip().splitlines()[-1] == plain
