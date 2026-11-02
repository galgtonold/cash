"""A loop that only reads the items of a list leaves the list's lineage alone.

``for sid, (uid, actions) in enumerate(data): rows.append((uid, sid, ...))``
binds ``actions`` to an element of ``data``, so the share check found
``data`` holding an object of the loop's outputs. Two things then moved
``data`` on with every run of the loop, though nothing changed it: every
output was linked to ``data``, ``rows`` too, so each append moved it; and the
target ``actions`` ended on the same element as on the run before, which read
as an output changed in place. The next run of the cell recorded ``rows`` as
built on that moved ``data``, and the cell below, reading only the frame built
from ``rows``, took the frame for one built on an older ``data`` and rebuilt
the whole chain: ``data``, the sort, ``rows`` and the frame.
"""

import pytest

pytestmark = [pytest.mark.timeout(300)]

MAKE = (
    "import random\nimport datetime\nrandom.seed(3)\n"
    "data_raw = []\nt = 1_600_000_000\n"
    "for s in range(2000):\n"
    "    acts = []\n"
    "    for k in range(random.randint(3, 8)):\n"
    "        t += random.randint(1, 50)\n"
    "        acts.append(f'a{k}|{t}')\n"
    "    data_raw.append(f'u{s % 90};' + ';'.join(acts))"
)
DATA = (
    "data = []\n"
    "for row in data_raw:\n"
    "    uid, *actions_raw = row.split(';')\n"
    "    actions = [(a.split('|')[0], datetime.datetime.fromtimestamp(int(a.split('|')[1]))) for a in actions_raw]\n"
    "    data.append((uid, actions))"
)
SORT = "data.sort(key=lambda x: x[-1][0][-1])"
ROWS = (
    "rows = []\n"
    "for sid, (uid, actions) in enumerate(data):\n"
    "    for a_name, a_time in actions:\n"
    "        rows.append((uid, sid, a_time, a_name))\n"
    "n_rows = len(rows)"
)
READ = "print(n_rows, len(data))"
STATEMENTS = "[s['code'] for s in get_ipython().magics_manager.registry['CashMagics']._last_cell_metrics['statements']]"
LINEAGE = "get_ipython().magics_manager.registry['CashMagics'].tracking_state.variable_lineage.get('data')"


@pytest.mark.fresh_kernel
def test_running_the_loop_twice_rebuilds_nothing_above_the_reader(nb_runner):
    nb_runner.create_notebook([MAKE, DATA, SORT, ROWS, READ])
    nb_runner.start_kernel()
    for cell in (1, 2, 3):
        nb_runner.run_cell(cell)
    before = nb_runner.peek(LINEAGE)

    nb_runner.run_cell(4)
    nb_runner.run_cell(4)

    assert nb_runner.peek(LINEAGE) == before
    nb_runner.run_cell(5)
    ran = nb_runner.peek(STATEMENTS)
    assert "data = []" not in ran
    assert "rows = []" not in ran
    total = nb_runner.peek("sum(len(a) for _, a in data)").strip()
    assert nb_runner.get_output(5).strip().splitlines()[-1] == f"{total} 2000"


#: ``(cells, what ``print(data)`` below them shows, whether the cells
#: after the first move ``data`` on)``: as restart and Run All give in plain
#: Python. None leaves the lineage unasserted: the loop rebinding its
#: target to a copy before changing it still moves ``data`` on, as the
#: change analysis does not follow the rebinding -- a rebuild too many,
#: never a stale value.
CASES = {
    "a target changed in place": (
        ["data = [[0], [0]]", "for r in data:\n    r.append(1)"],
        "[[0, 1], [0, 1]]",
        True,
    ),
    "an item set through the target": (
        ["data = [[1], [2]]", "for a in data:\n    a[0] = 5"],
        "[[5], [5]]",
        True,
    ),
    "a target rebound to a copy and changed": (
        ["data = [[1], [2]]", "for a in data:\n    a = list(a)\n    a.append(1)"],
        "[[1], [2]]",
        None,
    ),
    "an element changed through another name": (
        ["data = [[1], [2]]", "first = data[0]", "first.append(9)"],
        "[[1, 9], [2]]",
        True,
    ),
}


@pytest.mark.fresh_kernel
@pytest.mark.parametrize("case", list(CASES))
def test_what_the_loop_changes_through_an_element_is_seen_below(nb_runner, case):
    cells, plain, moves = CASES[case]
    nb_runner.create_notebook([*cells, "print(data)"])
    nb_runner.start_kernel()
    last = len(cells) + 1
    nb_runner.run_cell(1)
    before = nb_runner.peek(LINEAGE)
    for cell in range(2, last + 1):
        nb_runner.run_cell(cell)
    assert nb_runner.get_output(last).strip().splitlines()[-1] == plain
    if moves is not None:
        assert (nb_runner.peek(LINEAGE) != before) is moves

    nb_runner.run_all()
    assert nb_runner.get_output(last).strip().splitlines()[-1] == plain
    nb_runner.run_cell(2)
    nb_runner.run_cell(last)
    assert nb_runner.get_output(last).strip().splitlines()[-1] == plain
