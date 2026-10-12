"""A loop pass that changes an object through a name other than the one a
later statement of the pass reads.

An inner loop changing the outer loop's item (``for r in rows: for k in ks:
r.append(k + lr)``), or a change through a second name (``r = rows[i];
r.append(lr)``, ``w = model['w']`` before the loop), is a change to
``rows`` or ``model``: a later statement of the pass reading the whole
(``s = slow(total(rows))``) must not be served what it read for the old
``lr``, on a Run All and when only the cell below is run. An unchanged Run
All still restores the passes.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

FIRST = (
    "import time\nlr = 1\ndef total(rows):\n    return sum(sum(r) for r in rows)\n"
    "def slow(v):\n    time.sleep(0.25)\n    return v"
)
CASES = {
    "nested-loop": (
        "rows = [[1], [2], [3]]\nlog = []",
        "for r in rows:\n    for k in range(2):\n        r.append(k + lr)\n    s = slow(total(rows))\n    log.append(s)",
        "[9, 12, 15]",
        "[11, 16, 21]",
    ),
    "item-by-index": (
        "rows = [[1], [2], [3]]\nlog = []",
        "for i in range(len(rows)):\n    r = rows[i]\n    r.append(lr)\n    s = slow(total(rows))\n    log.append(s)",
        "[7, 8, 9]",
        "[8, 10, 12]",
    ),
    "dict-value": (
        "table = {'a': [1], 'b': [2]}\nlog = []",
        "for key in ['a', 'b']:\n    r = table[key]\n    r.append(lr)\n    s = slow(total(table.values()))\n    log.append(s)",
        "[4, 5]",
        "[5, 7]",
    ),
    "alias-before-the-loop": (
        "model = {'w': [0]}\nlog = []",
        "w = model['w']\nfor epoch in range(3):\n    w.append(lr)\n    s = slow(total([model['w']]))\n    log.append(s)",
        "[1, 2, 3]",
        "[2, 4, 6]",
    ),
}


def _notebook(nb_runner, case):
    make, loop, before, _after = CASES[case]
    nb_runner.create_notebook(["%cash_badge print", FIRST, make, loop, "print(log)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(5).strip().splitlines()[0] == before


@pytest.mark.parametrize("case", list(CASES))
def test_an_edit_above_reaches_every_pass_on_a_run_all(nb_runner, case):
    _notebook(nb_runner, case)
    nb_runner.set_cell_source(2, FIRST.replace("lr = 1", "lr = 2"))
    nb_runner.run_all()
    assert nb_runner.get_output(5).strip().splitlines()[0] == CASES[case][3], nb_runner.get_raw_output(4)


def test_an_edit_above_reaches_every_pass_when_the_cell_below_runs(nb_runner):
    _notebook(nb_runner, "nested-loop")
    nb_runner.set_cell_source(2, FIRST.replace("lr = 1", "lr = 2"))
    nb_runner.run_cell(5)
    assert nb_runner.peek("log") == CASES["nested-loop"][3]


@pytest.mark.parametrize("case", ["nested-loop", "item-by-index"])
def test_an_unchanged_run_all_restores_the_passes(nb_runner, case):
    _notebook(nb_runner, case)
    nb_runner.run_all()
    assert nb_runner.get_output(5).strip().splitlines()[0] == CASES[case][2]
    assert shows_cached(nb_runner.get_raw_output(4)), nb_runner.get_raw_output(4)
