"""A cell re-run on its own above a cell that changed its input in place
warns NOTEBOOK-CHANGED-BELOW, naming the variable and the cell below.

The re-run reads the input rebuilt as the cells above leave it, so it
computes without the change, and the variable no longer holds it.
"""

import warnings

import pytest

from tests._cell_driver import run_cash_cell

CELLS = ["import pandas as pd\ns = pd.Series([0.0, 1.0, 2.0])", "out = float(s.sum())", "s.iloc[0] = 1e9"]


def _changed_below(caught):
    return [str(w.message) for w in caught if getattr(w.message, "code", None) == "NOTEBOOK-CHANGED-BELOW"]


def _run(cash_magics, indices, cells=CELLS):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i in indices:
            run_cash_cell(cash_magics, cells[i], cells=cells)
    return _changed_below(caught)


def test_rerunning_the_cell_above_names_the_variable_and_the_cell(cash_magics):
    assert not _run(cash_magics, [0, 1, 2])
    found = _run(cash_magics, [1])
    assert len(found) == 1
    assert "'s'" in found[0] and "cell 3" in found[0] and "`s.iloc[0] = 1e9`" in found[0]
    # What the message says: the re-run computed without the change, and `s` lost it.
    assert cash_magics.shell.user_ns["out"] == 3.0
    assert cash_magics.shell.user_ns["s"].iloc[0] == 0.0


def test_a_second_run_from_the_top_is_quiet(cash_magics):
    """The same notebook warns on the re-run alone (positive control)."""
    assert not _run(cash_magics, [0, 1, 2, 0, 1, 2])
    assert _run(cash_magics, [1])


def test_a_cell_below_that_binds_a_new_value_is_quiet(cash_magics):
    """Nothing below changed the value in place; the same notebook with an
    in-place change warns (positive control in the test above)."""
    cells = [CELLS[0], CELLS[1], "s = s * 2"]
    assert not _run(cash_magics, [0, 1, 2], cells)
    assert not _run(cash_magics, [1], cells)
    assert _run(cash_magics, [0, 1, 2, 1])


#: A cell below that changes the input through what it calls: a function
#: setting a ``global``, a function changing it in place.
THROUGH_CALLS = {
    "a global set by a function": ["k = 1\ndef setk(x):\n    global k\n    k = x", "v = k + 0", "setk(2)"],
    "a function changing it in place": ["D = {'k': 1}\ndef fill():\n    D['k'] = 2", "v = D['k']", "fill()"],
}


@pytest.mark.parametrize("how", list(THROUGH_CALLS))
def test_a_change_made_through_a_call_below_warns(cash_magics, how):
    cells = THROUGH_CALLS[how]
    _run(cash_magics, [0, 1, 2], cells)
    found = _run(cash_magics, [1], cells)
    assert len(found) == 1, found
    assert "cell 3 below changed through what it calls" in found[0]


#: A cell below that changes the input through work it hands to a thread,
#: with the statement that handed it off.
THROUGH_THREADS = {
    "a joined thread": (
        [
            "import threading\nD = {'k': 1}",
            "v = D['k']",
            "def fill():\n    D['k'] = 2\nth = threading.Thread(target=fill)\nth.start()\nth.join()",
        ],
        "th.start()",
    ),
    "a thread pool": (
        [
            "from concurrent.futures import ThreadPoolExecutor\nD = {'k': 1}",
            "v = D['k']",
            "with ThreadPoolExecutor(1) as ex:\n    ex.submit(D.__setitem__, 'k', 2).result()",
        ],
        "with ThreadPoolExecutor(1) as ex:",
    ),
}


@pytest.mark.parametrize("how", list(THROUGH_THREADS))
def test_a_change_made_by_a_thread_below_warns_and_names_what_started_it(cash_magics, how):
    cells, statement = THROUGH_THREADS[how]
    _run(cash_magics, [0, 1, 2], cells)
    found = _run(cash_magics, [1], cells)
    assert len(found) == 1, found
    assert f"cell 3 below changed through work it handed to a thread (`{statement}`)" in found[0]
