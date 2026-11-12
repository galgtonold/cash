"""A cell re-run on its own above a cell that changed its input in place
warns NOTEBOOK-CHANGED-BELOW, naming the variable and the cell below.

The re-run reads the input rebuilt as the cells above leave it, so it
computes without the change, and the variable no longer holds it.
"""

import warnings

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


def test_a_cell_below_that_binds_a_new_value_is_named_too(cash_magics):
    """``s = s * 2`` below: the re-run reads ``s`` as the cells above leave
    it, not the doubled one plain Jupyter has, and says so."""
    cells = [CELLS[0], CELLS[1], "s = s * 2"]
    assert not _run(cash_magics, [0, 1, 2], cells)
    found = _run(cash_magics, [1], cells)
    assert len(found) == 1, found
    assert "'s'" in found[0] and "cell 3 below binds anew (`s = s * 2`)" in found[0], found
    assert cash_magics.shell.user_ns["out"] == 3.0


def test_a_binding_in_a_cell_above_is_quiet(cash_magics):
    """The binding the re-run reads is above it: what plain Jupyter has too."""
    cells = [CELLS[0], "s = s * 2", CELLS[1]]
    assert not _run(cash_magics, [0, 1, 2, 2], cells)


# The same binding twice, above and below the re-run cell, with an in-place
# change between them: the copy below ran last, so the live `s` is its value.
TWICE = [
    "import pandas as pd",
    "s = pd.Series([0.0, 1.0, 2.0])",
    "s.iloc[0] = 1e9",
    "out = float(s.sum())",
    "s = pd.Series([0.0, 1.0, 2.0])",
]
IDS = [f"c{i}" for i in range(6)]


def _run_ids(cash_magics, indices, cells):
    ids = IDS[: len(cells)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i in indices:
            run_cash_cell(cash_magics, cells[i], cells=cells, cell_ids=ids, cell_id=ids[i])
    return _changed_below(caught)


def test_a_binding_below_that_a_cell_above_also_holds_is_named(cash_magics):
    """The statement that bound `s` last is in cell 5 and in cell 2 above:
    the cell that ran it last decides, and the re-run of cell 4 says that
    cell 5 below bound the value it does not read."""
    assert not _run_ids(cash_magics, [0, 1, 2, 3, 4], TWICE)
    found = _run_ids(cash_magics, [3], TWICE)
    assert len(found) == 1, found
    assert "'s'" in found[0] and "cell 5 below binds anew" in found[0], found
    assert cash_magics.shell.user_ns["out"] == 1e9 + 3.0


def test_rerunning_the_copy_above_keeps_the_rerun_below_it_quiet(cash_magics):
    """Cell 2 run again binds `s` above cell 4: what plain Jupyter reads too."""
    assert not _run_ids(cash_magics, [0, 1, 2, 3, 4, 1], TWICE)
    assert not _run_ids(cash_magics, [3], TWICE)


def test_of_two_copies_below_the_one_that_ran_last_is_named(cash_magics):
    cells = [*TWICE, TWICE[4]]
    assert not _run_ids(cash_magics, [0, 1, 2, 3, 4, 5, 4], cells)
    found = _run_ids(cash_magics, [3], cells)
    assert len(found) == 1, found
    assert "cell 5 below binds anew" in found[0], found
