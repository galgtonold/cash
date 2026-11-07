"""A cell cash cannot find in the saved notebook says the cells above went unchecked.

Without a cell id a cell is found by its text. One that is new or edited and
not saved has no match, and the check of the cells above it was skipped
without a word: an edit saved above did not reach it, unlike a run from the
top. cash says so (NOTEBOOK-CELL-UNPLACED), once for each cell text.
"""

from __future__ import annotations

import warnings

from tests._cell_driver import run_cash_cell

SAVED = ["x = 1", "y = x + 1"]


def _codes(magics, code, cells, **ids):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(magics, code, cells=cells, **ids)
    return [getattr(w.message, "code", None) for w in caught]


def _run_all(magics):
    for cell in SAVED:
        assert "NOTEBOOK-CELL-UNPLACED" not in _codes(magics, cell, SAVED)


def test_an_unsaved_cell_reading_a_name_above_says_so_once(cash_magics, mock_shell):
    _run_all(cash_magics)
    assert _codes(cash_magics, "z = x * 10", SAVED).count("NOTEBOOK-CELL-UNPLACED") == 1
    assert "NOTEBOOK-CELL-UNPLACED" not in _codes(cash_magics, "z = x * 10", SAVED)
    assert mock_shell.user_ns["z"] == 10


def test_a_cell_id_the_file_lacks_says_so(cash_magics):
    for i, cell in enumerate(SAVED):
        run_cash_cell(cash_magics, cell, cells=SAVED, cell_ids=["a", "b"], cell_id="ab"[i])
    codes = _codes(cash_magics, "z = x * 10", SAVED, cell_ids=["a", "b"], cell_id="new")
    assert "NOTEBOOK-CELL-UNPLACED" in codes


def test_an_unsaved_cell_reading_nothing_above_is_quiet(cash_magics):
    _run_all(cash_magics)
    assert "NOTEBOOK-CELL-UNPLACED" not in _codes(cash_magics, "w = 3", SAVED)


def test_a_saved_cell_with_a_blank_line_of_spaces_is_found_and_quiet(cash_magics, mock_shell):
    saved = ["x = 1", "def f():\n    v = x + 1\n    \n    return v\n\nz = f()"]
    run_cash_cell(cash_magics, saved[0], cells=saved)
    ran = saved[1].replace("    \n", "\n") + "\n"  # as IPython runs it
    assert "NOTEBOOK-CELL-UNPLACED" not in _codes(cash_magics, ran, saved)
    assert mock_shell.user_ns["z"] == 2
