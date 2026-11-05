"""A value rebuilt on the way to a cell's input says it drops a change below.

``hist`` is built from ``played``, and a cell below adds a column to
``played`` in place. An edit to the helper makes the cell reading ``hist``
rebuild it, and with it ``played`` as the cells above leave it: the cell gets
what a run from the top gives it, but ``played`` loses the column until the
cell below runs again. NOTEBOOK-CHANGED-BELOW said so only for the cell's own
inputs, and ``played`` is none: it went without a word.
"""

from __future__ import annotations

import warnings

import pytest

from tests._cell_driver import run_cash_cell

CELLS = [
    "import pandas as pd\nsrc = pd.DataFrame({'h': [1, 2, 3], 'a': [0, 2, 1]})",
    "played = src.dropna()\nplayed = played.reset_index(drop=True)",
    "def run(m, k):\n    return m.assign(p=m['h'] * k)",
    "hist = run(played, 2)",
    "total = int(hist['p'].sum())",
    'played["gd"] = played["h"] - played["a"]  # goal difference',
]


def _said(cash_magics, cell, cells):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, cell, cells=cells)
    return [str(w.message) for w in caught if "NOTEBOOK-CHANGED-BELOW" in str(w.message)]


def test_a_value_rebuilt_for_an_input_says_it_lost_the_change_below(cash_magics):
    pytest.importorskip("pandas")
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    edited = list(CELLS)
    edited[2] = "def run(m, k):\n    return m.assign(p=m['h'] * k + 1)"

    said = _said(cash_magics, edited[4], edited)

    ns = cash_magics.shell.user_ns
    assert ns["total"] == 15, "not the value a run from the top gives"
    assert list(ns["played"].columns) == ["h", "a"]
    assert len(said) == 1, said
    assert "rebuilt 'played'" in said[0] and "cell 6 below" in said[0], said[0]


def test_nothing_is_said_when_nothing_is_rebuilt(cash_magics):
    """Control: the reader re-run with nothing edited rebuilds nothing."""
    pytest.importorskip("pandas")
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)

    said = _said(cash_magics, CELLS[4], CELLS)

    assert not said, said
    assert "gd" in cash_magics.shell.user_ns["played"].columns
