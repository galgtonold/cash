"""A frame a cell below changed in place is not taken for an edit above.

The upstream check tells an unsaved edit above the running cell (rebuild
what was made from the variable) from a change a cell below made (leave it).
It looked for the statement that last changed the variable in the cells
below as text, but cash records a statement unparsed: ``played['gd'] = ...``
is not found in a cell that writes ``played["gd"] = ...``. Re-running a cell
that read ``hist``, built from ``played`` above, then re-ran ``hist``'s
statement on the changed frame, and ``hist`` took a column no run from the
top gives it.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

CELLS = [
    "import pandas as pd\nsrc = pd.DataFrame({'h': [1, 2, 3], 'a': [0, 2, 1]})",
    "played = src.dropna()\nplayed = played.reset_index(drop=True)",
    "def run(m):\n    return m.assign(p=m['h'] * 2)",
    "hist = run(played)",
    "total = list(hist.columns)",
    'played["gd"] = played["h"] - played["a"]',
]


def test_a_reader_run_again_keeps_the_value_built_above(cash_magics, mock_shell):
    pytest.importorskip("pandas")
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    assert "gd" in mock_shell.user_ns["played"].columns

    run_cash_cell(cash_magics, CELLS[4], cells=CELLS)

    assert mock_shell.user_ns["total"] == ["h", "a", "p"]
