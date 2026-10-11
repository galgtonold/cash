"""A variable rebuilt before a cell runs takes what shares its object along.

``g = df.groupby('k')`` holds ``df``. After the loop changing ``df`` in
place is edited and that cell alone is run, cash rebuilds ``df`` as a new
object before the cell (its last change was edited out). ``g`` kept the old
frame and grouped values neither a plain kernel nor a run from the top
produces; it is rebuilt from the new ``df`` too.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

LOOP = "for i in range(1):\n    df.loc[i, 'v'] = {}"


def _cells(x):
    return [
        "import pandas as pd\ndf = pd.DataFrame({'k': [1, 1, 2], 'v': [1.0, 2.0, 3.0]})",
        "g = df.groupby('k')",
        LOOP.format(x),
        "total = g['v'].sum().tolist()",
    ]


def test_a_holder_of_the_rebuilt_frame_sees_the_edited_loop(cash_magics, mock_shell):
    pytest.importorskip("pandas")
    cells = _cells(100)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    cells = _cells(500)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["total"] == [502.0, 3.0]
    cells = _cells(7)
    run_cash_cell(cash_magics, cells[2], cells=cells)
    run_cash_cell(cash_magics, cells[3], cells=cells)
    ns = mock_shell.user_ns
    assert ns["g"].obj is ns["df"]
    assert ns["total"] == [9.0, 3.0]
