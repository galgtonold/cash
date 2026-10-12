"""An in-place method that changed nothing on its first run still counts as
a change.

A method no list knows, called on a frame, series or array, is looked at
before and after once per text, and what was seen is learned. ``arr.put(idx,
0.0)`` with no negative to clip changed nothing the first time, was learned
as reading only, and when an edit above brought a negative in, the array
changed with its lineage left alone: what was built on it was served stale.
"""

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")

SETUP = (
    "import time\n"
    "import numpy as np, pandas as pd\n"
    "def total(a):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return float(np.asarray(a).sum())\n"
)


@pytest.mark.parametrize(
    ("first", "second", "change", "expected"),
    [
        (
            "arr = np.array([1.0, 2.0, 3.0])",
            "arr = np.array([1.0, -2.0, 3.0])",
            "arr.put(np.flatnonzero(arr < 0), 0.0)",
            4.0,
        ),
        ("arr = np.array([1.0, 2.0, 3.0])", "arr = np.array([3.0, 2.0, 1.0])", "arr.sort()", 6.0),
        (
            "arr = pd.Series([1.0, 2.0, 3.0])",
            "arr = pd.Series([1.0, None, 3.0])",
            "arr.fillna(0.0, inplace=FILL)",
            4.0,
        ),
    ],
    ids=["put", "sort", "inplace_flag"],
)
def test_a_later_run_that_changes_the_receiver_reaches_its_readers(
    cash_magics, mock_shell, first, second, change, expected
):
    # The same reader before and after the change: one key while the
    # receiver's lineage stays put.
    cells = [SETUP + "FILL = True\n", first, "s = total(arr)", change, "s = total(arr)\nfirst = arr[0]"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    cells[1] = second
    for cell in cells[1:]:
        run_cash_cell(cash_magics, cell, cells=cells)

    assert mock_shell.user_ns["s"] == expected
    assert mock_shell.user_ns["first"] == np.asarray(mock_shell.user_ns["arr"])[0]


def test_a_method_that_returns_a_new_frame_still_leaves_the_frame_alone(cash_magics, mock_shell):
    """Positive control: ``df.sort_values('x')`` shown is looked at and
    learned as reading only, so what is built on ``df`` is kept."""
    cells = [SETUP, "df = pd.DataFrame({'x': [3.0, 1.0, 2.0]})", "df.sort_values('x')", "s = total(df.x)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    before = cash_magics.tracking_state.variable_lineage["df"]

    for cell in cells[2:]:
        run_cash_cell(cash_magics, cell, cells=cells)

    assert cash_magics.tracking_state.variable_lineage["df"] == before
    assert mock_shell.user_ns["s"] == 6.0
