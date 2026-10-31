"""``df = pd.DataFrame([], columns=[...])`` runs under cash as it does without.

The columns of such a frame share one object block of no rows. The walk of what
a frame holds reshaped it with -1, which numpy cannot infer for a block of shape
(3, 0), and the ValueError reached the notebook as if the user's line had raised it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cash.notebook import shared_objects as so
from tests._cell_driver import run_cash_cell


@pytest.mark.parametrize(
    ("source", "columns"),
    [
        ("df = pd.DataFrame([], columns=['a', 'b', 'c'])", ["a", "b", "c"]),
        ("df = pd.DataFrame(columns=['a', 'b', 'c'])", ["a", "b", "c"]),
        ("df = pd.DataFrame([], columns=['a'])", ["a"]),
    ],
    ids=["from an empty list", "from columns alone", "one column"],
)
def test_the_assignment_runs_and_binds_the_frame(cash_magics, mock_shell, source, columns):
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, "import pandas as pd")
    run_cash_cell(cash_magics, source)
    df = mock_shell.user_ns["df"]
    assert df.shape == (0, len(columns))
    assert list(df.columns) == columns


def test_what_a_frame_with_rows_holds_is_still_found():
    """The early exit is for a block of no rows only."""
    held = [1, 2]
    df = pd.DataFrame({"a": pd.Series([held, 2.5, "s"], dtype=object), "b": pd.Series([None, {"k": 1}, 3], dtype=object)})
    assert any(item is held for item in so._pandas_objects(df))
