"""A table a cell builds is as writable after cash stores it as before.

The RAM tier froze the statement's own table in place to share it, so the
next cell's ``df["x"].array[0] = v`` raised "assignment destination is
read-only" where plain Python works. Cash now freezes only its own copy.
"""

import pytest

pytest.importorskip("pandas")

from tests._cell_driver import run_cash_cell  # noqa: E402
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S  # noqa: E402

BUILD = (
    "import time\nimport numpy as np\nimport pandas as pd\n"
    "df = pd.DataFrame({'x': np.arange(1000, dtype=float), 'when': pd.date_range('2024', periods=1000, freq='min')})\n"
    f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"
)


def test_the_next_cell_writes_the_table_through_its_handles(cash_magics, mock_shell):
    run_cash_cell(cash_magics, BUILD)
    assert mock_shell.user_ns["df"] is not None
    run_cash_cell(
        cash_magics,
        "col = df['x']\ndf['x'].array[0] = -1.0\ncol.array[1] = -2.0\n"
        "df['when'].array[0] = pd.Timestamp('1999-01-01')",
    )
    df = mock_shell.user_ns["df"]
    assert df["x"].iloc[0] == -1.0 and df["x"].iloc[1] == -2.0
    assert df["when"].iloc[0].year == 1999
