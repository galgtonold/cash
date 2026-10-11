"""A table with a ``date_range`` index keeps its ``freq`` on a hit.

Code that reads the index's freq (``df.shift(1, freq=df.index.freq)``)
must print on every Run All, and after a restart, what it prints the first
time, as in a plain kernel.
"""

import pytest

from tests._nbharness.badge import shows_cached
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.timeout(300)]

CELLS = [
    "import cash\n%cash_on\n%cash_badge print",
    "import time, pandas as pd, numpy as np",
    (
        f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "daily = pd.DataFrame({'sales': np.arange(5.0)}, index=pd.date_range('2024-01-01', periods=5, freq='D'))"
    ),
    "prev = daily.shift(1, freq=daily.index.freq)\nr = (daily.index.freqstr, prev['sales'].tolist())",
]

EXPECTED = repr(("D", [0.0, 1.0, 2.0, 3.0, 4.0]))


def test_a_hit_of_a_dated_table_keeps_its_freq(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("r") == EXPECTED, "first Run All"
    nb_runner.run_all()
    raw = nb_runner.get_raw_output(3)
    assert shows_cached(raw), f"second Run All: the table cell is a hit\n{raw}"
    assert nb_runner.peek("r") == EXPECTED, "second Run All"
    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.peek("r") == EXPECTED, "after a restart"
