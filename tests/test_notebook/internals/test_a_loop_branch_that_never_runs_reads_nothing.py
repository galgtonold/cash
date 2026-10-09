"""An ``if`` in a loop body reads what its branch may change only when the
branch runs.

After every pass the ``if`` gave each variable its branches may change a new
lineage from a hash of its whole value, whether or not a branch ran:
``for k in range(7): if df.ss[k] is None: df.ss[k] = ...`` hashed a 1M-row
frame eight times (each pass and the loop's end) and took 23 s against 9 ms
plain. Pins the work, not the behaviour: a refactor may expect this to fail.
"""

from __future__ import annotations

from tests._cell_driver import run_cash_cell
from tests._work_counts import hashed_bytes

BIG = 4_000_000  # bytes in np.zeros(500_000)
LOOP = "for k in range(7):\n    if k == trig:\n        big[k] = 5"


def _hashed_by_the_loop(magics, trig: int) -> int:
    run_cash_cell(magics, f"trig = {trig}")
    with hashed_bytes() as hashed:
        run_cash_cell(magics, LOOP)
    return hashed.bytes


def test_a_branch_that_never_runs_hashes_nothing(cash_magics):
    run_cash_cell(cash_magics, "import numpy as np\nbig = np.zeros(500_000)")
    assert _hashed_by_the_loop(cash_magics, -1) < BIG // 100  # 8 x 4 MB before
    assert float(cash_magics.shell.user_ns["big"].sum()) == 0.0


def test_a_branch_that_runs_once_hashes_the_value_twice(cash_magics):
    """Once when it ran, for the passes after it, and once when the loop ends."""
    run_cash_cell(cash_magics, "import numpy as np\nbig = np.zeros(500_000)")
    hashed = _hashed_by_the_loop(cash_magics, 3)
    assert 2 * BIG <= hashed < 2 * BIG + BIG // 100
    assert float(cash_magics.shell.user_ns["big"].sum()) == 5.0


def test_a_nested_branch_reports_to_its_own_loop(cash_magics):
    run_cash_cell(cash_magics, "import numpy as np\nbig = np.zeros(500_000)\nflags = [0, 0, 1]")
    with hashed_bytes() as hashed:
        run_cash_cell(
            cash_magics,
            "for k in range(3):\n    if k > 5:\n        big[k] = 1\n    else:\n        if flags[k] == 7:\n            big[k] = 2",
        )
    assert hashed.bytes < BIG // 100
