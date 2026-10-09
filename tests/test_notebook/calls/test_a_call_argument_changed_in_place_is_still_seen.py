"""A call that changes its argument in place is seen doing it, however few
times the argument is hashed.

``y = f(x)`` hands one digest of ``x`` from the statement's in-place-change
check to the call's (`DigestHandoff`). Each change below must still be seen:
the call is not stored, the statement runs again, and ``x`` ends as it would
without cash.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

SETUP = (
    "import time\n"
    "import numpy as np, pandas as pd\n"
    "def bump(a):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    a += 1\n"
    "    return 0\n"
    "def bump_one(a, i=0):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    a[i] = a[i] + 1\n"
    "    return 0\n"
    "def count_in(d):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    d['n'] = d['n'] + 1\n"
    "    return 0\n"
    "def bump_in_place_of_a_column(d):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    a = d['n'].to_numpy()\n"
    "    a.flags.writeable = True\n"
    "    a[0] += 1\n"
    "    return 0\n"
    "def reads(a, log):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    import os\n"
    "    fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT)\n"
    "    os.write(fd, b'.')\n"
    "    os.close(fd)\n"
    "    return float(a.sum())\n"
)


@pytest.mark.parametrize(
    "make, statement, read",
    [
        ("x = np.zeros(100_000)", "y = bump(x)", "float(x[0])"),
        ("x = np.zeros(100_000)", "y = bump_one(x, i=99_999)", "float(x[99_999])"),
        ("x = np.zeros(100_000)", "bump_one(x)", "float(x[0])"),
        ("x = pd.DataFrame({'n': np.zeros(100_000)})", "y = count_in(x)", "float(x['n'].iloc[0])"),
        ("x = pd.DataFrame({'n': np.zeros(100_000)})", "y = bump_in_place_of_a_column(x)", "float(x['n'].iloc[0])"),
    ],
    ids=["array", "array, last item", "bare call", "frame column", "frame buffer past pandas"],
)
def test_each_run_changes_the_argument_as_without_cash(cash_magics, mock_shell, make, statement, read):
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, SETUP)
    run_cash_cell(cash_magics, make)
    for _ in range(3):
        run_cash_cell(cash_magics, statement)
    assert eval(read, mock_shell.user_ns) == 3.0, "a run skipped the change"


def test_an_argument_only_read_is_still_served(cash_magics, mock_shell, tmp_path):
    """Control: the shared digest still lets an untouched argument cache."""
    mock_shell.user_ns["__name__"] = "__main__"
    log = tmp_path / "runs"
    run_cash_cell(cash_magics, SETUP)
    run_cash_cell(cash_magics, f"x = np.arange(100_000.0)\nlog = {str(log)!r}")
    run_cash_cell(cash_magics, "y = reads(x, log)")
    run_cash_cell(cash_magics, "y = reads(x, log)")
    assert mock_shell.user_ns["y"] == float(np.arange(100_000.0).sum())
    assert log.read_bytes() == b".", "the call ran again"
