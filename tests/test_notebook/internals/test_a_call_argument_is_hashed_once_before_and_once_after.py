"""``y = normalize(x)`` hashes ``x`` once before the call and once after.

A statement handing a variable to a function of the user's checks that
argument twice over: the statement's in-place-change fingerprint before and
after it runs, and the routed call's argument hash before and after the
call. Four full reads of ``x`` on a first run: 19.1 s against 0.89 s
without cash over a 257 MB frame. When the statement is nothing but that
call, nothing can run between the statement's check and the call's, so
they share one digest each side (`DigestHandoff`). Anything more in the
statement and each check reads for itself.
"""

from __future__ import annotations

import pytest

from cash import content_hashers
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")


@pytest.fixture
def reads_of_x(monkeypatch, mock_shell):
    """How many times the value named ``x`` is hashed in full."""
    seen = []
    for name in ("hash_numpy", "hash_pandas"):
        real = getattr(content_hashers, name)

        def counting(value, *a, _real=real, **k):
            if value is mock_shell.user_ns.get("x"):
                seen.append(1)
            return _real(value, *a, **k)

        monkeypatch.setattr(content_hashers, name, counting)
    return seen


SETUP = (
    "import time\n"
    "import numpy as np, pandas as pd\n"
    "def normalize(a):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return (a - a.mean()) / a.std()\n"
    "def scaled(a, k=2.0):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return a * k\n"
)

ARRAY = "x = np.arange(300_000.0)"


@pytest.mark.parametrize(
    "statement",
    ["y = normalize(x)", "y = scaled(x, 3.0)", "y = scaled(x, k=3.0)"],
)
def test_a_first_run_reads_the_argument_twice(cash_magics, mock_shell, reads_of_x, statement):
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, SETUP)
    run_cash_cell(cash_magics, ARRAY)
    reads_of_x.clear()
    run_cash_cell(cash_magics, statement)
    # Was 4: the statement's fingerprint before it ran, the call's hash
    # before and after, and the statement's fingerprint after it ran.
    assert len(reads_of_x) == 2, f"x was hashed {len(reads_of_x)} times"


def test_a_frame_is_read_once_when_it_provably_did_not_change(cash_magics, mock_shell, reads_of_x):
    """A frame under copy-on-write is checked rather than read again
    (`ArgFingerprints`), by every check that asks."""
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, SETUP)
    run_cash_cell(cash_magics, "x = pd.DataFrame({'a': np.arange(300_000.0), 'b': np.ones(300_000)})")
    reads_of_x.clear()
    run_cash_cell(cash_magics, "y = normalize(x)")
    assert len(reads_of_x) == 1, f"x was hashed {len(reads_of_x)} times"


def test_a_statement_that_is_more_than_the_call_reads_for_each_check(cash_magics, mock_shell, reads_of_x):
    """Control: code may run between the checks, so none is shared."""
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, SETUP)
    run_cash_cell(cash_magics, ARRAY)
    reads_of_x.clear()
    run_cash_cell(cash_magics, "y = normalize(x) + 0")
    assert len(reads_of_x) == 4
