"""A call on a view and its base keys apart from the call on a copy.

A call on plain data is keyed by what it receives (`call_cache_key`'s
content keying), and an array by its content: ``shares(base, base[2:])``
and ``shares(base, base[2:].copy())`` hashed alike, and the second was
served the first's ``True``. The key now holds where arguments that keep
their data in the same memory sit in it; with none, it is unchanged.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

SETUP = """
import numpy as np, pandas as pd, time

def shares(a, b):
    time.sleep(0.05)
    return bool(np.shares_memory(a, b))

base = np.arange(10.0)
df = pd.DataFrame({"a": np.arange(3.0), "b": np.arange(3.0)})
"""


@pytest.fixture
def magics(cash_magics):
    run_cash_cell(cash_magics, SETUP)
    return cash_magics


@pytest.mark.parametrize(
    ("shared", "apart"),
    [
        ("shares(base, base[2:])", "shares(base, base[2:].copy())"),
        ("shares(base, base[::3])", "shares(base, base[::3].copy())"),
        ("shares(df['a'].to_numpy(), df['a'].to_numpy())", "shares(df['a'].to_numpy(), df['a'].to_numpy().copy())"),
    ],
    ids=["a view", "a strided view", "a column's array"],
)
def test_the_call_on_a_copy_is_not_served_the_shared_ones_result(magics, mock_shell, shared, apart):
    run_cash_cell(magics, f"r = {shared}")
    run_cash_cell(magics, f"q = {apart}")
    run_cash_cell(magics, f"s = {shared}")
    ns = mock_shell.user_ns
    assert (ns["r"], ns["q"], ns["s"]) == (True, False, True)
