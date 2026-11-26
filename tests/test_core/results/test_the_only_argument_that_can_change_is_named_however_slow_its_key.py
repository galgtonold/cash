"""The only argument of a call that can change is the one a change was made
to, whatever its key cost to build.

Naming which argument a call changed in place hashes each one before the
body, and is retired for a function once that costs more than
`purity_checks.MUTATION_CHECK_BUDGET_S` (or once building the key did): the
change is then said of "an argument", and the call is not stored with the
state it left the argument in. One argument that can change needs no hash to
be named, but the key's cost retired it too. On a loaded macOS runner one key
build of a three-row frame past 50 ms (a collection of a big heap, another
worker's turn) retired naming for the rest of the test, and ``work(df)``,
which writes into ``df``, ran on every call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cash.decorator import purity_checks


@pytest.fixture
def every_key_over_budget(monkeypatch):
    """Every key build and every argument hash costs more than naming may."""
    monkeypatch.setattr(purity_checks, "MUTATION_CHECK_BUDGET_S", -1.0)


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: np.arange(4.0), id="array"),
        pytest.param(lambda: pd.DataFrame({"a": [1.0, 2.0, 3.0]}), id="frame"),
    ],
)
def test_a_call_changing_its_only_argument_is_stored_however_slow_its_key(disk_cash, every_key_over_budget, make):
    runs = []

    @disk_cash.cache
    def work(v, factor=2):
        runs.append(1)
        v *= factor
        return float(np.asarray(v).sum())

    for _ in range(3):
        work(make())
    warm = len(runs)
    v = make()
    assert work(v) == float(np.asarray(make()).sum()) * 2
    assert len(runs) == warm, "a call changing its one argument in place was never stored"
    assert np.asarray(v).tolist() == (np.asarray(make()) * 2).tolist()


def test_with_two_arguments_that_can_change_none_is_named_past_the_budget(disk_cash, every_key_over_budget):
    """Control: with two, telling which one changed takes a hash of each,
    which the budget retires; the call runs every time."""
    runs = []

    @disk_cash.cache
    def work(a, b):
        runs.append(1)
        a *= 2
        return float(a.sum() + b.sum())

    for _ in range(3):
        work(np.arange(4.0), np.ones(2))
    assert len(runs) == 3
