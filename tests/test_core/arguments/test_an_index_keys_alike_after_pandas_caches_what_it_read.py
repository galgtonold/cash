"""An index keys alike after pandas caches what it read of it.

A pandas index handed to a cached function on its own was keyed by its
pickle, which holds what pandas has cached about the labels: reading
``index.unit``, or converting the index to another time zone, fills a
cache on the array under it. The same dates then keyed apart before and
after such a read, and a function that only read its index (pvlib's
``spa_python`` converting its times to UTC) was taken for one that changed
it in place: its result was never stored, and it ran on every call.

An index is keyed by its labels and what its schema names -- its class,
names, dtype and ``freq`` -- as a table's row labels are.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _times():
    return pd.date_range("2024-01-01", periods=500, freq="h", tz="Etc/GMT+7")


def test_reading_an_index_does_not_change_its_key(cash_instance):
    calls = []

    @cash_instance.cache
    def first(times):
        calls.append(1)
        return len(times)

    times = _times()
    first(times)
    times.tz_convert("UTC")
    times.unit
    first(times)
    first(_times())
    assert len(calls) == 1


def test_a_function_that_reads_its_index_is_stored(cash_instance):
    calls = []

    @cash_instance.cache
    def to_utc(times):
        calls.append(1)  # @cash:assume-safe
        return np.asarray(times.tz_convert("UTC").asi8 // 10**9)

    times = _times()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        to_utc(times)
        to_utc(times)
    assert not [w for w in caught if "argument mutation" in str(w.message)]
    assert len(calls) == 1


@pytest.mark.parametrize(
    "other",
    [
        lambda: _times().tz_convert("UTC"),
        lambda: pd.date_range("2024-01-01", periods=500, freq="h", tz="Etc/GMT+7", name="t"),
        lambda: pd.DatetimeIndex(list(_times()), freq=None),
        lambda: _times().shift(1),
        lambda: pd.Index(np.arange(500)),
        lambda: pd.RangeIndex(500),
    ],
    ids=["zone", "name", "freq", "labels", "ints", "range"],
)
def test_what_an_index_holds_keys_it_apart(cash_instance, other):
    calls = []

    @cash_instance.cache
    def describe(index):
        calls.append(1)
        return repr(index)

    assert describe(_times()) == repr(_times())
    assert describe(other()) == repr(other())
    assert len(calls) == 2


def test_an_int_index_and_a_range_key_apart(cash_instance):
    @cash_instance.cache
    def kind(index):
        return type(index).__name__

    assert kind(pd.Index(np.arange(5))) == "Index"
    assert kind(pd.RangeIndex(5)) == "RangeIndex"


def test_an_index_of_objects_keys_by_what_they_hold(cash_instance):
    @cash_instance.cache
    def total(index):
        return sum(len(cell) for cell in index)

    cells = [[1], [2, 3]]
    index = pd.Index(cells, dtype=object)
    assert total(index) == 3
    cells[1].append(4)
    assert total(index) == 4
