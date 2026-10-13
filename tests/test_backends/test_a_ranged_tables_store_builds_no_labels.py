"""Storing a table with a plain row range, and every hit of it, leaves the
range as a range.

A ``RangeIndex`` keeps no labels; reading its ``_data`` builds them, an
array of every row number, and keeps it. The RAM tier read it on every
store and hit while checking the axes for labels a shallow copy shares:
for a 21M-row ``read_parquet`` that was an extra 168 MB array, built in
0.8 s, on the caller's table too (it shares the range's cache).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cash.backends import frame_sharing
from cash.backends.memory_backend import InMemoryBackend

pytestmark = pytest.mark.skipif(not frame_sharing.enabled(), reason="needs pandas copy-on-write")


def _built(index: pd.Index) -> bool:
    return "_data" in index._cache


@pytest.mark.parametrize(
    "make",
    [
        lambda: pd.DataFrame({"x": np.arange(1000.0), "k": np.arange(1000) % 7}),
        lambda: pd.Series(np.arange(1000.0), name="x"),
    ],
    ids=["frame", "series"],
)
def test_a_store_and_its_hits_keep_the_range_unbuilt(make):
    value = make()
    backend = InMemoryBackend()
    backend.set("k", value, {"execution_time": 1.0})
    first, second = backend.get("k")[1], backend.get("k")[1]

    for table in (value, first, second):
        assert isinstance(table.index, pd.RangeIndex)
        assert not _built(table.index)
    pd.testing.assert_index_equal(first.index, value.index)
    pd.testing.assert_index_equal(second.index, value.index)
    # A hit is still its own table: a write to one reaches no other holder.
    first.iloc[0] = -1.0
    assert _first_cell(value) == 0.0
    assert _first_cell(second) == 0.0


def _first_cell(table) -> float:
    return table.iloc[0] if table.ndim == 1 else table.iloc[0, 0]
