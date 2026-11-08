"""A write into a table the RAM tier serves finds the arrays that kept the
old values.

A served table shares its memory with the tier's copy, so pandas copies a
column before writing it, and an array taken from the column before the
write (``v = y.to_numpy()``) keeps the old values, where a plain run writes
the column in place and the array sees it. `views_left_behind` reports the
memory such an array still reads, and only when something outside pandas'
blocks reads it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cash.backends import frame_sharing
from cash.backends.memory_backend import InMemoryBackend

pytestmark = pytest.mark.skipif(not frame_sharing.enabled(), reason="needs pandas copy-on-write")


def _served(backend: InMemoryBackend) -> pd.DataFrame:
    return backend.get("k")[1]


@pytest.fixture
def backend() -> InMemoryBackend:
    stored = InMemoryBackend()
    table = pd.DataFrame({"t": [0.0, 1.0, 2.0], "d": pd.date_range("2020-01-01", periods=3)})
    stored.set("k", table, {"execution_time": 1.0})
    return stored


TAKES = {
    "to_numpy": lambda df: df["t"].to_numpy(),
    "values": lambda df: df["t"].values,
    "array": lambda df: df["t"].array,
    "dates": lambda df: df["d"].to_numpy(),
}


@pytest.mark.parametrize("take", list(TAKES))
def test_an_array_taken_before_the_write_is_found(backend, take):
    df = _served(backend)
    kept = TAKES[take](df)
    frame_sharing.forget_asked()
    if take == "dates":
        df.iloc[0, 1] = pd.Timestamp("2001-01-01")
    else:
        df.iloc[0, 0] = 100.0
    roots = frame_sharing.views_left_behind()
    assert roots
    assert frame_sharing.reads_memory_of(kept, roots)


def test_the_reported_sequence_is_found(backend):
    df = _served(backend)
    y = df["t"]
    df = df.copy()
    v = y.to_numpy()
    frame_sharing.forget_asked()
    y.iloc[0] = 100
    assert v[0] == 0.0, "setup: the served column was copied before the write"
    assert frame_sharing.reads_memory_of(v, frame_sharing.views_left_behind())


@pytest.mark.parametrize(
    "write",
    [
        lambda df: df.iloc.__setitem__((0, 0), 100.0),
        lambda df: df.loc.__setitem__((0, "t"), 100.0),
    ],
    ids=["iloc", "loc"],
)
def test_a_write_with_nothing_else_reading_the_memory_reports_nothing(backend, write):
    df = _served(backend)
    frame_sharing.forget_asked()
    write(df)
    assert df["t"].iloc[0] == 100.0
    assert frame_sharing.views_left_behind() == []


def test_a_series_taken_from_the_table_is_not_an_array_left_behind(backend):
    """Control: a Series is a block pandas accounts for, not a reader."""
    df = _served(backend)
    y = df["t"]
    frame_sharing.forget_asked()
    y.iloc[0] = 100.0
    assert frame_sharing.views_left_behind() == []


def test_reading_the_table_is_no_write(backend):
    df = _served(backend)
    v = df["t"].to_numpy()
    frame_sharing.forget_asked()
    assert df["t"].sum() == 3.0
    assert frame_sharing.views_left_behind() == []
    del v


def test_each_write_is_reported_once(backend):
    df = _served(backend)
    v = df["t"].to_numpy()
    frame_sharing.forget_asked()
    df.iloc[0, 0] = 100.0
    assert frame_sharing.views_left_behind()
    assert frame_sharing.views_left_behind() == []
    del v


def test_a_write_past_pandas_says_why_it_is_refused(backend):
    df = _served(backend)
    with pytest.raises(ValueError, match="read-only") as raised:
        df["t"].array[0] = 5.0
    note = frame_sharing.read_only_note(raised.value)
    assert note is not None and "served from the cache" in note


def test_another_read_only_error_gets_no_note():
    array = np.zeros(3)
    array.flags.writeable = False
    with pytest.raises(ValueError, match="read-only") as raised:
        array[0] = 1.0
    assert frame_sharing.read_only_note(raised.value) is None
