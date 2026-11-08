"""A write into a table the RAM tier serves finds the arrays that kept the
old values.

A served table shares its memory with the tier's copy, so pandas copies a
column before writing it, and an array taken from the column before the
write (``v = y.to_numpy()``) keeps the old values, where a plain run writes
the column in place and the array sees it. Each write is counted as pandas
makes it; while reports are deferred (as in a notebook cell),
`views_left_behind` reports the writes that left such an array, and the
memory it still reads, and only when something outside pandas reads it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cash.backends import frame_sharing
from cash.backends.memory_backend import InMemoryBackend

pytestmark = pytest.mark.skipif(not frame_sharing.enabled(), reason="needs pandas copy-on-write")


@pytest.fixture(autouse=True)
def deferred():
    """Reports kept for `views_left_behind`, as in a notebook cell; outside
    one a write warns as it happens (test_a_write_into_a_served_table_with_an_array_taken_warns)."""
    frame_sharing.defer_reports(True)
    frame_sharing.forget_asked()
    yield
    frame_sharing.forget_asked()
    frame_sharing.defer_reports(False)


def _left() -> list:
    """The memory the writes since the last call left an array reading."""
    return frame_sharing.views_left_behind()[1]


def _clean() -> bool:
    """No write since the last call left an array behind."""
    writes, _ = frame_sharing.views_left_behind()
    return writes == 0


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
    roots = _left()
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
    assert frame_sharing.reads_memory_of(v, _left())


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
    assert _clean()


def test_a_series_taken_from_the_table_is_not_an_array_left_behind(backend):
    """Control: a Series is a block pandas accounts for, not a reader."""
    df = _served(backend)
    y = df["t"]
    frame_sharing.forget_asked()
    y.iloc[0] = 100.0
    assert _clean()


def test_reading_the_table_is_no_write(backend):
    df = _served(backend)
    v = df["t"].to_numpy()
    frame_sharing.forget_asked()
    assert df["t"].sum() == 3.0
    assert _clean()
    del v


def test_each_write_is_reported_once(backend):
    df = _served(backend)
    v = df["t"].to_numpy()
    frame_sharing.forget_asked()
    df.iloc[0, 0] = 100.0
    assert _left()
    assert _clean()
    del v


# Writes pandas makes in place through a block method (`Block._maybe_copy`,
# `Block._get_refs_and_copy`): the locals pandas keeps while writing are
# the write's own, not readers, and a shallow copy pandas makes when there
# is nothing to write is no write.
INPLACE = {
    "fillna": lambda t: t.fillna(0.0, inplace=True),
    "replace": lambda t: t.replace(1.0, 9.0, inplace=True),
    "where": lambda t: t.where(t > 0, 0.0, inplace=True),
    "mask": lambda t: t.mask(t > 1, 0.0, inplace=True),
    "clip": lambda t: t.clip(lower=0.5, inplace=True),
    "interpolate": lambda t: t.interpolate(inplace=True),
    "ffill": lambda t: t.ffill(inplace=True),
}


@pytest.fixture
def numbers() -> InMemoryBackend:
    stored = InMemoryBackend()
    table = pd.DataFrame({"t": [0.0, np.nan, 2.0], "u": [1.0, 1.0, np.nan]})
    stored.set("k", table, {"execution_time": 1.0})
    stored.set("s", table["t"].copy(), {"execution_time": 1.0})
    return stored


@pytest.mark.parametrize("method", list(INPLACE))
@pytest.mark.parametrize("shape", ["frame", "series"])
def test_a_write_in_place_with_no_array_held_reports_nothing(numbers, method, shape):
    table = numbers.get("k" if shape == "frame" else "s")[1]
    assert frame_sharing.has_pin(table._mgr.blocks[0]), "setup: the table is served"
    frame_sharing.forget_asked()
    INPLACE[method](table)
    assert _clean()


@pytest.mark.parametrize("method", ["fillna", "replace", "where"])
def test_a_write_in_place_with_an_array_held_is_found(numbers, method):
    df = _served(numbers)
    v = df["t"].to_numpy()
    frame_sharing.forget_asked()
    INPLACE[method](df)
    assert frame_sharing.reads_memory_of(v, _left())


def test_a_replace_in_place_of_an_arrow_column_reports_nothing():
    pytest.importorskip("pyarrow")
    stored = InMemoryBackend()
    table = pd.DataFrame({"s": pd.array(list("xyz"), dtype="string[pyarrow]"), "v": [0.0, 1.0, 2.0]})
    stored.set("k", table, {"execution_time": 1.0})
    df = _served(stored)
    frame_sharing.forget_asked()
    df.replace("x", "q", inplace=True)
    assert df["s"].iloc[0] == "q"
    assert _clean()


def test_an_array_gone_by_the_end_of_the_statement_is_still_counted(backend):
    """A function that takes an array and writes the table returns the old
    value where a plain run returns the new: counted at the write, though
    the array is gone when the statement ends."""

    def f(d):
        v = d["t"].to_numpy()
        d.iloc[0, 0] = 100.0
        return v[0]

    assert f(_served(backend)) == 0.0
    assert frame_sharing.views_left_behind()[0] == 1


@pytest.mark.xfail(
    reason="known limitation: an array keeps no trace of which hit it came from, so it counts against a write into another hit of the same entry",
    strict=True,
)
def test_an_array_of_another_hit_of_the_same_entry_is_not_a_reader(backend):
    """Two hits share the tier's memory, and numpy keeps no trace of which
    hit an array was taken from: writing one hit while an array of the
    other is held is reported, though in a plain run the two tables are
    apart and the array does not change either way."""
    a = _served(backend)
    b = _served(backend)
    vb = b["t"].to_numpy()
    frame_sharing.forget_asked()
    a.iloc[0, 0] = 100.0
    assert vb[0] == 0.0
    assert _clean()


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


def test_a_column_two_entries_keep_is_no_reader_of_either():
    """Two entries of one table share the columns the second did not
    change: each entry's copy is cash's own, not a reader of the other."""
    stored = InMemoryBackend()
    table = pd.DataFrame({"t": np.arange(4.0), "u": np.arange(4.0)})
    stored.set("a", table, {"execution_time": 1.0})
    table["w"] = 1
    stored.set("b", table, {"execution_time": 1.0})
    df = stored.get("b")[1]
    frame_sharing.forget_asked()
    df.iloc[0, 0] = 100.0
    assert _clean()
    assert stored.get("a")[1]["t"].iloc[0] == 0.0
