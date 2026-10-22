"""Tables with dates are shared on pandas 3.1 too, and a failed freeze changes nothing.

pandas 3.1 removed ``freq`` from DatetimeArray and TimedeltaArray (GH 24566):
an index keeps it alone. Freezing a date column read ``values.freq``, raised
AttributeError, and the start-up check of table sharing failed, so on pandas
3.1 every table was copied when stored and again on each hit. The block
reuse of a table stored again lost its date columns the same way.

The failed freeze had made the columns before the date one read-only, and
added no pin: pandas would then write such a block in place, and the write
would raise. A freeze that fails part way now puts back what it changed.
"""

from __future__ import annotations

import gc
import pickle

import numpy as np
import pandas as pd
import pytest

from cash.backends import frame_sharing, shared_blocks
from cash.backends.memory_backend import InMemoryBackend

pytestmark = pytest.mark.skipif(
    int(pd.__version__.split(".", 1)[0]) < 3, reason="needs pandas copy-on-write, pandas 3's only mode"
)


@pytest.fixture
def decided_again(monkeypatch):
    """Table sharing decided again, in this process as it is."""
    monkeypatch.setattr(frame_sharing, "_ENABLED", [])
    monkeypatch.setattr(frame_sharing, "_CHECKED_OK", [])


def _dated(n: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": np.arange(n, dtype=float),
            "when": pd.date_range("2020-01-01", periods=n, freq="h"),
            "took": pd.timedelta_range("1s", periods=n, freq="s"),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


@pytest.mark.usefixtures("decided_again")
def test_sharing_is_on():
    assert frame_sharing.enabled()
    assert frame_sharing._checked_works()


def test_a_table_with_dates_is_frozen_and_keeps_its_freq():
    stored = _dated()
    assert frame_sharing._freeze(stored)
    assert all(not a.flags.writeable for a in frame_sharing.block_arrays(stored))
    hit = frame_sharing.hand_out(stored)
    assert hit.index.freq == pd.offsets.Day()
    assert pickle.loads(pickle.dumps(hit)).index.freq == pd.offsets.Day()
    series = pd.Series(np.arange(4.0), index=pd.timedelta_range("1D", periods=4, freq="h"))
    assert frame_sharing._freeze(series)
    assert frame_sharing.hand_out(series).index.freq == pd.offsets.Hour()


@pytest.mark.usefixtures("decided_again")
def test_a_hit_of_a_table_with_dates_is_shared_and_can_be_changed():
    backend = InMemoryBackend()
    df = _dated()
    backend.set("k", {"variables": {"df": df}}, {"execution_time": 1.0})
    del df
    gc.collect()
    first = backend.get("k")[1]["variables"]["df"]
    second = backend.get("k")[1]["variables"]["df"]
    assert np.shares_memory(first["x"].to_numpy(), second["x"].to_numpy())
    first.iloc[0, 0] = -1.0
    first.iloc[0, 1] = pd.Timestamp("1999-01-01")
    pd.testing.assert_frame_equal(backend.get("k")[1]["variables"]["df"], _dated())
    assert first.index.freq == pd.offsets.Day()


def test_a_date_column_has_a_layout_for_block_reuse():
    values = _dated()._mgr.blocks[1].values
    assert type(values) is pd.arrays.DatetimeArray
    assert shared_blocks._layout(values) is not None


def test_a_freeze_that_fails_part_way_leaves_the_table_as_it_was(monkeypatch):
    def broken(index):
        raise AttributeError("an internal pandas changed")

    monkeypatch.setattr(frame_sharing, "_frozen_index", broken)
    df = _dated()
    assert not frame_sharing._freeze(df)
    assert all(a.flags.writeable for a in frame_sharing.block_arrays(df))
    assert not any(ref is frame_sharing._PIN for b in df._mgr.blocks for ref in b.refs.referenced_blocks)
    df["x"].array[0] = -1.0  # a handle writes it, as on any table
    df.iloc[1, 0] = -2.0
    assert df["x"].iloc[0] == -1.0 and df["x"].iloc[1] == -2.0


def test_with_sharing_off_a_hit_is_a_writable_copy_never_stale(monkeypatch):
    monkeypatch.setattr(frame_sharing, "_ENABLED", [False])
    monkeypatch.setattr(frame_sharing, "_CHECKED_OK", [False])
    backend = InMemoryBackend()
    df = _dated()
    backend.set("k", df, {"execution_time": 1.0})
    df["x"].array[0] = 123.0  # past pandas, into the caller's memory
    hit = backend.get("k")[1]
    pd.testing.assert_frame_equal(hit, _dated())
    hit["x"].array[1] = -1.0
    hit.iloc[2, 0] = -2.0
    pd.testing.assert_frame_equal(backend.get("k")[1], _dated())
