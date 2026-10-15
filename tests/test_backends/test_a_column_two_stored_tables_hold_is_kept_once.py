"""A column two stored tables hold is kept once in the RAM tier.

``h = df.assign(p=...)`` in a loop: every result holds ``df``'s columns and
adds one. The tier copied each result whole, so a grid of fifteen kept
fifteen copies of the same columns. A block whose content the frame memo
vouches for (the digest pieces of a hashed table that is unchanged since)
is now copied once and put into later stored tables as it is. These tests
pin that it is, and that nothing written since the hash, or held by the
caller, is ever taken for the stored copy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cash.backends import frame_sharing
from cash.backends.memory_backend import InMemoryBackend
from cash.decorator.arg_hashing import cow_frame_digest

pytestmark = pytest.mark.skipif(not frame_sharing.enabled(), reason="needs pandas copy-on-write")


def _frame(n: int = 5000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": np.arange(n, dtype=float),
            "when": pd.date_range("2024-01-01", periods=n, freq="min"),
            "team": pd.array([f"t{i % 13}" for i in range(n)], dtype=pd.StringDtype("python")),
        }
    )


def _stored(backend: InMemoryBackend, key: str) -> pd.DataFrame:
    return backend._store[key][1]


def _column_memory(frame: pd.DataFrame, name: str):
    values = frame._mgr.blocks[frame._mgr.blknos[frame.columns.get_loc(name)]].values
    return getattr(values, "_ndarray", values)


def _shares(a: pd.DataFrame, b: pd.DataFrame, name: str) -> bool:
    return np.shares_memory(_column_memory(a, name), _column_memory(b, name))


def test_a_derived_tables_shared_columns_are_not_copied_again():
    df = _frame()
    assert cow_frame_digest(df) is not None  # hashed, as a call's argument is
    backend = InMemoryBackend()
    backend.set("a", df.assign(p=1.0), {"execution_time": 1.0})
    from cash.backends import shared_blocks

    before = shared_blocks.reused_count()
    h = df.assign(p=2.0)
    backend.set("b", h, {"execution_time": 1.0})
    reused, nbytes = (now - then for now, then in zip(shared_blocks.reused_count(), before))
    assert reused == 3  # x, when, team: only p is copied
    assert nbytes >= df["x"].to_numpy().nbytes
    first, second = _stored(backend, "a"), _stored(backend, "b")
    for name in ("x", "when", "team"):
        assert _shares(first, second, name)
        assert not _shares(second, df, name)  # never the caller's memory
    assert not _shares(first, second, "p")
    pd.testing.assert_frame_equal(backend.get("b")[1], h)


def test_a_hit_written_in_place_changes_no_other_entry():
    df = _frame()
    cow_frame_digest(df)
    backend = InMemoryBackend()
    backend.set("a", df.assign(p=1.0), {"execution_time": 1.0})
    backend.set("b", df.assign(p=2.0), {"execution_time": 1.0})
    hit = backend.get("b")[1]
    hit.loc[0, "team"] = "changed"
    hit.loc[1, "x"] = -1.0
    hit["team"].array[2] = "also changed"
    assert backend.get("a")[1]["team"].iloc[0] == "t0"
    assert backend.get("a")[1]["team"].iloc[2] == "t2"
    assert backend.get("b")[1]["x"].iloc[1] == 1.0
    pd.testing.assert_frame_equal(backend.get("a")[1], df.assign(p=1.0))


def test_a_column_changed_since_its_hash_is_copied():
    df = _frame()
    cow_frame_digest(df)
    backend = InMemoryBackend()
    backend.set("a", df.assign(p=1.0), {"execution_time": 1.0})
    df.loc[0, "x"] = 99.0  # copy-on-write gives df a new block for x
    h = df.assign(p=2.0)
    backend.set("b", h, {"execution_time": 1.0})
    assert _stored(backend, "b")["x"].iloc[0] == 99.0
    assert _stored(backend, "a")["x"].iloc[0] == 0.0
    assert not _shares(_stored(backend, "a"), _stored(backend, "b"), "x")


def test_a_table_over_the_callers_array_is_copied():
    """The memo vouches for no table built over an array the caller still
    holds: it can write it past pandas. Such a table's columns are copied."""
    raw = np.arange(5000, dtype=float)
    df = pd.DataFrame({"x": raw}, copy=False)
    cow_frame_digest(df)
    backend = InMemoryBackend()
    backend.set("a", df.assign(p=1.0), {"execution_time": 1.0})
    raw[0] = 42.0
    from cash.backends import shared_blocks

    before = shared_blocks.reused_count()
    backend.set("b", df.assign(p=2.0), {"execution_time": 1.0})
    assert shared_blocks.reused_count()[0] == before[0]
    assert _stored(backend, "b")["x"].iloc[0] == 42.0
    assert _stored(backend, "a")["x"].iloc[0] == 0.0


def test_an_unhashed_table_is_copied_whole():
    """Positive control for the reuse: without a digest nothing is shared."""
    backend = InMemoryBackend()
    df = _frame()
    backend.set("a", df.assign(p=1.0), {"execution_time": 1.0})
    backend.set("b", df.assign(p=2.0), {"execution_time": 1.0})
    assert not _shares(_stored(backend, "a"), _stored(backend, "b"), "x")
