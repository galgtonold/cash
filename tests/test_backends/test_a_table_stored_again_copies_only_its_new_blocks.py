"""A table stored again after a column was added copies only that column.

A cleaning cell rebinds one table statement by statement, and a statement
that adds a column (``df["hour"] = ...``) stores the same frame again: every
other block is the array the RAM tier copied a statement before. The tier
copied all of it again, 440 MB a statement for a 2.9-million-row taxi
table. It now takes the block it copied before when the caller's array
holds the same bytes as that copy, and copies it when anything wrote the
array since, through pandas or past it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cash.backends import frame_sharing, shared_blocks
from cash.backends.memory_backend import InMemoryBackend

pytestmark = pytest.mark.skipif(not frame_sharing.enabled(), reason="needs pandas copy-on-write")


def _frame(n: int = 50_000) -> pd.DataFrame:
    # Built from a dict of new arrays: pandas copies them in, so the frame's
    # blocks are its own, as a filtered table's are.
    return pd.DataFrame(
        {
            "x": np.arange(n, dtype=float),
            "y": np.arange(n, dtype=float) * 2,
            "k": np.arange(n, dtype=np.int64),
            "when": pd.date_range("2024-01-01", periods=n, freq="min"),
        }
    )


def _memory(frame: pd.DataFrame, name: str):
    values = frame._mgr.blocks[frame._mgr.blknos[frame.columns.get_loc(name)]].values
    return getattr(values, "_ndarray", values)


def _shares(a: pd.DataFrame, b: pd.DataFrame, name: str) -> bool:
    return np.shares_memory(_memory(a, name), _memory(b, name))


def _stored(backend: InMemoryBackend, key: str) -> pd.DataFrame:
    return backend._store[key][1]


def _copied_bytes(monkeypatch) -> list[int]:
    """Bytes of the blocks the RAM tier copies from here on."""
    copied = [0]
    real = frame_sharing._copy_values

    def counting(values, arrow=True):
        copied[0] += int(values.nbytes)
        return real(values, arrow)

    monkeypatch.setattr(frame_sharing, "_copy_values", counting)
    return copied


def test_a_frame_stored_again_with_a_new_column_copies_only_it(monkeypatch):
    df = _frame()
    backend = InMemoryBackend()
    copied = _copied_bytes(monkeypatch)
    backend.set("a", df, {"execution_time": 1.0})
    first_store = copied[0]
    assert first_store >= df.memory_usage(index=False).sum()

    df["hour"] = df["when"].dt.hour
    before = shared_blocks.reused_count()[0]
    backend.set("b", df, {"execution_time": 1.0})

    # Only the new column is copied (and the column labels): 4 bytes a
    # row, where the first store copied 32.
    assert copied[0] - first_store <= df["hour"].to_numpy().nbytes + 4096
    assert shared_blocks.reused_count()[0] - before >= 3
    first, second = _stored(backend, "a"), _stored(backend, "b")
    for name in ("x", "y", "k", "when"):
        assert _shares(first, second, name)
        assert not _shares(second, df, name)  # never the caller's memory
    pd.testing.assert_frame_equal(backend.get("b")[1], df)


def test_a_block_written_past_pandas_since_it_was_stored_is_copied():
    df = _frame()
    original = df.copy()
    backend = InMemoryBackend()
    backend.set("a", df, {"execution_time": 1.0})
    # ``.array`` writes the block itself: copy-on-write does not see it.
    df["x"].array[0] = -1.0
    df["hour"] = df["when"].dt.hour
    backend.set("b", df, {"execution_time": 1.0})

    pd.testing.assert_frame_equal(backend.get("a")[1], original)
    assert backend.get("b")[1]["x"].iloc[0] == -1.0
    assert not _shares(_stored(backend, "a"), _stored(backend, "b"), "x")
    # Positive control: the blocks nobody wrote are still taken.
    assert _shares(_stored(backend, "a"), _stored(backend, "b"), "k")


def test_a_block_written_through_pandas_since_it_was_stored_is_copied():
    df = _frame()
    original = df.copy()
    backend = InMemoryBackend()
    backend.set("a", df, {"execution_time": 1.0})
    df.loc[3, "y"] = 99.0
    backend.set("b", df, {"execution_time": 1.0})

    pd.testing.assert_frame_equal(backend.get("a")[1], original)
    assert backend.get("b")[1]["y"].iloc[3] == 99.0
    assert not _shares(_stored(backend, "a"), _stored(backend, "b"), "y")
    assert _shares(_stored(backend, "a"), _stored(backend, "b"), "k")  # positive control


def test_writing_the_callers_frame_after_the_second_store_changes_neither_entry():
    df = _frame()
    original = df.copy()
    backend = InMemoryBackend()
    backend.set("a", df, {"execution_time": 1.0})
    df["hour"] = df["when"].dt.hour
    backend.set("b", df, {"execution_time": 1.0})
    stored_b = df.copy()
    assert _shares(_stored(backend, "a"), _stored(backend, "b"), "x")  # the entries share x

    df["x"].array[1] = -5.0
    df.loc[2, "k"] = -7
    df["when"].array[0] = pd.Timestamp("1999-01-01")

    pd.testing.assert_frame_equal(backend.get("a")[1], original)
    pd.testing.assert_frame_equal(backend.get("b")[1], stored_b)


def test_a_hit_written_in_place_changes_no_entry():
    df = _frame()
    original = df.copy()
    backend = InMemoryBackend()
    backend.set("a", df, {"execution_time": 1.0})
    df["hour"] = df["when"].dt.hour
    backend.set("b", df, {"execution_time": 1.0})
    assert _shares(_stored(backend, "a"), _stored(backend, "b"), "y")  # the entries share y
    hit = backend.get("b")[1]
    hit.loc[0, "x"] = 123.0
    with pytest.raises(ValueError):  # frozen: shared data refuses a write past pandas
        hit["y"].array[0] = 1.0

    pd.testing.assert_frame_equal(backend.get("a")[1], original)
    assert backend.get("b")[1]["x"].iloc[0] == 0.0


def test_a_stored_copy_that_is_gone_is_not_taken():
    df = _frame()
    backend = InMemoryBackend()
    backend.set("a", df, {"execution_time": 1.0})
    backend.delete("a")
    df["hour"] = df["when"].dt.hour
    before = shared_blocks.reused_count()[0]
    backend.set("b", df, {"execution_time": 1.0})
    assert shared_blocks.reused_count()[0] == before
    pd.testing.assert_frame_equal(backend.get("b")[1], df)
    # Positive control: the copy that lives is taken.
    df["minute"] = df["when"].dt.minute
    backend.set("c", df, {"execution_time": 1.0})
    assert _shares(_stored(backend, "b"), _stored(backend, "c"), "x")
