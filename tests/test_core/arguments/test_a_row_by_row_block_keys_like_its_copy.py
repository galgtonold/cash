"""A frame whose block is laid out row by row keys as its C-ordered copy.

A ``pivot`` and arithmetic on its result leave a block that holds each
column with a stride of the whole row. Hashing reads such a block a run
of columns at a time through one reused buffer (`_column_buffers`): every
column must still key by its own values, in its own place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cash.content_hashers import _SLAB_BYTES, _column_buffers, hash_pandas


def _panel(days: int, names: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    long = pd.DataFrame(
        {
            "date": np.repeat(pd.date_range("2013-01-01", periods=days), names),
            "ticker": np.tile([f"T{i}" for i in range(names)], days),
            "close": rng.random(days * names),
        }
    )
    return long.pivot(index="date", columns="ticker", values="close")


def _row_by_row(frame: pd.DataFrame) -> bool:
    values = frame._mgr.blocks[0].values
    return values.flags.f_contiguous and not values.flags.c_contiguous


@pytest.mark.parametrize(
    "build",
    [
        lambda p: p,
        lambda p: p.shift(3) / p.shift(9) - 1,
        lambda p: p > 0.5,
        lambda p: (p * 100).astype("int64"),
    ],
    ids=["pivot", "arithmetic", "mask", "ints"],
)
def test_a_row_by_row_block_keys_as_its_c_ordered_copy(build):
    # More rows of the block than one slab holds, so the buffer is filled several times.
    days = _SLAB_BYTES // (8 * 40) + 77
    frame = build(_panel(days, 300))
    assert _row_by_row(frame)
    # Built column by column: pandas stacks them into a block laid out column by column.
    copy = pd.DataFrame({name: frame[name].to_numpy() for name in frame.columns}, index=frame.index)
    copy.columns = frame.columns
    assert not _row_by_row(copy)
    assert hash_pandas(frame) == hash_pandas(copy)


def test_each_column_is_served_its_own_values():
    frame = _panel(_SLAB_BYTES // (8 * 50) + 5, 120)
    assert _row_by_row(frame)
    buffer = _column_buffers(frame)
    dtype = frame.dtypes.iloc[0]
    for pos in [0, 1, 119, 60, 3, 118, 60]:  # out of order: slabs refilled and reused
        np.testing.assert_array_equal(buffer(pos, dtype), frame.iloc[:, pos].to_numpy())


def test_a_change_to_one_value_changes_the_key():
    frame = _panel(500, 300)
    before = hash_pandas(frame)
    changed = _panel(500, 300)
    changed.iloc[499, 299] += 1.0  # the last value of the last slab
    assert _row_by_row(changed)
    assert hash_pandas(changed) != before
    swapped = frame[list(frame.columns[1:]) + [frame.columns[0]]]
    swapped.columns = frame.columns
    assert hash_pandas(swapped) != before
