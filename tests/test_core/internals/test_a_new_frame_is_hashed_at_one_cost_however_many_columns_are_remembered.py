"""Hashing a new DataFrame argument costs the same however many columns of
other frames cash remembers.

The frame-digest memo keeps a table ``column memory -> frames that hold
it``, and clears the ids of dropped frames from it once it is big. That
walk reads the whole table, and ran on every store while the frames that
are alive together held more than a fixed number of columns: one
20,000-column matrix alive made every call on a new one-row frame 50 times
slower, and 600 frames of 100 columns passed one after the other slowed
down as they went. The walk now runs once per doubling of the table.

A work count: the entries of the table that the walks read, not a timer.
"""

from __future__ import annotations

import pytest

from cash.decorator import arg_hashing
from cash.decorator.arg_hashing import FrameDigests

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

pytestmark = pytest.mark.core


class _CountedTable(dict):
    """The memo's column table, counting the entries a whole walk reads."""

    walked = 0

    def items(self):
        self.walked += len(self)
        return super().items()


@pytest.fixture
def memo(monkeypatch):
    if not pd.options.mode.copy_on_write and int(pd.__version__.split(".")[0]) < 3:
        pytest.skip("the frame memo needs copy-on-write")
    memo = FrameDigests(256)
    memo._columns = _CountedTable()
    monkeypatch.setattr(arg_hashing, "FRAME_DIGESTS", memo)
    return memo


def _hash_new_frames(memo, n):
    frames = [pd.DataFrame({"a": [float(i)], "b": [1.0]}) for i in range(n)]
    for frame in frames:
        assert memo.hash_frame(frame) is not None
    return frames


def test_a_wide_frame_alive_does_not_make_each_new_frame_walk_its_columns(memo):
    wide = pd.DataFrame(np.zeros((2, 20_000)))  # samples x genes
    assert memo.hash_frame(wide) is not None
    table = memo._columns
    assert len(table) >= 20_000  # the positive control: its columns are remembered
    before = table.walked
    kept = _hash_new_frames(memo, 100)
    # One walk at most while 200 columns join 20,000; it was one per frame,
    # 2,000,000 entries.
    assert table.walked - before <= 2 * len(table)
    assert memo.lookup(wide) is not None and memo.lookup(kept[0]) is not None


def test_many_frames_one_after_the_other_walk_the_table_a_few_times(memo):
    frames = [pd.DataFrame(np.full((2, 100), float(i))) for i in range(600)]
    table = memo._columns
    for frame in frames:
        assert memo.hash_frame(frame) is not None
    # 60,000 columns went in. Walks at a doubling read a few times that in
    # total; one walk per frame read 9,000,000 entries.
    assert 0 < table.walked <= 6 * 60_000
    # The walks did drop the ids of the frames the memo no longer holds.
    assert len(table) <= 2 * 64 * 256
