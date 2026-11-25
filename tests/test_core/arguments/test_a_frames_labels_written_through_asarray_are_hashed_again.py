"""A frame whose labels were written through ``np.asarray`` is read again.

Under pandas 3.1, ``np.asarray(df.index)`` of a numeric index is not a
copy and not a read-only view: it is the very array the index keeps,
writable as it stands. ``np.asarray(df.index)[0] = 99`` changes the
frame's labels with no flag to flip, and the frame memo, which recorded
only the read-only views pandas hands out, went on answering with the
digest of the old labels: a cached function of the frame served the result
computed for other labels.

A writable array one of these accessors returns is recorded like any
other handle. A copy among them is recorded for nothing: no frame keeps
its memory.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from cash.content_hashers import builtin_hash
from cash.decorator import arg_hashing
from cash.decorator.arg_hashing import cow_frame_digest

pytestmark = pytest.mark.skipif(
    not arg_hashing.is_cow_pandas(pd.Series([1.0])), reason="the frame memo runs under copy-on-write only"
)


def _frame(labels) -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}, index=pd.Index(labels))


def _write_and_drop(axis) -> bool:
    """Write the first label through ``np.asarray``; the handle is gone when
    this returns. False when this pandas hands out a copy or refuses."""
    handle = np.asarray(axis)
    before = axis[0]
    try:
        handle[0] = 99
    except ValueError:
        return False
    return axis[0] != before


@pytest.mark.parametrize("labels", [[10, 11, 12], [1.5, 2.5, 3.5]], ids=["integers", "floats"])
def test_the_digest_follows_a_label_written_through_asarray(labels):
    frame = _frame(labels)
    before = cow_frame_digest(frame)
    assert cow_frame_digest(frame) == before
    if not _write_and_drop(frame.index):
        pytest.skip("this pandas does not hand out the index's own array")
    assert cow_frame_digest(frame) == builtin_hash(frame) != before


def test_the_digest_follows_a_column_label_written_through_asarray():
    frame = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=pd.Index([7, 8]))
    before = cow_frame_digest(frame)
    if not _write_and_drop(frame.columns):
        pytest.skip("this pandas does not hand out the index's own array")
    assert cow_frame_digest(frame) == builtin_hash(frame) != before


def test_a_handle_still_held_counts_too():
    frame = _frame([10, 11, 12])
    before = cow_frame_digest(frame)
    handle = np.asarray(frame.index)
    try:
        handle[0] = 99
    except ValueError:
        pytest.skip("this pandas hands out a read-only array")
    if frame.index[0] != 99:
        pytest.skip("this pandas hands out a copy")
    assert cow_frame_digest(frame) == builtin_hash(frame) != before


def test_a_cached_function_sees_the_new_labels(disk_cash):
    @disk_cash.cache
    def label_total(df):
        return int(df.index.to_series().sum())

    frame = _frame([10, 11, 12])
    assert label_total(frame) == 33
    assert label_total(frame) == 33
    if not _write_and_drop(frame.index):
        pytest.skip("this pandas does not hand out the index's own array")
    assert label_total(frame) == 99 + 11 + 12


def test_a_frame_whose_labels_were_only_read_is_hashed_once_more_not_for_good(monkeypatch):
    """The read is recorded and costs one fresh hash; then the memo holds."""
    reads = []
    real = arg_hashing.FrameDigests.hash_frame

    def counting(self, obj):
        reads.append(1)
        return real(self, obj)

    monkeypatch.setattr(arg_hashing.FrameDigests, "hash_frame", counting)
    frame = _frame([10, 11, 12])
    first = cow_frame_digest(frame)
    np.asarray(frame.index).sum()
    assert cow_frame_digest(frame) == first
    assert cow_frame_digest(frame) == first
    assert cow_frame_digest(frame) == first
    assert len(reads) <= 2
