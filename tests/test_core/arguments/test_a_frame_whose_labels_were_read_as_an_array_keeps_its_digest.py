"""A frame keeps its digest after its labels were read as an array.

``Index.array`` keeps what it hands out in the index's cache: the labels'
own array, or (pandas 3.1) a read-only view of it. ``Index.to_numpy()``
goes through it for text labels, so any read of a ``str`` column index
held as Python strings (pandas 3 without pyarrow) left one reference
the memo counted as a caller's handle. The frame was then read in full
by every call: a 900k-row frame of team names passed to a 15-call grid
read about 3.5 GB where 0.3 GB do.

The cache's reference is pandas' own. A caller still holding the array
is one more reference, and every hand-out is recorded
(`watch_array_handles`), so a write through it is still read again.
Strings use pandas' Python storage explicitly, so these run with pyarrow
installed too.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from cash.content_hashers import builtin_hash
from cash.decorator import arg_hashing
from cash.decorator.arg_hashing import cow_frame_digest, frame_memoable

pytestmark = pytest.mark.skipif(
    not arg_hashing.is_cow_pandas(pd.Series([1.0])), reason="the frame memo runs under copy-on-write only"
)

N = 5_000
#: pandas 3's ``str`` dtype where pyarrow is not installed.
TEXT = pd.StringDtype("python", na_value=np.nan)


def _frame(index=None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "home": pd.array([f"t{i % 40}" for i in range(N)], dtype=TEXT),
            "away": pd.array([f"t{(i + 7) % 40}" for i in range(N)], dtype=TEXT),
            "goals": np.arange(N, dtype=float),
        },
        index=index,
    )
    frame.columns = pd.Index(list(frame.columns), dtype=TEXT)
    return frame


def _reads(monkeypatch) -> list:
    reads: list = []
    real = arg_hashing.FrameDigests.hash_frame

    def counting(self, obj):
        reads.append(obj.shape)
        return real(self, obj)

    monkeypatch.setattr(arg_hashing.FrameDigests, "hash_frame", counting)
    return reads


def _unlocked(array):
    """The array a write lands in: pandas 3.1 hands an index's out read-only."""
    if getattr(array, "_readonly", False):
        array._readonly = False
    return array


def _read_labels(frame) -> None:
    """What reading a frame's column names does in pandas: ``to_numpy``
    of text labels goes through ``Index.array``, which the cache keeps."""
    frame.columns.to_numpy(dtype=object)
    assert "array" in frame.columns._cache


def test_a_text_frame_whose_labels_were_read_is_read_once(monkeypatch):
    frame = _frame()
    _read_labels(frame)
    assert frame_memoable(frame)
    reads = _reads(monkeypatch)
    first = cow_frame_digest(frame)
    assert all(cow_frame_digest(frame) == first for _ in range(3))
    assert len(reads) == 1
    assert first == builtin_hash(frame.copy(deep=True))


def test_a_numeric_index_read_as_an_array_is_read_once(monkeypatch):
    frame = _frame(index=pd.Index(np.arange(N) * 3))
    frame.index.array  # cached by pandas, dropped here
    assert "array" in frame.index._cache
    assert frame_memoable(frame)
    reads = _reads(monkeypatch)
    first = cow_frame_digest(frame)
    assert cow_frame_digest(frame) == first
    assert len(reads) == 1


def test_a_view_of_an_index_shares_its_cache_and_is_counted_once():
    """Indexes made by ``_view`` share one cache: its reference is one."""
    frame = _frame()
    _read_labels(frame)
    other = frame.copy(deep=False)
    other.columns = frame.columns._view()
    assert other.columns._cache is frame.columns._cache
    assert frame_memoable(frame) and frame_memoable(other)
    held = frame.columns.array
    assert not frame_memoable(frame)
    assert not frame_memoable(other)
    del held


HANDLES = {
    "the column labels' array": lambda f: f.columns.array,
    "a text column's array": lambda f: f["home"].array,
    "a text column's values": lambda f: f["home"].values,
}


@pytest.mark.parametrize("handle", list(HANDLES.values()), ids=list(HANDLES))
def test_a_handle_still_held_is_not_vouched_for(handle):
    frame = _frame()
    _read_labels(frame)
    held = handle(frame)
    assert not frame_memoable(frame)
    del held
    assert frame_memoable(frame)


def _write_labels(frame):
    _unlocked(frame.columns.array)[0] = "renamed"


def _write_a_kept_label_handle(frame):
    held = _unlocked(frame.columns.array)
    cow_frame_digest(frame)
    held[0] = "renamed"


def _write_index_labels(frame):
    _unlocked(frame.index.array)[0] = -1


def _write_column(frame):
    frame["home"].array[0] = "ZZ"


def _write_kept_column_handle(frame):
    held = frame["home"].values
    cow_frame_digest(frame)
    held[1] = "QQ"


WRITES = {
    "labels through their array": _write_labels,
    "labels through a kept array": _write_a_kept_label_handle,
    "index labels through their array": _write_index_labels,
    "a text column through its array": _write_column,
    "a text column through kept values": _write_kept_column_handle,
}


@pytest.mark.parametrize("write", list(WRITES.values()), ids=list(WRITES))
def test_a_write_through_a_handle_is_read_again(monkeypatch, write):
    frame = _frame(index=pd.Index(np.arange(N) * 3))
    _read_labels(frame)
    frame.index.array  # both caches filled before the digest
    reads = _reads(monkeypatch)
    before = cow_frame_digest(frame)
    assert cow_frame_digest(frame) == before
    assert len(reads) == 1  # memoised: the write below is what it must see
    write(frame)
    after = cow_frame_digest(frame)
    assert after != before
    assert after == builtin_hash(frame.copy(deep=True))
