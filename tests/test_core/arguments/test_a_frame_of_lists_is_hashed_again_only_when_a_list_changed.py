"""A frame of lists (``groupby(...).agg(list)``) keeps its digest while each
list holds the same objects.

The content-hash memo trusted copy-on-write, which cannot see a list in a
cell change, so a block of lists was read in full every time: a notebook
loop over such a frame read 4.6 s of timestamps five times per cell. The
memo now keeps, for each list of immutable scalars, the objects it held
when the frame was hashed, and checks those instead of reading it: any
append, item set or delete is a re-read. A list holding anything that can
change is read every time, as before.
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


def _lists(n: int = 50_000) -> "pd.DataFrame":
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2020-01-01", periods=n, freq="min"),
            "v": np.arange(n) % 97,
            "day": [("a", "b", "c")[i % 3] for i in range(n)],
        }
    )
    return frame.groupby("day").agg(list)


def _reads(monkeypatch) -> list:
    reads: list = []
    real = arg_hashing.FrameDigests.hash_frame

    def counting(self, obj):
        reads.append(obj.shape)
        return real(self, obj)

    monkeypatch.setattr(arg_hashing.FrameDigests, "hash_frame", counting)
    return reads


def test_an_unchanged_frame_of_lists_is_read_once(monkeypatch):
    frame = _lists()
    reads = _reads(monkeypatch)
    first = cow_frame_digest(frame)
    assert all(cow_frame_digest(frame) == first for _ in range(3))
    assert len(reads) == 1
    assert first == builtin_hash(frame)


@pytest.mark.parametrize(
    "change",
    [
        lambda f: f.iloc[0, 1].append(5),
        lambda f: f.iloc[1, 0].pop(),
        lambda f: f.iloc[2, 1].__setitem__(0, 1000),
        lambda f: f.iloc[0, 0].__setitem__(3, pd.Timestamp("1999-01-01")),
        lambda f: f.iloc[0, 1].sort(reverse=True),
        lambda f: f.iloc[0, 1].__setitem__(0, f.iloc[0, 1][0] + 0.0),  # equal, but another object
    ],
)
def test_a_list_changed_in_place_is_read_again(change):
    frame = _lists()
    before = cow_frame_digest(frame)
    change(frame)
    assert cow_frame_digest(frame) == builtin_hash(frame)
    if builtin_hash(frame) != before:
        assert cow_frame_digest(frame) != before


def test_a_list_written_through_the_frame_is_read_again():
    frame = _lists()
    before = cow_frame_digest(frame)
    frame.iat[0, 1] = [1, 2, 3]
    assert cow_frame_digest(frame) == builtin_hash(frame) != before


def test_lists_of_lists_are_read_every_time(monkeypatch):
    frame = pd.DataFrame({"rows": [[[1], [2]] * 300_000, [[3]] * 300_000]})
    reads = _reads(monkeypatch)
    cow_frame_digest(frame)
    frame.iloc[0, 0][0].append(9)
    assert cow_frame_digest(frame) == builtin_hash(frame)
    assert len(reads) == 2
