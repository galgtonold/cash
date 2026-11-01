"""A frame whose object column holds tuples of immutable scalars (the
``(action, time)`` records an ``explode`` of sessions leaves) keeps its
digest until the frame is written.

The content-hash memo let through a block of immutable scalars or of lists
of them, not one of tuples, so such a frame was read in full by every
check: ``df.session.argmax()`` on two million records read them twice, 9 s
for a statement plain Python runs in a millisecond (student_8). Nothing in
such a block can change but a write to it, which copy-on-write turns into
a new block. A tuple holding anything that can change, or a pandas date,
which can take an attribute, is read every time, as before.
"""

from __future__ import annotations

import datetime

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from cash.content_hashers import builtin_hash
from cash.decorator import arg_hashing
from cash.decorator.arg_hashing import cow_frame_digest

pytestmark = pytest.mark.skipif(
    not arg_hashing.is_cow_pandas(pd.Series([1.0])), reason="the frame memo runs under copy-on-write only"
)

N = 20_000


def _records(make=None) -> pd.DataFrame:
    t0 = datetime.datetime(2020, 1, 1)
    make = make or (lambda i: (f"a{i % 9}", t0 + datetime.timedelta(seconds=i)))
    return pd.DataFrame({"user": np.arange(N) % 31, "actions": [make(i) for i in range(N)]})


def _reads(monkeypatch) -> list:
    reads: list = []
    real = arg_hashing.FrameDigests.hash_frame

    def counting(self, obj):
        reads.append(obj.shape)
        return real(self, obj)

    monkeypatch.setattr(arg_hashing.FrameDigests, "hash_frame", counting)
    return reads


def test_an_unchanged_frame_of_records_is_read_once(monkeypatch):
    frame = _records()
    reads = _reads(monkeypatch)
    first = cow_frame_digest(frame)
    assert all(cow_frame_digest(frame) == first for _ in range(3))
    assert len(reads) == 1
    assert first == builtin_hash(frame)


@pytest.mark.parametrize(
    "write",
    [
        lambda f: f.iat.__setitem__((3, 1), ("changed", 1)),
        lambda f: f.iat.__setitem__((5, 1), ("a0", datetime.datetime(1999, 1, 1))),
        lambda f: f.__setitem__("actions", f["actions"].map(lambda r: (r[0].upper(), r[1]))),
        lambda f: f.sort_values("user", inplace=True),
    ],
)
def test_a_write_to_the_frame_is_read_again(write):
    frame = _records()
    before = cow_frame_digest(frame)
    write(frame)
    assert cow_frame_digest(frame) == builtin_hash(frame)
    assert cow_frame_digest(frame) != before


def test_a_record_holding_a_list_is_read_every_time():
    frame = _records(lambda i: (f"a{i % 9}", [i]))
    before = cow_frame_digest(frame)
    frame["actions"].iloc[7][1].append(99)
    assert cow_frame_digest(frame) == builtin_hash(frame)
    assert cow_frame_digest(frame) != before


def test_a_record_holding_a_pandas_date_is_read_every_time():
    frame = _records(lambda i: (f"a{i % 9}", pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=i)))
    before = cow_frame_digest(frame)
    frame["actions"].iloc[7][1].tag = "set later"
    assert cow_frame_digest(frame) == builtin_hash(frame)
    if builtin_hash(frame) != before:
        assert cow_frame_digest(frame) != before


def test_the_records_are_read_once_when_the_frame_is_kept(monkeypatch):
    """The memo's shallow copy holds a view of the frame's block: the array
    that owns the memory is read once for both."""
    frame = _records()
    reads: list = []
    real = arg_hashing._plain_scalar_types

    def counting():
        reads.append(1)
        return real()

    monkeypatch.setattr(arg_hashing, "_plain_scalar_types", counting)
    cow_frame_digest(frame)
    assert len(reads) == 1
    reads.clear()
    cow_frame_digest(frame)
    assert reads == []
