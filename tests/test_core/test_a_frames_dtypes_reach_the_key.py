"""A pandas argument's dtypes are part of its cache key.

Found while attacking the decorator before round 26: the frame hash folded in
column and index LABELS plus ``hash_pandas_object``, which is value-only. So a
tz-naive and a tz-aware series shared one entry -- the tz-aware call was served
the naive one's ``TypeError: Cannot convert tz-naive timestamps`` -- and so did
``int64``/``Int64`` (pd.NA semantics), ``int64``/``int32``, ``bool``/``int64``,
and a categorical against an object column.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from cash import Cash
from cash.backends import InMemoryBackend


@pytest.fixture
def seen():
    cash = Cash(backend=InMemoryBackend(), register_magic=False)
    calls: list[str] = []

    @cash.cache
    def describe(frame):
        calls.append("ran")
        return str(frame.dtypes if hasattr(frame, "dtypes") else frame.dtype)

    return describe, calls


def _pair(seen, left, right):
    describe, calls = seen
    first, second = describe(left), describe(right)
    return first, second, len(calls)


@pytest.mark.parametrize(
    "left, right",
    [
        (
            pd.Series(pd.to_datetime(["2020-06-01 12:00"])),
            pd.Series(pd.to_datetime(["2020-06-01 12:00"]).tz_localize("UTC")),
        ),
        (pd.Series([1, 2], dtype="int64"), pd.Series([1, 2], dtype="Int64")),
        (pd.Series([1, 2], dtype="int64"), pd.Series([1, 2], dtype="int32")),
        (pd.Series([1, 0], dtype="bool"), pd.Series([1, 0], dtype="int64")),
        (pd.Series(["a", "b"], dtype="object"), pd.Series(["a", "b"], dtype="category")),
        (pd.DataFrame({"a": [1, 2]}, dtype="int64"), pd.DataFrame({"a": [1, 2]}, dtype="int32")),
    ],
)
def test_two_dtypes_are_two_entries(seen, left, right):
    first, second, ran = _pair(seen, left, right)
    assert first != second, f"both calls answered {first!r}"
    assert ran == 2


def test_an_index_dtype_is_part_of_the_key(seen):
    describe, calls = seen
    describe(pd.DataFrame({"a": [1, 2]}, index=pd.Index([0, 1], dtype="int64")))
    describe(pd.DataFrame({"a": [1, 2]}, index=pd.Index([0, 1], dtype="int32")))
    assert len(calls) == 2


def test_the_same_frame_still_hits(seen):
    describe, calls = seen
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    describe(frame)
    describe(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
    assert len(calls) == 1, "an unchanged frame must still hit"
