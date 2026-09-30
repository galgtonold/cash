"""A frame whose memory something outside pandas can write is hashed afresh.

The copy-on-write memo trusts that a frame only changes by getting new block
arrays. These writes reach a frame's data with every array identity kept, and
the memo served the content hash from before the write:

* the caller's array a Series was built over with ``copy=False``, written
  and then dropped before the next call (the check ran only at lookup, when
  nothing referenced it any more);
* the array a frame's index was built over: ``pd.DataFrame(..., index=arr)``
  and ``df.index = arr`` do not copy it, and the index was never checked;
* the array under a date column (``pd.Series(dates, copy=False)``): only the
  date wrapper was checked, not the array inside it;
* the ndarray an Arrow-backed float column was built over zero-copy;
* ``.array`` of an index that shares a column's memory
  (``pd.Index(df["a"])``, ``df.set_index("a").index``),
  ``pd.array(s, copy=False)`` and a date index's ``asi8``: writable handles
  that were not recorded.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from cash import Cash
from cash.backends import InMemoryBackend
from cash.decorator import arg_hashing

pytestmark = pytest.mark.skipif(
    not arg_hashing.is_cow_pandas(pd.Series([1.0])), reason="the frame memo runs under copy-on-write only"
)


@pytest.fixture
def seen():
    cash = Cash(backend=InMemoryBackend(), register_magic=False)

    @cash.cache
    def seen(value):
        return value.to_csv()

    return seen


def _twice(fn, value):
    fn(value)
    fn(value)


def test_a_series_over_a_dropped_array_written_before_the_call(seen):
    arr = np.array([1.0, 2.0, 3.0])
    series = pd.Series(arr, copy=False)
    _twice(seen, series)
    arr[0] = 100.0
    del arr
    assert seen(series) == seen.__wrapped__(series)
    assert "100.0" in seen(series)


def test_an_index_over_the_callers_array(seen):
    arr = np.array([1.0, 2.0, 3.0])
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=arr)
    _twice(seen, frame)
    arr[0] = 100.0
    assert seen(frame) == seen.__wrapped__(frame)
    assert "100.0" in seen(frame)


def test_an_index_assigned_from_the_callers_array(seen):
    arr = np.array([1.0, 2.0, 3.0])
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    frame.index = arr
    _twice(seen, frame)
    arr[0] = 100.0
    del arr
    assert seen(frame) == seen.__wrapped__(frame)


def test_a_date_column_over_the_callers_array(seen):
    arr = np.array(["2020-01-01", "2020-01-02"], dtype="M8[ns]")
    series = pd.Series(arr, copy=False)
    _twice(seen, series)
    arr[0] = np.datetime64("1999-12-31", "ns")
    assert seen(series) == seen.__wrapped__(series)
    assert "1999" in seen(series)


def test_an_arrow_column_over_the_callers_array(seen):
    pa = pytest.importorskip("pyarrow")
    arr = np.array([1.0, 2.0, 3.0])
    series = pd.Series(pd.arrays.ArrowExtensionArray(pa.array(arr)))
    _twice(seen, series)
    arr[0] = 100.0
    assert seen(series) == seen.__wrapped__(series)


HANDLES = {
    "pd.Index(df['a']).array": lambda d: pd.Index(d["a"]).array.__setitem__(0, 100.0),
    "df.set_index('a').index.array": lambda d: d.set_index("a").index.array.__setitem__(0, 100.0),
    "pd.array(df['a'], copy=False)": lambda d: pd.array(d["a"], copy=False).__setitem__(0, 100.0),
    "pd.DatetimeIndex(df['t']).asi8": lambda d: pd.DatetimeIndex(d["t"]).asi8.__setitem__(0, 0),
}


@pytest.mark.parametrize("write", list(HANDLES.values()), ids=list(HANDLES))
def test_a_write_through_a_handle_that_shares_a_column(write, seen):
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "t": pd.date_range("2020-01-01", periods=3)})
    _twice(seen, frame)
    before = seen.__wrapped__(frame)
    write(frame)
    assert seen.__wrapped__(frame) != before  # the write did land in the frame
    assert seen(frame) == seen.__wrapped__(frame)


def test_an_index_over_the_callers_array_after_a_label_lookup(seen):
    """``.loc`` gives the index a lookup table holding its array, which the
    memo's copy of the frame shares. Counted once per copy, it stood in for
    the caller's reference and the write was missed."""
    arr = np.array([1.0, 2.0, 3.0])
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=arr)
    frame.loc[2.0]
    _twice(seen, frame)
    arr[0] = 100.0
    assert seen(frame) == seen.__wrapped__(frame)
