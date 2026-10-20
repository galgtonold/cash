"""A result the caller writes right after the call is stored, and served as
it was returned.

The RAM tier kept a big polars result as a clone, and a big pandas table
uncopied, each with the digest of its data, and dropped the entry when a hit
found the digest changed. ``a = np.arange(n); s = load(a); a[0] = 1`` wrote
the series' memory, and the function ran on every call, with only a debug
log. Now a result is copied at its store unless nothing outside cash can
write its memory: pandas under copy-on-write keeps the table uncopied, and a
write past pandas, which still drops the entry, says so once
(STORE-RESULT-WRITTEN). A numpy array is copied unless no one can ever write
it (`memory_backend._unwritable`). A result with two parts over the same
memory (an array and a view of it, or a frame made over it), which a hit
would hand back apart, is not stored (STORE-PARTS-SHARED).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from cash.backends import frame_sharing, memory_backend
from cash.backends.memory_backend import InMemoryBackend
from cash.decorator.result_parts import split_by_a_hit

#: Rows enough for 16 MiB of numbers: a table kept uncopied.
ROWS = frame_sharing.CHECKED_MIN_BYTES // 16 * 11 // 10


def _table() -> pd.DataFrame:
    raw = pd.DataFrame({"fare": np.linspace(-1.0, 100.0, ROWS), "n": np.arange(ROWS)})
    return raw[raw["fare"] > 0]


def _counted(cash, build):
    runs = []

    def load(*args):
        runs.append(1)
        return build(*args)

    return cash.cache(load), runs


def _codes(caught) -> list[str]:
    return [getattr(w.message, "code", None) for w in caught]


def test_a_table_the_caller_writes_through_pandas_is_served_as_returned(cash_instance):
    load, runs = _counted(cash_instance, _table)
    r = load()
    r["fare"] = 0.0  # right after the call
    r.iloc[0, 1] = -5
    served = load()
    assert len(runs) == 1, "stored, and run 2 is a hit"
    pd.testing.assert_frame_equal(served, _table())


def test_a_polars_series_over_an_array_the_caller_writes_is_served_as_returned(cash_instance):
    pl = pytest.importorskip("polars")
    load, runs = _counted(cash_instance, lambda a: pl.Series("s", a))  # over the caller's memory
    a = np.arange(float(ROWS))
    first = load(a)
    a[0] = 1.0  # right after the call: reaches `first`, not the entry
    assert first[0] == 1.0
    served = load(np.arange(float(ROWS)))
    assert len(runs) == 1, "stored, and run 2 is a hit"
    assert served[0] == 0.0


@pytest.mark.skipif(not frame_sharing._checked_works(), reason="needs pandas copy-on-write")
def test_a_table_written_past_pandas_after_its_store_is_dropped_and_said_once(cash_instance):
    load, runs = _counted(cash_instance, _table)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            r = load()
            r["fare"].array[0] = 123.0  # past pandas: the block itself
    assert len(runs) == 3, "never served what the caller wrote"
    assert _codes(caught).count("STORE-RESULT-WRITTEN") == 1


def test_an_array_and_a_polars_series_over_it_are_not_stored(cash_instance):
    pl = pytest.importorskip("polars")

    def build():
        a = np.arange(10.0)
        return a, pl.Series("s", a)

    load, runs = _counted(cash_instance, build)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            a, s = load()
            a[0] = -1.0
            assert s[0] == -1.0  # the sharing holds on every run
    assert len(runs) == 3
    assert _codes(caught).count("STORE-PARTS-SHARED") == 1
    message = next(str(w.message) for w in caught if getattr(w.message, "code", None) == "STORE-PARTS-SHARED")
    assert "result[1]" in message and "result[0]" in message


def test_parts_that_share_memory_are_found_where_a_result_keeps_them():
    pl = pytest.importorskip("polars")

    class Holder:
        def __init__(self, a):
            self.raw = a
            self.frame = pl.DataFrame({"x": a, "y": np.zeros(len(a))})

    a = np.arange(10.0)
    assert split_by_a_hit({"raw": a, "s": pl.Series(a)}) == ("result['raw']", "result['s']")
    assert split_by_a_hit([a[2:], pl.Series(a)]) == ("result[0]", "result[1]")
    assert split_by_a_hit(Holder(a)) == ("result.raw", "result.frame")


def test_parts_that_share_nothing_are_stored():
    pl = pytest.importorskip("polars")
    a = np.arange(10.0)
    assert split_by_a_hit((a, pl.Series(a.copy()))) is None
    assert split_by_a_hit((a, pl.Series(a * 2))) is None
    assert split_by_a_hit((a, a)) is None  # one object: a copy keeps it one
    grid = np.arange(20.0).reshape(4, 5)
    assert split_by_a_hit((grid[:, :-1], grid[:, -1])) is None  # one base, no element in common
    assert split_by_a_hit((grid[::2], grid[1::2])) is None
    assert split_by_a_hit((pl.Series(a), pl.Series(a))) is None  # two frames: polars writes neither
    assert split_by_a_hit(a) is None


def test_two_arrays_whose_data_overlap_are_found():
    a = np.arange(20.0).reshape(4, 5)
    assert split_by_a_hit((a, a[1:3])) == ("result[0]", "result[1]")
    assert split_by_a_hit({"t": a.T, "x": a[0]}) == ("result['t']", "result['x']")
    assert split_by_a_hit([a[1:3], a[2:4]]) == ("result[0]", "result[1]")


def test_an_array_and_a_view_of_it_are_not_stored(cash_instance):
    def build():
        a = np.arange(10.0)
        return a, a[2:5]

    load, runs = _counted(cash_instance, build)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            a, v = load()
            a[2] = -1.0
            assert v[0] == -1.0
    assert len(runs) == 2
    assert _codes(caught).count("STORE-PARTS-SHARED") == 1


def test_a_pandas_frame_over_an_array_beside_it_is_found():
    a = np.arange(10.0)
    series = pd.Series(a, copy=False)
    if not np.shares_memory(series.to_numpy(), a):
        pytest.skip("this pandas copies the array")
    assert split_by_a_hit((a, series)) == ("result[0]", "result[1]")


def test_a_read_only_array_that_owns_its_data_is_copied():
    a = np.arange(10.0)
    a.flags.writeable = False
    copied = memory_backend._copy_array(a)
    assert copied is not a and not copied.flags.writeable
    a.setflags(write=True)  # an owner can be made writeable again
    a[0] = -1.0
    assert copied[0] == 0.0


def test_a_read_only_view_over_a_writeable_base_is_copied():
    base = np.arange(10.0)
    view = base[2:5]
    view.flags.writeable = False
    copied = memory_backend._copy_array(view)
    assert copied is not view
    base[2] = -1.0  # reaches the view
    assert view[0] == -1.0 and copied[0] == 2.0


def test_a_read_only_view_of_a_read_only_owner_is_copied():
    owner = np.arange(10.0)
    owner.flags.writeable = False
    view = owner[2:5]
    assert memory_backend._copy_array(view) is not view


@pytest.mark.parametrize(
    "make",
    [
        lambda: np.frombuffer(np.arange(10.0).tobytes()),
        lambda: np.frombuffer(np.arange(10.0).tobytes())[2:5],
        lambda: np.frombuffer(memoryview(np.arange(10.0).tobytes())),
    ],
    ids=["over bytes", "a view of one", "over a memoryview of bytes"],
)
def test_an_array_over_bytes_is_kept_as_it_is(make):
    array = make()
    with pytest.raises(ValueError):
        array.setflags(write=True)
    assert memory_backend._copy_array(array) is array
    backend = InMemoryBackend()
    backend.set("k", array, {"execution_time": 1.0, "copy_required": True})
    _meta, hit = backend.get("k")
    np.testing.assert_array_equal(hit, array)


def test_an_array_over_a_bytearray_is_copied():
    array = np.frombuffer(bytearray(np.arange(10.0).tobytes()))
    array.flags.writeable = False
    assert memory_backend._copy_array(array) is not array
