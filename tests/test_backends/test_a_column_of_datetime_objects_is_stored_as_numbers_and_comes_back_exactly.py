"""An object column of dates and times is stored as columns of numbers and
comes back exactly as pickle would bring it back.

Pickled item by item, a million ``datetime`` objects took 1.8 s to write
(a 15 s stall on Windows for one user's logs). `datetime_columns` stores the
fields pickle keeps of each item, a column per kind; these pin that every
item comes back with its type, value, zone, fold and unit, that ``None``
and ``NaT`` stay where they were, and that anything the columns could not
give back exactly (two zones, a subclass, a ``pytz`` zone) is pickled as
before.
"""

from __future__ import annotations

import datetime as dt
import pickle
import pickletools
import re
import zoneinfo

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from cash.backends.datetime_columns import DATETIME_OBJECTS_MIN_ITEMS, reduce_datetime_objects
from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.serialization import PickleSerializer
from cash.backends.tiered_backend import TieredBackend

N = DATETIME_OBJECTS_MIN_ITEMS + 37
BERLIN = zoneinfo.ZoneInfo("Europe/Berlin")
PLUS_5 = dt.timezone(dt.timedelta(hours=5), "PLUS5")


class MyDatetime(dt.datetime):
    """A datetime subclass: pickled as before."""


class Fixed(dt.tzinfo):
    """A zone of the user's own: pickled as before."""

    def utcoffset(self, when):
        return dt.timedelta(hours=1)

    def dst(self, when):
        return dt.timedelta(0)


def _objects(items):
    out = np.empty(len(items), dtype=object)
    out[:] = items
    return out


def _naive(i):
    return dt.datetime(1, 1, 1) + dt.timedelta(days=i * 3653, seconds=i * 7, microseconds=i * 13)


def _naive_column():
    items = [_naive(i % 1000) for i in range(N)]
    items[3] = dt.datetime(9999, 12, 31, 23, 59, 59, 999999)
    items[4] = dt.datetime(2024, 10, 27, 2, 30, fold=1)
    items[5] = dt.datetime(1969, 12, 31, 23, 59, 59, 1)
    items[6] = dt.datetime(2000, 2, 29, 12)
    return _objects(items)


def _aware_column(tz):
    start = dt.datetime(2024, 10, 26, 22, 0)
    items = [(start + dt.timedelta(minutes=17 * i)).replace(tzinfo=tz) for i in range(N)]
    items[1] = dt.datetime(2024, 10, 27, 2, 30, fold=1, tzinfo=tz)  # the hour Berlin goes through twice
    items[2] = dt.datetime(2024, 10, 27, 2, 30, fold=0, tzinfo=tz)
    items[3] = dt.datetime(2024, 3, 31, 2, 30, tzinfo=tz)  # a time Berlin skips
    return _objects(items)


def _timestamps(tz=None, unit="ns"):
    index = pd.date_range("2024-10-26 22:00", periods=N, freq="17min", tz=tz).as_unit(unit)
    return _objects(list(index.astype(object)))


def _dates():
    return _objects([dt.date(1, 1, 1) + dt.timedelta(days=(i % 1000) * 3650) for i in range(N)])


def _mixed():
    items = []
    for i in range(N):
        items.append(
            [
                _naive(i % 1000),
                pd.Timestamp("2024-01-01 00:00:00.000000001") + pd.Timedelta(seconds=i),
                dt.date(2024, 1, 1) + dt.timedelta(days=i % 300),
                None,
                pd.NaT,
            ][i % 5]
        )
    return _objects(items)


def _missing():
    items = [None if i % 3 else pd.NaT for i in range(N)]
    items[7] = dt.datetime(2024, 1, 1)
    return _objects(items)


TAKEN = {
    "naive datetimes": _naive_column,
    "datetimes in a ZoneInfo zone": lambda: _aware_column(BERLIN),
    "datetimes in a fixed offset": lambda: _aware_column(PLUS_5),
    "datetimes in UTC": lambda: _aware_column(dt.timezone.utc),
    "naive Timestamps": _timestamps,
    "Timestamps in seconds": lambda: _timestamps(unit="s"),
    "Timestamps in a ZoneInfo zone": lambda: _timestamps(BERLIN, "us"),
    "Timestamps in UTC": lambda: _timestamps(dt.timezone.utc),
    "dates": _dates,
    "mixed kinds": _mixed,
    "None and NaT around a datetime": _missing,
}


def _two_zones():
    items = list(_aware_column(BERLIN))
    items[9] = items[9].astimezone(PLUS_5)
    return _objects(items)


def _subclass():
    items = list(_naive_column())
    items[9] = MyDatetime(2024, 1, 1)
    return _objects(items)


def _two_units():
    items = list(_timestamps(unit="ns"))
    items[9] = pd.Timestamp("2024-01-01").as_unit("s")
    return _objects(items)


def _nan_for_missing():
    items = list(_naive_column())
    items[9] = float("nan")
    return _objects(items)


def _a_string():
    items = list(_naive_column())
    items[-1] = "2024-01-01"
    return _objects(items)


LEFT = {
    "two zones": _two_zones,
    "a datetime subclass": _subclass,
    "a zone of the user's own": lambda: _aware_column(Fixed()),
    "two Timestamp units": _two_units,
    "a NaN for missing": _nan_for_missing,
    "a string among them": _a_string,
    "only None": lambda: _objects([None] * N),
    "too short": lambda: _naive_column()[: DATETIME_OBJECTS_MIN_ITEMS - 1],
}


def _same_items(back, expected):
    """*back* holds what *expected* (pickle's own round trip) holds, item by item."""
    assert type(back) is np.ndarray and back.dtype == object
    assert back.shape == expected.shape
    assert back.flags.c_contiguous == expected.flags.c_contiguous
    assert back.flags.f_contiguous == expected.flags.f_contiguous
    for got, want in zip(back.ravel(order="K"), expected.ravel(order="K"), strict=True):
        assert type(got) is type(want)
        if want is None or want is pd.NaT:
            assert got is want
            continue
        assert _repr(got) == _repr(want)
        assert got == want or (want != want and got != got)  # NaN
        if isinstance(want, dt.datetime):
            assert got.fold == want.fold
            assert type(got.tzinfo) is type(want.tzinfo)
            assert _repr(got.tzinfo) == _repr(want.tzinfo)
            assert got.utcoffset() == want.utcoffset()
        if isinstance(want, pd.Timestamp):
            assert got.unit == want.unit
            assert got.asm8 == want.asm8
    # The items share their zone objects as pickle's own copy shares them.
    zones = [
        {id(getattr(x, "tzinfo", None)) for x in a.ravel() if isinstance(x, dt.datetime)} for a in (back, expected)
    ]
    assert len(zones[0]) == len(zones[1])


def _repr(item):
    """*item*'s repr without the address of a zone of the user's own."""
    return re.sub(r" at 0x[0-9a-fA-F]+", "", repr(item))


def _round_trip(value):
    serializer = PickleSerializer()
    stream, buffers = serializer.serialize_split(value)
    return serializer.deserialize_split(stream, buffers)


@pytest.mark.parametrize("name", TAKEN)
def test_a_column_the_columns_take_comes_back_exactly(name):
    column = TAKEN[name]()
    assert reduce_datetime_objects(column) is not None
    _same_items(_round_trip(column), pickle.loads(pickle.dumps(column, protocol=5)))


@pytest.mark.parametrize("name", LEFT)
def test_a_column_the_columns_cannot_give_back_is_pickled_as_before(name):
    column = LEFT[name]()
    assert reduce_datetime_objects(column) is None
    _same_items(_round_trip(column), pickle.loads(pickle.dumps(column, protocol=5)))


def test_a_pytz_zone_is_pickled_as_before():
    pytz = pytest.importorskip("pytz")
    zone = pytz.timezone("Europe/Berlin")
    column = _objects([zone.localize(dt.datetime(2024, 1, 1) + dt.timedelta(days=i)) for i in range(N)])
    assert reduce_datetime_objects(column) is None
    _same_items(_round_trip(column), pickle.loads(pickle.dumps(column, protocol=5)))


def test_the_column_is_not_pickled_item_by_item():
    """What the change is for: before it, the stream held a reduce per item."""
    frame = pd.DataFrame({"t": pd.Series(list(_naive_column()), dtype=object)})
    stream = PickleSerializer().serialize(frame)
    reduces = sum(1 for op, _arg, _pos in pickletools.genops(stream) if op.name == "REDUCE")
    assert reduces < 50


def test_a_frame_with_other_columns_comes_back_as_it_was_stored():
    frame = pd.DataFrame(
        {
            "when": pd.Series(list(_aware_column(BERLIN)), dtype=object),
            "day": pd.Series(list(_dates()), dtype=object),
            "mixed": pd.Series(list(_mixed()), dtype=object),
            "n": np.arange(N),
            "x": np.linspace(0, 1, N),
            "s": pd.Series([f"r{i}" for i in range(N)], dtype=object),
            "t64": pd.date_range("2024", periods=N, freq="s"),
        },
        index=pd.Index(list(_naive_column()), dtype=object, name="logged"),
    )
    expected = pickle.loads(pickle.dumps(frame, protocol=5))
    back = _round_trip(frame)
    pd.testing.assert_frame_equal(back, expected)
    assert list(back.dtypes) == list(frame.dtypes)
    for column in ("when", "day", "mixed", "s"):
        _same_items(back[column].to_numpy(), expected[column].to_numpy())
    _same_items(back.index.to_numpy(), expected.index.to_numpy())
    back_plain = PickleSerializer().deserialize(PickleSerializer().serialize(frame))
    pd.testing.assert_frame_equal(back_plain, expected)


def test_a_fortran_ordered_block_comes_back_in_its_order():
    block = np.empty((N, 2), dtype=object, order="F")
    block[:, 0] = _naive_column()
    block[:, 1] = _dates()
    assert reduce_datetime_objects(block) is not None
    _same_items(_round_trip(block), pickle.loads(pickle.dumps(block, protocol=5)))
    _same_items(_round_trip(np.ascontiguousarray(block)), np.ascontiguousarray(block))
    _same_items(_round_trip(block[::2]), np.ascontiguousarray(block[::2]))


def test_a_restored_column_digests_as_the_stored_one():
    """The notebook digests a value over its store pickle: an unchanged
    column read back from disk must digest alike, or every statement
    reading it misses after a restart."""
    from cash.stored_pickle import object_digest

    column = pd.Series(list(_mixed()), dtype=object, name="t")
    serializer = PickleSerializer()
    assert object_digest(serializer.deserialize(serializer.serialize(column))) == object_digest(column)
    changed = column.copy()
    changed[0] = changed[0] + dt.timedelta(microseconds=1)
    assert object_digest(changed) != object_digest(column)


def test_a_disk_hit_through_the_tiers_gives_the_column_back(tmp_path):
    """The RAM tier's pickle, written to disk as it is, and read back by
    a reader whose RAM tier is empty."""
    frame = pd.DataFrame({"t": pd.Series(list(_aware_column(BERLIN)), dtype=object), "v": np.arange(N)})
    writer = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path), flush_interval=0)])
    writer.set("k", frame, {"execution_time": 5.0, "decorator_entry": True})
    writer.shutdown()
    reader = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path), flush_interval=0)])
    meta, back = reader.get("k")
    assert meta["source"] == "DISK"
    expected = pickle.loads(pickle.dumps(frame, protocol=5))
    pd.testing.assert_frame_equal(back, expected)
    _same_items(back["t"].to_numpy(), expected["t"].to_numpy())
