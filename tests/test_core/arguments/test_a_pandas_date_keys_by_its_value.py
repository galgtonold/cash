"""An exact pandas ``Timestamp``, ``Timedelta`` or ``NaT`` keys by its value.

Pickled, each timestamp went through two reducers and pickle's own: 2.4 s a
key for a column of seven lists of 140,000 timestamps, which a loop over the
frame keyed eight times in one cell. Each now keys as the number it holds,
its unit, and a Timestamp's ``fold`` and zone; a list of them as one array
of those numbers. Code can tell equal instants in two units or two zones
apart, and so do their keys. One carrying attributes is pickled as before
(as a subclass would be: pandas builds a plain Timestamp for one), and keys
of values without dates are the bytes they were.
"""

from __future__ import annotations

import datetime
import sys
import tracemalloc
from collections import namedtuple
from dataclasses import dataclass

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from cash import canonical_form
from cash.canonical_form import canonical_bytes
from cash.content_hashers import BUILTIN_CONTENT


def key(value):
    return canonical_bytes(value, BUILTIN_CONTENT)


def old_key(value, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(canonical_form, "pandas_dates", lambda: frozenset())
        return key(value)


@dataclass
class Row:
    name: str
    when: datetime.datetime


Pair = namedtuple("Pair", "a b")


def _without_dates():
    shared = [1, 2]
    yield "numbers", [1, 2.5, True, None, 10**30]
    yield "strings", ["a", "b" * 300, ""]
    yield "nested", {"a": [shared, shared], "b": (1, {2, 3}), "c": frozenset({"x"})}
    yield "datetimes", [datetime.datetime(2020, 1, 2, 3), datetime.date(2020, 1, 2), datetime.timedelta(3)] * 5
    yield "zoned datetime", datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)
    yield "numpy dates", [np.datetime64("2020-01-01"), np.timedelta64(3, "s")] * 5
    yield "numpy numbers", [np.int64(i) for i in range(20)]
    yield "records", [Row(str(i), datetime.datetime(2020, 1, i + 1)) for i in range(10)]
    yield "namedtuples", [Pair(i, str(i)) for i in range(10)]
    yield "frame", pd.DataFrame({"t": pd.date_range("2020", periods=5), "x": list("abcde")})
    yield "series of objects", pd.Series([[1, 2], [3]], dtype=object)


def _with_dates():
    stamps = list(pd.date_range("2020-01-01", periods=20, freq="h"))
    yield "one timestamp", pd.Timestamp("2020-01-01")
    yield "a list", stamps
    yield "zoned", list(pd.date_range("2020-01-01", periods=10, tz="Europe/Berlin"))
    yield "timedeltas", [pd.Timedelta(i, "s") for i in range(10)]
    yield "NaT", [pd.NaT, 1]
    yield "in a dict", {"start": stamps[0], "end": stamps[-1]}
    yield "frame of lists", pd.DataFrame({"t": [stamps[:10], stamps[10:]]})


@pytest.mark.parametrize(("name", "value"), list(_without_dates()), ids=[n for n, _ in _without_dates()])
def test_a_key_without_dates_is_the_bytes_it_was(monkeypatch, name, value):
    assert key(value) == old_key(value, monkeypatch)


@pytest.mark.parametrize(("name", "value"), list(_with_dates()), ids=[n for n, _ in _with_dates()])
def test_a_key_holding_dates_changes(monkeypatch, name, value):
    assert key(value) != old_key(value, monkeypatch)
    assert key(value) == key(value)


def _both(value):
    """Alone, item by item, and in a list long enough to key as one array."""
    return key([value]), key([value] * 10)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01").as_unit("s")),
        (pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-01-01", tz="UTC").tz_convert("Europe/Berlin")),
        (pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-01-01")),
        (pd.Timestamp("2020-10-25 02:30"), pd.Timestamp("2020-10-25 02:30").replace(fold=1)),
        (pd.Timedelta(1, "s"), pd.Timedelta(1, "s").as_unit("ms")),
        (pd.Timedelta(1, "s"), pd.Timedelta(2, "s")),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01 00:00:00.000000001")),
        (pd.NaT, None),
    ],
    ids=["unit", "zone", "zone or none", "fold", "timedelta unit", "timedelta", "a nanosecond", "NaT"],
)
def test_what_code_can_tell_apart_keys_apart(a, b):
    for left, right in zip(_both(a), _both(b)):
        assert left != right


def test_equal_dates_built_apart_key_alike():
    built = list(pd.date_range("2020-01-01", periods=12, freq="D"))
    parsed = [pd.Timestamp(f"2020-01-{day:02d}").as_unit(built[0].unit) for day in range(1, 13)]
    assert key(built) == key(parsed)
    assert key(built[0]) == key(parsed[0])


def test_one_changed_item_changes_the_key():
    stamps = list(pd.date_range("2020-01-01", periods=12, freq="D"))
    other = list(stamps)
    other[5] = other[5] + pd.Timedelta(1, "ns")
    assert key(stamps) != key(other)


def test_an_attribute_on_a_timestamp_stays_in_its_key(monkeypatch):
    stamps = list(pd.date_range("2020-01-01", periods=12, freq="D"))
    noted = list(stamps)
    noted[3] = pd.Timestamp(stamps[3].isoformat()).as_unit(stamps[3].unit)  # another object
    noted[3].source = "manual"
    assert key(noted) != key(stamps)
    assert key(noted[3]) == old_key(noted[3], monkeypatch)


def test_zoned_timestamps_of_one_zone_key_as_one_array():
    zoned = list(pd.date_range("2020-01-01", periods=12, tz="Europe/Berlin"))
    mixed = zoned[:6] + [t.tz_convert("UTC") for t in zoned[6:]]
    assert key(zoned) != key(mixed)
    assert key(mixed) == key(list(mixed))


@pytest.mark.skipif(sys.version_info < (3, 11), reason="no object.__getstate__: a dict is made to look")
def test_keying_makes_no_dict_for_each_timestamp():
    stamps = list(pd.date_range("2020-01-01", periods=200_000, freq="min"))
    tracemalloc.start()
    try:
        key(stamps)
        key(stamps[0])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # 1.6 MB of numbers in the key; a dict each would be about 70 MB.
    assert peak < 20_000_000, peak


def test_a_cached_call_on_dates_hits_and_misses_by_their_values(tmp_path):
    from cash import Cash
    from cash.backends import InMemoryBackend

    cash = Cash(backend=InMemoryBackend(), register_magic=False)
    calls = []

    @cash.cache
    def first_day(days):
        calls.append(1)
        return min(days)

    stamps = list(pd.date_range("2020-01-01", periods=30, freq="D"))
    assert first_day(stamps) == stamps[0]
    assert first_day(list(stamps)) == stamps[0]
    assert len(calls) == 1
    seconds = [t.as_unit("s") for t in stamps]
    assert first_day(seconds).unit == "s"
    assert len(calls) == 2
