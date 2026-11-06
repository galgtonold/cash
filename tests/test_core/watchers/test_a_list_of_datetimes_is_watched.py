"""A value holding naive datetimes is watched like one holding numbers.

A participant's notebook held the parsed log as a list of ``(user,
[(action, datetime), ...])`` records and a 2-million-item list of
``datetime`` objects. Neither was watched (a datetime was "anything else"),
so each was pickled whole every time cash looked at it: around every cell
run as written, 0.2 s and 0.15 s at a tenth of the data. A naive
``datetime``, ``date``, ``time`` or ``timedelta`` is immutable and pickles as
its class and its bytes: it is now an atom, held by address with every byte
it pickles from recorded beside it, and its class recorded as a plain
class's is (how pickle finds it by name, ``copyreg.dispatch_table`` at it).
One with a ``tzinfo`` holds an object of its own and is still not watched.
"""

from __future__ import annotations

import copyreg
import datetime

import pytest

from cash import value_watch
from cash.value_hash import _compute_hash

T0 = datetime.datetime(2020, 1, 2, 3, 4, 5, 678)


def _records(n=200):
    return [(f"@User{i % 7}", [("Action_1", T0 + datetime.timedelta(minutes=i)), ("Action_2", T0)]) for i in range(n)]


def _watched(value, counted):
    for _ in range(3):
        value_watch.digest(value, counted)
    assert value_watch._cw.info(value) is not None, "not watched"
    calls = counted.calls
    assert value_watch.digest(value, counted) == _compute_hash(value)
    assert counted.calls == calls, "an unchanged value was read again"


VALUES = {
    "datetimes": lambda: [T0 + datetime.timedelta(seconds=i) for i in range(500)],
    "records": _records,
    "dates and times": lambda: [(T0.date(), T0.time(), datetime.timedelta(days=i, microseconds=7)) for i in range(300)],
    "a dict of datetimes": lambda: {f"k{i}": T0 + datetime.timedelta(hours=i) for i in range(300)},
}


@pytest.mark.parametrize("case", list(VALUES))
def test_an_unchanged_value_is_not_read_again(watching, counted, case):
    value = VALUES[case]()
    watching["value"] = value
    _watched(value, counted)


EDITS = {
    "an item replaced": lambda v: v.__setitem__(5, v[5] + datetime.timedelta(microseconds=1)),
    "a record's list grown": lambda v: v[3][1].append(("Action_3", T0)),
    "a fold": lambda v: v.__setitem__(0, v[0].replace(fold=1)),
}


@pytest.mark.parametrize("edit", list(EDITS))
def test_a_change_is_read(watching, counted, edit):
    value = VALUES["records"]() if edit == "a record's list grown" else VALUES["datetimes"]()
    watching["value"] = value
    _watched(value, counted)
    EDITS[edit](value)
    assert value_watch.digest(value, counted) == _compute_hash(value), "a stale digest was served"


def test_a_datetime_made_where_the_old_one_lived_is_read(watching, counted):
    value = [datetime.datetime(2020, 1, 1, 0, 0, 1), {"pad": 1}]
    watching["value"] = value
    _watched(value, counted)
    old = id(value[0])
    for i in range(1000):
        value[0] = None
        value[0] = datetime.datetime(2021, 1, 1, 0, 0, i % 60)
        if id(value[0]) == old:
            break
    else:
        pytest.skip("the allocator never made the new datetime where the old one was")
    assert value_watch.digest(value, counted) == _compute_hash(value), "a stale digest was served"


def test_a_datetime_with_a_tzinfo_is_not_watched(watching, counted):
    value = [T0.replace(tzinfo=datetime.timezone.utc)] * 50 + [{"pad": 1}]
    watching["value"] = value
    for _ in range(3):
        value_watch.digest(value, counted)
    assert value_watch._cw.info(value) is None


def test_a_reducer_registered_for_datetime_is_read(watching, counted):
    value = VALUES["datetimes"]()
    watching["value"] = value
    _watched(value, counted)
    copyreg.pickle(datetime.datetime, lambda d: (datetime.datetime.fromisoformat, (d.isoformat(),)))
    try:
        calls = counted.calls
        assert value_watch.digest(value, counted) == _compute_hash(value)
        assert counted.calls == calls + 1, "a digest from before the reducer was served"
    finally:
        del copyreg.dispatch_table[datetime.datetime]


def test_a_watched_datetime_costs_its_pointer_and_one_word(watching, counted):
    """The parsed log at full size (2 million datetimes in records) needed a
    344 MB table, past the budget of a 16 GB machine (2% of RAM), and was
    hashed every time: a datetime is now printed in one word, and a walked
    root's arrays are shrunk to what they hold."""
    n = 20_000
    value = [T0 + datetime.timedelta(seconds=i) for i in range(n)]
    watching["value"] = value
    _watched(value, counted)
    # 8 bytes of pointer and 9 of print per item, and the root's own
    assert value_watch._cw.stats()["table_bytes"] < 18 * n + 200_000
