"""A value the watchers gave up on is skipped while it lives, not after.

A value that changes on every check, or is too small to be worth watching,
is skipped for a while: hashed each time, never armed. The skip was kept by
address alone, and an address outlives its object: the next dict made lands
where the last one was freed. That dict inherited the skip, so a big dict
that never changed went unwatched for up to a million checks: a notebook
variable "never vouched for", a memo that never served its argument, both
after whatever test had left a skip at that address.
"""

from __future__ import annotations

from cash import value_watch

SKIP = value_watch._cw.SKIP if value_watch.enabled() else None


def _skipped_for_changing(counted, make):
    value = make()
    for i in range(8):
        value["n"] = i
        value_watch.digest(value, counted)
    assert value_watch._cw.check(value, value_watch.KIND_VALUE) == SKIP, "the changing value was not skipped"
    return value


def _at_the_same_address(make, address):
    """A new dict made where a freed one lived (the dict free list)."""
    held = []
    for _ in range(100):
        value = make()
        if id(value) == address:
            return value
        held.append(value)
    raise AssertionError("no new value landed at the freed address")


def _watched_after_two_hashes(value, counted):
    counted.calls = 0
    for _ in range(4):
        value_watch.digest(value, counted)
    return counted.calls == 2


def _rows():
    return {"rows": list(range(50))}


def test_a_dict_made_where_a_changing_one_was_freed_is_watched(watching, counted):
    address = id(_skipped_for_changing(counted, _rows))
    fresh = _at_the_same_address(_rows, address)
    assert _watched_after_two_hashes(fresh, counted), "the new value inherited the freed one's skip"


def test_a_dict_made_where_a_too_small_one_was_freed_is_watched(watching, counted):
    value_watch._cw.configure(min_bytes=1 << 20)
    small = {"a": 1}
    for _ in range(3):
        value_watch.digest(small, counted)
    assert value_watch._cw.check(small, value_watch.KIND_VALUE) == SKIP
    address = id(small)
    del small
    value_watch._cw.configure(min_bytes=0)
    fresh = _at_the_same_address(lambda: {"a": 1}, address)
    assert _watched_after_two_hashes(fresh, counted)


def test_a_skipped_value_still_alive_stays_skipped(watching, counted):
    """The control: the skip itself still holds while its value lives."""
    value = _skipped_for_changing(counted, _rows)
    counted.calls = 0
    for _ in range(4):
        value_watch.digest(value, counted)
    assert counted.calls == 4
