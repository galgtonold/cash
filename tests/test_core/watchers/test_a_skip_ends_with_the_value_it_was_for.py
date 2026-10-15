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


#: The sighting table: 2**13 slots, by a mix of the address (cwatch.c's
#: ``hmix`` and ``SIGHT_BITS``).
_SIGHT_SLOTS = 1 << 13
_U64 = (1 << 64) - 1


def _sight_slot(address: int) -> int:
    x = address
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & _U64
    x ^= x >> 33
    x = (x * 0xC4CEB9FE1A85EC53) & _U64
    x ^= x >> 33
    return x % _SIGHT_SLOTS


class _Changing:
    def __init__(self) -> None:
        self.n = 0


def test_a_value_sharing_a_skipped_values_slot_is_watched(watching, counted):
    """A sighting is kept in a slot chosen by address, and a skip kept its
    slot from every other value for up to a million checks: a value whose
    address landed there was seen for the first time on every check, and
    never armed. It counts as seen twice instead."""
    changing = _Changing()
    for i in range(8):
        changing.n = i
        value_watch.digest(changing, counted)
    assert value_watch._cw.check(changing, value_watch.KIND_VALUE) == SKIP
    held = []
    while True:
        value = _rows()
        if _sight_slot(id(value)) == _sight_slot(id(changing)):
            break
        held.append(value)
    counted.calls = 0
    for _ in range(4):
        value_watch.digest(value, counted)
    assert counted.calls < 4, "a value in a skipped value's slot was never armed"
