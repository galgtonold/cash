"""A value the watchers gave up on is skipped while it lives, not after.

A value that changes on every check, or is too small to be worth watching,
is skipped for a while: hashed each time, never armed. The skip was kept by
address alone, and an address outlives its object: the next dict made lands
where the last one was freed. That dict inherited the skip, so a big dict
that never changed went unwatched for up to a million checks: a notebook
variable "never vouched for", a memo that never served its argument, both
after whatever test had left a skip at that address. A list or a tuple,
which no weak reference can name, is told by what holds its items as well:
the next list made lands where the last one was freed too.
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


def _a_dict_where_a_list_was_freed(make_list):
    """A dict made where a list made by *make_list* was freed, and that list's
    address. Lists kept on the list free list are not given back, so it is
    filled first; the dict free list is drained by the dicts held here."""
    for _ in range(5):
        others = [[i] for i in range(300)]
        listed = make_list()
        address = id(listed)
        del others, listed
        held = []
        for _ in range(20000):
            value = {"a": 1, "b": 2}  # no list of its own, to land where one was
            if id(value) == address:
                return value
            held.append(value)
    raise AssertionError("no dict landed where the list was freed")


def test_a_dict_made_where_a_skipped_list_was_freed_is_watched(watching, counted):
    """A list has no weak reference to prove a skip is still about it, and
    was skipped by address alone: the notebook variable a dict was "never
    vouched for" when it was made where a list some earlier test left
    skipped had lived."""
    value_watch._cw.configure(min_bytes=1 << 20)

    def skipped_list():
        small = [1, 2, 3]
        watching["small"] = small  # a list is watched while a namespace holds it
        for _ in range(3):
            value_watch.digest(small, counted)
        assert value_watch._cw.check(small, value_watch.KIND_VALUE) == SKIP, "the small list was not skipped"
        del watching["small"]
        return small

    fresh = _a_dict_where_a_list_was_freed(skipped_list)
    value_watch._cw.configure(min_bytes=0)
    assert _watched_after_two_hashes(fresh, counted), "the dict inherited the freed list's skip"


def _a_list_where_a_skipped_one_was_freed(watching, counted, make):
    """A list made by *make* where a three-item list skipped as too small
    was freed just before: CPython hands a new list the one freed last."""
    value_watch._cw.configure(min_bytes=1 << 20)
    small = [1, 2, 3]
    watching["small"] = small
    for _ in range(3):
        value_watch.digest(small, counted)
    assert value_watch._cw.check(small, value_watch.KIND_VALUE) == SKIP, "the small list was not skipped"
    address = id(small)
    del watching["small"], small
    fresh = make()
    value_watch._cw.configure(min_bytes=0)
    assert id(fresh) == address, "the new list was not made where the skipped one was freed"
    return fresh


def test_a_list_made_where_a_skipped_list_was_freed_is_watched(watching, counted):
    """A list has no weak reference to prove a skip is still about it, and
    was skipped by its address and type alone: a table of 30,000 records
    made where a small list some earlier statement left skipped had lived
    was hashed whole on every check, never watched."""
    rows = _a_list_where_a_skipped_one_was_freed(
        watching, counted, lambda: [{"id": i, "tags": ["a"]} for i in range(30_000)]
    )
    watching["rows"] = rows
    assert _watched_after_two_hashes(rows, counted), "the new list inherited the freed list's skip"


def test_a_tuple_made_where_a_skipped_tuple_was_freed_is_watched(watching, counted):
    value_watch._cw.configure(min_bytes=1 << 20)
    small = tuple([1, 2, "c"])  # not a constant, which its code would keep alive
    watching["small"] = small
    for _ in range(3):
        value_watch.digest(small, counted)
    assert value_watch._cw.check(small, value_watch.KIND_VALUE) == SKIP, "the small tuple was not skipped"
    address = id(small)
    del watching["small"], small
    value_watch._cw.configure(min_bytes=0)
    held = []
    for i in range(20000):
        fresh = (i, [i] * 50, "x" * 100)
        if id(fresh) == address:
            break
        held.append(fresh)
    else:
        raise AssertionError("no tuple landed where the skipped one was freed")
    watching["fresh"] = fresh
    assert _watched_after_two_hashes(fresh, counted), "the new tuple inherited the freed tuple's skip"


def test_a_skipped_list_still_alive_stays_skipped(watching, counted):
    """The control for lists: the skip holds while the list lives, its
    items set in place or not."""
    rows = [{"id": i} for i in range(200)]
    watching["rows"] = rows
    for i in range(8):
        rows[0] = {"id": -i}
        value_watch.digest(rows, counted)
    assert value_watch._cw.check(rows, value_watch.KIND_VALUE) == SKIP, "the changing list was not skipped"
    rows[1] = {"id": -1}
    counted.calls = 0
    for _ in range(4):
        value_watch.digest(rows, counted)
    assert counted.calls == 4


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
