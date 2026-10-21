"""Two values whose sightings share a slot are both watched in the end.

A value is watched from its second hash: the first one is noted in a table
of sightings indexed by address. Two values whose addresses land in the
same slot, hashed in turn (a notebook reads ``data`` and then ``rows`` in
every statement), each wrote over the other's sighting, so every hash of
either was a first: neither was ever watched, both were hashed on every
read. A notebook variable "never vouched for" on one CI run in thousands,
whenever the addresses that run handed out happened to meet in the slot.
"""

from __future__ import annotations

from cash import value_watch

_MASK64 = (1 << 64) - 1


def _slot(value) -> int:
    """The sightings slot of *value* (``sight_of`` in cwatch.c)."""
    x = id(value)
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & _MASK64
    x ^= x >> 33
    x = (x * 0xC4CEB9FE1A85EC53) & _MASK64
    x ^= x >> 33
    return x & ((1 << value_watch._cw.SIGHT_BITS) - 1)


def _sharing_a_slot_with(value):
    held = []
    for i in range(500_000):
        other = {"k": [i] * 20}
        if _slot(other) == _slot(value):
            return other
        held.append(other)
    raise AssertionError("no dict landed in the value's slot")


def test_two_values_hashed_in_turn_are_both_watched(watching, counted):
    first = {"rows": [{"v": i} for i in range(50)]}
    second = _sharing_a_slot_with(first)
    for _ in range(4):
        value_watch.digest(first, counted)
        value_watch.digest(second, counted)
    calls = counted.calls
    for _ in range(3):
        value_watch.digest(first, counted)
        value_watch.digest(second, counted)
    assert counted.calls == calls, "a value sharing a sighting slot is never watched"
