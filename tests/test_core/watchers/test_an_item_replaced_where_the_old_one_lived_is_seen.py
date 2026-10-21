"""An item replaced by a new one made where the old one lived is seen.

A watched list's record is its item pointers, compared on the next check.
The pointers were borrowed: ``scores[0] = f()`` freed the old float, the
new one landed at its address, the pointers compared equal and the digest
of the old list was served for the new one. The same went for a set's
items, an instance's slots, and the items of a tuple or frozenset, which
had no record at all. Every value such a record holds by address now has
what it reads as recorded beside it (a float's bits, an int's sign and
magnitude, a string's hash, length and first bytes), compared too. Held
instead, its address could not be
reused, but the watch kept the user's old strings alive for as long as it
stood: a list emptied in place kept its 50 MB string until it was read
again, outside the watch's memory budget.
"""

from __future__ import annotations

import sys

import pytest

from cash import value_watch
from cash.value_hash import _compute_hash


class Slotted:
    __slots__ = ("a", "b", "__weakref__")

    def __init__(self, a, b):
        self.a = a
        self.b = b


def _new_int(n):
    return int(str(n))  # a fresh int object, not a cached small one


# past the small-object allocator, whose pools reuse an address less surely
_SAME_LENGTH = [("x" * 598, f"z{i % 10}") for i in range(1000)]


def _watched(value, counted):
    for _ in range(3):
        value_watch.digest(value, counted)
    assert value_watch._cw.info(value) is not None, "not watched"


def _replace_at_the_same_address(read, write, make, tries=200):
    """Replace the item *read* gives with one *make* builds, until the new
    one lands where the old one was freed (the allocator's free list)."""
    old = id(read())
    for i in range(tries):
        write(None)
        write(make(i))
        if id(read()) == old:
            return True
    return False


CASES = {
    "a list's item": (
        lambda: [_new_int(1000001), {"pad": [1, 2]}],
        lambda v: v[0],
        lambda v, x: v.__setitem__(0, x),
        lambda i: _new_int(2000002 + i),
    ),
    "a float in a list": (
        lambda: [float("1.5"), {"pad": 1}],
        lambda v: v[0],
        lambda v, x: v.__setitem__(0, x),
        lambda i: float(f"2.{i + 5}"),
    ),
    "a tuple in a list": (
        lambda: [(_new_int(7000), "a"), {"pad": 1}],
        lambda v: v[0],
        lambda v, x: v.__setitem__(0, x),
        lambda i: (_new_int(8000 + i), "b"),
    ),
    "an int in a tuple in a dict": (
        lambda: {"t": [(_new_int(5000), "x")]},
        lambda v: v["t"][0],
        lambda v, x: v["t"].__setitem__(0, x),
        lambda i: (_new_int(6000 + i), "x"),
    ),
    "a slot's value": (
        lambda: {"o": Slotted(_new_int(3000001), 1)},
        lambda v: v["o"].a,
        lambda v, x: setattr(v["o"], "a", x),
        lambda i: _new_int(4000002 + i),
    ),
    "a str in a list": (
        lambda: ["".join(["a" * 598, "c7"]), {"pad": 1}],
        lambda v: v[0],
        lambda v, x: v.__setitem__(0, x),
        lambda i: "".join(_SAME_LENGTH[i]),
    ),
    "an int's sign": (
        lambda: [_new_int(1000001), {"pad": 1}],
        lambda v: v[0],
        lambda v, x: v.__setitem__(0, x),
        lambda i: _new_int(-1000001),
    ),
    "a frozenset in a list": (
        lambda: [frozenset({float("1.5"), "k"}), {"pad": 1}],
        lambda v: v[0],
        lambda v, x: v.__setitem__(0, x),
        lambda i: frozenset({float(f"2.{i + 5}"), "k"}),
    ),
}


def _set_float(v):
    return next(x for x in v["s"] if type(x) is float)


def _set_write(v, x):
    if x is None:
        v["s"].discard(_set_float(v))
    else:
        v["s"].add(x)


CASES["a set's item"] = (
    lambda: {"s": {float("1.5"), "k"}},
    _set_float,
    _set_write,
    lambda i: float(f"5.{i + 5}"),
)


@pytest.mark.parametrize("case", list(CASES))
def test_the_new_item_is_read(watching, counted, case):
    build, read, write, make = CASES[case]
    value = build()
    watching["value"] = value
    _watched(value, counted)
    # A new item lands at the old one's address within a few tries.
    if not _replace_at_the_same_address(lambda: read(value), lambda x: write(value, x), make, tries=1000):
        pytest.skip("the allocator never made the new item where the old one was")
    assert value_watch.digest(value, counted) == _compute_hash(value), "a stale digest was served"


def test_the_watch_keeps_no_item_alive(watching, counted):
    big = "x" * 5_000_000
    before = sys.getrefcount(big)
    value = [big, float("1.5"), _new_int(9000001)]
    watching["value"] = value
    _watched(value, counted)
    assert sys.getrefcount(big) == before + 1, "the watch holds the list's string"
    value.clear()
    assert sys.getrefcount(big) == before, "an emptied list's string stays alive"
    assert value_watch._cw.stats()["table_bytes"] < 1_000_000


def test_an_int_past_63_bits_is_not_held(watching, counted):
    big = int(str(2**63 + 5))
    before = sys.getrefcount(big)
    value = [big, {"pad": 1}]
    watching["value"] = value
    _watched(value, counted)
    assert sys.getrefcount(big) == before + 1, "the watch holds an int that fits 64 bits"


def test_an_equal_item_made_at_the_same_address_is_served(watching, counted):
    value = [float("1.5"), {"pad": 1}]
    watching["value"] = value
    _watched(value, counted)
    assert _replace_at_the_same_address(lambda: value[0], lambda x: value.__setitem__(0, x), lambda i: float("1.5"))
    calls = counted.calls
    assert value_watch.digest(value, counted) == _compute_hash(value)
    assert counted.calls == calls, "an equal float where the old one lived was hashed again"
