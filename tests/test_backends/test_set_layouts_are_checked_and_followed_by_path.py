"""Set layouts are checked before a table is touched, and a hit follows only
the paths to the sets.

A layout record laid a set's elements into the slots it named without asking
whether they made a table: a damaged record could crash the process or leave
an element where a lookup never finds it. Each record is now covered by a
checksum, and each set's slots must make a table CPython could have built --
a power-of-two size with an empty slot left, no slot twice, every element on
its hash's probe path behind taken slots only -- or the set keeps the order
its rebuild gave it.

A value with no set was walked twice on a store (once per tier) and a value
with one was walked whole again on every hit; a store now notes that a value
holds none, and a hit follows each set's recorded path.
"""

from __future__ import annotations

import pickle
import random

import pytest

from cash.backends import set_order

pytestmark = pytest.mark.skipif(set_order._KERNELS is None, reason="no C extension for this Python")

K = set_order._KERNELS


def _value():
    r = random.Random(3)
    big = set(r.sample(range(10**6), 3000))
    for n in r.sample(sorted(big), 1000):
        big.discard(n)
    return {"a": big, "b": [{(1, 2), (3, 4), (5, 6)}, frozenset(r.random() for _ in range(40))]}


def _mix64(x):
    m = (1 << 64) - 1
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & m
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & m
    return x ^ (x >> 31)


def _signed(body: bytes) -> bytes:
    """*body* with the checksum `csets.c` ends a record with."""
    m = (1 << 64) - 1
    h, i = 0x6A09E667F3BCC908 ^ len(body), 0
    while i + 8 <= len(body):
        h = (_mix64(h ^ int.from_bytes(body[i : i + 8], "little")) + 0x9E3779B97F4A7C15) & m
        i += 8
    tail = int.from_bytes(body[i:], "little") if i < len(body) else 0
    return body + _mix64(h ^ tail).to_bytes(8, "little")


def _intact(back, value):
    assert back == value
    for got, want in ((back["a"], value["a"]), (back["b"][0], value["b"][0]), (back["b"][1], value["b"][1])):
        assert len(got) == len(want) and all(x in got for x in want)


def test_the_checksum_model_matches():
    layouts = K.set_layouts(_value(), set_order.MAX_NODES)
    assert _signed(layouts[:-8]) == layouts


def test_a_damaged_record_is_refused_whole():
    value = _value()
    layouts = K.set_layouts(value, set_order.MAX_NODES)
    r = random.Random(5)
    for trial in range(400):
        damaged = bytearray(layouts)
        if trial % 2:
            damaged[r.randrange(len(damaged))] ^= 1 << r.randrange(8)
        else:
            damaged = damaged[: r.randrange(len(damaged))]
        back = pickle.loads(pickle.dumps(value, 5))
        with pytest.raises(ValueError):
            K.apply_set_layouts(back, bytes(damaged), set_order.MAX_NODES)
        _intact(back, value)


def test_slots_that_make_no_table_are_never_laid_out():
    """Damage the slots and sign the record again: a set is laid out only into
    a table every lookup works on, else it keeps its rebuilt table."""
    value = _value()
    layouts = K.set_layouts(value, set_order.MAX_NODES)
    r = random.Random(7)
    refused = 0
    for _ in range(600):
        body = bytearray(layouts[:-8])
        body[r.randrange(40, len(body))] ^= 1 << r.randrange(8)
        back = pickle.loads(pickle.dumps(value, 5))
        try:
            refused += K.apply_set_layouts(back, _signed(bytes(body)), set_order.MAX_NODES)[2]
        except ValueError:  # a header no longer parses
            refused += 1
        _intact(back, value)
    assert refused > 300


def test_a_value_without_sets_is_walked_once_and_a_hit_does_nothing():
    value = {"rows": [{"id": i, "x": [i, str(i)]} for i in range(1000)]}
    metadata = {}
    set_order.record(value, metadata)
    assert metadata[set_order.FIELD] == b""  # the next tier sees it was walked
    assert set_order.restore(pickle.loads(pickle.dumps(value, 5)), metadata) is None


def test_a_hit_lays_out_a_set_deep_in_a_big_value_and_counts_what_it_did():
    r = random.Random(1)
    tags = set()
    for n in r.sample(range(1 << 40), 600):  # big ints collide in a small table
        tags.add(n)
    for n in r.sample(sorted(tags), 200):
        tags.discard(n)
    value = {"rows": [{"id": i, "name": f"n{i}"} for i in range(50_000)], "deep": [[{"k": (1, tags)}]], "same": tags}
    metadata = {}
    set_order.record(value, metadata)
    back = pickle.loads(pickle.dumps(value, 5))
    assert list(back["deep"][0][0]["k"][1]) != list(tags)  # the rebuild moved them
    assert set_order.restore(back, metadata) == (1, 0, 0)
    assert list(back["deep"][0][0]["k"][1]) == list(tags)
    assert back["same"] is back["deep"][0][0]["k"][1]  # shared, laid out once
    # Laid out so already: counted apart from those laid out and those left.
    assert set_order.restore(back, metadata) == (0, 1, 0)
