"""The RAM tier sizes and copies plain data without a Python call per element.

Promoting a disk hit of two million parsed rows into the RAM tier took 6.3 s:
a recursive sizeof walk (3.5 s) and ``copy.deepcopy`` (1.5 s, again on every
RAM hit). Lists and tuples over primitives are now sized a level at a time and
copied without the per-element walk -- and a copy must still be one: nothing
the caller does to what it got back may reach the stored entry.
"""
from __future__ import annotations

import sys

import pytest

from cash import _plain_data
from cash.backends.memory_backend import InMemoryBackend

pytestmark = [pytest.mark.core]


def _stored(backend, key):
    return backend._store[key][1]


def test_rows_of_tuples_come_back_as_a_new_list_every_time():
    b = InMemoryBackend()
    rows = [(i, f"x{i}") for i in range(1000)]
    b.set("k", rows)
    _meta, first = b.get("k")
    _meta, second = b.get("k")
    assert first == rows and second == rows
    assert first is not second and first is not _stored(b, "k")
    first.append((-1, "caller's"))
    assert b.get("k")[1] == rows, "a caller's append reached the stored entry"


def test_nested_lists_are_copied_at_every_level():
    b = InMemoryBackend()
    grid = [[i, [i, i + 1]] for i in range(500)]
    b.set("g", grid)
    got = b.get("g")[1]
    got[0][1].append("caller's")
    got[1][0] = "caller's"
    assert b.get("g")[1] == [[i, [i, i + 1]] for i in range(500)]
    grid[2][1].append("the original's")
    assert b.get("g")[1][2] == [2, [2, 3]], "the stored entry shares the original's lists"


def test_a_bytearray_leaf_is_not_shared():
    """bytearray is plain for keying, but it can be changed in place."""
    b = InMemoryBackend()
    b.set("b", [(1, bytearray(b"ab"))])
    got = b.get("b")[1]
    got[0][1][0] = ord("z")
    assert b.get("b")[1][0][1] == bytearray(b"ab")


def test_plain_sizes_track_the_recursive_estimate():
    rows = [(i, f"x{i}") for i in range(2000)]
    fast = _plain_data.size_of(rows)
    recursive = sys.getsizeof(rows) + sum(
        sys.getsizeof(t) + sum(sys.getsizeof(x) for x in t) for t in rows)
    assert fast == recursive


def test_a_big_level_is_sized_from_a_sample_that_lands_close():
    """Above `SIZE_EXACT_UP_TO` items a level is sampled. Rows flatten to a
    repeating int, int, str pattern, which a fixed stride could have landed on
    one column of; texts vary in length."""
    for value in ([(i, i * 2, "x" * (i % 300)) for i in range(100_000)],
                  ["y" * ((i * 7919) % 5000) for i in range(100_000)]):
        exact = sys.getsizeof(value) + sum(
            sum(map(sys.getsizeof, flat)) for flat, _t in _plain_data._levels(value))
        estimate = _plain_data.size_of(value)
        assert abs(estimate - exact) / exact < 0.02
        assert _plain_data.size_of(value) == estimate, "one value, two sizes"


def test_rows_are_looked_at_once_on_the_way_in(monkeypatch):
    """Sizing, the immutability check and the copy share one walk: three were
    most of promoting two million parsed rows into the RAM tier."""
    walks = []
    real = _plain_data._levels
    monkeypatch.setattr(_plain_data, "_levels", lambda v: walks.append(1) or real(v))
    InMemoryBackend().set("k", [(i, str(i)) for i in range(1000)])
    assert len(walks) == 1


def test_non_plain_data_takes_the_old_paths():
    assert _plain_data.size_of({"a": 1}) is None
    assert _plain_data.copy_plain([{1, 2}]) == (False, None)
    assert not _plain_data.immutable_below([[1]])
    assert _plain_data.immutable_below([(1, (2, "x"))])


def test_dict_rows_come_back_as_new_dicts_every_time():
    """Round 20 (r20s2 F10): a list of dicts was deep-copied on every RAM hit;
    with immutable values a new dict per row is the whole copy."""
    b = InMemoryBackend()
    rows = [{"id": i, "city": "x"} for i in range(1000)]
    b.set("d", rows)
    got = b.get("d")[1]
    assert got == rows and got is not rows and got[0] is not rows[0]
    got[0]["city"] = "caller's"
    assert b.get("d")[1][0]["city"] == "x", "a caller's write reached the stored entry"
    rows[1]["city"] = "the original's"
    assert b.get("d")[1][1]["city"] == "x", "the stored entry shares the original's dicts"
    assert "d" in b._dict_rows
    b.set("nested", [{"a": [1]}])                  # a mutable value: the old path
    assert "nested" not in b._dict_rows
