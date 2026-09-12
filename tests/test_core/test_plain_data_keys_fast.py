"""A big plain-data argument is keyed by its content, without walking it in Python.

A warm hit on a function receiving two million parsed rows took 8.4 s against a
0.04 s body: three Python-level walks over every element (`_contains_set`,
`_canonicalize_dict_order`, `_iter_code_carriers`) that can find nothing in
lists and tuples of primitives, then a pickle whose memo -- a dict entry per
tuple and string -- was 80 % of what was left. Plain data is now proven so at C
speed and pickled without the memo: 0.37 s. Without the memo the key is the
content alone, so what matters is that equal content keys equal and anything
else does not.
"""
from __future__ import annotations

import time

import pytest

import cash.core as core
from cash import Cash, _plain_data

pytestmark = [pytest.mark.core]

SHARED = (1, "shared")

SHAPES = {
    "rows": ([(i, i * 2, f"x{i % 7}") for i in range(500)],),
    "nested lists": ([[i, [i, i + 1.5], b"b", None, True, 2j] for i in range(200)],),
    "empty tuples inside": ([(), (1, ()), [(), ()]],),
    "kwargs": ((), {"rows": [(1, 2), (3, 4)], "n": 5}),
    "a list and a scalar": ([1, 2, 3], "tag", 4.5),
    "shared tuple inside": ([SHARED, SHARED, (2,)],),
    "shared across arguments": ([SHARED], [SHARED]),
    "the same list twice": ([1, 2], ) * 2,
    "a dict inside": ([{"b": 1, "a": 2}],),
    "a set inside": ([{3, 1, 2}],),
    "a list subclass": (type("L", (list,), {})([1, 2]),),
    "bytearray leaves": ([bytearray(b"xy"), 1],),
}


def _args(shape):
    value = SHAPES[shape]
    if len(value) == 2 and isinstance(value[1], dict):
        return value[0], value[1]
    return value, {}


@pytest.fixture
def key(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "cache"))
    return lambda *args, **kwargs: c._hash_arg_payload(args, kwargs)


def test_equal_content_keys_equal(key):
    """Built twice, and with a string shared between rows against a copy per
    row: the memo wrote a shared object once and then referred back to it."""
    tag = "x" * 20
    shared = [(i, tag) for i in range(100)]
    fresh = [(i, "".join(["x"] * 20)) for i in range(100)]
    assert shared[0][1] is shared[1][1] and fresh[0][1] is not fresh[1][1]
    assert key(shared) == key(fresh)
    assert key([(1, 2)] * 3) == key([(1, 2), (1, 2), (1, 2)])
    assert key(rows=[(1, "a")], n=2) == key(n=2, rows=[(1, "a")])


@pytest.mark.parametrize("a, b", [
    ([(1,)], [(1.0,)]),
    ([(1,)], [(True,)]),
    ([(1, 2)], [[1, 2]]),
    (["a"], [b"a"]),
    ([(1, (2,))], [(1, 2)]),
    ([1, 2], [2, 1]),
    ([bytearray(b"a")], [b"a"]),
    ([None], [()]),
])
def test_different_content_keys_apart(key, a, b):
    assert key(a) != key(b)


def test_a_list_reached_twice_keys_like_two_equal_ones(key):
    """As it always has: the general path rebuilt every list, which dropped
    the sharing too. The key is the content."""
    inner = [1]
    assert key([inner, inner]) == key([[1], [1]])
    assert key(inner, inner) == key([1], [1])


def test_plain_data_is_recognised_and_other_data_is_not():
    rows = [(i, str(i)) for i in range(10)]
    assert core._is_plain(rows)
    assert core._is_plain([[1, [2, [3]]]])
    assert not core._is_plain([{"a": 1}])
    assert not core._is_plain([{1}])
    assert not core._is_plain([object()])
    cyclic: list = [1]
    cyclic.append(cyclic)
    assert not core._is_plain(cyclic)


def test_a_warm_hit_on_many_rows_does_not_walk_them(tmp_path, monkeypatch):
    """Counted, not timed: none of the three walks is entered for the rows."""
    c = Cash(cache_dir=str(tmp_path / "cache"))
    rows = [(i, i * 2, f"x{i % 100}") for i in range(50_000)]

    @c.cache
    def total(rows):
        return sum(r[0] for r in rows)

    total(rows)
    calls = {"_contains_set": 0, "_canonicalize_dict_order": 0, "carriers": 0}
    for name in ("_contains_set", "_canonicalize_dict_order"):
        real = getattr(core, name)
        monkeypatch.setattr(core, name, lambda *a, _r=real, _n=name, **k:
                            calls.__setitem__(_n, calls[_n] + 1) or _r(*a, **k))
    real_iter = c._iter_code_carriers
    monkeypatch.setattr(c, "_iter_code_carriers", lambda *a, **k:
                        calls.__setitem__("carriers", calls["carriers"] + 1) or real_iter(*a, **k))
    t0 = time.perf_counter()
    assert total(rows) == sum(range(50_000))
    # Walking the rows is a call per row and more; the key itself takes a few.
    assert max(calls.values()) < 50, f"walked the rows: {calls}"
    assert time.perf_counter() - t0 < 30        # a hang guard, not a benchmark


FAST = {"rows", "nested lists", "empty tuples inside", "kwargs", "a list and a scalar",
        "shared tuple inside", "shared across arguments", "the same list twice",
        "bytearray leaves", "a dict inside"}   # a list of one dict: dict rows


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_which_shapes_take_the_fast_path(tmp_path, monkeypatch, shape):
    """Plain data is pickled without the memo; a set or a subclass
    takes the general path, which handles them."""
    c = Cash(cache_dir=str(tmp_path / "cache"))
    unshared = []
    real = _plain_data.pickle_unshared
    monkeypatch.setattr(_plain_data, "pickle_unshared",
                        lambda value: unshared.append(1) or real(value))
    c._hash_arg_payload(*_args(shape))
    assert bool(unshared) == (shape in FAST), shape


# -- round 20: dates, dict rows, and one argument not sinking the rest -------

import datetime as _dt  # noqa: E402
import decimal as _decimal  # noqa: E402


def test_rows_holding_dates_and_decimals_are_plain():
    """r20s3: rows with a date column cost 16x their body per call to key --
    a date took the whole list off the fast path."""
    rows = [(_dt.date(2024, 1, 1 + i % 28), _decimal.Decimal("1.50"), i) for i in range(1000)]
    assert core._is_plain(rows)
    assert core._is_plain([[_dt.datetime(2024, 1, 1, 12), _dt.timedelta(hours=2)]])


def test_date_rows_key_by_content(key):
    a = [(_dt.date(2024, 1, 2), 1)]
    assert key(a) == key([(_dt.date(2024, 1, 2), 1)])
    assert key(a) != key([(_dt.date(2024, 1, 3), 1)])
    assert key([(_decimal.Decimal("1.5"),)]) != key([(1.5,)])


def test_dict_rows_key_by_content_whatever_the_insertion_order(key):
    """r20s2 F10: csv.DictReader rows went down the general path, every dict
    walked and rebuilt in Python -- ~10x the same data as tuples."""
    rows = [{"id": i, "city": f"c{i % 5}"} for i in range(1000)]
    same = [{"city": f"c{i % 5}", "id": i} for i in range(1000)]
    assert core._plain_census(rows)[0] == "dict_rows"
    assert key(rows) == key(same)
    other = [dict(r) for r in rows]
    other[500]["city"] = "elsewhere"
    assert key(rows) != key(other)
    assert key([{"a": 1}]) != key([(1,)]), "dict rows and tuple rows keyed alike"
    assert key([{"a": 1}]) != key([{"b": 1}])


def test_a_small_dict_beside_a_big_list_leaves_the_list_on_the_fast_path(tmp_path, monkeypatch):
    """r20s2 F2: one tiny dict argument next to 2M tuples made the key 8x
    slower -- the fast path applied only when every argument was plain."""
    c = Cash(cache_dir=str(tmp_path / "cache"))
    rows = [(i, str(i)) for i in range(50_000)]
    unshared, walked = [], []
    real_pickle, real_canon = _plain_data.pickle_unshared, core._canonicalize_dict_order
    monkeypatch.setattr(_plain_data, "pickle_unshared",
                        lambda v: unshared.append(v) or real_pickle(v))
    monkeypatch.setattr(core, "_canonicalize_dict_order",
                        lambda *a, **k: walked.append(1) or real_canon(*a, **k))
    c._hash_arg_payload((rows,), {"opts": {"b": 1, "a": 2}})
    assert any(v is rows for v in unshared), "the list was not keyed on its own"
    assert len(walked) < 100, f"the general path walked the rows ({len(walked)} calls)"
