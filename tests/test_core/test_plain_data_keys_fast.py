"""A big plain-data argument is keyed without walking it in Python -- to the same key.

A warm hit on a function receiving two million parsed rows took 8.4 s against a
0.04 s body: three Python-level walks over every element (`_contains_set`,
`_canonicalize_dict_order`, `_iter_code_carriers`) that can find nothing in
lists and tuples of primitives. That is now proven at C speed and the walks
are skipped. The key must not move: an entry written before is still found,
so every shape here is keyed both ways and compared.
"""
from __future__ import annotations

import time

import pytest

import cash.core as core
from cash import Cash

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


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_key_is_the_one_the_walks_gave(tmp_path, monkeypatch, shape):
    c = Cash(cache_dir=str(tmp_path / "cache"))
    args, kwargs = _args(shape)
    fast = c._hash_arg_payload(args, kwargs)
    monkeypatch.setattr(core, "_plain_payload_values", lambda values: None)
    walked = c._hash_arg_payload(args, kwargs)
    assert fast == walked, f"{shape}: the key moved"


def test_plain_data_is_recognised_and_other_data_is_not():
    rows = [(i, str(i)) for i in range(10)]
    assert core._plain_container_ids(rows) is not None
    assert core._plain_container_ids([[1, [2, [3]]]]) is not None
    assert core._plain_container_ids([{"a": 1}]) is None
    assert core._plain_container_ids([{1}]) is None
    assert core._plain_container_ids([object()]) is None
    cyclic: list = [1]
    cyclic.append(cyclic)
    assert core._plain_container_ids(cyclic) is None


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
        "bytearray leaves"}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_which_shapes_take_the_fast_path(tmp_path, monkeypatch, shape):
    """The comparison above is only worth something if the fast path ran for
    the plain shapes, and fell back for a shared object, a dict, a set, a
    subclass -- where keeping the object would change the bytes."""
    c = Cash(cache_dir=str(tmp_path / "cache"))
    seen = []
    real = core._plain_payload_values
    monkeypatch.setattr(core, "_plain_payload_values",
                        lambda values: seen.append(real(values)) or seen[-1])
    c._hash_arg_payload(*_args(shape))
    assert (seen[-1] is not None) == (shape in FAST), shape
