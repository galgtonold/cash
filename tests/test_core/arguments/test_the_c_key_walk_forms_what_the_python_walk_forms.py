"""The C walk of plain data (``cplain.c``, `canonical_form._plain_walk`) forms
a key exactly as the Python walk does.

It reads exact dicts, lists and tuples and the arrays of numbers inside them;
every other part -- a subclass, a set, an object, an array of objects or
records, a masked array, a list that could be records -- goes back to the
Python walk. Both walks run here on many random structures and on the edge
cases one at a time, and must give the same canonical bytes, the same
``seen`` entries in the same order, the same errors, and the same argument
digests through a `Cash`'s hasher. Without the C extension (Python 3.10,
``CASH_PURE_PYTHON``) the Python walk keys as it always did.
"""

from __future__ import annotations

import collections
import random

import numpy as np
import pytest

from cash import Cash, _native, canonical_form
from cash.canonical_form import CyclicValueError, canonical_bytes, stable_key_repr
from cash.content_hashers import BUILTIN_CONTENT
from cash.decorator.arg_hashing import quiet_digests

needs_c = pytest.mark.skipif(
    getattr(_native.kernels, "PlainWalk", None) is None, reason=f"no C walk of plain data: {_native.REASON}"
)


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


@pytest.fixture
def python_only(monkeypatch):
    """Run the Python walk alone, as without the C extension."""

    def use():
        monkeypatch.setattr(canonical_form, "_PLAIN_WALK", None)

    return use


Point = collections.namedtuple("Point", "x y")


class Record:
    def __init__(self, n):
        self.n = n


class Text(str):
    pass


def _arrays(rng: random.Random) -> list:
    base = np.arange(24, dtype=np.float64).reshape(4, 6) * rng.random()
    return [
        base,
        np.asfortranarray(base),
        base[:, ::2],
        base.T[::2],
        base[::-1],
        np.array(rng.random()),
        np.zeros((0, 3)),
        np.zeros((3, 0), order="F"),
        np.zeros(0, dtype=np.int8),
        np.arange(5, dtype=">i4"),
        np.arange(6, dtype="<u2").reshape(2, 3),
        np.array([True, False]),
        np.arange(3, dtype=np.float16),
        np.arange(4, dtype=np.complex128),
        np.array(["ab", "c"]),
        np.array([b"x", b"yz"]),
        np.broadcast_to(np.arange(3.0), (4, 3)),
        np.arange(3).astype("datetime64[s]"),
        np.zeros(2, dtype=[("a", "<f8"), ("b", "u1")]),
        np.array([1, "a", None], dtype=object),
        np.array([[1, 2], [3, 4]], dtype=object),
        np.ma.masked_array([1.0, 2.0], mask=[0, 1]),
        np.zeros(3, dtype=("f8", (2,))),
    ]


def _leaf(rng: random.Random, pool: list) -> object:
    choice = rng.randrange(14)
    if choice == 0:
        return rng.randrange(-5, 5)
    if choice == 1:
        return rng.choice(["a", "bb", "", "é"])
    if choice == 2:
        return rng.random()
    if choice == 3:
        return rng.choice([True, False, None])
    if choice == 4:
        return b"by"
    if choice == 5:
        return complex(1, rng.randrange(3))
    if choice == 6:
        return rng.choice(pool)  # held twice: an alias
    if choice == 7:
        return rng.choice([{1, 2}, frozenset({"a"}), bytearray(b"q"), Point(1, 2), Text("t"), np.float64(2.5)])
    if choice == 8:
        return Record(rng.randrange(3))
    return rng.choice(_arrays(rng))


def _tree(rng: random.Random, depth: int, pool: list) -> object:
    if depth == 0 or rng.random() < 0.25:
        return _leaf(rng, pool)
    kind = rng.randrange(4)
    width = rng.choice([0, 1, 3, 8, 12])
    parts = [_tree(rng, depth - 1, pool) for _ in range(width)]
    if kind == 0:
        node: object = {(rng.choice(["k", 1, (1, 2), 2.5]), i): p for i, p in enumerate(parts)}
    elif kind == 1:
        node = list(parts)
    elif kind == 2:
        node = tuple(parts)
    else:
        node = {f"k{i}": p for i, p in enumerate(parts)}
    if type(node) is not tuple:
        pool.append(node)
    return node


def _both(value, monkeypatch, *, content=BUILTIN_CONTENT, hook=None):
    """``(bytes, seen numbering)`` by the C walk, then by the Python walk."""
    out = []
    for fast in (True, False):
        if not fast:
            monkeypatch.setattr(canonical_form, "_PLAIN_WALK", None)
        seen: dict = {}
        try:
            got = canonical_bytes(value, content, hook=hook, seen=seen)
        except (CyclicValueError, TypeError) as exc:
            got = (type(exc), str(exc))
        out.append((got, [(oid, met[0]) for oid, met in seen.items()]))
    monkeypatch.undo()
    return out


@needs_c
@pytest.mark.parametrize("seed", range(60))
def test_random_nested_structures_key_alike_in_both_walks(seed, monkeypatch):
    rng = random.Random(seed)
    pool: list = [[1, 2], {"z": np.arange(3)}, np.ones((2, 2))]
    value = {"bags": [_tree(rng, 4, pool) for _ in range(4)], "shared": pool[:3]}
    fast, slow = _both(value, monkeypatch)
    assert fast == slow


@needs_c
@pytest.mark.parametrize("index", range(len(_arrays(random.Random(0)))))
def test_each_kind_of_array_keys_alike_in_both_walks(index, monkeypatch):
    array = _arrays(random.Random(1))[index]
    for value in (array, [array], {"a": array, "b": [array]}, (array, array)):
        fast, slow = _both(value, monkeypatch)
        assert fast == slow


@needs_c
def test_arg_digests_through_the_hasher_match(c, monkeypatch):
    rng = random.Random(7)
    pool: list = [[0], {"z": np.arange(2)}]
    value = [_tree(rng, 4, pool) for _ in range(6)]
    fast = c._args.hash_payload((value,), {"k": np.arange(4)})
    with quiet_digests():
        quiet_fast = [c._args.hash_payload((value,), {"k": np.arange(4)}) for _ in range(2)]
    monkeypatch.setattr(canonical_form, "_PLAIN_WALK", None)
    slow = c._args.hash_payload((value,), {"k": np.arange(4)})
    assert fast == slow
    assert quiet_fast == [slow, slow]


@needs_c
def test_the_c_walk_reads_only_plain_parts(c, monkeypatch):
    """A list of records, numpy numbers or namedtuples is read by the Python
    walk whole, and a registered hasher keeps the C walk off."""
    calls = []
    real = canonical_form._walk_below

    def counted(value, walk):
        calls.append(type(value))
        return real(value, walk)

    monkeypatch.setattr(canonical_form, "_walk_below", counted)
    canonical_form._PLAIN_SET_UP.clear()
    try:
        value = {"r": [Record(i) for i in range(9)], "n": [np.float64(i) for i in range(9)], "a": [np.arange(2)]}
        stable_key_repr(value, BUILTIN_CONTENT)
        assert calls == [list, list]  # the arrays' list was read in C
        assert c._args._content.plain_arrays is not None
        c.register_hasher(Record, lambda r: r.n)
        walk = canonical_form._Walk(c._args._content, {}, c._args._nested_hasher, None)
        assert walk.fast is None
    finally:
        canonical_form._PLAIN_SET_UP.clear()


@needs_c
def test_in_a_quiet_window_the_c_walk_reads_each_array_once(c, monkeypatch):
    """Two hashes of one argument in a `quiet_digests` window read each array
    once; outside one, every hash reads every array."""
    import hashlib
    import types

    read = []

    def counting(data):
        read.append(1)
        return hashlib.sha256(data)

    monkeypatch.setattr(canonical_form, "hashlib", types.SimpleNamespace(sha256=counting))
    canonical_form._PLAIN_SET_UP.clear()
    try:
        value = [{"v": np.arange(i, i + 3), "m": np.ones((2, 2))} for i in range(20)]
        with quiet_digests():
            first = c._args.hash_payload((value,), {})
            second = c._args.hash_payload((value,), {})
        assert first == second
        assert len(read) == 40
        c._args.hash_payload((value,), {})
        assert len(read) == 80
    finally:
        canonical_form._PLAIN_SET_UP.clear()


@needs_c
def test_a_container_that_holds_itself_is_refused_alike(monkeypatch):
    inner: list = [np.arange(2)]
    outer = {"x": (1, inner)}
    inner.append(outer)
    fast, slow = _both(outer, monkeypatch)
    assert fast == slow
    assert fast[0][0] is CyclicValueError


@needs_c
def test_a_very_deep_value_keys_alike(monkeypatch):
    value: object = np.arange(3)
    for i in range(3000):
        value = [value, i] if i % 2 else {"d": value}
    fast, slow = _both(value, monkeypatch)
    assert fast == slow


@pytest.mark.parametrize("pure", [False, True])
def test_a_changed_value_deep_inside_misses(c, python_only, pure):
    if pure:
        python_only()
    calls = []

    @c.cache
    def total(bags):
        calls.append(1)
        return float(sum(b["values"].sum() for b in bags["data_bags"]))

    bags = {"data_bags": [{"values": np.ones((3, 2)), "mask": np.arange(2)} for _ in range(50)]}
    assert total(bags) == 300.0
    assert total(bags) == 300.0
    assert len(calls) == 1
    bags["data_bags"][37]["values"][1, 1] = 2.0
    assert total(bags) == 301.0
    assert len(calls) == 2
    bags["data_bags"][12]["mask"] = np.asfortranarray(np.arange(2))  # same values, a new array: hit
    assert total(bags) == 301.0
    bags["data_bags"][12]["values"] = np.asfortranarray(bags["data_bags"][12]["values"])
    total(bags)  # F-ordered now: keyed apart
    assert len(calls) == 3


def test_without_the_c_walk_the_python_walk_keys(c, python_only):
    python_only()
    walk = canonical_form._Walk(c._args._content, {}, c._args._nested_hasher, None)
    assert walk.fast is None
    value = {"a": [np.arange(3), (1, "x")], "b": {"c": np.zeros((2, 2), order="F")}}
    assert c._args.hash_payload((value,), {}) == c._args.hash_payload((value,), {})
