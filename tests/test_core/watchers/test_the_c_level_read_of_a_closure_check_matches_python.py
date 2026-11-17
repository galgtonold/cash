"""The closure check's level read in C (``fresh_items`` into a ``PlaceSet``)
gives exactly what its Python version (`_fresh_in_python` into a
`_KeptSet`) gives: the same objects in the same order, the same places,
the clock read after the same chunks, and a part that raises losing the
same chunk. Both run on random levels of leaves, records, containers,
objects of a class whose metaclass hashes its own way, and objects met
before.
"""

from __future__ import annotations

import gc
import random
import weakref

import pytest

from cash import _native, closure_sharing
from cash.closure_sharing import _PLACE, ClosureCheck, closures_sharing

K = _native.kernels

pytestmark = pytest.mark.skipif(
    K is None or not hasattr(K, "fresh_items"), reason=f"no C walk kernels: {_native.REASON}"
)


class Meta(type):
    def __hash__(cls):
        return 7

    def __eq__(cls, other):
        return cls is other


class Odd(metaclass=Meta):
    pass


class Thing:
    pass


class Boom(Exception):
    pass


class Clock:
    """A limit whose time is up at its *at*-th read; counts its reads."""

    def __init__(self, at: int | None = None):
        self.at = at
        self.reads = 0

    def passed(self) -> bool:
        self.reads += 1
        return self.at is not None and self.reads >= self.at


def _pool(rng: random.Random) -> list:
    shared = [Thing() for _ in range(6)] + [[1], {"a": 1}, Odd(), (Thing(),)]
    pool: list = []
    for _ in range(rng.randrange(0, 60)):
        pick = rng.randrange(12)
        if pick == 0:
            pool.append(rng.choice([1, 2.5, "s", b"b", None, True, 10**30]))
        elif pick == 1:
            pool.append(tuple(rng.choice([1, "x", None, 2.0]) for _ in range(rng.randrange(0, 4))))
        elif pick == 2:
            pool.append((1, Thing()) if rng.random() < 0.5 else ("a", [2]))
        elif pick == 3:
            pool.append([rng.randrange(5)])
        elif pick == 4:
            pool.append({rng.randrange(5): Thing()})
        elif pick == 5:
            pool.append(Odd())
        elif pick == 6:
            pool.append(frozenset({1, 2}) if rng.random() < 0.5 else {3})
        else:
            pool.append(rng.choice(shared))
    return pool


def _held(rng: random.Random, pool: list, raise_at: int | None):
    """A recipe for a level's parts, made afresh for each side."""
    cuts = sorted(rng.randrange(len(pool) + 1) for _ in range(rng.randrange(0, 4)))
    pieces = [pool[a:b] for a, b in zip([0, *cuts], [*cuts, len(pool)])]
    dicts = [{rng.randrange(9): rng.choice(pool) if pool else 1 for _ in range(rng.randrange(0, 3))} for _ in range(3)]
    forms = [rng.randrange(5) for _ in pieces]
    raise_in = rng.randrange(len(pieces) + 5)

    def make():
        held: list = []
        for piece, form in zip(pieces, forms):
            if form == 0:
                held.append(list(piece))
            elif form == 1:
                held.append(tuple(piece))
            elif form == 2:
                held.append(iter(piece))
            else:
                held.append(x for x in piece)
        held.append(closure_sharing._Dicts((dicts, False)))
        held.append(closure_sharing._Dicts((dicts, True)))
        held.append(dicts[0])
        held.append(dicts[1].values())
        if raise_at is not None:
            held.insert(raise_in, _raising(pool, raise_at))
        return held

    return make


def _raising(pool: list, at: int):
    yield from pool[:at]
    raise Boom


def _read(make, places, before: list, chunk: int, clock_at: int | None, monkeypatch):
    monkeypatch.setattr(closure_sharing, "_CHUNK", chunk)
    _put(places, before)
    clock = Clock(clock_at)
    try:
        out = closure_sharing._fresh(make(), places, clock)
    except Boom:
        out = Boom
    return out, clock.reads


def _put(places, objects: list) -> None:
    """Put *objects* in *places* as an earlier level would have."""
    closure_sharing._fresh([objects], places, Clock())


def _keys(places) -> set:
    return set(places.keys()) if isinstance(places, K.PlaceSet) else set(places)


@pytest.mark.parametrize("seed", range(300))
def test_a_level_reads_the_same_in_c_and_in_python(seed, monkeypatch):
    rng = random.Random(seed)
    pool = _pool(rng)
    before = rng.sample(pool, rng.randrange(len(pool) + 1)) if pool else []
    raise_at = rng.randrange(len(pool) + 1) if rng.random() < 0.3 else None
    make = _held(rng, pool, raise_at)
    chunk = rng.choice([1, 2, 3, 5, 8, 1 << 20])
    clock_at = rng.choice([None, None, 1, 2, 3])
    c_places, py_places = K.PlaceSet(), closure_sharing._KeptSet()
    c_out, c_reads = _read(make, c_places, before, chunk, clock_at, monkeypatch)
    py_out, py_reads = _read(make, py_places, before, chunk, clock_at, monkeypatch)
    assert type(c_out) is type(py_out)
    if isinstance(py_out, list):
        assert len(c_out) == len(py_out)
        assert all(a is b for a, b in zip(c_out, py_out))
    else:
        assert c_out is py_out  # None (time up) or Boom
    assert c_reads == py_reads
    assert _keys(c_places) == _keys(py_places)
    assert len(c_places) == len(py_places)
    for item in pool:
        assert (_PLACE(item) in c_places) == (_PLACE(item) in py_places)


def test_a_place_set_holds_what_it_met_and_lets_it_go_with_it():
    places = K.PlaceSet()
    thing = Thing()
    gone = weakref.ref(thing)
    assert closure_sharing._fresh([[thing, thing]], places, Clock()) == [thing]
    assert _PLACE(thing) in places and len(places) == 1
    assert "a key" not in places and 2**200 not in places and -1 not in places
    del thing
    gc.collect()
    assert gone() is not None  # held: its address stands for it alone
    del places
    gc.collect()
    assert gone() is None


def test_a_place_set_grows_past_many_objects():
    places = K.PlaceSet()
    things = [Thing() for _ in range(50_000)]
    assert closure_sharing._fresh([things, things[::-1]], places, Clock()) == things
    assert len(places) == len(things)
    assert all(_PLACE(t) in places for t in things)
    assert _PLACE(Thing()) not in places


@pytest.mark.parametrize("seed", range(40))
def test_the_closure_check_answers_alike_with_and_without_the_kernel(seed, monkeypatch):
    rng = random.Random(seed)
    box = {"a": [Thing() for _ in range(rng.randrange(1, 5))]}
    inner = rng.choice(box["a"])
    stray = Thing()
    outputs = {
        "result": (box, [Thing(), (1, 2), (inner,)], lambda: inner if rng.random() < 0.5 else stray),
        "other": {"k": (lambda: box) if rng.random() < 0.5 else 1},
    }
    with_kernel = closures_sharing(outputs, 60.0)
    monkeypatch.setattr(closure_sharing, "_FRESH_ITEMS", None)
    assert isinstance(closure_sharing._places(), closure_sharing._KeptSet)
    without = closures_sharing(outputs, 60.0)
    assert isinstance(with_kernel, ClosureCheck)
    assert with_kernel == without
