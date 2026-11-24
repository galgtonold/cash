"""A hit of nested plain dicts hands back the copy the last hit handed
out, under the rules a copy of an object read out of pickle bytes is.

A dictionary trie built of nested dicts is kept in the RAM tier as marshal
bytes (`_Marshalled`), and every hit read a new copy out of them: 70 ms a
hit for 450,000 dicts, where ``functools.lru_cache`` hands back the one
object. The copy a hit hands out is now kept, and the next hit hands that
very object back while the entry is the same, nothing outside holds any
part of it, and the watchers prove nothing in it changed. Anything else --
a copy changed at the top or deep down, held, or held through an inner
dict, or a new entry stored -- gets a copy equal to what was stored.
"""

from __future__ import annotations

import copy
import random

import pytest

from cash import _plain_data, value_watch
from cash.backends import memory_backend
from cash.backends.memory_backend import InMemoryBackend

pytestmark = pytest.mark.skipif(
    not value_watch.enabled(),
    reason="a copy is handed back only where the C extension's watchers prove it unchanged",
)


def trie(n=3000, seed=5):
    rnd = random.Random(seed)
    root: dict = {}
    for _ in range(n):
        node = root
        for ch in "".join(rnd.choice("abcdefgh") for _ in range(rnd.randint(3, 9))):
            node = node.setdefault(ch, {})
        node["$"] = True
    return root


PLAIN = trie()


@pytest.fixture
def loads(monkeypatch):
    """Copies read out of marshal bytes."""
    counted: list = []
    real = _plain_data.marshal_loads

    def counting(data):
        counted.append(1)
        return real(data)

    monkeypatch.setattr(_plain_data, "marshal_loads", counting)
    return counted


@pytest.fixture
def backend():
    backend = InMemoryBackend()
    backend.set("k", trie(), {"size": 100_000})
    assert type(backend._store["k"][1]) is memory_backend._Marshalled, "kept as marshal bytes"
    _meta, first = backend.get("k")  # read out, and kept for the next hit
    assert first == PLAIN
    del first
    return backend


def test_an_unchanged_copy_nobody_holds_is_handed_back(backend, loads):
    _meta, second = backend.get("k")
    second_id = id(second)
    del second
    _meta, third = backend.get("k")
    assert loads == [], "no hit read a new copy"
    assert id(third) == second_id
    assert third == PLAIN


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda t: t.pop("a"), id="a-top-key-removed"),
        pytest.param(lambda t: t["a"]["b"].clear(), id="a-deep-dict-cleared"),
        pytest.param(lambda t: t["c"]["d"].update(z={}), id="a-node-added"),
    ],
)
def test_a_copy_its_caller_changed_is_not_handed_back(backend, loads, change):
    _meta, first = backend.get("k")
    change(first)
    del first
    _meta, second = backend.get("k")
    assert len(loads) == 1, "a new copy was read"
    assert second == PLAIN


def test_a_copy_its_caller_holds_is_not_handed_back(backend):
    _meta, first = backend.get("k")
    _meta, second = backend.get("k")
    assert second is not first
    second.clear()
    assert first == PLAIN, "each caller's copy is its own"


def test_a_copy_held_through_an_inner_dict_is_not_handed_back(backend):
    _meta, first = backend.get("k")
    inner = first["a"]
    kept = copy.deepcopy(inner)
    del first
    _meta, second = backend.get("k")
    assert second["a"] is not inner
    second["a"].clear()
    assert inner == kept
    assert second.keys() == PLAIN.keys()


def test_a_new_entry_under_the_key_gets_its_own_copy(backend):
    other = trie(2500, seed=9)
    backend.set("k", copy.deepcopy(other), {"size": 100_000})
    _meta, got = backend.get("k")
    assert got == other
    del got
    _meta, again = backend.get("k")
    assert again == other
