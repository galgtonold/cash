"""JSON-like data below an object is asked about as one tree, not walked.

A graph object's ``adj`` dict of 100,000 dicts was walked one dict at a time
by every check of the statement, though only a root of such data was read a
level at a time. Data below an object is now read the same way, counted from
what holds its containers beyond their parents. What must hold: the walk finds
the same objects shared as the walk that reads every container, on graphs
where some containers inside the data are held from outside too.
"""

from __future__ import annotations

import random
import time

import pytest

from cash.notebook import shared_objects as so

pytestmark = pytest.mark.core


class Box:
    """An object of the notebook's own, holding data in its attributes."""


def _graph(seed: int):
    """A Box holding JSON-like data, and a list holding some containers from
    inside that data (or the data itself) from outside."""
    rng = random.Random(seed)
    outside: list = []

    def data(depth: int):
        if depth == 0 or rng.random() < 0.2:
            return rng.choice([1, 2.5, "s", None, True])
        if rng.random() < 0.5:
            made = {k: data(depth - 1) for k in range(rng.choice([1, 3, 40]))}
        else:
            made = [data(depth - 1) for _ in range(rng.choice([1, 3, 40]))]
        if rng.random() < 0.05:
            outside.append(made)
        return made

    box = Box()
    for attr in "abc":
        setattr(box, attr, data(4))
    if rng.random() < 0.3:
        inner = Box()
        inner.data = data(3)
        box.inner = inner
    return box, outside


def _shared(group, fast):
    value_types = so.VALUE_TYPES + so.library_value_types()
    nodes, inbound, checked, owner, _breaks = so._walk(group, [], value_types, fast=fast)
    found = {(key, owner[key]) for key in so.excess_refs(nodes, inbound, checked)}
    del nodes, inbound, checked, owner
    return found


@pytest.mark.parametrize("seed", range(60))
def test_the_tree_read_finds_what_the_full_walk_finds(seed):
    box, outside = _graph(seed)
    assert _shared({"box": box}, fast=True) == _shared({"box": box}, fast=False)
    assert len(outside) >= 0  # held to the end: it is the outside holder


def test_a_container_inside_the_data_held_from_outside_is_found():
    box = Box()
    box.adj = {i: {i + 1: {"w": 1.0}} for i in range(1_000)}
    kept = box.adj[500]
    found = _shared({"box": box}, fast=True)
    assert (id(kept), "box") in found
    assert found == _shared({"box": box}, fast=False)


def test_data_held_only_by_its_parents_is_not_walked_a_dict_at_a_time():
    box = Box()
    box.adj = {i: {i + 1: {"w": 1.0}} for i in range(200_000)}
    value_types = so.VALUE_TYPES + so.library_value_types()
    start = time.perf_counter()
    nodes = so._walk({"box": box}, [], value_types, fast=True)[0]
    took = time.perf_counter() - start
    assert len(nodes) <= 3, len(nodes)  # the box and its adj dict, no dict below
    assert took < 2.0, took


def _parts(value, seed: int) -> list:
    """A few objects from inside *value*'s data, to stand as sources."""
    rng = random.Random(seed)
    found, stack = [], [value]
    while stack:
        obj = stack.pop()
        children = so.children_of(obj)
        if children:
            for child in children:
                if type(child) in (dict, list):
                    found.append(child)
                    stack.append(child)
    return rng.sample(found, min(3, len(found)))


@pytest.mark.parametrize("seed", range(40))
def test_holds_part_of_answers_as_the_full_walk_does(seed, monkeypatch):
    box, outside = _graph(seed)
    other, _ = _graph(seed + 1000)
    for sources in ([other], [*_parts(box, seed)], [*outside[:2]], [{"x": [1, 2]}]):
        read = so.holds_part_of(box, sources)
        with monkeypatch.context() as m:
            m.setattr(so, "_TREE_BELOW_MIN", 10**9)
            walked = so.holds_part_of(box, sources)
        assert read == walked


@pytest.mark.parametrize("seed", range(20))
def test_the_borrowers_found_are_those_the_full_walk_finds(seed, monkeypatch):
    np = pytest.importorskip("numpy")
    from cash.notebook import borrowed_memory as bm

    box, _outside = _graph(seed)
    base = np.arange(100)
    box.c = {"rows": [{"k": i} for i in range(50)], "view": base[2:5], "deep": [[base[1:3]] for _ in range(40)]}
    read = [id(obj) for obj, _family in bm._borrowers_in([box])]
    with monkeypatch.context() as m:
        m.setattr(bm, "_TREE_MIN", 10**9)
        walked = [id(obj) for obj, _family in bm._borrowers_in([box])]
    assert sorted(read) == sorted(walked)
    assert len(read) == 41
