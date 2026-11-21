"""The share check takes a list of ``SimpleNamespace`` records in one step,
as it takes records of the notebook's own classes, and finds what the walk a
record at a time finds.

A cell building 1.5 million ``SimpleNamespace(i=i, x=..., name=...)``
records had its share check walk every record one Python step at a time
until its time ran out, then search every object the collector tracks for
holders: about 8 s of a 1.5 s cell, ending in NOTEBOOK-SHARE-UNCHECKED.
``SimpleNamespace`` is a C class, which the step left out, but it keeps its
attributes in the ``__dict__`` Python reads, as a class of the notebook's
does. The check exists for correctness, so these tests build many shapes at
random and compare the verdict with the step off.
"""

from __future__ import annotations

import random
import types

import pytest

from cash.notebook import shared_objects
from cash.notebook.shared_objects import share_group, shared_names


class Sub(types.SimpleNamespace):
    pass


def _namespace(seed: int):
    """``(namespace, names, loop, elsewhere)``: a list of records, names
    bound to some of them or their parts, and holders that are no variable."""
    rng = random.Random(seed)
    kind = rng.choice(["flat", "flat", "holding", "sub", "mixed"])
    records: list = []
    for i in range(rng.randint(32, 120)):
        if kind == "holding" and rng.random() < 0.2:
            records.append(types.SimpleNamespace(i=i, items=[i]))
        elif kind == "sub":
            records.append(Sub(i=i, x=i * 0.5))
        elif kind == "mixed" and rng.random() < 0.3:
            records.append(rng.choice([i, "s", 1.5, None]))
        else:
            records.append(types.SimpleNamespace(i=i, x=i * 0.5, name=f"n{i}"))
    ns: dict = {"records": records}
    elsewhere: list = []
    objects = [r for r in records if isinstance(r, types.SimpleNamespace)]
    for i in range(rng.randint(0, 5)):
        target = rng.choice(objects)
        how = rng.random()
        if how < 0.3:
            ns[f"v{i}"] = target  # a name bound to a record: a loop variable
        elif how < 0.45:
            ns[f"v{i}"] = [target]  # a name holding one through a container
        elif how < 0.6:
            elsewhere.append(target)  # a holder that is no variable
        elif how < 0.75:
            records.append(target)  # a record held twice in the list
        elif how < 0.9 and hasattr(target, "items"):
            ns[f"v{i}"] = target.items  # a part of a record
        else:
            ns[f"w{i}"] = records[rng.randrange(len(records)) :]  # another list of the same records
    names = ["records"] + [name for name in ns if name != "records" and rng.random() < 0.5]
    loop = [name for name in ns if name.startswith("v") and name not in names]
    return ns, names, loop, elsewhere


@pytest.fixture
def one_at_a_time(monkeypatch):
    def off():
        monkeypatch.setattr(shared_objects, "_FLAT_RECORDS_MIN", 10**9)

    return off


def _roots(ns: dict, names: list) -> dict:
    return {name: ns[name] for name in names}


@pytest.mark.parametrize("seed", range(300))
def test_the_step_finds_the_names_the_walk_finds(seed, one_at_a_time):
    ns, names, _loop, elsewhere = _namespace(seed)
    fast = shared_names(_roots(ns, names), (ns,))
    one_at_a_time()
    assert shared_names(_roots(ns, names), (ns,)) == fast
    assert elsewhere is not None


@pytest.mark.parametrize("seed", range(300))
def test_the_step_finds_the_holders_the_walk_finds(seed, one_at_a_time):
    ns, names, loop, elsewhere = _namespace(seed)
    outputs = loop or names
    captured = _roots(ns, outputs)
    holders, shared = share_group(outputs, captured, ns)
    fast = (sorted(holders), shared)
    del holders
    one_at_a_time()
    holders, shared = share_group(outputs, captured, ns)
    assert (sorted(holders), shared) == fast
    assert elsewhere is not None


def test_a_loop_variable_left_on_a_record_finds_its_list():
    ns = {"records": [types.SimpleNamespace(i=i, x=i * 0.5) for i in range(5000)]}
    ns["r"] = ns["records"][-1]

    holders, shared = share_group(["r"], {"r": ns["r"]}, ns)

    assert set(holders) == {"records"} and shared == set()


def test_a_record_held_by_a_container_no_name_holds_still_refuses():
    ns = {"records": [types.SimpleNamespace(i=i, x=i * 0.5) for i in range(5000)]}
    registry = [ns["records"][5]]

    holders, shared = share_group(["records"], {"records": ns["records"]}, ns)

    assert holders == {} and shared == {"records"}
    assert registry


def test_a_long_list_of_namespaces_is_not_walked_a_record_at_a_time(monkeypatch):
    ns = {"records": [types.SimpleNamespace(i=i, x=i * 0.5, name=f"n{i}") for i in range(20_000)]}
    asked = []
    real = shared_objects.children_of
    monkeypatch.setattr(shared_objects, "children_of", lambda value: asked.append(1) or real(value))

    assert shared_names({"records": ns["records"]}, (ns,)) == set()
    assert len(asked) < 100
