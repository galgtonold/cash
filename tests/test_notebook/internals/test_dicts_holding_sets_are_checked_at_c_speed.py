"""The share check takes a list of dicts each holding a small set in one
step, and finds what the walk a record at a time finds.

A cell building 1.5 million ``{"i": i, "tags": {i % 7, i % 11}}`` records
had its share check walk every dict and every set one Python step at a time
until its time ran out, then search every object the collector tracks for
holders: about 11 s of a 2.4 s cell, ending in NOTEBOOK-SHARE-UNCHECKED. A
dict holding values alone was taken by reference count already; a set of
values that only its dict holds, once, can be shared no more than the dict
can, and a frozenset or tuple of values is no part the walk counts at all,
so a dict holding those is taken the same way. The check exists for
correctness, so these tests build many shapes at random and compare the
verdict with the step off.
"""

from __future__ import annotations

import random

import pytest

from cash.notebook import shared_objects
from cash.notebook.shared_objects import share_group, shared_names


def _record(kind: str, i: int, rng: random.Random):
    if kind == "set":
        return {"i": i, "tags": {i % 7, i % 11}}
    if kind == "frozen":
        return {"i": i, "tags": frozenset({i % 7}), "pair": (i, "a"), frozenset({i}): 1}
    if kind == "list":
        return [i, {i % 3}]
    if kind == "deeper":
        # A set holding a tuple, and a list in a set's place: walked.
        return {"i": i, "tags": {(i, "a")} if rng.random() < 0.5 else [i]}
    return {"i": i, "tags": {i}} if rng.random() < 0.7 else {"i": i, "tags": (i, [i])}


def _namespace(seed: int):
    """``(namespace, names, loop, elsewhere)``: a list of records, names
    bound to some of them or their sets, and holders that are no variable."""
    rng = random.Random(seed)
    kind = rng.choice(["set", "set", "frozen", "list", "deeper", "mixed"])
    records = [_record(kind, i, rng) for i in range(rng.randint(32, 120))]
    ns: dict = {"records": records}
    elsewhere: list = []
    for i in range(rng.randint(0, 5)):
        target = rng.choice(records)
        parts = target.values() if type(target) is dict else target
        inner = next((v for v in parts if type(v) in (set, list)), None)
        how = rng.random()
        if how < 0.2:
            ns[f"v{i}"] = target  # a name bound to a record: a loop variable
        elif how < 0.35 and inner is not None:
            ns[f"v{i}"] = inner  # a name bound to a record's set
        elif how < 0.45 and inner is not None and type(target) is dict:
            rng.choice([r for r in records if type(r) is dict])["again"] = inner  # one set in two places
        elif how < 0.55 and inner is not None:
            elsewhere.append(inner)  # a set held by no variable
        elif how < 0.65:
            elsewhere.append(target)
        elif how < 0.8:
            records.append(target)  # a record held twice in the list
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


def test_a_set_bound_to_a_name_is_found_holding_part_of_the_list():
    ns = {"records": [{"i": i, "tags": {i}} for i in range(5000)]}
    ns["tags"] = ns["records"][7]["tags"]

    holders, shared = share_group(["records"], {"records": ns["records"]}, ns)

    assert set(holders) == {"tags"} and shared == set()


def test_a_set_held_by_no_variable_still_refuses():
    ns = {"records": [{"i": i, "tags": {i}} for i in range(5000)]}
    registry = [ns["records"][7]["tags"]]

    holders, shared = share_group(["records"], {"records": ns["records"]}, ns)

    assert holders == {} and shared == {"records"}
    assert registry


def test_a_set_held_twice_by_its_dict_is_walked():
    ns = {"records": [{"i": i, "tags": {i}} for i in range(5000)]}
    ns["records"][7]["again"] = ns["records"][7]["tags"]

    assert shared_names({"records": ns["records"]}, (ns,)) == set()


def test_a_long_list_of_dicts_holding_sets_is_not_walked_a_record_at_a_time(monkeypatch):
    ns = {"records": [{"i": i, "tags": {i % 7, i % 11}} for i in range(20_000)]}
    asked = []
    real = shared_objects.children_of
    monkeypatch.setattr(shared_objects, "children_of", lambda value: asked.append(1) or real(value))

    assert shared_names({"records": ns["records"]}, (ns,)) == set()
    assert len(asked) < 100
