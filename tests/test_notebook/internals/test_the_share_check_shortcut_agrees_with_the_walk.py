"""The share check's shortcut for JSON-like data finds what the full walk finds.

A list of records some of whose containers another name holds too -- the
record a loop variable is left bound to, the list of pairs ``act`` from
``for (name, act) in parsed:`` -- was walked one container at a time on every
run: ``parsed = sorted(parsed, ...)`` took 10 s instead of 0.04 s. Such a root
is now read a level at a time, and only the containers held besides their
parent are taken on their own (`shared_objects._take_breaks`). The check
exists for correctness (a restored copy must never stand in for an object
another name holds), so the shortcut must reach the same verdict as the walk
for every shape: these tests build many shapes at random, with names, other
containers and repeats reaching into them at random depths, and compare the
two.
"""

from __future__ import annotations

import datetime
import random

import pytest

from cash.notebook import holder_patches as hp
from cash.notebook import shared_objects
from cash.notebook.shared_objects import share_group, shared_names


def _leaf(rng: random.Random):
    return rng.choice([1, 2.5, "s", None, True, b"b", datetime.datetime(2020, 1, 2), ("a", 1)])


def _tree(rng: random.Random, depth: int, containers: list):
    """A random JSON-like value; every container in it is appended to *containers*."""
    kind = rng.choice(["list", "dict", "tuple"]) if depth else "leaf"
    if kind == "leaf":
        return _leaf(rng)
    items = [_tree(rng, depth - 1, containers) if rng.random() < 0.6 else _leaf(rng) for _ in range(rng.randint(0, 4))]
    if kind == "list":
        value = items
    elif kind == "dict":
        value = {f"k{i}": item for i, item in enumerate(items)}
    else:
        value = tuple(items)
    containers.append(value)
    return value


def _namespace(seed: int):
    """``(namespace, outputs, elsewhere)``: records, the names reaching into
    them, and containers outside the namespace reaching into them."""
    rng = random.Random(seed)
    containers: list = []
    records = [_tree(rng, rng.randint(1, 4), containers) for _ in range(rng.randint(5, 40))]
    ns = {"records": records}
    elsewhere = []
    inner = [c for c in containers if c is not records]
    for i in range(rng.randint(0, 4)):
        if not inner:
            break
        target = rng.choice(inner)
        how = rng.random()
        if how < 0.4:
            ns[f"v{i}"] = target  # a name bound to a part
        elif how < 0.55:
            ns[f"v{i}"] = [target]  # a name holding a part through a container
        elif how < 0.7:
            elsewhere.append(target)  # a holder that is no variable
        elif how < 0.85:
            records.append(target)  # a part held twice in the records
        else:
            host = rng.choice([c for c in inner if type(c) in (list, dict)] or [records])
            if type(host) is list:
                host.append(target)  # a part held at another depth too
            else:
                host[f"x{i}"] = target
    names = ["records"] + [name for name in ns if name != "records" and rng.random() < 0.6]
    return ns, names, elsewhere


@pytest.fixture
def full_walk(monkeypatch):
    """Runs the share check with the shortcut off: every container walked."""
    real = shared_objects._walk

    def walk(*args, **kwargs):
        kwargs["fast"] = False
        return real(*args, **kwargs)

    def off():
        monkeypatch.setattr(shared_objects, "_walk", walk)

    return off


def _roots(ns: dict, names: list) -> dict:
    return {name: ns[name] for name in names}


@pytest.mark.parametrize("seed", range(300))
def test_the_shortcut_finds_the_names_the_walk_finds(seed, full_walk):
    ns, names, elsewhere = _namespace(seed)
    fast = shared_names(_roots(ns, names), (ns,))
    full_walk()
    full = shared_names(_roots(ns, names), (ns,))

    assert fast == full
    assert elsewhere is not None


@pytest.mark.parametrize("seed", range(300))
def test_the_shortcut_finds_the_holders_the_walk_finds(seed, full_walk):
    ns, names, elsewhere = _namespace(seed)
    captured = _roots(ns, names)
    fast = _verdict(names, captured, ns)
    full_walk()
    full = _verdict(names, captured, ns)

    assert fast == full
    assert elsewhere is not None


def _verdict(names: list, captured: dict, ns: dict) -> tuple[list, set]:
    """`share_group`'s answer by name: the holders' values it returns would
    be one more holder for the next check to count."""
    holders, shared = share_group(names, captured, ns)
    return sorted(holders), shared


@pytest.mark.parametrize("seed", range(200))
def test_the_shortcut_counts_every_node_as_the_walk_does(seed):
    """The counts behind the verdict: each object the shortcut keeps has the
    count the walk gives it, and each it leaves out has no reference but its
    parent's."""
    ns, names, elsewhere = _namespace(seed)
    value_types = shared_objects.VALUE_TYPES + shared_objects.library_value_types()
    roots = _roots(ns, names)
    fast = shared_objects._walk(roots, [ns], value_types, fast=True)
    full = shared_objects._walk(roots, [ns], value_types, fast=False)
    fast_nodes, fast_inbound, fast_checked = fast[0], fast[1], fast[2]
    full_nodes, full_inbound, full_checked = full[0], full[1], full[2]

    assert set(fast_nodes) <= set(full_nodes)
    assert {k: fast_inbound[k] for k in fast_nodes} == {k: full_inbound[k] for k in fast_nodes}
    assert set(fast_checked) <= set(full_checked)
    left_out = [k for k in full_checked if k not in fast_nodes]
    assert all(full_inbound[k] == 1 for k in left_out)
    assert elsewhere is not None


def test_a_loop_variable_left_on_a_record_is_counted_through_the_shortcut():
    ns = {"recs": [{"id": i, "tags": ["a"]} for i in range(100)]}
    ns["r"] = ns["recs"][-1]

    holders, shared = share_group(["r"], {"r": ns["r"]}, ns)

    assert set(holders) == {"recs"} and shared == set()


def test_a_record_held_by_a_container_no_name_holds_still_refuses():
    ns = {"recs": [{"id": i, "tags": ["a"]} for i in range(100)]}
    ns["r"] = ns["recs"][-1]
    registry = [ns["recs"][5]]

    holders, shared = share_group(["r"], {"r": ns["r"]}, ns)

    assert holders == {} and shared == {"r"}
    assert registry


def test_parts_of_a_part_another_name_holds_are_walked():
    """A container held besides its parent, inside another one: the
    shortcut would count the edge between them twice, so it walks."""
    ns = {"recs": [{"id": i, "tags": ["a"]} for i in range(50)]}
    ns["rec"] = ns["recs"][3]
    ns["tags"] = ns["recs"][3]["tags"]

    assert shared_names({"recs": ns["recs"], "rec": ns["rec"], "tags": ns["tags"]}, (ns,)) == set()
    assert shared_names({"recs": ns["recs"], "rec": ns["rec"]}, (ns,)) == {"recs"}


def test_shared_pairs_of_values_are_not_holders():
    """Tuples of values alone that something else holds too (the RAM tier's
    copy of parsed pairs shares them) have no identity to keep."""
    pairs = [("a", datetime.datetime(2020, 1, i + 1)) for i in range(20)]
    ns = {"parsed": [("u", pairs[:10]), ("v", pairs[10:])]}
    kept = list(pairs)

    assert shared_names({"parsed": ns["parsed"]}, (ns,)) == set()
    assert kept


@pytest.mark.parametrize("seed", range(200))
def test_holder_patches_read_a_level_at_a_time_match_the_walk(seed, monkeypatch):
    ns, names, _elsewhere = _namespace(seed)
    holders = {name: ns[name] for name in ns if name != "records"}
    outputs = {"records": ns["records"]}
    fast = hp.holder_patches(holders, outputs)
    monkeypatch.setattr(hp, "_tree_group", lambda *args: None)
    full = hp.holder_patches(holders, outputs)

    assert fast == full
    assert names
