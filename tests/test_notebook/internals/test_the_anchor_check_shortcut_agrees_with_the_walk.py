"""The anchor check reads JSON-like outputs a level at a time and still takes
back the anchors the whole walk takes back.

A variable found holding part of a statement's outputs can be bound again to
its live object on a restore instead of copied (an anchor), but not when its
object holds an object of the outputs: the live anchor would hold the old
object, not the restored copy (`shared_objects._take_back_holding`). That
was a walk over every container of the outputs for each such round:
``general_log.append(create_row(...))`` over 1.5 million parsed lines spent
5 s of the loop's 23 s there. The containers only their parent holds are now
left out (`shared_objects._group_nodes`): a walk from an anchor reaches one
only through its parent. These tests build outputs and holders at random --
names bound to parts, to the whole, to containers around parts, parts held
twice, cycles -- and ask `share_group` both ways for the same answer.
"""

from __future__ import annotations

import datetime
import random

import pytest

from cash.notebook import shared_objects
from cash.notebook.shared_objects import share_group


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


def _namespace(seed: int) -> tuple[dict, list[str], list]:
    """``(namespace, outputs, elsewhere)``: records and other outputs, names
    holding parts of them in every way a notebook does, and containers no
    name holds that reach into them."""
    rng = random.Random(seed)
    containers: list = []
    records = [_tree(rng, rng.randint(1, 4), containers) for _ in range(rng.randint(3, 30))]
    ns: dict = {"records": records}
    elsewhere: list = []
    mutable = [c for c in containers if type(c) in (list, dict)]
    if rng.random() < 0.5:
        ns["other"] = _tree(rng, 3, containers) if rng.random() < 0.7 else [1, 2, 3]
    for i in range(rng.randint(1, 6)):
        part = rng.choice(containers) if containers else records
        how = rng.random()
        if how < 0.2:
            ns[f"v{i}"] = part  # a loop variable left on a part
        elif how < 0.35:
            ns[f"v{i}"] = [part, _leaf(rng)]  # a container around a part
        elif how < 0.45:
            ns[f"v{i}"] = {"all": records}  # a container around the whole
        elif how < 0.55:
            ns[f"v{i}"] = records  # an alias of an output
        elif how < 0.65:
            records.append(part)  # a part held twice
        elif how < 0.75 and mutable:
            host = rng.choice(mutable)
            if type(host) is list:
                host.append(rng.choice([records, part]))  # a cycle, or a part at another depth
            else:
                host[f"x{i}"] = rng.choice([records, part])
        elif how < 0.8:
            elsewhere.append(part)  # a holder that is no variable
        elif how < 0.9:
            ns[f"v{i}"] = part  # a part and, held by another name, a part of it
            below = [c for c in containers if c is not part and type(part) in (list, tuple) and c in part]
            below += [c for c in containers if type(part) is dict and any(c is v for v in part.values())]
            if below:
                ns[f"w{i}"] = [rng.choice(below)]
        else:
            ns[f"v{i}"] = _tree(rng, 2, [])  # a variable sharing nothing
    names = ["records"] + [name for name in ns if name != "records" and rng.random() < 0.3]
    return ns, names, elsewhere


def _verdict(names: list[str], ns: dict, anchorable) -> tuple:
    """`share_group`'s whole answer by name and identity: holders, shared
    names, anchors, and the variables linked through anchors."""
    anchors: dict = {}
    links: set[str] = set()
    holders, shared = share_group(
        names, {name: ns[name] for name in names}, ns, anchorable=anchorable, anchors=anchors, anchor_links=links
    )
    return sorted(holders), shared, sorted((name, id(obj)) for name, obj in anchors.items()), links


def _anchorable(seed: int):
    rng = random.Random(seed * 7 + 1)
    choice = rng.random()
    if choice < 0.6:
        return lambda name: True
    taken = {f"{c}{i}" for c in "vw" for i in range(6) if rng.random() < 0.5} | {"other"}
    return lambda name: name in taken


@pytest.fixture
def whole_walk(monkeypatch):
    """The anchor check as it was: every container of the outputs walked."""
    real = shared_objects._group_nodes

    def off():
        monkeypatch.setattr(
            shared_objects, "_group_nodes", lambda group, types, leaves, memo: real(group, types, leaves, None)
        )

    return off


@pytest.mark.parametrize("seed", range(600))
def test_the_level_read_takes_back_the_anchors_the_walk_takes_back(seed, whole_walk):
    ns, names, elsewhere = _namespace(seed)
    fast = _verdict(names, ns, _anchorable(seed))
    whole_walk()
    whole = _verdict(names, ns, _anchorable(seed))

    assert fast == whole
    assert elsewhere is not None


@pytest.mark.timeout(120)
def test_the_generated_shapes_take_the_level_read(monkeypatch):
    """The shapes above reach the anchor check, and most take the level read
    rather than falling back to the walk: the comparison is not empty.

    Most of its time is `gc.get_referrers` (a holder that is no variable is
    found by a scan of the heap, one per level): about 140 scans over the
    first 300 shapes, each in proportion to everything the process holds.
    Alone that is 3 s; in a test worker that has run thousands of tests, on
    a loaded machine, it passed 30 s. So it reads half the shapes (208
    asked, 93 by level, twice what it asks), and has a longer limit."""
    real = shared_objects._group_nodes
    calls = {"asked": 0, "level": 0}

    def counting(group, types, leaves, memo):
        nodes = real(group, types, leaves, memo)
        if memo is not None:
            calls["asked"] += 1
            calls["level"] += len(nodes) < len(real(group, types, leaves, None))
        return nodes

    monkeypatch.setattr(shared_objects, "_group_nodes", counting)
    for seed in range(300):
        ns, names, _elsewhere = _namespace(seed)
        _verdict(names, ns, _anchorable(seed))

    assert calls["asked"] >= 100 and calls["level"] >= 50, calls


def _records(n: int) -> list[dict]:
    return [{"date": "2020-01-02", "level": "INFO", "took": [i, 2.5], "tags": {"a": i}} for i in range(n)]


def test_an_anchor_holding_a_record_is_taken_back():
    """``last = {"row": general_log[-1]}``: bound again live, it would hold
    the old record, not the restored list's."""
    ns = {"log": _records(200)}
    ns["last"] = {"row": ns["log"][-1]["tags"]}

    assert _verdict(["log"], ns, lambda name: True) == (["last"], set(), [], set())


def test_an_anchor_bound_to_an_output_itself_is_taken_back():
    """An alias of the output is a root of the group: what only it holds is
    left out of the level read, so it is asked of the whole walk."""
    ns = {"log": _records(200)}
    ns["alias"] = ns["log"]

    assert _verdict(["log"], ns, lambda name: True) == (["alias"], set(), [], set())


def test_an_anchor_beside_the_records_stays_an_anchor():
    """A record left in a loop variable that holds nothing of the outputs'
    stays an anchor: bound again to the live record."""
    ns = {"log": _records(200), "rows": [1, 2]}
    ns["r"] = ns["log"][7]

    whole = _verdict(["rows", "r"], ns, lambda name: True)

    assert whole[1] == set()


def test_an_anchor_holding_a_part_of_another_anchor_stays_an_anchor():
    """``rec = log[3]`` and ``tags = [rec["tags"]]``, both anchors: bound
    again live, ``tags`` holds what the live ``rec`` holds, not an object of
    the restored ``log``. The level read must not walk below ``rec``, which
    the whole walk stops at."""
    ns = {"log": _records(200)}
    ns["rec"] = ns["log"][3]
    ns["tags"] = [ns["rec"]["tags"]]

    assert _verdict(["log"], ns, lambda name: True) == ([], set(), [("rec", id(ns["rec"]))], {"rec", "tags"})
