"""The share check takes a list of small records in one step, and finds what
the walk a record at a time finds.

``for p in points: total += p.x`` over 200,000 small objects ran in 16 ms and
took cash 640 ms: the loop variable ``p`` is an item of ``points``, so the
share check walked every record of it, one Python step each, twice per
statement, and then searched every object the collector tracks for ``p``'s
holders. Now the records that nothing but their list holds, told by
reference count at C speed, are left out of the walk (`_records_left`): no
count can find one shared, and nothing below one is walked. The rest are
walked as before. The check exists for correctness, so these tests build
many shapes at random and compare the verdict with the step off.
"""

from __future__ import annotations

import dataclasses
import random

import pytest

from cash.notebook import shared_objects
from cash.notebook.shared_objects import share_group, shared_names


@dataclasses.dataclass(eq=False)
class Rec:
    a: int
    b: float


@dataclasses.dataclass(eq=False)
class Holding:
    a: int
    items: list


@dataclasses.dataclass(eq=False, slots=True)
class Slotted:
    a: int


class _Getattr:
    def __init__(self, a):
        self.a = a

    def __getattr__(self, name):
        raise AttributeError(name)


Looked = dataclasses.dataclass(eq=False)(_Getattr)


def _namespace(seed: int):
    """``(namespace, names, elsewhere)``: a list of records, names bound to
    some of them, and holders that are no variable."""
    rng = random.Random(seed)
    kind = rng.choice(["flat", "flat", "mixed", "holding", "slotted", "looked"])
    n = rng.randint(32, 120)
    records: list = []
    for i in range(n):
        if kind == "holding" and rng.random() < 0.2:
            records.append(Holding(i, [i]))
        elif kind == "mixed" and rng.random() < 0.3:
            records.append(rng.choice([i, "s", 1.5, None]))
        elif kind == "slotted":
            records.append(Slotted(i))
        elif kind == "looked":
            records.append(Looked(i))
        else:
            records.append(Rec(i, i * 0.5))
    ns: dict = {"records": records}
    elsewhere: list = []
    objects = [r for r in records if not isinstance(r, (int, str, float, type(None)))]
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
        elif how < 0.9 and isinstance(target, Holding):
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


@pytest.mark.parametrize("seed", range(200))
def test_the_step_finds_the_anchors_the_walk_finds(seed, one_at_a_time):
    ns, names, loop, elsewhere = _namespace(seed)
    outputs = loop or names

    def verdict():
        anchors: dict = {}
        links: set = set()
        holders, shared = share_group(
            outputs, _roots(ns, outputs), ns, anchorable=lambda name: True, anchors=anchors, anchor_links=links
        )
        return sorted(holders), shared, sorted(anchors), sorted(links)

    fast = verdict()
    one_at_a_time()
    assert verdict() == fast
    assert elsewhere is not None


@pytest.mark.parametrize("seed", range(200))
def test_every_record_left_out_has_no_holder_but_its_list(seed):
    ns, names, _loop, _elsewhere = _namespace(seed)
    value_types = shared_objects.VALUE_TYPES + shared_objects.library_value_types()
    nodes = shared_objects._walk(_roots(ns, names), [ns], value_types)[0]
    holders: dict[int, int] = {}
    for value in ns.values():
        for item in value if isinstance(value, list) else [value]:
            holders[id(item)] = holders.get(id(item), 0) + 1
    for record in ns["records"]:
        if type(record) in (Rec, Holding, Slotted, Looked) and id(record) not in nodes:
            assert type(record) is Rec, "only a flat record of a plain class is left out"
            assert holders[id(record)] == 1


def test_a_loop_variable_left_on_a_record_finds_its_list():
    ns = {"points": [Rec(i, i * 0.5) for i in range(5000)]}
    ns["p"] = ns["points"][-1]

    holders, shared = share_group(["p"], {"p": ns["p"]}, ns)

    assert set(holders) == {"points"} and shared == set()


def test_a_record_held_by_a_container_no_name_holds_still_refuses():
    ns = {"points": [Rec(i, i * 0.5) for i in range(5000)]}
    ns["p"] = ns["points"][-1]
    registry = [ns["points"][5]]

    holders, shared = share_group(["points"], {"points": ns["points"]}, ns)

    assert holders == {} and shared == {"points"}
    assert registry


def test_a_long_list_of_records_is_not_walked_a_record_at_a_time(monkeypatch):
    ns = {"points": [Rec(i, i * 0.5) for i in range(20_000)]}
    asked = []
    real = shared_objects.children_of
    monkeypatch.setattr(shared_objects, "children_of", lambda value: asked.append(1) or real(value))

    assert shared_names({"points": ns["points"]}, (ns,)) == set()
    assert len(asked) < 100
