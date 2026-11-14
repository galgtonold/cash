"""A set read back from the cache iterates in the order the stored one did.

Unpickled or copied, a set is rebuilt from its elements, and many of them land
in other slots of its table than in the set that was stored -- whose order is
the one the user's run saw. The store records where they sat (`set_order`),
in the entry's metadata, and a read puts them back: from the RAM tier, from
disk after a restart, and for sets inside lists, tuples, dicts and other sets,
and inside what holds them otherwise: subclasses of those (a defaultdict, an
OrderedDict, a namedtuple) and objects' attributes, in a ``__dict__`` or in
``__slots__``.
A set whose elements hash differently in the reading process (strings under
another ``PYTHONHASHSEED``) keeps its rebuilt order, and stays a working set.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import os
import pickle
import random
import subprocess
import sys

import pytest

from cash.backends import set_order
from cash.backends.factory import build_backend_from_config
from cash.config.schema import CashConfig

pytestmark = pytest.mark.skipif(set_order._KERNELS is None, reason="no C extension for this Python")


def _sets():
    r = random.Random(1)
    moved = [set() for _ in range(4)]
    for n in range(4000):
        moved[n % 4].add(n)
    for n in r.sample(range(4000), 1500):  # deletions leave markers in the table
        moved[n % 4].discard(n)
        moved[(n * 7 + 3) % 4].add(n)
    return {
        "ints": set(r.sample(range(1000), 100)),
        "floats": {r.random() for _ in range(300)},
        "pairs": {(r.randrange(10**6), r.randrange(9)) for _ in range(2000)},
        "nested": [(frozenset(r.sample(range(10**9), 40)), {frozenset(r.sample(range(500), 30)): 1})],
        "moved": moved,
    }


def _orders(value):
    return {
        "ints": list(value["ints"]),
        "floats": list(value["floats"]),
        "pairs": list(value["pairs"]),
        "nested": [list(value["nested"][0][0]), [list(k) for k in value["nested"][0][1]]],
        "moved": [list(s) for s in value["moved"]],
    }


class SetOf(set):
    pass


class FrozenSetOf(frozenset):
    pass


Pair = collections.namedtuple("Pair", "name members")


class Plain:
    def __init__(self, members):
        self.name = "plain"
        self.members = members


@dataclasses.dataclass
class Data:
    name: str
    members: set


class Slotted:
    __slots__ = ("name", "members")

    def __init__(self, members):
        self.name = "slotted"
        self.members = members


def _colliding(seed):
    """A set of ints many of which sit off their first slot: a rebuild puts
    them elsewhere."""
    r = random.Random(seed)
    s = set(r.sample(range(4000), 600))
    for n in r.sample(sorted(s), 200):
        s.discard(n)
    s.update(r.sample(range(4000, 8000), 100))
    return s


def _held():
    by_group = collections.defaultdict(set)
    by_group["a"] |= _colliding(1)
    by_group["b"] |= _colliding(2)
    ordered = collections.OrderedDict([("x", _colliding(3)), ("y", _colliding(4))])
    return {
        "subclass": SetOf(_colliding(5)),
        "frozen_subclass": FrozenSetOf(_colliding(6)),
        "defaultdict": by_group,
        "ordereddict": ordered,
        "namedtuple": Pair("p", _colliding(7)),
        "object": Plain(_colliding(8)),
        "dataclass": Data("d", _colliding(9)),
        "slots": Slotted(_colliding(10)),
        "two_deep": [Plain(Data("inner", _colliding(11))), Slotted({"k": Pair("q", _colliding(12))})],
    }


def _held_orders(value):
    def walk(x):
        if isinstance(x, (set, frozenset)):
            return list(x)
        if isinstance(x, dict):
            return [(k, walk(v)) for k, v in x.items()]
        if isinstance(x, (list, tuple)):
            return [walk(v) for v in x]
        if isinstance(x, (Plain, Data, Slotted)):
            return [walk(x.name), walk(x.members)]
        return x

    return {name: walk(x) for name, x in value.items()}


def test_sets_inside_subclasses_and_objects_change_order_in_a_pickle_round_trip():
    """The control: each holder's sets come back in another order without
    the layouts."""
    value = _held()
    before, after = _held_orders(value), _held_orders(pickle.loads(pickle.dumps(value, 5)))
    assert [n for n in value if before[n] == after[n]] == []


def test_sets_inside_subclasses_and_objects_come_back_in_the_stored_order(tmp_path):
    value = _held()
    backend = _backend(tmp_path)
    backend.set("stmt:held", value, {"execution_time": 5.0})
    _, from_ram = backend.get("stmt:held")
    assert _held_orders(from_ram) == _held_orders(value)

    meta, from_disk = _backend(tmp_path).get("stmt:held")
    assert meta.get("source") == "DISK"
    assert _held_orders(from_disk) == _held_orders(value)
    assert type(from_disk["subclass"]) is SetOf and from_disk["subclass"] == value["subclass"]
    assert all(x in from_disk["slots"].members for x in value["slots"].members)


def test_an_ordereddict_moved_out_of_its_dict_order_keeps_working(tmp_path):
    """Its copy is built in its order, not in the order of the dict under it:
    a path recorded in one would not find its set in the other, so its sets
    are not laid out -- never given another set's layout."""
    ordered = collections.OrderedDict([("x", _colliding(3)), ("y", {-1, -2})])
    ordered.move_to_end("x")
    backend = _backend(tmp_path)
    backend.set("stmt:moved", ordered, {"execution_time": 5.0})
    _, got = _backend(tmp_path).get("stmt:moved")
    assert list(got) == ["y", "x"] and got == ordered
    assert all(n in got["x"] for n in ordered["x"])


class Reduced:
    """Copied by its own __reduce__, which may hand back anything."""

    def __init__(self):
        self.members = _colliding(13)

    def __reduce__(self):
        return (Reduced, ())


def test_an_object_with_its_own_reduce_is_not_walked_into():
    from cash import _native

    assert _native.kernels.set_layouts([Reduced()], 1000) == b""
    assert _native.kernels.set_layouts([Plain(_colliding(13))], 1000) != b""


def _backend(tmp_path):
    return build_backend_from_config(CashConfig(cache_dir=str(tmp_path / ".cash"), compress=False))


def test_a_pickle_round_trip_alone_changes_the_order():
    """The control: without the layouts, the rebuilt sets iterate otherwise."""
    value = _sets()
    assert _orders(pickle.loads(pickle.dumps(value, 5))) != _orders(value)


def test_the_ram_tier_and_the_disk_tier_give_the_stored_order(tmp_path):
    value = _sets()
    backend = _backend(tmp_path)
    backend.set("stmt:sets", value, {"execution_time": 5.0})
    _, from_ram = backend.get("stmt:sets")
    assert from_ram == value
    assert _orders(from_ram) == _orders(value)

    restarted = _backend(tmp_path)
    meta, from_disk = restarted.get("stmt:sets")
    assert meta.get("source") == "DISK"
    assert from_disk == value
    assert _orders(from_disk) == _orders(value)
    # Working sets: every element found, and later edits behave as on the original.
    for name in ("ints", "floats", "pairs"):
        assert all(x in from_disk[name] for x in value[name])
        value[name].add(-1)
        from_disk[name].add(-1)
        value[name].discard(next(iter(value[name])))
        from_disk[name].discard(next(iter(from_disk[name])))
        assert list(from_disk[name]) == list(value[name])


STORE = """
import json, random, sys
from cash.backends.factory import build_backend_from_config
from cash.config.schema import CashConfig
r = random.Random(1)
value = {"words": {f"w{i}" for i in r.sample(range(10000), 300)}, "ints": set(r.sample(range(1000), 100))}
backend = build_backend_from_config(CashConfig(cache_dir=sys.argv[1], compress=False))
if sys.argv[2] == "store":
    backend.set("stmt:words", value, {"execution_time": 5.0})
    backend.shutdown()  # the disk write finishes before the process ends
    got = value
else:
    _, got = backend.get("stmt:words")
    assert got == value, "equal sets"
    assert all(w in got["words"] for w in value["words"]) and "nope" not in got["words"], "lookups work"
orders = {"words": list(got["words"]), "ints": list(got["ints"])}
got["words"].add("x"); got["words"].discard("w1"); assert "x" in got["words"] and "w1" not in got["words"]
print(json.dumps(orders))
"""


def _run(cache_dir, mode, seed):
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    out = subprocess.run(
        [sys.executable, "-c", STORE, cache_dir, mode], env=env, capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_another_hash_seed_keeps_strings_working_and_ints_in_order(tmp_path):
    cache_dir = str(tmp_path / ".cash")
    stored = _run(cache_dir, "store", 0)
    same_seed = _run(cache_dir, "read", 0)
    assert same_seed == stored
    other_seed = _run(cache_dir, "read", 7)
    # Ints hash alike under every seed: the stored order.
    assert other_seed["ints"] == stored["ints"]
    # Strings do not: the rebuilt order, in a set that works (checked in the child).
    assert sorted(other_seed["words"]) == sorted(stored["words"])
