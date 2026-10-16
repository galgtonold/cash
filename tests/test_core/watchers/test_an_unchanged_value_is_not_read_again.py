"""A value hashed before comes back with the same digest, unread, while the
watchers prove it unchanged; any change anywhere in it is read again.

Each test hashes a value through `value_watch.digest` with a counting hash:
the first hash records a sighting, the second arms the watchers, and from
then on an unchanged value is not read. After a change, the next digest is
read from the value and equals a fresh hash.
"""

from __future__ import annotations

import copyreg
import gc
import sys
import time
import types

import pytest

from cash import value_watch
from cash.value_hash import _compute_hash, compute_hash


class Point:
    def __init__(self, x, y=None):
        self.x = x
        self.y = y


class Other:
    def __init__(self, x, y=None):
        self.x = x
        self.y = y


class Slotted:
    __slots__ = ("a", "b", "__weakref__")

    def __init__(self, a, b):
        self.a = a
        self.b = b


class Mixed(Slotted):
    __slots__ = ("c", "__dict__")


class Outer:
    class Inner:
        def __init__(self, v):
            self.v = v


class OwnState:
    def __init__(self):
        self.a = 1

    def __getstate__(self):
        return {"a": self.a}


def helper(x):
    return x


def other_helper(x):
    return -x


def watched(value, hash_fn):
    """Hash *value* until it is watched: a sighting, then the arming hash."""
    first = value_watch.digest(value, hash_fn)
    second = value_watch.digest(value, hash_fn)
    assert first == second
    return first


def reread(value, hash_fn) -> bool:
    """Is *value* read again by its next digest? Asserts that digest is a
    fresh hash's."""
    before = hash_fn.calls
    got = value_watch.digest(value, hash_fn)
    assert got == _compute_hash(value)
    return hash_fn.calls > before


def test_an_unchanged_dict_is_hashed_once_after_it_is_watched(watching, counted):
    value = {"rows": [{"v": i} for i in range(10)], "meta": ("a", 1, None)}
    watched(value, counted)
    calls = counted.calls
    for _ in range(5):
        assert value_watch.digest(value, counted) == _compute_hash(value)
    assert counted.calls == calls


def test_the_first_hash_only_notes_the_value(watching, counted):
    value = {"a": [1, 2, 3]}
    value_watch.digest(value, counted)
    assert value_watch._cw.info(value) is None, "watched on first sight"
    value_watch.digest(value, counted)
    assert value_watch._cw.info(value) is not None


CHANGES = {
    "a key set": lambda v: v["meta"].__setitem__("n", 2),
    "a key deleted": lambda v: v["meta"].pop("n"),
    "a dict cleared": lambda v: v["meta"].clear(),
    "a dict updated": lambda v: v["meta"].update(z=1),
    "a key moved to the end": lambda v: v.__setitem__("meta", v.pop("meta")),
    "a list appended": lambda v: v["rows"].append(1),
    "a list item replaced": lambda v: v["rows"].__setitem__(0, "x"),
    "a list reversed": lambda v: v["rows"].reverse(),
    "a nested dict in a list": lambda v: v["rows"][1].__setitem__("v", 99),
    "a set grown": lambda v: v["tags"].add("new"),
    "a set emptied": lambda v: v["tags"].clear(),
    "an attribute set": lambda v: setattr(v["point"], "x", 7),
    "an attribute deleted": lambda v: delattr(v["point"], "y"),
    "a __dict__ replaced": lambda v: setattr(v["point"], "__dict__", {"x": 1, "y": 2}),
    "a __class__ swapped": lambda v: setattr(v["point"], "__class__", Other),
    "a slot set": lambda v: setattr(v["slotted"], "a", 5),
    "a slot deleted": lambda v: delattr(v["slotted"], "b"),
    "a list in a slot appended": lambda v: v["slotted"].b.append(3),
    "a nested class's instance": lambda v: setattr(v["inner"], "v", 3),
    "a mixed object's dict": lambda v: setattr(v["mixed"], "d", 4),
    "a mixed object's slot": lambda v: setattr(v["mixed"], "c", 4),
    "an equal string, another object": lambda v: v["meta"].__setitem__("s", "".join(["ab", "c"])),
    "an instance __getstate__": lambda v: setattr(v["point"], "__getstate__", lambda: {"x": 0}),
}


#: Changes after which the value holds what the walk cannot see.
HASHED_FROM_THEN_ON = {"an instance __getstate__"}


def _value():
    mixed = Mixed(1, 2)
    mixed.c = 3
    mixed.d = 1
    return {
        "rows": [1, {"v": 2}, [3, 4]],
        "meta": {"n": 1, "s": "abc"},
        "tags": {"a", "b"},
        "point": Point(1, [2]),
        "slotted": Slotted(1, [2]),
        "inner": Outer.Inner(1),
        "mixed": mixed,
        "frozen": (1, frozenset({2, 3}), ("t", b"b", 1.5)),
        "callables": [helper, Point, Outer.Inner],
    }


@pytest.mark.parametrize("change", list(CHANGES))
def test_a_change_anywhere_is_read_again(watching, counted, change):
    value = _value()
    watched(value, counted)
    CHANGES[change](value)
    assert reread(value, counted)
    # and then it is watched again, and unchanged -- unless the change made
    # it a value the walk cannot see (a lambda as its own __getstate__)
    value_watch.digest(value, counted)
    assert reread(value, counted) == (change in HASHED_FROM_THEN_ON)


def test_a_change_through_an_alias_is_seen_in_both_values(watching, counted):
    shared = {"n": 1}
    a = {"shared": shared, "a": 1}
    b = [shared, 2]
    watching["b"] = b
    watched(a, counted)
    watched(b, counted)
    alias = shared
    alias["n"] = 2
    assert reread(a, counted)
    assert reread(b, counted)


def test_a_class_attribute_that_changes_the_pickle_is_seen(watching, counted):
    value = {"p": Point(1)}
    watched(value, counted)
    Point.tag = "class level"  # changes nothing pickled, but the class moved
    try:
        assert reread(value, counted)
        value_watch.digest(value, counted)
        Point.__getstate__ = lambda self: {"only": "this"}
        assert reread(value, counted)
        assert compute_hash(value) == _compute_hash(value)
    finally:
        del Point.tag
        if "__getstate__" in Point.__dict__:
            del Point.__getstate__


def test_a_base_class_change_is_seen_through_the_subclass(watching, counted):
    value = {"m": Mixed(1, 2)}
    watched(value, counted)
    Slotted.extra = 1
    try:
        assert reread(value, counted)
    finally:
        del Slotted.extra


@pytest.fixture
def module_with_classes():
    mod = types.ModuleType("_cwatch_test_mod")
    exec(
        "class K:\n    def __init__(self, v):\n        self.v = v\nclass L(K):\n    pass\ndef f():\n    return 1\n",
        mod.__dict__,
    )
    sys.modules[mod.__name__] = mod
    yield mod
    sys.modules.pop(mod.__name__, None)
    copyreg.dispatch_table.pop(mod.K, None)


def test_rebinding_the_name_a_class_is_pickled_by_is_seen(watching, counted, module_with_classes):
    mod = module_with_classes
    value = {"k": mod.K(1)}
    watched(value, counted)
    original = mod.K
    mod.K = mod.L  # pickle now fails to find the class: the identity hash
    assert reread(value, counted)
    mod.K = original
    assert reread(value, counted)


def test_an_unrelated_global_of_the_module_is_not_a_change(watching, counted, module_with_classes):
    mod = module_with_classes
    value = {"k": mod.K(1)}
    watched(value, counted)
    mod.unrelated = 5
    assert not reread(value, counted)


def test_rebinding_the_module_in_sys_modules_is_seen(watching, counted, module_with_classes):
    mod = module_with_classes
    value = {"k": mod.K(1)}
    watched(value, counted)
    sys.modules[mod.__name__] = types.ModuleType(mod.__name__)
    assert reread(value, counted)
    sys.modules[mod.__name__] = mod
    assert reread(value, counted)


def test_a_reducer_registered_for_the_class_is_seen(watching, counted, module_with_classes):
    mod = module_with_classes
    value = {"k": mod.K(1)}
    watched(value, counted)
    copyreg.pickle(mod.K, lambda obj: (mod.K, (2,)))
    assert reread(value, counted)


def test_a_function_held_as_a_value_is_watched_by_name(watching, counted, module_with_classes):
    mod = module_with_classes
    value = {"f": mod.f}
    watched(value, counted)
    mod.f.__code__ = (lambda: 2).__code__  # not pickled, but the function changed
    assert reread(value, counted)
    value_watch.digest(value, counted)
    mod.f.__qualname__ = "g"  # pickle cannot find it under that name
    assert reread(value, counted)


def test_a_list_variable_rebound_in_its_namespace_is_not_reused(watching, counted):
    ns = watching
    for i in range(50):
        # each new list may take the address the last one freed, holding a
        # new string at the address the last string freed
        ns["x"] = [str(i) * 3, [i]]
        value_watch.digest(ns["x"], counted)
        assert value_watch.digest(ns["x"], counted) == _compute_hash(ns["x"])
        # the youngest generation only: what the last round left is young.
        # A full collection walks every object the worker holds -- 1-2 s
        # each in a long-lived worker, and 50 of them overran the timeout.
        gc.collect(0)


def test_a_list_no_namespace_holds_is_not_watched(watching, counted):
    value = [1, 2, {"a": 1}]
    value_watch.digest(value, counted)
    value_watch.digest(value, counted)
    assert value_watch._cw.info(value) is None


def test_an_object_freed_and_replaced_at_its_address_is_read(watching, counted):
    for i in range(50):
        obj = Point(str(i) * 2, [i])
        value_watch.digest(obj, counted)
        assert value_watch.digest(obj, counted) == _compute_hash(obj)
        del obj


def test_a_dict_freed_and_replaced_at_its_address_is_read(watching, counted):
    for i in range(50):
        d = {"k": str(i) * 2, "l": [i]}
        value_watch.digest(d, counted)
        assert value_watch.digest(d, counted) == _compute_hash(d)
        del d


INELIGIBLE = {
    "a class with its own __getstate__": lambda: {"o": OwnState()},
    "a lambda": lambda: {"f": lambda: 1},
    "a bytearray": lambda: {"b": bytearray(b"ab")},
    "a complex number": lambda: {"c": 1j},
    "a dict subclass": lambda: {"d": __import__("collections").OrderedDict(a=1)},
    "a builtin function": lambda: {"f": len},
    "an object of a local class": lambda: {"o": type("Local", (), {})()},
}


@pytest.mark.parametrize("case", list(INELIGIBLE))
def test_a_value_holding_what_the_walk_cannot_see_is_hashed_every_time(watching, counted, case):
    value = INELIGIBLE[case]()
    for _ in range(3):
        value_watch.digest(value, counted)
    assert counted.calls == 3
    assert value_watch._cw.info(value) is None


def test_a_small_value_is_not_watched_by_default(watching, counted):
    value_watch._cw.configure(min_bytes=64 << 10)
    small = {"a": list(range(10))}
    big = {"a": list(range(20000))}
    for value in (small, big):
        for _ in range(3):
            value_watch.digest(value, counted)
    assert value_watch._cw.info(small) is None
    assert value_watch._cw.info(big) is not None


def test_a_value_changed_on_every_check_stops_being_watched(watching, counted):
    value = {"n": 0, "l": list(range(10))}
    for i in range(12):
        value["n"] = i
        assert value_watch.digest(value, counted) == _compute_hash(value)
    assert value_watch._cw.info(value) is None
    assert value_watch.stats()["dropped_changing"] >= 1


def _slow(hash_fn):
    """*hash_fn*, taking 20 ms more: a value worth more than one hashed in
    microseconds. What a value is worth is the time its hash took, so the
    tests below set it rather than leave it to the machine's timing."""

    def slow(value):
        time.sleep(0.02)
        return hash_fn(value)

    return slow


def _full_table(hash_fn):
    """Four watched values, and a budget one byte under the table they fill."""
    cw = value_watch._cw
    values = [{"i": i, "l": [{"x": j} for j in range(200)]} for i in range(4)]
    for value in values:
        watched(value, hash_fn)
    assert cw.stats()["roots"] == 4
    cw.configure(budget=cw.stats()["table_bytes"] - 1)
    return values


def test_over_budget_the_least_worth_value_is_dropped(watching, counted):
    cw = value_watch._cw
    values = _full_table(counted)
    extra = {"l": [{"x": j} for j in range(200)]}
    watched(extra, _slow(counted))
    assert cw.stats()["evicted"] >= 1
    assert cw.stats()["table_bytes"] <= cw.stats()["budget"]
    assert cw.info(extra) is not None
    for value in [*values, extra]:
        assert value_watch.digest(value, counted) == _compute_hash(value)


def test_over_budget_a_value_worth_less_than_every_watched_one_is_not_watched(watching, counted):
    cw = value_watch._cw
    values = _full_table(_slow(counted))
    table_bytes = cw.stats()["table_bytes"]
    extra = {"l": [{"x": j} for j in range(200)]}
    watched(extra, counted)
    assert cw.stats()["over_budget"] >= 1
    assert cw.stats()["evicted"] == 0
    assert cw.info(extra) is None
    assert cw.stats()["table_bytes"] <= table_bytes
    for value in [*values, extra]:
        assert value_watch.digest(value, counted) == _compute_hash(value)


def test_kinds_keep_their_own_digest(watching, counted):
    value = {"a": [1, 2]}
    watched(value, counted)
    other = value_watch.digest(value, lambda v: ("marker", "x"), kind=value_watch.KIND_PLAIN_KEY)
    assert other == ("marker", "x")
    assert value_watch.digest(value, lambda v: ("marker", "y"), kind=value_watch.KIND_PLAIN_KEY) == ("marker", "x")
    value["a"].append(3)
    # a change invalidates every kind
    assert value_watch.digest(value, lambda v: ("marker", "z"), kind=value_watch.KIND_PLAIN_KEY) == ("marker", "z")


def test_peek_counts_no_sighting(watching, counted):
    value = {"a": [1]}
    for _ in range(3):
        assert value_watch.peek(value) is None
    assert value_watch._cw.info(value) is None
    watched(value, counted)
    assert value_watch.peek(value) == _compute_hash(value)
    value["a"].append(2)
    assert value_watch.peek(value) is None


def test_forget_and_reset_drop_the_watch(watching, counted):
    value = {"a": [1]}
    watched(value, counted)
    value_watch.forget(value)
    assert value_watch._cw.info(value) is None
    watched(value, counted)
    value_watch.reset()
    assert value_watch.stats()["roots"] == 0


def test_the_identity_fallback_is_never_kept(watching):
    class Unpicklable:  # a local class: pickling fails, the hash is its id
        pass

    holder = {"u": Unpicklable()}
    first = compute_hash(holder)
    assert compute_hash(holder) == first
    assert value_watch._cw.info(holder) is None or not value_watch._cw.info(holder)["kinds"]


def test_verify_mode_raises_on_a_stale_digest(watching, counted, monkeypatch):
    value = {"a": [1]}
    watched(value, counted)
    value_watch._cw.commit(value, value_watch.KIND_VALUE, "stale", 0.0)  # what a bug would leave
    monkeypatch.setattr(value_watch, "VERIFY", True)
    with pytest.raises(value_watch.WatchMismatch):
        value_watch.digest(value, counted)


def test_compute_hash_goes_through_the_watchers(watching):
    value = {"rows": [{"v": i} for i in range(100)]}
    first = compute_hash(value)
    hits = value_watch.stats()["hits"]
    for _ in range(3):
        assert compute_hash(value) == first
    assert value_watch.stats()["hits"] == hits + 3 - 1  # the second hash armed it
