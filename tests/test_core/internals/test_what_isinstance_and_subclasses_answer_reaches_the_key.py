"""What ``isinstance`` and ``__subclasses__`` answer, when no namespace
the call reads changes, still reaches the key and the fast hit path.

* A user class registered with a library ABC (``Mapping.register(Box)``).
* ``register()`` through a metaclass of one's own; an ``__instancecheck__``
  reading a module global; a tuple of user ABCs.
* A subclass collected and a new one made at its address: the fast hit
  path holds the classes below by weakref, not by ``id()``.
* A method added to a class a ``runtime_checkable`` protocol is checked
  against.
* A subclass defined again under the same name, edited (a notebook cell
  run again): the key folds each class below by what it holds.
"""

from __future__ import annotations

import abc
import collections.abc
import gc
import os
import sys
import time
import typing

import pytest

from cash import Cash
from cash.decorator import hit_memo
from cash.exceptions import CashCacheIneffectiveWarning

pytestmark = [pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning")]


#: Nothing records what a call ran: before 3.12, or with ``CASH_TRACE=0``.
UNTRACED = sys.version_info < (3, 12) or os.environ.get("CASH_TRACE", "").strip().lower() in ("0", "off", "false", "no")


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def warm(f, *a):
    for _ in range(4):
        r = f(*a)
    return r


# 1. a library ABC, a user class registered with it
class Box:
    pass


def box_is_mapping(x):
    return (x, isinstance(Box(), collections.abc.Mapping))


def test_a_user_class_registered_with_a_library_abc(c):
    f = c.cache(box_is_mapping)
    assert warm(f, 1) == (1, False)
    collections.abc.Mapping.register(Box)
    assert f(1) == (1, True)


# 2. register() through a metaclass of the user's own
class Registering(type):
    def register(cls, sub):
        cls._members.add(sub)
        return sub

    def __instancecheck__(cls, inst):
        return type(inst) in cls._members


class Kind(metaclass=Registering):
    _members: set = set()


class Thing:
    pass


def thing_is_kind(x):
    return (x, isinstance(Thing(), Kind))


def test_register_through_a_metaclass_of_ones_own(c):
    """On 3.12+ the trace of what ran sees the set ``register()`` filled.
    Before, nothing does: the call runs uncached, and says so."""
    f = c.cache(thing_is_kind)
    if UNTRACED:
        with pytest.warns(CashCacheIneffectiveWarning, match="KEY-UNTRACKABLE-DEP"):
            assert warm(f, 1) == (1, False)
    else:
        assert warm(f, 1) == (1, False)
    Kind.register(Thing)
    try:
        assert f(1) == (1, True)
        if UNTRACED:
            assert f.cache_info()["hits"] == 0
    finally:
        Kind._members.clear()


# 3. an instancecheck reading a module global
MODE = {"strict": False}


class Loose(type):
    def __instancecheck__(cls, inst):
        return not MODE["strict"]


class Any_(metaclass=Loose):
    pass


def anything(x):
    return (x, isinstance(3, Any_))


def test_an_instancecheck_reading_mutable_state(c):
    f = c.cache(anything)
    assert warm(f, 1) == (1, True)
    MODE["strict"] = True
    try:
        assert f(1) == (1, False)
    finally:
        MODE["strict"] = False


# 4. a subclass collected and another made at its address
class Plugin:
    pass


def plugin_names(x):
    return (x, sorted(k.__name__ for k in Plugin.__subclasses__()))


def test_a_new_subclass_at_a_collected_ones_address(c):
    f = c.cache(plugin_names)
    first = type("First", (Plugin,), {})
    old = id(first)
    assert warm(f, 1) == (1, ["First"])
    del first
    gc.collect()
    give_up = time.monotonic() + 5
    while time.monotonic() < give_up:
        second = type("Second", (Plugin,), {})
        if id(second) == old:
            break
        del second
        gc.collect(0)  # a class is a cycle: collected young
    else:
        pytest.skip("no address reuse")
    assert f(1) == (1, ["Second"])


# 5. a runtime-checkable protocol
@typing.runtime_checkable
class Sized_(typing.Protocol):
    def size(self) -> int: ...


class Bag:
    pass


def bag_is_sized(x):
    return (x, isinstance(Bag(), Sized_))


def test_a_runtime_checkable_protocol(c):
    f = c.cache(bag_is_sized)
    assert warm(f, 1) == (1, False)
    Bag.size = lambda self: 0
    try:
        assert f(1) == (1, True)
    finally:
        del Bag.size


# 6. a tuple of ABCs
class Animal(abc.ABC):
    pass


class Plant(abc.ABC):
    pass


class Rock:
    pass


def rock_is_alive(x):
    return (x, isinstance(Rock(), (Animal, Plant)))


def test_a_tuple_of_abcs(c):
    f = c.cache(rock_is_alive)
    assert warm(f, 1) == (1, False)
    Plant.register(Rock)
    assert f(1) == (1, True)


class Step:
    pass


def run_steps(x):
    return (x, [k().out() for k in Step.__subclasses__()])


def test_a_subclass_redefined_with_the_same_name(c):
    ns = {"Step": Step}
    exec("class Double(Step):\n    def out(self):\n        return 2\n", ns)
    f = c.cache(run_steps)
    assert warm(f, 1) == (1, [2])
    exec("class Double(Step):\n    def out(self):\n        return 3\n", ns)  # the cell run again, edited
    gc.collect()
    h = hit_memo.status().get("fast_hits", 0)
    r = f(1)
    fast = hit_memo.status().get("fast_hits", 0) > h
    r2 = f(1)
    assert (r, fast, r2) == ((1, [3]), False, (1, [3]))
