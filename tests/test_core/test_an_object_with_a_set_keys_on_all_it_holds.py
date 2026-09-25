"""An object holding a set keys on everything pickle would store for it.

To sort the set, such an object was keyed by its ``__dict__`` alone, which
drops what a builtin or C base holds: an ndarray subclass's data, a ``str``
subclass's text, a ``deque`` subclass's items. Two tagged arrays over
``[1, 2]`` and ``[30, 40]`` shared an entry. A dict subclass with
``__slots__`` lost its slot values the same way.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

from cash import Cash, FileBackend


class Tagged(np.ndarray):
    pass


def tagged(values, tags=("raw",)):
    array = np.asarray(values).view(Tagged)
    array.tags = set(tags)
    return array


class Name(str):
    pass


def name(text):
    value = Name(text)
    value.aliases = {"x"}
    return value


class Window(collections.deque):
    pass


def window(items):
    value = Window(items)
    value.seen = {"a"}
    return value


class Slotted(dict):
    __slots__ = ("source",)


def slotted(source):
    value = Slotted(a=1)
    value.source = source
    return value


class Plain:
    def __init__(self, n):
        self.n = n
        self.tags = {"b", "a"}


@pytest.fixture
def key(tmp_path):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))
    return lambda value: c._args.hash_payload((value,), {})


@pytest.mark.parametrize(
    "left, right",
    [
        (tagged([1, 2]), tagged([30, 40])),
        (name("alice"), name("bob")),
        (window([1, 2]), window([3, 4])),
        (slotted("db://a"), slotted("db://b")),
    ],
    ids=["ndarray subclass", "str subclass", "deque subclass", "dict subclass with slots"],
)
def test_what_the_base_holds_reaches_the_key(key, left, right):
    assert key(left) != key(right)


def test_equal_objects_still_key_alike(key):
    """Positive control: equal values built twice, sets in any order."""
    assert key(tagged([1, 2], ("a", "b"))) == key(tagged([1, 2], ("b", "a")))
    assert key(name("bob")) == key(name("bob"))
    assert key(Plain(1)) == key(Plain(1))
    assert key(Plain(1)) != key(Plain(2))


def test_a_class_defined_in_a_function_still_keys(key):
    class Local:
        def __init__(self):
            self.tags = {"a"}

    assert key(Local()) == key(Local())


def test_the_decorator_serves_each_its_own_total(tmp_path):
    c = Cash(backend=FileBackend(cache_dir=str(tmp_path)))

    @c.cache
    def total(a):
        return int(a.sum())

    assert total(tagged([1, 2])) == 3
    assert total(tagged([30, 40])) == 70
