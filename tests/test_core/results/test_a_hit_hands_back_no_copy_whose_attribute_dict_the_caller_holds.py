"""A hit hands back no copy whose instance attribute dict the caller holds.

A cached function returned an object whose attribute ``node`` is a plain
instance. The caller kept ``d = vars(r.node)``, dropped ``r`` and called
again: the hit handed back the very copy, so ``vars(r2.node) is d`` and a
later ``d["name"] = ...`` showed in ``r2``, where plain Python returns a
new object. Since 3.13 an instance can keep its attributes inline even once
its ``__dict__`` exists, and the garbage collector's walk then reaches the
attributes and not the dict, so the reference the caller holds to the dict
went uncounted. The same holds for ``r.node.__dict__``, for the top object's
own dict, and for an instance deeper down; and a bound builtin method of a
list inside the result (``r.add = r.items.append``) holds that list. Each
gets a new copy, equal to what plain Python returns.
"""

from __future__ import annotations

from abc import ABC

import pytest

from cash import value_watch

pytestmark = pytest.mark.skipif(
    not value_watch.enabled(),
    reason="the copy is handed back only where the C extension's watchers prove it unchanged",
)

#: How many copies were read out of pickle bytes (`Result.__setstate__`).
LOADS: list[int] = []


class Leaf:
    def __init__(self, name, child=None):
        self.name = name
        self.items = [1, 2]
        self.child = child


class Result(ABC):  # an ABC, as libraries' results often are
    def __init__(self):
        self.node = Leaf("n", Leaf("deep"))
        self.pad = [Leaf(i) for i in range(200)]
        # many objects: the RAM tier keeps it as pickle bytes
        self.groups = [[Leaf(j) for j in range(50)] for _ in range(200)]

    def __setstate__(self, state):
        LOADS.append(1)
        self.__dict__.update(state)


@pytest.fixture
def result(cash_instance):
    @cash_instance.cache
    def make():
        return Result()

    make()  # the miss: stored
    first = make()  # the first hit: a copy, kept
    del first
    # the copy is handed back while nobody holds it: what the cases below test
    before = len(LOADS)
    second = make()
    del second
    assert len(LOADS) == before, "an unheld copy is handed back"
    return make


def _deep(r):
    return r.node.child


@pytest.mark.parametrize(
    ("grab", "owner"),
    [
        pytest.param(lambda r: vars(r.node), lambda r: r.node, id="vars-of-a-nested-instance"),
        pytest.param(lambda r: r.node.__dict__, lambda r: r.node, id="dunder-dict-of-a-nested-instance"),
        pytest.param(lambda r: vars(r), lambda r: r, id="vars-of-the-result"),
        pytest.param(lambda r: vars(_deep(r)), _deep, id="vars-of-a-deeper-instance"),
    ],
)
def test_a_copy_whose_attribute_dict_is_held_is_not_handed_back(result, grab, owner):
    first = result()
    held = grab(first)
    del first
    second = result()
    assert vars(owner(second)) is not held
    held["name"] = "MUT"
    held["extra"] = 1
    assert getattr(owner(second), "name", None) != "MUT"
    assert not hasattr(owner(second), "extra")
    assert [leaf.name for leaf in second.pad[:3]] == [0, 1, 2]


def test_the_copy_after_it_is_handed_back_while_nobody_holds_it(result):
    first = result()
    held = vars(first.node)
    del first
    second = result()
    assert vars(second.node) is not held
    del held, second
    before = len(LOADS)
    third = result()
    assert len(LOADS) == before, "the new copy, unheld, is handed back"
    assert third.node.name == "n"


class Bag:
    def __init__(self):
        self.items = [1, 2, 3]
        self.add = self.items.append
        self.groups = [[Leaf(j) for j in range(50)] for _ in range(200)]


def test_a_held_bound_method_of_a_list_in_the_result_gets_a_new_copy(cash_instance):
    @cash_instance.cache
    def make():
        return Bag()

    make()
    first = make()
    del first
    second = make()
    add = second.add
    del second
    third = make()
    assert third.add is not add
    add("MUT")
    assert third.items == [1, 2, 3]
    third.add(4)
    assert third.items == [1, 2, 3, 4], "the copy's method is bound to the copy's own list"
