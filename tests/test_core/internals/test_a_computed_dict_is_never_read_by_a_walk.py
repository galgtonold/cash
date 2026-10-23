"""An object whose class computes ``__dict__`` (a property) is never asked
for it by cash.

``object.__getattribute__(obj, "__dict__")`` does not skip such a property:
the class's data descriptor wins, so the user's code ran, and a lazy tree's
made a new child at each read. In a notebook whose namespace is the
``__main__`` module, as in IPython, the look for a shared output read it
5,000 times. Every walk now reads the dict Python keeps (``own_dict``).
"""

from __future__ import annotations

import pytest

from cash import jit_functions, source_reading
from cash.analysis import ast_util, handed_callables
from cash.notebook import shared_objects
from cash.notebook.control_structures import single_unit_policy
from cash.notebook.statement import hidden_generators

RUNS: list[str] = []


class Lazy:
    __module__ = "__main__"

    def __init__(self):
        object.__setattr__(self, "kept", 1)

    @property
    def __dict__(self):
        RUNS.append("ran")
        return {"child": Lazy.__new__(Lazy)}


class LazyWithAClassValue(Lazy):
    """``kept`` on the class too: ``getattr_static`` finds it, and the
    plain-attribute read goes on to the object's own dict."""

    kept = 0


class LazyList(list):
    @property
    def __dict__(self):
        RUNS.append("ran")
        return {"child": Lazy.__new__(Lazy)}


READS = {
    "shared attributes": lambda: shared_objects.attributes_of(Lazy()),
    "shared children": lambda: shared_objects.children_of(Lazy()),
    "shared leaf attributes": lambda: shared_objects._attributes_in(Lazy(), {}, frozenset()),
    "plain attribute": lambda: single_unit_policy._plain_attribute(LazyWithAClassValue(), "kept"),
    "generator walk": lambda: list(hidden_generators._children(Lazy())),
    "instance dict": lambda: handed_callables._instance_dict(Lazy()),
    "container items": lambda: list(handed_callables._items(LazyList([1]))),
    "in instance dict": lambda: ast_util._in_instance_dict(Lazy(), "kept"),
    "wrapper layers": lambda: source_reading.wrapper_layers(Lazy()),
    "jit dict": lambda: jit_functions._own_dict(Lazy()),
}


@pytest.mark.parametrize("read", READS.values(), ids=READS.keys())
def test_a_computed_dict_does_not_run(read):
    del RUNS[:]
    read()
    assert RUNS == []


class Attrs(dict):
    """A ``dict`` subclass set as an object's ``__dict__``: its own
    methods never run, and what it holds is still read."""

    def values(self):
        RUNS.append("ran")
        return super().values()

    def items(self):
        RUNS.append("ran")
        return super().items()

    def get(self, *args):
        RUNS.append("ran")
        return super().get(*args)

    def __contains__(self, key):
        RUNS.append("ran")
        return super().__contains__(key)

    def __iter__(self):
        RUNS.append("ran")
        return super().__iter__()


def helper():
    return 1


class Held:
    kept = 0

    def __init__(self):
        self.__dict__ = Attrs(fn=helper, kept=helper, __wrapped__=helper)


class HeldList(list):
    def __init__(self):
        super().__init__([1])
        self.__dict__ = Attrs(fn=helper)


SUBCLASS_READS = {
    "instance dict": lambda: handed_callables._instance_dict(Held())["fn"] is helper,
    "container items": lambda: helper in list(handed_callables._items(HeldList())),
    "in instance dict": lambda: ast_util._in_instance_dict(Held(), "fn"),
    "generator walk": lambda: (".fn", helper) in list(hidden_generators._children(Held())),
    "wrapper layers": lambda: source_reading.wrapper_layers(Held())[-1] is helper,
    "jit dict": lambda: jit_functions._own_dict(Held())["fn"] is helper,
    "plain attribute": lambda: single_unit_policy._plain_attribute(Held(), "kept") is helper,
}


@pytest.mark.parametrize("read", SUBCLASS_READS.values(), ids=SUBCLASS_READS.keys())
def test_a_dict_subclass_as_the_dict_is_read_without_its_methods(read):
    """``self.__dict__ = Attrs(fn=helper)``: the walks still find
    ``helper`` (a held function its globals are keyed by), as before."""
    del RUNS[:]
    assert read()
    assert RUNS == []
