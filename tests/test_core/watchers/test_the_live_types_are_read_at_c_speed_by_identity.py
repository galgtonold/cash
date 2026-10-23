"""The types alive on the heap are read at C speed, each once, by identity.

`unseen_writes._with_objects_alive` takes the type of every object the
collector tracks: 0.4 s in Python for the 2.4 million objects of a long
test run, on each statement reading a value too big to look through. The C
kernel gives the same types, and never runs a metaclass's ``__hash__``.
"""

from __future__ import annotations

import gc

import pytest

from cash import _native

pytestmark = pytest.mark.skipif(
    getattr(_native.kernels, "distinct_types", None) is None, reason=f"no C kernel: {_native.REASON}"
)


class Loud(type):
    def __hash__(cls):
        raise AssertionError("a metaclass's __hash__ ran")

    def __eq__(cls, other):
        raise AssertionError("a metaclass's __eq__ ran")


class Odd(metaclass=Loud):
    pass


def test_the_types_are_those_python_finds_each_once():
    kept = [Odd() for _ in range(5)] + [[i] for i in range(50_000)]
    objects = gc.get_objects()
    found = _native.kernels.distinct_types(objects)
    expected = {id(type(o)): type(o) for o in objects}
    assert len(found) == len(expected)
    assert {id(t) for t in found} == set(expected)
    assert any(t is Odd for t in found)
    del kept


def test_an_empty_list_has_no_types():
    assert _native.kernels.distinct_types([]) == []


def test_anything_but_a_list_is_refused():
    with pytest.raises(TypeError):
        _native.kernels.distinct_types((1, 2))
