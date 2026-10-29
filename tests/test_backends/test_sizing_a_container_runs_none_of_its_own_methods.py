"""Sizing a container of a class of the user's runs none of its methods.

A dict, list or set subclass whose ``__len__`` or ``values()`` prints and
counts showed ``CALLED __len__`` in the cell that bound it and in the one
that read it, and its counter went up, where plain Python never asked:
cash sized it (for the RAM tier and the restore budget) through the
class's own methods. It is read by the builtin type's methods now.
"""

from __future__ import annotations

import pytest

from cash.notebook.upstream.changed_below import _held_directly
from cash.sizing import estimate_object_size, memory_footprint

CALLS: list[str] = []


def _counting(base: type) -> type:
    def method(name):
        def run(self, *args, **kwargs):
            CALLS.append(name)
            return getattr(base, name)(self, *args, **kwargs)

        return run

    names = ["__iter__", "__len__", "__contains__", "__getitem__", "__reversed__", "copy"]
    if base is dict:
        names += ["keys", "values", "items", "get"]
    return type(f"Counting{base.__name__.title()}", (base,), {n: method(n) for n in names if hasattr(base, n)})


def _filled(base: type):
    made = _counting(base)()
    if base is dict:
        for k in range(5):
            dict.__setitem__(made, k, [k])
    elif base is list:
        for k in range(5):
            list.append(made, [k])
    else:
        for k in range(5):
            set.add(made, (k,))
    return made


@pytest.mark.parametrize("base", [dict, list, set])
def test_sizing_and_the_look_below_call_nothing_of_the_class(base):
    value = _filled(base)
    CALLS.clear()
    assert estimate_object_size(value) > 0
    assert memory_footprint(value) > 0
    held, whole = _held_directly(value)
    assert whole and len(held) == 5
    assert CALLS == []
