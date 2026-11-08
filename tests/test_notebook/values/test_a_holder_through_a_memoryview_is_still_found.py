"""A variable holding an output's array through a memoryview shares it.

``buf = memoryview(data)`` holds ``data``, and a write through ``buf``
shows in ``out.data`` in plain Python. An object in the outputs that holds
its array through a C-level view too (a search tree's typed memoryview,
one built when the object is loaded) must not make that holder look like
the output's own: the share check counts references, and runs none of the
outputs' own code (``__reduce__``, ``__setstate__``) to do it.
"""

from __future__ import annotations

import pytest

from cash.notebook.shared_objects import share_group

np = pytest.importorskip("numpy")

LOADS: list[int] = []


class Index:
    """Builds a memoryview over its data when loaded by pickle."""

    def __init__(self, data):
        self.data = data

    def __setstate__(self, state):
        self.__dict__.update(state)
        LOADS.append(1)
        self._mv = memoryview(self.data)


def _namespace():
    data = np.arange(5.0)
    return {"buf": memoryview(data), "out": Index(data)}


def test_a_holder_outside_the_outputs_is_found():
    user_ns = _namespace()

    _found, shared = share_group(["out"], user_ns, user_ns)

    assert shared == {"out"}, shared


def test_the_share_check_loads_nothing():
    LOADS.clear()
    user_ns = _namespace()

    share_group(["out"], user_ns, user_ns)

    assert LOADS == [], LOADS
