"""A RAM hit of a stored object whose class has since been defined again
(a notebook re-running its class cell, ``importlib.reload``) hands back an
instance of the class the module names now, as a disk hit does.

The RAM copy is a pickle round trip, and pickle refuses an instance of a
class its module no longer names ("not the same object as __main__.Fit").
The fallback, ``deepcopy``, kept the old class: ``isinstance(m, Fit)`` and
``m == Fit(...)`` were False and pickling the hit failed. For a frame of
such objects the fallback was pandas' deep copy, which shares the cell
objects with the entry, so a change to a hit's cell reached every later hit.
"""

from __future__ import annotations

import pickle
import sys
import types

import pytest

from cash.backends.memory_backend import InMemoryBackend

pytestmark = [pytest.mark.core]

_SOURCE = """
import dataclasses, enum

@dataclasses.dataclass
class Fit:
    w: list

class Kind(enum.Enum):
    A = 1

class Cell:
    def __init__(self, w):
        self.w = w
"""


@pytest.fixture
def module():
    """A module whose classes `define` defines again, as a re-run cell does."""
    mod = types.ModuleType("cash_test_redefined_classes")
    sys.modules[mod.__name__] = mod

    def define():
        exec(_SOURCE, mod.__dict__)
        return mod

    yield define
    del sys.modules[mod.__name__]


def test_a_hit_is_an_instance_of_the_class_defined_now(module):
    mod = module()
    b = InMemoryBackend()
    b.set("k", {"fit": mod.Fit([1.0]), "kind": mod.Kind.A, "fits": [mod.Fit([2.0])]}, {"copy_required": True})
    b.set("bare", mod.Fit([3.0]), {"copy_required": True})
    module()  # the class cell runs again

    got = b.get("k")[1]
    assert isinstance(got["fit"], mod.Fit)
    assert got["fit"] == mod.Fit([1.0])
    assert got["kind"] is mod.Kind.A
    assert isinstance(got["fits"][0], mod.Fit)
    assert b.get("bare")[1] == mod.Fit([3.0])
    pickle.dumps(got)


def test_a_frame_of_such_objects_does_not_share_its_cells(module):
    pd = pytest.importorskip("pandas")
    mod = module()
    b = InMemoryBackend()
    b.set("k", pd.DataFrame({"m": [mod.Cell([1]), mod.Cell([2])], "n": [1, 2]}), {"copy_required": True})
    module()

    first = b.get("k")[1]
    assert type(first["m"].iloc[0]) is mod.Cell
    first["m"].iloc[0].w.append(9)
    assert b.get("k")[1]["m"].iloc[0].w == [1]
