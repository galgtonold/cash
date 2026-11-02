"""A RAM hit of a value holding a function and an instance of a class,
both defined again since (a notebook re-running their cell), holds the
function its module names now when that is the same definition, as a disk
hit loads it.

Pickle refuses the old class, so the copy is made again with each class
and function named again swapped for its stand-in; that kept the old
function, and pickling the hit (``joblib.dump`` of a pipeline holding
``column_ratio`` and a notebook transformer) failed: "not the same object
as __main__.column_ratio". A function defined again with other code is
still the one the value held: the one named now does something else.
"""

from __future__ import annotations

import pickle
import sys
import types

import pytest

from cash.backends.memory_backend import InMemoryBackend

pytestmark = [pytest.mark.core]

_SOURCE = """
def column_ratio(x):
    return x * 2

class Similarity:
    def __init__(self, k):
        self.k = k
"""


@pytest.fixture
def module():
    """A module whose cell `run` runs (again), as a notebook does."""
    mod = types.ModuleType("cash_test_function_and_class_defined_again")
    sys.modules[mod.__name__] = mod

    def run(source: str = _SOURCE):
        exec(source, mod.__dict__)
        return mod

    yield run
    del sys.modules[mod.__name__]


def _stored(mod) -> InMemoryBackend:
    b = InMemoryBackend()
    b.set("k", {"steps": [mod.column_ratio, mod.Similarity(3)]}, {"copy_required": True})
    return b


def test_a_hit_holds_the_function_defined_now(module):
    mod = module()
    b = _stored(mod)
    module("\n\n" + _SOURCE)  # the cell runs again, lines apart

    steps = b.get("k")[1]["steps"]
    assert steps[0] is mod.column_ratio
    assert type(steps[1]) is mod.Similarity
    pickle.dumps(steps)


def test_a_function_defined_again_with_other_code_is_the_one_held(module):
    mod = module()
    held = mod.column_ratio
    b = _stored(mod)
    module(_SOURCE.replace("x * 2", "-x"))

    steps = b.get("k")[1]["steps"]
    assert steps[0] is held
    assert steps[0](5) == 10
    assert type(steps[1]) is mod.Similarity
