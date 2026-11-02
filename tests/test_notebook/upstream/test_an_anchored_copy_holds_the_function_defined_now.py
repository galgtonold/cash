"""The copy an entry stores with references to variables' objects holds a
function whose cell ran again as the RAM and disk tiers do: the one its
module names now when that is the same definition, else the one held.

A pipeline holding ``column_ratio`` beside a variable's object is stored
with a reference to that variable; the copy kept the first ``column_ratio``
even when the cell defining it had only run again, so a hit held a function
no name in the notebook holds, and pickling it failed.
"""

from __future__ import annotations

import pickle
import sys
import types

import pytest

from cash.notebook.anchors import AnchorRef, with_anchor_refs

SOURCE = "def column_ratio(x):\n    return x * 2\n"


@pytest.fixture
def module():
    """A module whose cell `run` runs (again), as a notebook does."""
    mod = types.ModuleType("cash_test_anchored_function_defined_again")
    sys.modules[mod.__name__] = mod

    def run(source: str = SOURCE):
        exec(source, mod.__dict__)
        return mod

    yield run
    del sys.modules[mod.__name__]


def test_the_same_definition_is_the_one_named_now(module):
    mod = module()
    shared = {"fitted": True}
    value = [mod.column_ratio, shared]
    module("\n" + SOURCE)

    copy = with_anchor_refs(value, {"shared": shared})
    assert copy[0] is mod.column_ratio
    assert copy[1] == AnchorRef("shared")
    pickle.dumps(copy)


def test_other_code_is_the_function_held(module):
    mod = module()
    held = mod.column_ratio
    shared = {"fitted": True}
    module(SOURCE.replace("x * 2", "-x"))

    copy = with_anchor_refs([held, shared], {"shared": shared})
    assert copy[0] is held
    assert copy[1] == AnchorRef("shared")
