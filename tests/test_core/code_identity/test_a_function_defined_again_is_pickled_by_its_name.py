"""A value holding a function whose cell ran again since is pickled with
it by its name, but only when the function its name holds now is the same
definition.

Pickle refuses such a function ("not the same object as
__main__.column_ratio"), so a value holding it was never stored, and its
content digest fell back to its identity, new in every process. Pickled by
its name, it loads as the function defined now, which is right only when
that one does the same: a changed body, default, closure or global
namespace is refused, and the value is not stored.
"""

from __future__ import annotations

import pickle
import sys
import types

import pytest

from cash import kept_state, stored_pickle
from cash.backends.memory_backend import InMemoryBackend
from cash.value_hash import compute_hash, is_identity_fallback_hash

pytestmark = [pytest.mark.core]

FIRST = "def column_ratio(x, k=2):\n    'Ratio.'\n    return x * k\n"


@pytest.fixture
def module():
    """A module whose cell `run` runs (again), as a notebook does."""
    mod = types.ModuleType("cash_test_defined_again")
    sys.modules[mod.__name__] = mod

    def run(source: str):
        exec(source, mod.__dict__)
        return mod

    yield run
    del sys.modules[mod.__name__]


def _held_then_run_again(module, again: str) -> tuple[list, types.ModuleType]:
    mod = module(FIRST)
    held = [mod.column_ratio, 3]
    module(again)
    assert held[0] is not mod.column_ratio
    return held, mod


def test_the_same_definition_is_pickled_by_name_and_loads_as_the_one_defined_now(module):
    # Lines apart: the cell above it grew.
    held, mod = _held_then_run_again(module, "\n\n" + FIRST)
    with pytest.raises(pickle.PicklingError):
        pickle.dumps(held)

    loaded = pickle.loads(kept_state.dumps(held, protocol=5))
    assert loaded[0] is mod.column_ratio
    assert loaded[0](5) == 10


def test_its_digest_is_its_content_and_the_same_in_every_process(module):
    held, _ = _held_then_run_again(module, FIRST)
    digest = compute_hash(held)
    assert not is_identity_fallback_hash(held, digest)
    assert stored_pickle.object_digest(held) == stored_pickle.object_digest([held[0], 3])


@pytest.mark.parametrize(
    "again",
    [
        "def column_ratio(x, k=2):\n    'Ratio.'\n    return x * k + 1\n",
        "def column_ratio(x, k=3):\n    'Ratio.'\n    return x * k\n",
        "def column_ratio(x, *, k=2):\n    'Ratio.'\n    return x * k\n",
        "def column_ratio(x, k=2):\n    'Other.'\n    return x * k\n",
        "def column_ratio(x, k=2):\n    'Ratio.'\n    return x * k\ncolumn_ratio.scale = 2\n",
        "def column_ratio(x: int, k=2):\n    'Ratio.'\n    return x * k\n",
    ],
    ids=["body", "default", "keyword_only", "docstring", "attribute", "annotation"],
)
def test_another_definition_is_refused(module, again):
    held, _ = _held_then_run_again(module, again)
    with pytest.raises(kept_state.DefinedAgain, match="column_ratio"):
        kept_state.dumps(held, protocol=5)
    with pytest.raises(kept_state.DefinedAgain):
        stored_pickle.object_digest(held)


def test_a_closure_is_refused(module):
    mod = module("def make():\n    k = 2\n    def inner(x):\n        return x * k\n    return inner\n")
    old = mod.make()
    new = mod.make()
    assert not kept_state.same_definition(old, new)


def test_another_namespace_is_refused(module):
    mod = module(FIRST)
    elsewhere: dict = {}
    exec(FIRST, elsewhere)
    assert not kept_state.same_definition(mod.column_ratio, elsewhere["column_ratio"])


def test_a_ram_copy_keeps_the_function_the_value_held(module):
    held, mod = _held_then_run_again(module, "def column_ratio(x, k=2):\n    'Ratio.'\n    return -x\n")
    b = InMemoryBackend()
    b.set("k", {"held": held}, {"copy_required": True})
    hit = b.get("k")[1]["held"]
    assert hit[0] is held[0]
    assert hit[0](5) == 10
