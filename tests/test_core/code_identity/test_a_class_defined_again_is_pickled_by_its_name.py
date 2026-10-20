"""A value holding a class whose cell ran again since, or an instance of
it, is pickled by the class's name, but only when the class its name holds
now is the same definition.

Pickle refuses such a class ("not the same object as __main__.Fit"), so an
instance of it was never stored (E6: a fitted pipeline holding a
``FeatureFromRegressor`` defined in a cell run twice). Pickled by its name,
it loads as an instance of the class defined now, which is right only when
that one does the same: a changed method, base, class attribute or slot is
refused, and the value is not stored.
"""

from __future__ import annotations

import pickle
import sys
import types

import pytest

from cash import kept_state, stored_pickle
from cash.value_hash import compute_hash, is_identity_fallback_hash

pytestmark = [pytest.mark.core]

FIRST = '''
class Base:
    pass

class Fit(Base):
    """Scales."""
    unit = "m"

    def __init__(self, k=2):
        self.k = k

    def go(self, x):
        return x * self.k

    @property
    def twice(self):
        return 2 * self.k

    @staticmethod
    def name():
        return "fit"

    @classmethod
    def make(cls):
        return cls(3)

    def __repr__(self):
        return f"Fit({self.k}) " + super().__repr__()
'''


@pytest.fixture
def module():
    """A module whose cell `run` runs (again), as a notebook does."""
    mod = types.ModuleType("cash_test_class_defined_again")
    sys.modules[mod.__name__] = mod

    def run(source: str):
        exec(source, mod.__dict__)
        return mod

    yield run
    del sys.modules[mod.__name__]


def _held_then_run_again(module, again: str) -> tuple[list, types.ModuleType]:
    mod = module(FIRST)
    held = [mod.Fit(5), mod.Fit]
    module(again)
    assert type(held[0]) is not mod.Fit
    return held, mod


def test_the_same_definition_is_pickled_by_name_and_loads_as_the_one_defined_now(module):
    # Lines apart: the cell above it grew.
    held, mod = _held_then_run_again(module, "\n\n" + FIRST)
    with pytest.raises(pickle.PicklingError):
        pickle.dumps(held)

    loaded = pickle.loads(kept_state.dumps(held, protocol=5))
    assert type(loaded[0]) is mod.Fit
    assert loaded[1] is mod.Fit
    assert loaded[0].go(2) == 10
    assert loaded[0].twice == 10


def test_its_digest_is_its_content_and_the_same_in_every_process(module):
    held, _ = _held_then_run_again(module, FIRST)
    digest = compute_hash(held)
    assert not is_identity_fallback_hash(held, digest)
    # The next process runs the same cells: a new class, held, defined again.
    again, mod = _held_then_run_again(module, FIRST)
    assert held[1] is not again[1]
    assert stored_pickle.object_digest(again) == stored_pickle.object_digest(held)
    # What it loads as after a restart, the class defined now, digests the same.
    fresh = [mod.Fit(5), mod.Fit]
    assert stored_pickle.object_digest(fresh) == stored_pickle.object_digest(held)
    assert compute_hash(fresh) == compute_hash(held)
    assert kept_state.dumps(fresh, protocol=5) == kept_state.dumps(held, protocol=5)
    # The module names the class defined now all along: an object pickled
    # in the same dump finds it there.
    seen = []

    class Looks:
        def __reduce__(self):
            seen.append(mod.Fit)
            return (int, (1,))

    kept_state.dumps([*held, Looks()], protocol=5)
    assert seen and all(cls is fresh[1] for cls in seen)
    assert mod.Fit is fresh[1]


def test_a_base_defined_again_with_it_counts_when_it_is_the_same(module):
    # Running the cell again defines Base again too: Fit's new base is the new Base.
    held, mod = _held_then_run_again(module, FIRST)
    assert held[1].__bases__[0] is not mod.Base
    assert kept_state.same_class(held[1], mod.Fit)


@pytest.mark.parametrize(
    "again",
    [
        FIRST.replace("return x * self.k", "return x * self.k + 1"),
        FIRST.replace('unit = "m"', 'unit = "cm"'),
        FIRST.replace('"""Scales."""', '"""Scales more."""'),
        FIRST.replace("return 2 * self.k", "return 3 * self.k"),
        FIRST.replace('return "fit"', 'return "fit2"'),
        FIRST.replace("return cls(3)", "return cls(4)"),
        FIRST.replace("class Base:\n    pass", "class Base:\n    k = 1"),
        FIRST.replace("class Fit(Base):", "class Fit(Base, dict):"),
        FIRST + "\nFit.extra = 1\n",
        FIRST.replace('unit = "m"', 'unit = "m"\n    __slots__ = ("j",)'),
    ],
    ids=[
        "method",
        "attribute",
        "docstring",
        "property",
        "staticmethod",
        "classmethod",
        "base",
        "bases",
        "added",
        "slots",
    ],
)
def test_another_definition_is_refused(module, again):
    held, _ = _held_then_run_again(module, again)
    with pytest.raises(kept_state.DefinedAgain, match="Fit"):
        kept_state.dumps(held, protocol=5)
    with pytest.raises(kept_state.DefinedAgain):
        stored_pickle.object_digest(held)


def test_a_class_body_it_cannot_compare_is_refused(module):
    # A dataclass's fields are objects of the library's: not compared, so refused.
    source = "import dataclasses\n@dataclasses.dataclass\nclass Fit:\n    k: int = 2\n"
    mod = module(source)
    held = [mod.Fit()]
    module(source)
    with pytest.raises(kept_state.DefinedAgain):
        kept_state.dumps(held, protocol=5)


REDUCING = """
def rebuild(k):
    return Obj(k)

class Obj:
    def __init__(self, k):
        self.k = k

    def __reduce__(self):
        return (rebuild, (self.k,))
"""


def test_an_object_its_own_reduce_would_load_otherwise_is_named(module):
    mod = module(REDUCING)
    held = {"a": [mod.Obj(3)], "b": 1}
    assert kept_state.loads_otherwise(held, mod.__name__) is None
    module("def rebuild(k):\n    return Obj(k + 1)\n")
    assert kept_state.loads_otherwise(held, mod.__name__) == "Obj"
    # A class of another module is not this one's to judge.
    assert kept_state.loads_otherwise(held, "elsewhere") is None


WRAPPING = """
import functools

def wrap(f, label):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        return (label, f(*args, **kwargs))
    return wrapped

class Fit:
    keys = {"transform", "fit"}

    def transform(self, x):
        return x * 2

    transform = wrap(transform, "out")

    @functools.cached_property
    def size(self):
        return 3
"""


def test_a_wrapped_method_sets_and_a_cached_property_are_compared_through(module):
    mod = module(WRAPPING)
    held = [mod.Fit()]
    module(WRAPPING)
    assert kept_state.same_class(type(held[0]), mod.Fit)
    loaded = pickle.loads(kept_state.dumps(held, protocol=5))
    assert type(loaded[0]) is mod.Fit
    assert loaded[0].transform(2) == ("out", 4)


@pytest.mark.parametrize(
    "again",
    [
        WRAPPING.replace("return x * 2", "return x * 3"),
        WRAPPING.replace('wrap(transform, "out")', 'wrap(transform, "other")'),
        WRAPPING.replace('{"transform", "fit"}', '{"transform"}'),
        WRAPPING.replace("return 3", "return 4"),
    ],
    ids=["wrapped", "wrapper_closure", "set", "cached_property"],
)
def test_a_change_inside_them_is_refused(module, again):
    mod = module(WRAPPING)
    held = [mod.Fit()]
    module(again)
    assert not kept_state.same_class(type(held[0]), mod.Fit)


def test_a_scikit_learn_transformer_defined_again_is_stored():
    pytest.importorskip("sklearn")
    source = (
        "from sklearn.base import BaseEstimator, TransformerMixin\n"
        "class Similarity(BaseEstimator, TransformerMixin):\n"
        "    def __init__(self, n=2):\n"
        "        self.n = n\n"
        "    def fit(self, X, y=None, sample_weight=None):\n"
        "        return self\n"
        "    def transform(self, X):\n"
        "        return X\n"
    )
    mod = types.ModuleType("cash_test_class_defined_again_sklearn")
    sys.modules[mod.__name__] = mod
    try:
        exec(source, mod.__dict__)
        held = [mod.Similarity(4)]
        exec(source, mod.__dict__)
        loaded = pickle.loads(kept_state.dumps(held, protocol=5))
        assert type(loaded[0]) is mod.Similarity
        assert loaded[0].n == 4
    finally:
        del sys.modules[mod.__name__]


ANNOTATED = """
class Fit:
    k: int = 2

    def go(self, x: "Later") -> int:
        return x
"""


def test_annotations_are_compared_as_names_where_not_defined_yet(module):
    mod = module(ANNOTATED)
    old = mod.Fit
    module(ANNOTATED)
    assert kept_state.same_class(old, mod.Fit)
    module(ANNOTATED.replace("-> int", "-> str"))
    assert not kept_state.same_class(old, mod.Fit)
    module(ANNOTATED.replace("k: int", "k: float"))
    assert not kept_state.same_class(old, mod.Fit)
