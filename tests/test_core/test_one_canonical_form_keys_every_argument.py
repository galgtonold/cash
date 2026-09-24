"""Arguments are keyed through one canonical form, and every type in it is named.

There used to be two forms. An argument list holding a set went through one
that sorted the set and rewrote EVERY object in the arguments as its bare class
name plus its attributes; any other argument list went through a second form
that only put dicts in order. The bare class name dropped the module, so two
classes both called ``Config`` in two modules, holding the same attributes,
keyed alike as soon as a set was passed beside them, and the second call was
served the first one's result.
"""

from __future__ import annotations

import collections
import threading
import warnings

import pytest

from cash import Cash
from cash.exceptions import CashCacheIneffectiveWarning

pytestmark = [pytest.mark.core]


def _config_class(module: str) -> type:
    cls = type("Config", (), {"__init__": lambda self: setattr(self, "rate", 1)})
    cls.__module__ = module
    return cls


@pytest.fixture
def two_configs(monkeypatch):
    """Two picklable classes called ``Config``, in two different modules."""
    import sys
    import types

    made = []
    for name in ("cash_test_mod_a", "cash_test_mod_b"):
        mod = types.ModuleType(name)
        mod.Config = _config_class(name)
        monkeypatch.setitem(sys.modules, name, mod)
        made.append(mod.Config)
    return made


def test_same_named_classes_from_two_modules_key_apart_beside_a_set(tmp_path, two_configs):
    a_config, b_config = two_configs
    c = Cash(cache_dir=str(tmp_path / "cache"))
    seen = []

    @c.cache
    def describe(tags, cfg):
        seen.append(type(cfg).__module__)
        return type(cfg).__module__

    assert describe({"x"}, a_config()) == "cash_test_mod_a"
    assert describe({"x"}, b_config()) == "cash_test_mod_b"
    assert seen == ["cash_test_mod_a", "cash_test_mod_b"]


def test_containers_of_different_types_key_apart(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "cache"))

    @c.cache
    def kind(value):
        return type(value).__name__

    assert kind([1, 2]) == "list"
    assert kind((1, 2)) == "tuple"
    assert kind({1, 2}) == "set"
    assert kind(frozenset({1, 2})) == "frozenset"
    assert kind({"a": 1}) == "dict"
    assert kind(collections.OrderedDict(a=1)) == "OrderedDict"


def test_equal_values_key_alike_whatever_their_order(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "cache"))
    runs = []

    @c.cache
    def count(opts, tags):
        runs.append(1)
        return len(opts) + len(tags)

    assert count({"a": 1, "b": {2, 3}}, {"x", "y"}) == 4
    assert count({"b": {3, 2}, "a": 1}, {"y", "x"}) == 4
    assert len(runs) == 1


def _helper_with_a_lock_default(x, lock=threading.Lock()):  # the unhashable default under test
    return x


def test_a_helper_default_that_cannot_be_hashed_makes_the_call_uncached(tmp_path):
    """A helper's defaults are folded like the function's own: one that cannot
    be hashed runs the call uncached, rather than keying without it."""
    c = Cash(cache_dir=str(tmp_path / "cache"))
    runs = []

    @c.cache
    def outer(x):
        runs.append(1)
        return _helper_with_a_lock_default(x)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert outer(1) == 1
        assert outer(1) == 1
    assert len(runs) == 2
    assert any("KEY-UNHASHABLE-DEFAULT" in str(w.message) for w in caught if w.category is CashCacheIneffectiveWarning)
