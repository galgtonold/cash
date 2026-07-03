"""CAS-105: c.cache(obj.method) — a bound method's instance state must be
part of the cache key.

Before the fix, ``self`` never appeared in ``args`` for bound-method
decoration, so two instances with different state shared one entry and
silently returned each other's results.
"""
from __future__ import annotations

import threading
import warnings

import pytest

from cash import Cash, FileBackend
from cash.exceptions import CashCacheIneffectiveWarning


class Repo:
    def __init__(self, base):
        self.base = base

    def compute(self, x):
        return self.base + x


def _cash(tmp_path):
    return Cash(backend=FileBackend(cache_dir=str(tmp_path)))


def test_two_instances_do_not_collide(tmp_path):
    c = _cash(tmp_path)
    m1 = c.cache(Repo(10).compute)
    m2 = c.cache(Repo(1000).compute)
    assert m1(1) == 11
    assert m2(1) == 1001, "two instances with different state shared a key"
    assert m1(1) == 11


def test_same_instance_hits(tmp_path):
    c = _cash(tmp_path)
    calls = {"n": 0}

    class Counted:
        def __init__(self):
            self.base = 5

        def compute(self, x):
            calls["n"] += 1
            return self.base + x

    inst = Counted()
    m = c.cache(inst.compute)
    assert m(1) == 6
    assert m(1) == 6
    assert calls["n"] == 1, "same instance + same args must hit"


def test_instance_mutation_invalidates(tmp_path):
    """self is hashed per call: changing instance state must recompute."""
    c = _cash(tmp_path)
    inst = Repo(10)
    m = c.cache(inst.compute)
    assert m(1) == 11
    inst.base = 500
    assert m(1) == 501, "mutated instance state served a stale result"


def test_register_hasher_applies_to_bound_self(tmp_path):
    c = _cash(tmp_path)
    calls = {"n": 0}

    class Loader:
        def __init__(self, dataset_id, junk):
            self.dataset_id = dataset_id
            self.junk = junk  # incidental state excluded by the hasher

        def load(self, version):
            calls["n"] += 1
            return f"{self.dataset_id}:{version}"

    c.register_hasher(Loader, lambda ld: f"ds:{ld.dataset_id}")
    m1 = c.cache(Loader("abc", junk=1).load)
    m2 = c.cache(Loader("abc", junk=999).load)  # same identity per hasher
    assert m1(1) == "abc:1"
    assert m2(1) == "abc:1"
    assert calls["n"] == 1, "hasher-equal instances must share the entry"


def test_unhashable_self_warns_and_stays_distinct(tmp_path):
    c = _cash(tmp_path)

    class Holder:
        def __init__(self, base):
            self.base = base
            self.lock = threading.Lock()  # unpicklable

        def compute(self, x):
            return self.base + x

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m1 = c.cache(Holder(10).compute)
        m2 = c.cache(Holder(1000).compute)
        assert m1(1) == 11
        assert m2(1) == 1001, "identity fallback must keep instances distinct"
        assert any(issubclass(x.category, CashCacheIneffectiveWarning) for x in w), (
            "unhashable bound self must warn once"
        )


def test_in_class_decoration_unchanged(tmp_path):
    """Control: in-class decoration passes self via args and stays correct."""
    c = _cash(tmp_path)

    class Box:
        def __init__(self, base):
            self.base = base

        @c.cache
        def compute(self, x):
            return self.base + x

    assert Box(10).compute(1) == 11
    assert Box(1000).compute(1) == 1001
