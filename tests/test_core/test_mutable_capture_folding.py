"""CAS-104: closures from one factory with different MUTABLE captures must not
share a cache key — but accumulator captures must not make keys drift.

Read-only mutable captures (a weights list, a config dict, an array) are
folded by content hash; captures the body reassigns or may mutate in place
keep the old skip behavior.
"""
from __future__ import annotations

import numpy as np
import pytest

from cash import Cash, FileBackend


def _cash(tmp_path):
    return Cash(backend=FileBackend(cache_dir=str(tmp_path)))


def _make_scorer(weights):
    def score(x):
        return sum(w * x for w in weights)
    return score


def test_list_capture_factories_do_not_collide(tmp_path):
    c = _cash(tmp_path)
    f = c.cache(_make_scorer([1, 2]))      # 3x
    g = c.cache(_make_scorer([10, 20]))    # 30x
    assert f(5) == 15
    assert g(5) == 150, "closures over different list captures shared a key"
    assert f(5) == 15


def test_dict_capture_factories_do_not_collide(tmp_path):
    c = _cash(tmp_path)

    def make(cfg):
        def apply(x):
            return x * cfg["factor"]
        return apply

    f = c.cache(make({"factor": 2}))
    g = c.cache(make({"factor": 50}))
    assert f(3) == 6
    assert g(3) == 150, "closures over different dict captures shared a key"


def test_ndarray_capture_factories_do_not_collide(tmp_path):
    c = _cash(tmp_path)

    def make(arr):
        def total(x):
            return float((arr * x).sum())
        return total

    f = c.cache(make(np.ones(5)))
    g = c.cache(make(np.full(5, 100.0)))
    assert f(2) == 10.0
    assert g(2) == 1000.0, "closures over different array captures shared a key"


def test_read_only_capture_still_hits(tmp_path):
    c = _cash(tmp_path)
    calls = {"n": 0}

    def make(weights):
        def score(x):
            calls["n"] += 1
            return sum(w * x for w in weights)
        return score

    f = c.cache(make([1, 2]))
    assert f(5) == 15
    assert f(5) == 15
    assert calls["n"] == 1, "unchanged read-only capture must keep hitting"


def test_accumulator_capture_key_does_not_drift(tmp_path):
    """A capture the body mutates (log.append) must keep the old skip
    behavior: the key stays stable so identical calls still hit."""
    c = _cash(tmp_path)
    runs = {"n": 0}

    def make(log):
        def work(x):
            runs["n"] += 1
            log.append(x)
            return x * 2
        return work

    f = c.cache(make([]))
    assert f(4) == 8
    assert f(4) == 8
    assert runs["n"] == 1, "accumulator capture made the key drift (miss)"


def test_external_mutation_of_folded_capture_invalidates(tmp_path):
    """Bonus semantic of per-call content hashing: mutating a read-only
    capture from OUTSIDE the function invalidates its entries."""
    c = _cash(tmp_path)
    weights = [1, 2]
    f = c.cache(_make_scorer(weights))
    assert f(5) == 15
    weights.append(3)          # external mutation, body never mutates
    assert f(5) == 30, "externally mutated capture served a stale result"


def test_nonlocal_counter_capture_unaffected(tmp_path):
    """Reassigned (nonlocal) captures were already excluded — control."""
    c = _cash(tmp_path)

    def make():
        n = 0

        def bump(x):
            nonlocal n
            n += 1
            return x + 1
        return bump

    f = c.cache(make())
    assert f(1) == 2
    assert f(1) == 2
