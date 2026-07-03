"""CAS-109: wrappers must key on the function they execute, not on the live
``module.qualname`` registry slot.

Redefining a function (notebook cell re-run) or decorating two lambdas (which
share the ``<lambda>`` qualname) overwrites the shared registry slot; before
the fix a stale wrapper then stored its results under the NEW function's
state hash, poisoning it — and two lambdas collided outright.
"""
from __future__ import annotations

import pytest

from cash import Cash, FileBackend


def _cash(tmp_path):
    return Cash(backend=FileBackend(cache_dir=str(tmp_path)))


def test_redefinition_does_not_poison_new_wrapper(tmp_path):
    c = _cash(tmp_path)

    def score_v1(x):
        return x + 1

    def score_v2(x):
        return x + 100

    # Emulate a notebook cell re-run: same name/qualname, new body.
    score_v2.__qualname__ = score_v1.__qualname__
    score_v2.__name__ = score_v1.__name__

    w1 = c.cache(score_v1)
    assert w1(1) == 2
    w2 = c.cache(score_v2)      # re-registration overwrites the registry slot
    assert w1(1) == 2           # stale wrapper keeps ITS OWN identity
    assert w2(1) == 101, "v1's result was planted under v2's state hash"
    assert w1(1) == 2


def test_two_lambdas_distinct_entries(tmp_path):
    c = _cash(tmp_path)
    f = c.cache(lambda x: x + 1)
    g = c.cache(lambda x: x + 100)
    assert f(1) == 2
    assert g(1) == 101, "different lambdas shared one cache entry"
    assert f(1) == 2


def test_same_line_lambdas_distinct_entries(tmp_path):
    c = _cash(tmp_path)
    f, g = c.cache(lambda x: x + 1), c.cache(lambda x: x + 100)  # one source line
    assert f(1) == 2
    assert g(1) == 101, (
        "same-line lambdas share source text; the code fingerprint must "
        "disambiguate them"
    )


def test_named_function_keys_stable_across_instances(tmp_path):
    """The pin equals the registration-time source hash for named functions,
    so persisted entries keep hitting from a fresh Cash instance."""
    calls = {"n": 0}

    def compute(x):
        calls["n"] += 1
        return x * 3

    c1 = _cash(tmp_path)
    assert c1.cache(compute)(2) == 6
    assert calls["n"] == 1

    c2 = _cash(tmp_path)
    assert c2.cache(compute)(2) == 6
    assert calls["n"] == 1, "fresh instance missed a persisted entry"


def test_clean_reregistration_recomputes(tmp_path):
    """Without a stale alias in play, redefinition must still invalidate."""
    c = _cash(tmp_path)

    def metric_v1(x):
        return x + 1

    def metric_v2(x):
        return x + 100

    metric_v2.__qualname__ = metric_v1.__qualname__
    metric_v2.__name__ = metric_v1.__name__

    assert c.cache(metric_v1)(1) == 2
    assert c.cache(metric_v2)(1) == 101
