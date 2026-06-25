"""Logically-identical calls written in different forms must share one key.

`f(1)`, `f(1, 10)`, `f(1, y=10)`, `f(x=1, y=10)` are the same call; without
signature-binding they each produce a different cache key (a wasteful miss, the
same effectiveness gap `functools.lru_cache` has). Cash binds the call to the
function signature and applies defaults so all these forms collapse to one key.
kwargs supplied in different orders collapse too.

Normalization is best-effort: anything `inspect.signature(...).bind(...)` can't
handle (builtins, `*args` passthroughs, deliberately mismatched calls) falls
back to the raw argument form, so behavior never regresses for those.
"""
from __future__ import annotations

import tempfile

from cash import Cash, FileBackend


def _cash() -> Cash:
    return Cash(backend=FileBackend(cache_dir=tempfile.mkdtemp()))


# Module-level so instances pickle by reference (local classes are unpicklable,
# which would silently disable caching and make the method test vacuous).
class _Model:
    def __init__(self, scale):
        self.scale = scale


def test_positional_vs_keyword_same_key():
    c = _cash()

    @c.cache
    def f(x, y):
        return x + y

    assert f.explain(1, 2).cache_key == f.explain(1, y=2).cache_key
    assert f.explain(1, 2).cache_key == f.explain(x=1, y=2).cache_key
    assert f.explain(1, 2).cache_key == f.explain(y=2, x=1).cache_key


def test_omitted_default_equals_explicit_default():
    c = _cash()

    @c.cache
    def f(x, y=10):
        return x + y

    assert f.explain(1).cache_key == f.explain(1, 10).cache_key
    assert f.explain(1).cache_key == f.explain(1, y=10).cache_key
    # A non-default value must still differ.
    assert f.explain(1).cache_key != f.explain(1, 11).cache_key


def test_kwargs_order_independent():
    c = _cash()

    @c.cache
    def k(a, b, c_=3):
        return (a, b, c_)

    assert k.explain(a=1, b=2).cache_key == k.explain(b=2, a=1).cache_key


def test_normalization_collapses_calls_to_one_compute():
    c = _cash()
    calls = {"n": 0}

    @c.cache
    def f(x, y=10):
        calls["n"] += 1
        return x + y

    assert f(1) == 11
    assert f(1, 10) == 11          # same call, different form
    assert f(1, y=10) == 11        # same call, keyword form
    assert f(x=1, y=10) == 11      # all keyword
    assert calls["n"] == 1, "all four forms should hit the same entry"


def test_distinct_calls_still_distinct():
    c = _cash()
    calls = {"n": 0}

    @c.cache
    def f(x, y=10):
        calls["n"] += 1
        return x + y

    f(1)
    f(2)
    f(1, 11)
    assert calls["n"] == 3


def test_var_positional_passthrough_does_not_crash():
    """A *args function has a signature bind() accepts; extra positionals stay
    positional and distinct arities remain distinct keys."""
    c = _cash()
    calls = {"n": 0}

    @c.cache
    def f(*nums):
        calls["n"] += 1
        return sum(nums)

    assert f(1, 2, 3) == 6
    assert f(1, 2, 3) == 6
    assert calls["n"] == 1
    assert f(1, 2) == 3
    assert calls["n"] == 2


def test_var_keyword_order_independent():
    c = _cash()
    calls = {"n": 0}

    @c.cache
    def f(**opts):
        calls["n"] += 1
        return sorted(opts.items())

    f(a=1, b=2)
    f(b=2, a=1)          # same **kwargs, different order
    assert calls["n"] == 1
    f(a=1, b=3)
    assert calls["n"] == 2


def test_unpicklable_default_falls_back_to_raw_and_still_caches():
    """Applying defaults must not fold an unpicklable default into the key and
    thereby stop a call from caching that hashed fine before. The raw form is
    retried so f(1) keeps caching."""
    c = _cash()
    calls = {"n": 0}
    sentinel = lambda: None  # noqa: E731 - unpicklable default

    @c.cache
    def f(x, _cb=sentinel):
        calls["n"] += 1
        return x * 2

    assert f(1) == 2
    assert f(1) == 2          # must hit despite the unpicklable default
    assert calls["n"] == 1


def test_method_self_still_keys_per_instance():
    """Binding includes `self`; two instances with different state stay
    distinct, the same instance hits, and positional/keyword forms collapse."""
    c = _cash()
    calls = {"n": 0}

    @c.cache
    def predict(model, x):
        calls["n"] += 1
        return x * model.scale

    m1 = _Model(2)
    m2 = _Model(3)
    assert predict(m1, 5) == 10
    assert predict(m2, 5) == 15        # different instance state -> distinct key
    assert predict(m1, x=5) == 10      # keyword form -> same key as predict(m1, 5)
    assert calls["n"] == 2, "only the two distinct instances should compute"
