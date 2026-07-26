"""The runtime half of sub-expression caching (CAS-243).

``__cash_call__(fn)`` resolves a callee to the thing that should actually be
called. Structural eligibility is decided from the AST; this is the
*object-level* gate, which needs the live callable in hand:

- an ordinary function -> a cached counterpart,
- one already ``@cash.cache``-decorated -> itself, untouched (it is already on
  this path; wrapping again would mint a second key for the same work),
- a builtin or anything not worth keying -> itself.

The resolver must never be the reason user code breaks: anything it does not
understand is handed back unchanged.
"""

import time

import pytest

import cash
from cash.notebook.call_interception import CallCache
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


@pytest.fixture
def call_cache(tmp_path):
    return CallCache(cash.Cash(cache_dir=str(tmp_path / "cc")))


def test_undecorated_function_is_cached(call_cache):
    """The point of the feature: the body runs once across two identical calls."""
    calls = []

    def compute(x):
        calls.append(x)
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)   # above the cost-model floor
        return x + 1

    cached = call_cache.resolve(compute)
    assert cached(5) == 6
    assert cached(5) == 6
    assert calls == [5], "body re-ran; the call was not cached"


def test_different_arguments_are_separate_entries(call_cache):
    calls = []

    def compute(x):
        calls.append(x)
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        return x + 1

    cached = call_cache.resolve(compute)
    assert cached(1) == 2
    assert cached(2) == 3
    assert cached(1) == 2
    assert calls == [1, 2], "argument identity was not part of the key"


def test_already_decorated_function_is_returned_unchanged(call_cache, tmp_path):
    """Must be identity — a second wrapper would double-key the same call."""
    other = cash.Cash(cache_dir=str(tmp_path / "other"))

    @other.cache
    def compute(x):
        return x + 1

    assert call_cache.resolve(compute) is compute


def test_builtins_are_returned_unchanged(call_cache):
    """A loop body calling len()/str() thousands of times must not pay for keys."""
    for fn in (len, str, range, print, isinstance):
        assert call_cache.resolve(fn) is fn


def test_non_function_callables_are_returned_unchanged(call_cache):
    """Classes and arbitrary callables are out of scope; hand them back."""

    class Widget:
        def __call__(self):
            return 1

    widget = Widget()
    assert call_cache.resolve(Widget) is Widget
    assert call_cache.resolve(widget) is widget
    assert call_cache.resolve(None) is None


def test_wrapper_is_reused_for_the_same_function(call_cache):
    """Resolving twice must not build a fresh wrapper each time.

    In a loop this runs once per iteration; re-wrapping would allocate, redo
    source introspection, and split the cache across wrappers.
    """
    def compute(x):
        return x + 1

    assert call_cache.resolve(compute) is call_cache.resolve(compute)


def test_exceptions_propagate_and_are_not_cached(call_cache):
    """A failure must not become a cached value, and must raise identically.

    The sleep is load-bearing: below the cost-model floor nothing is stored at
    all, so ``calls == [1, 1]`` would hold whether or not failures are cached
    and the test would prove nothing.
    """
    calls = []

    def compute(x):
        calls.append(x)
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        raise ValueError("boom")

    cached = call_cache.resolve(compute)
    for _ in range(2):
        with pytest.raises(ValueError, match="boom"):
            cached(1)
    assert calls == [1, 1], "a raising call was suppressed or cached"


def test_unhashable_arguments_still_execute(call_cache):
    """If the key cannot be built the call must still run, not fail."""
    def compute(gen):
        return sum(gen)

    cached = call_cache.resolve(compute)
    assert cached(iter([1, 2, 3])) == 6
