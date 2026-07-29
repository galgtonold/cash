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

**Sites are registered** (``call_cache.set_sites([...])``) before every
``resolve()`` call below, matching how production actually reaches
``resolve()`` — ``_code_and_tree_for_execution`` never binds ``__cash_call__``
into ``user_ns`` without a non-empty site table. A CAS-243 review (round 2,
Critical C3) found this file previously tested ONLY the no-site fallback
branch (the pre-Task-5 decorator path), which is unreachable in real notebook
execution — that gap is exactly why a wrapper-cache staleness bug (C1) shipped
with a green suite. See ``test_call_interception_no_site_fallback.py`` for the
one file that deliberately keeps testing the fallback branch on its own terms.
"""

import time

import pytest

import cash
from cash.notebook.call_interception import CallCache, CallSite
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


@pytest.fixture
def call_cache(tmp_path):
    return CallCache(cash.Cash(cache_dir=str(tmp_path / "cc")))


def _site(source="compute(x)", names=("compute", "x"), computed_arg_positions=(0,)):
    """A one-arg call site: the argument is treated as a computed (non-Name)
    value, so its live value -- not some name's lineage -- is what the key
    discriminates on. Matches how every test below actually calls the wrapper
    (with a literal, not a variable read from a notebook's ``user_ns``).
    """
    return CallSite(
        source=source, free_names=frozenset(names), occurrence_index=0,
        computed_arg_positions=computed_arg_positions,
    )


def test_undecorated_function_is_cached(call_cache):
    """The point of the feature: the body runs once across two identical calls."""
    calls = []

    def compute(x):
        calls.append(x)
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)   # above the cost-model floor
        return x + 1

    call_cache.set_sites([_site()])
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

    call_cache.set_sites([_site()])
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

    call_cache.set_sites([_site()])
    assert call_cache.resolve(compute) is compute


def test_builtins_are_returned_unchanged(call_cache):
    """A loop body calling len()/str() thousands of times must not pay for keys."""
    call_cache.set_sites([_site(source="fn()", names=("fn",), computed_arg_positions=())])
    for fn in (len, str, range, print, isinstance):
        assert call_cache.resolve(fn) is fn


def test_non_function_callables_are_returned_unchanged(call_cache):
    """Classes and arbitrary callables are out of scope; hand them back."""

    class Widget:
        def __call__(self):
            return 1

    widget = Widget()
    call_cache.set_sites([_site(source="w()", names=("w",), computed_arg_positions=())])
    assert call_cache.resolve(Widget) is Widget
    assert call_cache.resolve(widget) is widget
    assert call_cache.resolve(None) is None


def test_wrapper_is_reused_for_the_same_function(call_cache):
    """Resolving twice must not build a fresh wrapper each time.

    In a loop this runs once per iteration; re-wrapping would allocate, redo
    source introspection, and split the cache across wrappers. The SAME site
    object is registered before each ``resolve()`` call, matching how a loop
    body actually re-executes: a fresh, but EQUAL, ``CallSite`` each
    iteration (see the ``_wrappers`` keying note in ``call_interception.py``).
    """
    def compute(x):
        return x + 1

    call_cache.set_sites([_site()])
    first = call_cache.resolve(compute)
    call_cache.set_sites([_site()])
    second = call_cache.resolve(compute)
    assert first is second


def test_two_sites_at_the_same_index_get_distinct_wrappers_and_keys(call_cache):
    """The exact shape CAS-243 review C1 found broken: two different
    statements resolving the SAME function at ``site_index=0`` (every
    statement's own site list starts at 0) must not share a wrapper, because
    each wrapper closes over its own ``CallSite`` and must key independently.
    """
    calls = []

    def compute(x):
        calls.append(x)
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        return x + 1

    call_cache.set_sites([_site(source="compute(a)", names=("compute", "a"))])
    wrapped_a = call_cache.resolve(compute, site_index=0)

    call_cache.set_sites([_site(source="compute(a + 100)", names=("compute", "a"))])
    wrapped_b = call_cache.resolve(compute, site_index=0)

    assert wrapped_a is not wrapped_b, (
        "two different call sites at the same index shared one wrapper"
    )
    assert wrapped_a(5) == 6
    assert wrapped_b(5) == 6, (
        "the second site's wrapper served the first site's cached value "
        "(CAS-243 review C1 -- distinct sites must key distinctly even when "
        "called with the identical literal argument)"
    )
    assert calls == [5, 5], "both distinct sites should have executed their own call"


def test_editing_a_cell_recomputes_instead_of_reusing_the_stale_wrapper(call_cache):
    """The production repro from CAS-243 review C1, at the ``CallCache``
    level: re-registering site_index 0 with a DIFFERENT site (an edited
    statement) must not resolve to the wrapper the previous statement built.
    """
    calls = []

    def compute(x):
        calls.append(x)
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        return x + 1

    # Mirrors the exact production repro: `compute(a)` -- `a` a bare Name, so
    # `computed_arg_positions=()` and the key depends on `a`'s LINEAGE, not on
    # whatever literal the wrapper happens to be called with. `compute(a +
    # 100)` is a BinOp, not a bare Name, so `computed_arg_positions=(0,)` and
    # the live value now IS part of the key. A wrapper stale from the first
    # site ignores its argument's value entirely (still thinks position 0 is
    # a bare Name) and so cannot discriminate `(1)` from `(101)` -- the two
    # calls collide on ONE key regardless of which literal is passed, which is
    # precisely why the "different arguments" test above did not also catch
    # C1: that test never changes a site's `computed_arg_positions` shape,
    # only which value flows through an unchanged one.
    call_cache.set_sites([_site(source="compute(a)", names=("compute", "a"), computed_arg_positions=())])
    call_cache.resolve(compute, site_index=0)(1)

    call_cache.set_sites([_site(source="compute(a + 100)", names=("compute", "a"), computed_arg_positions=(0,))])
    result = call_cache.resolve(compute, site_index=0)(101)

    assert result == 102, "the edited statement's call was served a stale value"
    assert calls == [1, 101], "the edited statement's call did not actually run"


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

    call_cache.set_sites([_site()])
    cached = call_cache.resolve(compute)
    for _ in range(2):
        with pytest.raises(ValueError, match="boom"):
            cached(1)
    assert calls == [1, 1], "a raising call was suppressed or cached"


def test_unhashable_arguments_still_execute(call_cache):
    """If the key cannot be built the call must still run, not fail."""
    def compute(gen):
        return sum(gen)

    call_cache.set_sites([_site(source="compute(gen)", names=("compute", "gen"))])
    cached = call_cache.resolve(compute)
    assert cached(iter([1, 2, 3])) == 6
