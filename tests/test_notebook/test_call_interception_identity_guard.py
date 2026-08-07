"""An intercepted call must not cache an identity-coupled result (CAS-243).

Found by adversarial probing, and it produced a genuinely wrong file on disk.

``statement/processor.py`` refuses to cache a matplotlib Figure/Axes via
``identity_coupled_reason``: the RAM tier deep-copies on store, and
``Figure.__setstate__`` re-registers the COPY as pyplot's *current figure*. A
later bare ``plt.savefig()`` then writes the cache's snapshot instead of the
figure the user drew on — on the FIRST run, silently.

The call path routes through the decorator, which has no such guard, so
``# @cash:cache-calls`` reintroduced the exact failure the statement path
refuses. Measured: with the directive, ``fig.savefig()`` and ``plt.savefig()``
wrote *different images* and ``plt.gcf() is fig`` was False; without caching,
identical and True.

The decorator used to have the same hole when a user wrote ``@cash.cache`` by
hand (confirmed by the same probe). That was filed as CAS-245 and is now closed
— ``Cash._refuses_identity_coupled`` gates all four decorator store sites, with
``tests/test_core/test_decorator_identity_guard.py`` guarding it. This file
still guards the path that applies caching *without the user asking*, which is
the one that owes a higher duty of care.

**Migrated to real sites (CAS-243 Task 6).** This file used to be the one
deliberate holdout exercising only the no-site decorator-fallback branch of
``resolve()`` — ``CallUnit._storable`` was a stub returning ``True`` before
Task 6, so a call routed through a REAL ``CallSite`` had no guard at all (see
``tests/test_notebook_integration/test_cache_calls_figure_guard.py``'s
history for the matching, now-removed ``xfail``). Task 6 implemented the
guard in ``CallUnit._storable`` itself, so this file now registers sites via
``set_sites`` like its siblings (``test_call_interception_runtime.py``) and
exercises the production path end to end: ``CallCache.resolve`` ->
``CallUnit.wrap`` -> ``_storable``.

**Every test sleeps above ``CallUnit``'s ``_COST_FLOOR_S`` (0.01s) and, where
there is a second call, asserts on ``call_cache.drain_call_log()``'s
``cache_hit`` flags — not only on object identity.** An object-identity
check alone (``first is not second``) is true whether or not the guard fired:
the RAM tier deep-copies on *both* store and retrieval, so a genuine cache hit
still hands back an object distinct from the one a fresh call would produce.
Confirmed by mutation testing during Task 6: disabling the guard left every
identity-only assertion in this file passing.
"""
from __future__ import annotations

import time

import pytest

import cash
from cash.notebook.call_interception import CallCache, CallSite
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture
def call_cache(tmp_path):
    return CallCache(cash.Cash(cache_dir=str(tmp_path / "cc")))


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _site(source="make_fig()", names=("make_fig",)):
    return CallSite(source=source, free_names=frozenset(names), occurrence_index=0)


def test_a_figure_returning_call_does_not_hijack_pyplot(call_cache):
    """After the call, pyplot's current figure must still be the returned one.

    The sleep is load-bearing: the guard is only even consulted once the call
    clears ``CallUnit``'s cost floor (``wrap``'s
    ``elapsed >= _COST_FLOOR_S and self._storable(...)``); without it this
    call is never a store candidate at all and the test would pass whether or
    not the guard exists.
    """
    def make_fig():
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        fig, ax = plt.subplots()
        return fig

    call_cache.set_sites([_site()])
    fig = call_cache.resolve(make_fig)()
    assert plt.gcf() is fig, (
        "storing the result registered a deep copy as pyplot's current figure; "
        "a later bare plt.savefig() would write that copy instead of this figure"
    )


def test_a_figure_returning_call_is_not_cached(call_cache):
    """The second call must recompute, not hit -- checked via the call log,
    since a hit's deep-copied return is *always* a distinct object from the
    first call's live one, guard or no guard (see module docstring).
    """
    def make_fig():
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        fig, ax = plt.subplots()
        return fig

    call_cache.set_sites([_site()])
    cached = call_cache.resolve(make_fig)
    first, second = cached(), cached()
    assert first is not second, "a Figure was served from cache"
    assert [e["cache_hit"] for e in call_cache.drain_call_log()] == [False, False], (
        "an identity-coupled result must never be served from cache -- the "
        "second call has to recompute, not hit"
    )


def test_ordinary_results_are_still_cached(call_cache):
    """Positive control: the guard must not disable caching in general."""
    calls = []

    def compute(x):
        calls.append(x)
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        return x + 1

    call_cache.set_sites([_site(source="compute(x)", names=("compute", "x"))])
    cached = call_cache.resolve(compute)
    assert cached(3) == 4
    assert cached(3) == 4
    assert calls == [3], "the identity guard suppressed ordinary caching"
    assert [e["cache_hit"] for e in call_cache.drain_call_log()] == [False, True], (
        "an ordinary, non-identity-coupled result should still hit on the "
        "second call"
    )


def test_a_container_of_figures_is_also_refused(call_cache):
    """`fig, axes = plt.subplots(2, 2)` shapes hide the Figure in a tuple."""
    def make_pair():
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        fig, ax = plt.subplots()
        return fig, ax

    call_cache.set_sites([_site(source="make_pair()", names=("make_pair",))])
    cached = call_cache.resolve(make_pair)
    first, second = cached(), cached()
    assert first[0] is not second[0], "a Figure inside a tuple was served from cache"
    assert [e["cache_hit"] for e in call_cache.drain_call_log()] == [False, False], (
        "a container holding an identity-coupled value must never be served "
        "from cache -- the second call has to recompute, not hit"
    )
