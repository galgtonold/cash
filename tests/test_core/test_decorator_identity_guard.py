"""``@cash.cache`` must not store an identity-coupled result (CAS-245).

A ``Figure`` is only correct while it IS the object pyplot's process-wide
registry points at.  The RAM tier deep-copies on store, and
``Figure.__setstate__`` re-registers the COPY as the current figure — so the
act of *caching* redirects ``plt.gcf()`` away from the object the caller holds.
The user then draws on their figure while a later bare ``plt.savefig()`` writes
the cache's private snapshot.  Silently, on the FIRST call, during the store —
which is why "nothing is cached yet, it can't be that" sends you to the wrong
place.

Measured before the fix, three arms in one process:

    no caching     gcf-is-fig=True   object=5185B  pyplot=5185B  MATCH
    @c.cache       gcf-is-fig=False  object=5185B  pyplot=7442B  DIVERGED

``statement/processor.py`` and ``call_unit.py`` had gated on
``identity_coupled_reason`` for a while; the decorator was the remaining hole.
See ``tests/test_notebook/test_call_interception_identity_guard.py`` for the
call-interception half.

**Why these tests do not assert object identity between two calls.** The RAM
tier deep-copies on *both* store and retrieval, so ``first is not second``
holds whether or not the guard fired — an identity-only assertion passes
against a completely disabled guard.  Every test here asserts on an external
observable instead: the bytes pyplot writes, or an execution counter.
"""
from __future__ import annotations

import asyncio
import time
import warnings

import pytest

import cash
from cash.exceptions import CashCacheIneffectiveWarning
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

pytestmark = pytest.mark.libraries


def _draw_and_compare(fig, ax, tmp_path, label):
    """Draw on *fig*, then save it twice: via the object, and via pyplot.

    Returns ``(gcf_is_fig, bytes_match)``.  The two routes agree only while
    pyplot's registry still points at the caller's figure.
    """
    ax.bar(["a", "b"], [3, 6])
    via_object = tmp_path / f"{label}_object.png"
    via_pyplot = tmp_path / f"{label}_pyplot.png"
    fig.savefig(via_object)
    plt.savefig(via_pyplot)
    return plt.gcf() is fig, via_object.read_bytes() == via_pyplot.read_bytes()


def _make_builder():
    """A figure factory that CREATES without drawing, plus its call counter.

    Creation must be split from drawing: a function that draws and returns in
    one go leaves nothing for the caller to add, so the two savefig routes
    agree even when the registry has been hijacked, and the test goes vacuous.
    """
    calls = []

    def build(n):
        calls.append(n)
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        fig, ax = plt.subplots()
        return fig, ax

    return build, calls


def test_an_uncached_figure_keeps_pyplots_registry(tmp_path):
    """Control arm: this is the behaviour the cached arms must match."""
    plt.close("all")
    build, _ = _make_builder()
    fig, ax = build(1)
    is_current, bytes_match = _draw_and_compare(fig, ax, tmp_path, "control")
    assert is_current, "plain Python already fails — the harness is wrong, not cash"
    assert bytes_match


def test_a_cached_figure_does_not_hijack_pyplots_current_figure(tmp_path):
    plt.close("all")
    c = cash.Cash(cache_dir=str(tmp_path / "cache"))
    build, _ = _make_builder()
    fig, ax = c.cache(build)(1)

    is_current, bytes_match = _draw_and_compare(fig, ax, tmp_path, "cached")
    assert is_current, "caching detached plt.gcf() from the returned figure"
    assert bytes_match, (
        "fig.savefig() and plt.savefig() wrote different images — pyplot is "
        "writing the cache's deep copy, not the figure the caller drew on"
    )


def test_the_figure_is_never_stored(tmp_path):
    """The guard must REFUSE, not merely hand back an equivalent object.

    Two calls with the same argument: the body has to run both times.  This is
    what distinguishes "not stored" from "stored, and the hit deep-copied".
    """
    plt.close("all")
    c = cash.Cash(cache_dir=str(tmp_path / "cache"))
    build, calls = _make_builder()
    cached = c.cache(build)

    cached(1)
    cached(1)
    assert calls == [1, 1], f"expected two real executions, got {calls}"


def test_an_ordinary_result_still_caches(tmp_path):
    """Non-regression: the gate must not refuse everything.

    Without this, a guard that always returned True would pass every other
    test in this file.
    """
    c = cash.Cash(cache_dir=str(tmp_path / "cache"))
    calls = []

    def compute(n):
        calls.append(n)
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        return n * 10

    cached = c.cache(compute)
    assert cached(3) == 30
    assert cached(3) == 30
    assert calls == [3], "an ordinary value stopped caching"


def test_an_axes_array_is_refused_too(tmp_path):
    """``fig, axes = plt.subplots(2, 2)`` binds an object-array of Axes.

    A top-level type check alone would miss it and cache the array, which
    drags the Figure along.  Covered by the bounded container scan.
    """
    plt.close("all")
    c = cash.Cash(cache_dir=str(tmp_path / "cache"))
    calls = []

    def build_grid(n):
        calls.append(n)
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        _fig, axes = plt.subplots(2, 2)
        return axes

    cached = c.cache(build_grid)
    cached(1)
    cached(1)
    assert calls == [1, 1], f"the Axes array was cached: {calls}"


def test_the_refusal_warns_once(tmp_path):
    """Silently declining leaves the user wondering why nothing speeds up."""
    plt.close("all")
    c = cash.Cash(cache_dir=str(tmp_path / "cache"))
    build, _ = _make_builder()
    cached = c.cache(build)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cached(1)
        cached(1)

    ours = [w for w in caught if issubclass(w.category, CashCacheIneffectiveWarning)
            and "not cached" in str(w.message)]
    assert len(ours) == 1, f"expected exactly one refusal warning, got {len(ours)}"
    assert "matplotlib Figure" in str(ours[0].message)


def test_the_async_wrapper_refuses_too(tmp_path):
    """The async store path is a separate site and needs its own gate."""
    plt.close("all")
    c = cash.Cash(cache_dir=str(tmp_path / "cache"))
    calls = []

    async def build(n):
        calls.append(n)
        await asyncio.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        fig, ax = plt.subplots()
        return fig, ax

    cached = c.cache(build)

    async def run_twice():
        await cached(1)
        await cached(1)

    asyncio.run(run_twice())
    assert calls == [1, 1], f"the async path cached the figure: {calls}"
