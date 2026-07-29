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

The decorator has the same hole when a user writes ``@cash.cache`` by hand
(confirmed by the same probe) — that is filed separately. This guards the path
that applies caching *without the user asking*, which is the one that owes a
higher duty of care.

**Deliberately still exercising the no-site decorator-fallback branch of
``resolve()``, unlike every other ``test_call_interception_*`` file** (a CAS-
243 review, round 2, asked all of them to register a real site and reconsider
whether the fallback still earns its keep — this file's answer is "yes, kept,
on purpose, for now"). ``CallUnit``'s post-execution refusals
(``CallUnit._storable``) are still a stub returning ``True`` -- Task 6's job,
not this one's -- so a call routed through a REAL site does not apply this
guard yet and these tests would fail (matching
``tests/test_notebook_integration/test_cache_calls_figure_guard.py``'s
tracked, ``strict=True`` xfail for the same reason). Registering a site here
would mean marking every test in this file xfail instead of the one
integration test that already covers it, so the no-site fallback path is kept
under deliberate test here until Task 6 lands ``CallUnit``'s own guard --
at which point this file should be migrated to real sites like its siblings,
and the integration xfail removed.
"""
from __future__ import annotations

import pytest

import cash
from cash.notebook.call_interception import CallCache

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


def test_a_figure_returning_call_does_not_hijack_pyplot(call_cache):
    """After the call, pyplot's current figure must still be the returned one."""
    def make_fig():
        fig, ax = plt.subplots()
        return fig

    fig = call_cache.resolve(make_fig)()
    assert plt.gcf() is fig, (
        "storing the result registered a deep copy as pyplot's current figure; "
        "a later bare plt.savefig() would write that copy instead of this figure"
    )


def test_a_figure_returning_call_is_not_cached(call_cache):
    """Two calls must produce two distinct live figures, not a restored copy."""
    def make_fig():
        import time
        time.sleep(0.2)   # above the cost-model floor, or nothing is stored
        fig, ax = plt.subplots()
        return fig

    cached = call_cache.resolve(make_fig)
    first, second = cached(), cached()
    assert first is not second, "a Figure was served from cache"


def test_ordinary_results_are_still_cached(call_cache):
    """Positive control: the guard must not disable caching in general."""
    calls = []

    def compute(x):
        import time
        calls.append(x)
        time.sleep(0.2)
        return x + 1

    cached = call_cache.resolve(compute)
    assert cached(3) == 4
    assert cached(3) == 4
    assert calls == [3], "the identity guard suppressed ordinary caching"


def test_a_container_of_figures_is_also_refused(call_cache):
    """`fig, axes = plt.subplots(2, 2)` shapes hide the Figure in a tuple."""
    def make_pair():
        import time
        time.sleep(0.2)   # above the cost-model floor
        fig, ax = plt.subplots()
        return fig, ax

    cached = call_cache.resolve(make_pair)
    first, second = cached(), cached()
    assert first[0] is not second[0], "a Figure inside a tuple was served from cache"
