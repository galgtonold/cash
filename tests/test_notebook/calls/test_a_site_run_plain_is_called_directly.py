"""A call site the many-cheap-calls guard runs plain gets the function itself.

``d = [f(i) for i in range(200_000)]`` sent every call through cash's wrapper
even after the guard had decided the site runs plain: a warnings relay and a
record per call, ~8 us each, so the first run took 1.6 s where the plain
kernel took 0.03 s. Once the site is plain, ``resolve`` hands back the
function, which the user's own line then calls; the badge still counts every
call.
"""

from __future__ import annotations

import warnings

import pytest

import cash
from cash.notebook import call_unit as cu
from cash.notebook.call_interception import CallSite
from tests._call_cache import make_call_cache

SITE = CallSite(
    source="f(i)",
    free_names=frozenset({"f", "i"}),
    occurrence_index=0,
    computed_arg_positions=(0,),
    local_arg_positions=(0,),
)
N = cu._GUARD_AFTER_CALLS + cu._PLAIN_SAMPLES + 200


@pytest.fixture
def call_cache(tmp_path):
    cc = make_call_cache(cash.Cash(cache_dir=str(tmp_path / "cc")))
    cc.set_sites([SITE])
    return cc


def _plain_after_guard(call_cache, fn):
    """Call *fn* through ``resolve`` until the guard decides; return what
    ``resolve`` gives from then on."""
    for i in range(N):
        assert call_cache.resolve(fn, 0)(i) == fn(i)
    return call_cache.resolve(fn, 0)


def test_the_function_itself_once_the_site_runs_plain(call_cache):
    def f(i):
        return i * 2

    assert _plain_after_guard(call_cache, f) is f
    events = call_cache.drain_call_log()
    assert sum(e["calls"] for e in events) == N + 1, "a call made without the wrapper went uncounted"
    plain = [e for e in events if e["ran_plain"]]
    assert len(plain) == 1 and plain[0]["execution_time"] > 0


def test_a_new_statement_run_goes_through_the_wrapper_again(call_cache):
    def f(i):
        return i * 2

    assert _plain_after_guard(call_cache, f) is f
    call_cache.set_sites([SITE])
    assert call_cache.resolve(f, 0) is not f


def test_another_function_at_the_site_is_not_passed_through(call_cache):
    def f(i):
        return i * 2

    def g(i):
        return i * 3

    assert _plain_after_guard(call_cache, f) is f
    assert call_cache.resolve(g, 0) is not g


def test_after_the_log_is_drained_calls_are_logged_again(call_cache):
    def f(i):
        return i * 2

    assert _plain_after_guard(call_cache, f) is f
    call_cache.drain_call_log()
    for i in range(5):
        call_cache.resolve(f, 0)(i)
    assert sum(e["calls"] for e in call_cache.drain_call_log()) == 5


def test_calls_counted_before_the_next_statement_run_reach_the_log(call_cache):
    """``resolve`` counts a plain call in a slot of its own; a statement run
    starting over must not drop what it counted."""

    def f(i):
        return i * 2

    assert _plain_after_guard(call_cache, f) is f
    for i in range(7):
        call_cache.resolve(f, 0)(i)
    call_cache.set_sites([SITE])
    events = call_cache.drain_call_log()
    assert sum(e["calls"] for e in events) == N + 8


def test_a_warning_from_a_plain_call_names_the_users_line(call_cache):
    def f(i):
        warnings.warn("careful", UserWarning, stacklevel=2)
        return i

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn = _plain_after_guard(call_cache, f)
        caught.clear()
        fn(1)  # this line: what the rewritten cell runs
    assert [w.filename for w in caught] == [__file__]
