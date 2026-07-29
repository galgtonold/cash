"""Call interception must honour the refusals cash already supports (CAS-243).

Found while analysing whether the directive could be made default-on. Both are
live bugs in the opt-in feature, not hypotheticals:

- ``@cash.stateful`` is *the* documented way to say "never cache this
  function". ``decide_cacheability`` honours it for statements. ``CallCache``
  did not, so a stateful callee was cached and returned a stale value on the
  FIRST run: ``[1, 1]`` where plain Python gives ``[1, 2]``.
- ``# @cash:no-cache`` is an explicit instruction about a statement. With
  ``cache-calls`` also present, the expensive call inside it was cached anyway
  — the directive silently overrode the refusal.

Both are the same shape as the Figure bug: the call path bypasses a guard the
statement path enforces.

**Sites are registered before every ``resolve()`` call**, matching production
(see ``test_call_interception_runtime.py``'s module docstring for why a
no-site ``resolve()`` call is not representative of real notebook execution).
For the refusal tests this also strengthens the claim being made: with a real
site registered, these prove the refusal wins over the CallUnit real-site
path too, not merely over the decorator-fallback path a no-site call takes.
"""
from __future__ import annotations

import time

import pytest

import cash
from cash.notebook.call_interception import CallCache, CallSite


@pytest.fixture
def call_cache(tmp_path):
    return CallCache(cash.Cash(cache_dir=str(tmp_path / "cc")))


def _site(source="compute(x)", names=("compute", "x"), computed_arg_positions=(0,)):
    return CallSite(
        source=source, free_names=frozenset(names), occurrence_index=0,
        computed_arg_positions=computed_arg_positions,
    )


def test_stateful_callee_is_never_wrapped(call_cache):
    """`@cash.stateful` means never cache. The call path must obey it too."""
    calls = []

    @cash.stateful
    def next_id():
        calls.append(1)
        time.sleep(0.2)          # above the cost-model floor
        return len(calls)

    call_cache.set_sites([_site(source="next_id()", names=("next_id",), computed_arg_positions=())])
    resolved = call_cache.resolve(next_id)
    assert resolved is next_id, "a @stateful function was wrapped for caching"
    assert resolved() == 1
    assert resolved() == 2, "a @stateful function was served from cache"


def test_stateful_callee_is_not_recorded_as_intercepted(call_cache):
    """It must not show up on the badge as something cash cached."""
    @cash.stateful
    def next_id():
        return 1

    call_cache.set_sites([_site(source="next_id()", names=("next_id",), computed_arg_positions=())])
    call_cache.resolve(next_id)
    assert call_cache.wrapped_names == set()


def test_ordinary_callee_still_wrapped(call_cache):
    """Positive control: the refusal check must not disable caching generally."""
    calls = []

    def compute(x):
        calls.append(x)
        time.sleep(0.2)
        return x + 1

    call_cache.set_sites([_site()])
    cached = call_cache.resolve(compute)
    assert cached(3) == 4
    assert cached(3) == 4
    assert calls == [3], "the refusal check suppressed ordinary caching"
