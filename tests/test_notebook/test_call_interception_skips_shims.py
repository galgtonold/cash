"""Cash's own file-tracking shims must never be intercepted (CAS-246).

``file_tracker`` replaces ``open``, ``pd.read_csv`` and friends with tracking
wrappers. A wrapper is a plain ``types.FunctionType``, so the builtin exclusion
in ``resolve`` does not cover it -- and wrapping one means trying to cache a
file handle. The audit-log repro raised and wrote nothing:

    # @cash:cache-calls
    open('audit.log', 'a').write(f'run {next_seq()}\n')
    -> __cash_call__(open)('audit.log', 'a').write(...)   # CashImpurityWarning
                                                          # on cash's own shim

``_is_file_tracker_patch`` is the sentinel every install site already sets, and
which ``cache_key.py`` already reads defensively for the same reason (CAS-214,
where this shim poisoned a cache key). Reusing it means new shims are covered
with no second list to maintain -- which the coverage test below pins.

A site is registered before every ``resolve()`` call: production always has
one, and refusing a shim EVEN WHEN a real ``CallUnit`` site exists for it is
the stronger claim -- the refusal check runs before any site is consulted, so
it must win either way.
"""
from __future__ import annotations

import builtins

import pytest

import cash
from cash.notebook.call_interception import CallCache, CallSite
from cash.notebook.file_tracker import FileAccessTracker


@pytest.fixture
def call_cache(tmp_path):
    return CallCache(cash.Cash(cache_dir=str(tmp_path / "cc")))


def _site_for(fn) -> CallSite:
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", "shim")
    return CallSite(
        source=f"{name}(a)", free_names=frozenset({name, "a"}), occurrence_index=0,
        computed_arg_positions=(0,),
    )


def test_the_open_shim_is_passed_through(call_cache):
    ns: dict = {}
    with FileAccessTracker(ns):
        shim = ns.get("open", builtins.open)
        if not getattr(shim, "_is_file_tracker_patch", False):
            pytest.skip("no open shim installed in this configuration")
        call_cache.set_sites([_site_for(shim)])
        # Identity, not merely equality: only a genuinely-wrapped callee could
        # ever produce an intercepted-call event, so this alone proves the
        # shim never gets a chance to.
        assert call_cache.resolve(shim) is shim
        assert call_cache.drain_call_log() == []


def test_every_installed_shim_is_passed_through(call_cache):
    """Coverage guard: a NEW shim must not silently become interceptable.

    Walks what the tracker actually installed rather than a hand-written list,
    so adding a tracked loader cannot regress this without failing here.
    """
    import numpy, pandas

    ns: dict = {}
    with FileAccessTracker(ns):
        shims = [
            fn
            for mod in (builtins, pandas, numpy)
            for fn in vars(mod).values()
            if getattr(fn, "_is_file_tracker_patch", False)
        ]
        shims += [f for f in ns.values() if getattr(f, "_is_file_tracker_patch", False)]
        assert shims, "no shims installed; this test would be vacuous"
        for shim in shims:
            call_cache.set_sites([_site_for(shim)])
            assert call_cache.resolve(shim) is shim, (
                f"{getattr(shim, '__qualname__', shim)} would be intercepted"
            )
