"""`CallUnit._ttl_fresh` edge semantics (CAS-268).

The end-to-end contract is pinned in
`tests/test_notebook_integration/test_call_unit_ttl.py`. These cover the three
decisions a real kernel cannot reach without sleeping or corrupting an entry,
each of which is a place a plausible implementation goes subtly wrong.
"""
from __future__ import annotations

import time

import pytest

from cash.backends import InMemoryBackend
from cash.core import Cash
from cash.notebook.call_unit import CallUnit


def _unit(ttl):
    cash = Cash(backend=InMemoryBackend(), register_magic=False)
    return CallUnit(cash, ctx_provider=lambda: None, ttl_provider=lambda: ttl)


def test_no_ttl_never_expires_an_entry():
    """`None` is "no TTL", the path every unannotated statement takes."""
    assert _unit(None)._ttl_fresh({"timestamp": 0.0}) is True


def test_ttl_zero_expires_without_consulting_the_clock():
    """`ttl=0` is a REQUEST ("expire immediately"), not an absent setting.

    Decided without arithmetic on purpose: a same-tick re-read can measure
    `age == 0.0` on a coarse timer, and `0.0 > 0` is False -- which would hand
    back the very entry `ttl=0` exists to reject. An entry stamped *now* is
    the case that catches it.
    """
    assert _unit(0)._ttl_fresh({"timestamp": time.time()}) is False


def test_a_falsy_ttl_is_not_treated_as_absent():
    """The distinction `is not None` protects, stated directly. Reading `if
    ttl:` here would make `ttl=0` mean "no TTL" -- which is precisely what
    CAS-221 was at the statement layer."""
    assert _unit(0)._ttl_fresh({"timestamp": time.time()}) is False
    assert _unit(None)._ttl_fresh({"timestamp": time.time()}) is True


@pytest.mark.parametrize("age, ttl, fresh", [
    (0.0, 30, True),      # just written
    (10.0, 30, True),     # inside the window
    (100.0, 30, False),   # past it
])
def test_expiry_tracks_the_recorded_timestamp(age, ttl, fresh):
    metadata = {"timestamp": time.time() - age}
    assert _unit(ttl)._ttl_fresh(metadata) is fresh


@pytest.mark.parametrize("metadata", [
    {},                          # no timestamp recorded
    {"timestamp": None},
    {"timestamp": "not-a-time"},
])
def test_an_entry_without_a_usable_timestamp_expires(metadata):
    """Fail SAFE, not fast. An unreadable timestamp reads as the epoch, so the
    entry expires under any TTL rather than being served forever to a caller
    who has explicitly asked for freshness. Treating it as fresh would be the
    same silent-staleness failure this ticket exists to remove.
    """
    assert _unit(30)._ttl_fresh(metadata) is False


def test_a_unit_built_without_a_ttl_provider_behaves_as_before():
    """Every direct construction predating this parameter -- tests included --
    must keep the old "TTL is not consulted" behaviour."""
    cash = Cash(backend=InMemoryBackend(), register_magic=False)
    unit = CallUnit(cash, ctx_provider=lambda: None)

    assert unit._ttl_fresh({"timestamp": 0.0}) is True
