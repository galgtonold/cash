"""``# @cash:ttl=0`` must expire immediately (CAS-221).

The docs promise this twice — ``docs/annotations.md`` says a ``ttl`` of 0
"means 'immediately expired' — every run will be a miss" — and the decorator
path honours it, because ``core.py`` asks ``if ttl is not None``. The statement
path asked ``if ttl:``, so a ttl of 0 skipped the expiry check altogether and
the entry lived forever: the exact inversion of what was asked for.

These tests drive :class:`CacheFreshnessChecker` directly with a stub backend,
so they pin the contract without a kernel, a notebook, or a real clock wait.
"""

from __future__ import annotations

import time

import pytest

from cash.notebook.statement.freshness import CacheFreshnessChecker


class _StubBackend:
    """Return one fixed ``(metadata, value)`` pair for any key."""

    def __init__(self, metadata: dict | None, value: object) -> None:
        self._metadata = metadata
        self._value = value

    def get(self, key: str):  # noqa: ARG002 - key is irrelevant to the stub
        return self._metadata, self._value


def _checker(age_seconds: float = 0.0, value: object = "cached-value"):
    """A checker whose single entry was stored *age_seconds* ago."""
    metadata = {
        'key': 'k',
        'timestamp': time.time() - age_seconds,
        # No file deps: this suite is about TTL and nothing else.
        'file_dependencies': {},
    }
    return CacheFreshnessChecker(backend=_StubBackend(metadata, value))


def _check(checker: CacheFreshnessChecker, ttl):
    """Run a lookup with no inputs, so only the TTL branch can fire."""
    _metadata, cached_data, _elapsed = checker.check_cache(
        tracking_state=None,  # unreachable: the input-file check needs `inputs`
        cache_key='k',
        ttl=ttl,
        inputs=None,
    )
    return cached_data


def test_ttl_zero_expires_a_fresh_entry():
    """The regression. An entry stored *this instant* is already stale at ttl=0."""
    assert _check(_checker(age_seconds=0.0), ttl=0) is None


def test_ttl_zero_expires_an_entry_stored_a_moment_ago():
    """Same claim, off the same-tick boundary, so a coarse clock cannot pass it."""
    assert _check(_checker(age_seconds=0.5), ttl=0) is None


def test_ttl_zero_reports_why_it_missed():
    """The badge must attribute the miss to TTL rather than leaving it blank."""
    checker = _checker(age_seconds=0.0)
    _check(checker, ttl=0)
    assert checker.last_miss_reason is not None
    assert 'TTL' in checker.last_miss_reason


def test_ttl_none_never_expires():
    """Guard the other direction: unset TTL must not start invalidating."""
    assert _check(_checker(age_seconds=10_000.0), ttl=None) == "cached-value"


def test_positive_ttl_still_hits_within_the_window():
    """The fix must not turn every TTL'd entry into a miss."""
    assert _check(_checker(age_seconds=1.0), ttl=60) == "cached-value"


def test_positive_ttl_still_expires_past_the_window():
    """And must not stop expiring the ones that are genuinely too old."""
    assert _check(_checker(age_seconds=120.0), ttl=60) is None


@pytest.mark.parametrize('ttl', [0, 0.0])
def test_ttl_zero_expires_for_int_and_float(ttl):
    """``%cash_on ttl=0`` can deliver a float where a pragma delivers an int."""
    assert _check(_checker(age_seconds=0.0), ttl=ttl) is None
