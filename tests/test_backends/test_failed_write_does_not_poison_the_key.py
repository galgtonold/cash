"""One failed write must not brick a cache key for the rest of the session.

A failed future used to stay in ``_pending`` forever, and ``wait(key)`` runs on
every lookup of that key -- so a single write failure re-raised on every
subsequent ``get()`` until the process ended, INCLUDING after the underlying
condition had cleared. Measured before the fix, with writes failing then
recovering:

    call 1  -> returned 42
    call 2  -> RAISED CacheBackendError
    call 3  -> RAISED CacheBackendError
    call 4  -> RAISED CacheBackendError
    -- writes work again from here --
    call 5  -> RAISED CacheBackendError
    call 6  -> RAISED CacheBackendError

So a transient antivirus lock or a momentarily full disk permanently bricked
one cached function, surfacing as ``CacheBackendError`` raised out of the
user's own call, with a kernel restart the only way back. A cache that makes
code uncallable is the exact inverse of what a cache is for -- and note the
value was computable throughout: call 1 proved it.

The failure is still recorded. ``_run_task`` appends it to the discarded-writes
registry as it happens, which is what the badge row and ``%cash_stats`` read,
so nothing is hidden by declining to re-raise it forever.
"""
from __future__ import annotations

import pytest

from cash.backends._base import PendingWrites
from cash.exceptions import CashCacheStoreFailedWarning

pytestmark = pytest.mark.expects_failed_writes


class Boom(RuntimeError):
    pass


def _failing(pw: PendingWrites, key: str = "k") -> None:
    pw.submit(key, lambda: (_ for _ in ()).throw(Boom("disk full")))


def test_a_failure_surfaces_once_not_forever():
    """The core regression: the second lookup must not re-raise."""
    pw = PendingWrites()
    try:
        _failing(pw)
        with pytest.warns(CashCacheStoreFailedWarning):
            pw.wait("k")

        # Before the fix this raised too, and kept raising for the life of the
        # process. Nothing is pending any more -- the write did not land.
        pw.wait("k")
        pw.wait("k")
    finally:
        pw.shutdown(wait=True)


def test_the_key_works_again_once_writes_recover():
    """A transient failure must not outlive the condition that caused it."""
    pw = PendingWrites()
    try:
        _failing(pw)
        with pytest.warns(CashCacheStoreFailedWarning):
            pw.wait("k")

        done = []
        pw.submit("k", lambda: done.append(1))
        pw.wait("k")                      # must not raise the OLD failure
        assert done == [1], "the recovered write did not run"
    finally:
        pw.shutdown(wait=True)


def test_a_live_write_submitted_after_the_failure_is_not_discarded():
    """The drop must only remove the future it actually waited on.

    Between the failed ``result()`` and taking the lock, another thread can
    submit a fresh write for the same key. Dropping whatever is in the dict at
    that point -- rather than the specific future -- would throw away a live
    write and lose an entry that was about to land.
    """
    pw = PendingWrites()
    try:
        _failing(pw)
        landed = []
        # Replace the pending future before anyone waits on the failed one.
        pw.submit("k", lambda: landed.append("new"))
        pw.wait("k")

        assert landed == ["new"], "the replacement write was lost"
    finally:
        pw.shutdown(wait=True)


def test_an_unrelated_key_is_unaffected():
    """Control arm: the drop must be scoped to the failing key."""
    pw = PendingWrites()
    try:
        _failing(pw, "bad")
        ok = []
        pw.submit("good", lambda: ok.append(1))

        with pytest.warns(CashCacheStoreFailedWarning):
            pw.wait("bad")
        pw.wait("good")

        assert ok == [1]
    finally:
        pw.shutdown(wait=True)


def test_the_failure_is_still_recorded_for_reporting():
    """Declining to re-raise forever must not cost the evidence.

    The badge row and %cash_stats both read the discarded-writes registry, so
    this is the channel that replaced the permanent re-raise.
    """
    from cash.backends import _base

    _base.reset_discarded_writes()
    pw = PendingWrites()
    try:
        _failing(pw)
        with pytest.warns(CashCacheStoreFailedWarning):
            pw.wait("k")
        pw.wait("k")          # consumed; still no re-raise

        recorded = _base.discarded_writes()
        assert [k for k, _ in recorded] == ["k"], recorded
    finally:
        pw.shutdown(wait=True)
        _base.reset_discarded_writes()
