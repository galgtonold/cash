"""A cache write that fails must never disappear silently.

Writes are asynchronous, so a failure lands on a future rather than at the call
site. ``wait(key)`` re-raises it — but only if someone asks for that key again.
Nothing does at process exit, and ``shutdown()`` used to wait on the executor
without ever inspecting the futures, while ``wait_all()`` swallowed exceptions
with a bare ``pass``.

The result was a failure mode with no evidence anywhere: a write fails, the
entry is permanently absent, and the user sees only unexplained cache misses.
Whatever else is decided about durability (CAS-209), a failure must leave a
trace.
"""
from __future__ import annotations

import logging

import pytest

from cash.backends._base import PendingWrites


class Boom(RuntimeError):
    pass


def _failing_writes() -> PendingWrites:
    pw = PendingWrites()
    pw.submit("k-bad", lambda: (_ for _ in ()).throw(Boom("disk full")))
    return pw


class TestFailureIsRecorded:
    def test_failed_write_is_reported(self):
        pw = _failing_writes()
        try:
            pw.shutdown(wait=True)
            failed = pw.failed_writes()
            assert [k for k, _ in failed] == ["k-bad"]
            assert isinstance(failed[0][1], Boom)
        finally:
            pw.shutdown(wait=True)

    def test_successful_writes_report_nothing(self):
        pw = PendingWrites()
        try:
            pw.submit("k-ok", lambda: None)
            pw.wait("k-ok")
            assert pw.failed_writes() == []
        finally:
            pw.shutdown(wait=True)

    def test_shutdown_warns_about_an_unobserved_failure(self, caplog):
        """The failure has to reach a level a user would actually see.

        It was previously logged at DEBUG from inside the backend, if at all —
        invisible in every normal configuration.
        """
        pw = _failing_writes()
        with caplog.at_level(logging.WARNING, logger="cash.backends._base"):
            pw.shutdown(wait=True)
        assert any("cache write(s) failed" in r.message or
                   "cache write(s) failed" in r.getMessage()
                   for r in caplog.records), (
            f"no WARNING recorded; saw {[r.getMessage() for r in caplog.records]}"
        )

    def test_shutdown_never_raises(self):
        """shutdown() usually runs from atexit — raising there helps nobody."""
        pw = _failing_writes()
        pw.shutdown(wait=True)  # must not raise

    def test_observed_failure_still_raises_at_the_call_site(self):
        """Reporting at shutdown must not weaken wait(key)'s contract."""
        pw = _failing_writes()
        try:
            with pytest.raises(Boom):
                pw.wait("k-bad")
        finally:
            pw.shutdown(wait=True)

    def test_wait_all_does_not_raise_but_does_not_hide(self, caplog):
        """Bulk reads must survive one bad entry, while leaving a trace."""
        pw = _failing_writes()
        try:
            with caplog.at_level(logging.DEBUG, logger="cash.backends._base"):
                pw.wait_all()  # must not raise
            assert any("Pending write failed" in r.getMessage()
                       for r in caplog.records)
        finally:
            pw.shutdown(wait=True)
