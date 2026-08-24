"""A discarded cache write must show up on the badge, not only in %cash_stats.

The badge is the ambient surface -- it appears after every cell whether or not
you asked for it. %cash_stats is a command you have to know to type, and a user
whose writes are silently failing has no reason to suspect anything, so they
never type it. Windows spent an unknown period discarding a write on every run
(fixed in 0.4.1) and the only report was a logger warning at kernel shutdown,
which in a notebook means never.

The row appears ONLY when something failed, which is what makes it acceptable
on a surface whose own design notes warn that noise teaches people to ignore
it (see badge_renderer/_reasons.py). Zero cost in the healthy case.
"""
from __future__ import annotations

import pytest

from cash.backends import _base as _backend_base
from cash.notebook.ipython.cell_executor import discarded_writes_notification


@pytest.fixture(autouse=True)
def _clean_registry():
    _backend_base.reset_discarded_writes()
    yield
    _backend_base.reset_discarded_writes()


def _record(key: str, message: str) -> None:
    """Append as the write worker would, without needing a real failing disk."""
    _backend_base._DISCARDED_WRITES.append((key, message))


def test_no_row_when_nothing_failed():
    """The negative arm: a healthy cell must not carry a scary row.

    This is the assertion that keeps the feature acceptable on a glanceable
    surface -- if it fired routinely it would train the reader to skip the
    badge, which is the thing the badge cannot afford.
    """
    row, seen = discarded_writes_notification(0)
    assert row is None
    assert seen == 0


def test_a_failed_write_produces_a_warning_row():
    _record("model:abc", "CacheBackendError: [WinError 5] Access is denied")
    row, seen = discarded_writes_notification(0)

    assert row is not None
    assert row["status"] == "WARNING"
    assert seen == 1
    # Says what it costs, not just that it happened -- a count alone is
    # bookkeeping, and the reader has no reason to care about bookkeeping.
    assert "recompute" in row["code"]


def test_only_new_failures_are_reported():
    """The watermark must not re-report a failure the user already saw.

    Without this the row would reappear on every subsequent cell for the rest
    of the session, which is exactly the noise that gets a badge ignored.
    """
    _record("k1", "CacheBackendError: disk full")
    row, seen = discarded_writes_notification(0)
    assert row is not None and seen == 1

    row2, seen2 = discarded_writes_notification(seen)
    assert row2 is None, "re-reported a failure already announced"
    assert seen2 == 1

    _record("k2", "CacheBackendError: disk full")
    row3, seen3 = discarded_writes_notification(seen2)
    assert row3 is not None, "a NEW failure must be announced"
    assert seen3 == 2
    assert "1 cache write failed" in row3["code"], row3["code"]


def test_the_count_is_of_new_failures_not_the_total():
    """Three failures at once read as three, not as the running total."""
    for i in range(3):
        _record(f"k{i}", "CacheBackendError: disk full")
    row, _ = discarded_writes_notification(0)
    assert "3 cache writes failed" in row["code"], row["code"]


def test_the_row_is_ascii_and_short_enough_for_print_mode():
    """Print mode caps a row at 80 chars and the badge asserts ASCII.

    Both constraints are documented on the sibling notifications; a row that
    breaks either is silently mangled rather than loudly wrong, so it is worth
    a test rather than a comment.
    """
    _record("k", "CacheBackendError: [WinError 5] Access is denied")
    row, _ = discarded_writes_notification(0)
    code = row["code"]
    assert code.isascii(), code
    # 80 is the print renderer's row cap. Measured truncation at 89 chars:
    # the row rendered as "... See %ca", losing the pointer that makes it
    # actionable. A too-generous ceiling here let that through once already.
    assert len(code) <= 80, f"{len(code)} chars, print mode will truncate: {code}"


def test_a_broken_backend_registry_cannot_break_the_cell(monkeypatch):
    """A diagnostic must never be able to take down a user's execution.

    Same guarantee the staleness notifications carry, and for the same reason:
    the worst acceptable outcome is a missed warning.
    """
    def boom():
        raise RuntimeError("registry exploded")

    monkeypatch.setattr(_backend_base, "discarded_writes", boom)
    row, seen = discarded_writes_notification(7)
    assert row is None
    assert seen == 7, "watermark must survive a failed read unchanged"
