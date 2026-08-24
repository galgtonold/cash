"""A discarded cache write must be visible where the user asks if caching works.

A failed cache write does not raise. The entry is simply absent, the work is
recomputed, and none of the counters in ``%cash_stats`` can show it -- a
discarded write is not a miss, it is a hit that never got the chance to exist.
The only other report is a logger warning from ``_report_failed_writes`` at
shutdown, which in a notebook means at kernel death: in practice, never.

That combination is how Windows spent an unknown period silently discarding
writes on effectively every run while every test passed and every stats line
looked healthy. The user-visible symptom was "cash didn't really save me much"
-- which nobody reports as a bug.

So ``%cash_stats`` is the one surface that has to say it, and these pin that it
does, in both the human and JSON forms, without inventing a problem when there
isn't one.
"""
from __future__ import annotations

import json

import pytest

from cash.backends._base import _DISCARDED_WRITES, discarded_writes, reset_discarded_writes


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_discarded_writes()
    yield
    reset_discarded_writes()


def _record(key, message):
    """Append as the write worker would, without needing a real failing disk."""
    _DISCARDED_WRITES.append((key, message))


def test_a_healthy_session_says_nothing_about_discarded_writes(cash_magics, capsys):
    """The negative arm: no failures, no scary line."""
    cash_magics.cash_stats("")
    out = capsys.readouterr().out
    assert "Discarded" not in out, out


def test_a_discarded_write_is_reported_with_its_cause(cash_magics, capsys):
    _record("model:abc", "CacheBackendError: [WinError 5] Access is denied")
    cash_magics.cash_stats("")
    out = capsys.readouterr().out

    assert "Discarded writes:    1" in out, out
    # The cause, not just the count -- "1 discarded write" alone is not
    # actionable, and this is the only place the user will ever see it.
    assert "WinError 5" in out, out
    # Say what it costs, or the number reads as bookkeeping.
    assert "recomputes every run" in out, out


def test_the_count_survives_more_than_one_failure(cash_magics, capsys):
    for i in range(3):
        _record(f"k{i}", f"CacheBackendError: disk full ({i})")
    cash_magics.cash_stats("")
    out = capsys.readouterr().out

    assert "Discarded writes:    3" in out, out
    assert "and 2 more" in out, out


def test_json_mode_carries_the_count(cash_magics, capsys):
    _record("k", "CacheBackendError: disk full")
    cash_magics.cash_stats("json")
    payload = json.loads(capsys.readouterr().out)

    assert payload["discarded_writes"] == 1, payload


def test_reset_of_the_registry_is_not_wired_to_stats_reset(cash_magics, capsys):
    """A stats reset must not erase evidence of a broken cache.

    ``%cash_stats reset`` forgets counters the user chose to forget. A
    discarded write is not a counter -- it is an unresolved fault, and the
    entry is still missing from disk afterwards. Clearing it here would let a
    user reset their way into believing the cache is healthy.
    """
    _record("k", "CacheBackendError: disk full")
    cash_magics.cash_stats("reset")
    capsys.readouterr()

    assert len(discarded_writes()) == 1, "a stats reset erased a real fault"
    cash_magics.cash_stats("")
    assert "Discarded writes:    1" in capsys.readouterr().out
