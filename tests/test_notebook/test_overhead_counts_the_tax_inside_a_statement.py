"""Cash's own time INSIDE a statement counts as overhead, not as your compute.

A paired Restart & Run All measured cash 370 s slower than
plain Jupyter while ``%cash_stats`` reported 210 s of overhead -- "understated
again (harmonise loop alone +240 s)". Two other reports had the same
shape.

Overhead was computed as ``cell wall time - the execution_time of the
statements that ran``, and ``execution_time`` is raw wall time: everything
cash spends inside a statement -- recording file reads, keying and hashing the
arguments of the calls it routes, storing them -- sat inside that number and
cancelled out. Cash already measures that tax (``file_tracker.
tracking_seconds`` and ``CallUnit.overhead_s``, used for ``compute_cost``);
the session totals just did not subtract it.
"""

from __future__ import annotations

import json

import pytest

from cash.notebook.cache_status import CacheStatus


def _stats_json(magics, capsys) -> dict:
    capsys.readouterr()
    magics.cash_stats("json")
    return json.loads(capsys.readouterr().out.strip())


def test_the_tax_inside_a_statement_is_overhead(cash_magics, capsys):
    """A 10 s cell in which the user's code ran 6 s and cash spent 4 s hashing
    inside it: 4 s of tax plus the 0.2 s around it, not 0.2 s."""
    cash_magics._update_session_stats(
        [{"status": CacheStatus.COMPUTED, "execution_time": 9.8, "cash_tax": 4.0, "code": "m = fit(big)"}],
        cell_total_time=10.0,
    )
    data = _stats_json(cash_magics, capsys)
    assert data["total_overhead"] == pytest.approx(4.2)
    assert data["total_compute_time"] == pytest.approx(5.8)


def test_a_statement_without_a_tax_is_unchanged(cash_magics, capsys):
    """Control: the accounting for a statement cash spent nothing inside."""
    cash_magics._update_session_stats(
        [{"status": CacheStatus.COMPUTED, "execution_time": 9.8, "code": "m = fit()"}],
        cell_total_time=10.0,
    )
    data = _stats_json(cash_magics, capsys)
    assert data["total_overhead"] == pytest.approx(0.2)
    assert data["total_compute_time"] == pytest.approx(9.8)


def test_the_tax_never_makes_compute_negative(cash_magics, capsys):
    """A tax larger than the statement's wall time (clock skew, partial
    timing) floors at zero rather than crediting cash with negative work."""
    cash_magics._update_session_stats(
        [{"status": CacheStatus.COMPUTED, "execution_time": 1.0, "cash_tax": 5.0, "code": "x = f()"}],
        cell_total_time=1.2,
    )
    data = _stats_json(cash_magics, capsys)
    assert data["total_compute_time"] == pytest.approx(0.0)
    assert data["total_overhead"] == pytest.approx(1.2)


def test_a_verified_saving_is_credited_at_the_users_cost(cash_magics, capsys):
    """The baseline a later hit is credited against is the user's code time,
    not the time cash spent keying it: crediting the tax would pay cash for
    its own overhead."""
    cash_magics._update_session_stats(
        [{"status": CacheStatus.COMPUTED, "execution_time": 9.8, "cash_tax": 4.0, "code": "m = fit(big)"}],
        cell_total_time=10.0,
    )
    cash_magics._update_session_stats(
        [{"status": CacheStatus.RESTORED, "saved_time": 9.8, "execution_time": 0.0, "code": "m = fit(big)"}],
        cell_total_time=0.3,
    )
    data = _stats_json(cash_magics, capsys)
    assert data["total_verified_saved"] == pytest.approx(5.8)
