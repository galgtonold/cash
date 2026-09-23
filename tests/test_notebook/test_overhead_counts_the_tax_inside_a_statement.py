"""Cash's own time INSIDE a statement counts as overhead, not as your compute.

Round 30, r30s4: a paired Restart & Run All measured cash 370 s slower than
plain Jupyter while ``%cash_stats`` reported 210 s of overhead -- "understated
again (harmonise loop alone +240 s)". Two other testers reported the same
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
from unittest.mock import MagicMock

import pytest
from traitlets.config import Configurable

from cash.backends import InMemoryBackend
from cash.core import Cash
from cash.notebook.cache_status import CacheStatus
from cash.notebook.ipython.magics import CashMagics


class _MockShell(Configurable):
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns
        self.display_pub = type("MockDisplayPub", (), {"publish": MagicMock()})()


@pytest.fixture
def magics_fixture():
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = _MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    yield magics, shell, backend
    backend.clear()
    shell.user_ns.clear()


def _stats_json(magics, capsys) -> dict:
    capsys.readouterr()
    magics.cash_stats("json")
    return json.loads(capsys.readouterr().out.strip())


def test_the_tax_inside_a_statement_is_overhead(magics_fixture, capsys):
    """A 10 s cell in which the user's code ran 6 s and cash spent 4 s hashing
    inside it: 4 s of tax plus the 0.2 s around it, not 0.2 s."""
    magics, _shell, _backend = magics_fixture
    magics._update_session_stats(
        [{"status": CacheStatus.COMPUTED, "execution_time": 9.8, "cash_tax": 4.0, "code": "m = fit(big)"}],
        cell_total_time=10.0,
    )
    data = _stats_json(magics, capsys)
    assert data["total_overhead"] == pytest.approx(4.2)
    assert data["total_compute_time"] == pytest.approx(5.8)


def test_a_statement_without_a_tax_is_unchanged(magics_fixture, capsys):
    """Control: the accounting for a statement cash spent nothing inside."""
    magics, _shell, _backend = magics_fixture
    magics._update_session_stats(
        [{"status": CacheStatus.COMPUTED, "execution_time": 9.8, "code": "m = fit()"}],
        cell_total_time=10.0,
    )
    data = _stats_json(magics, capsys)
    assert data["total_overhead"] == pytest.approx(0.2)
    assert data["total_compute_time"] == pytest.approx(9.8)


def test_the_tax_never_makes_compute_negative(magics_fixture, capsys):
    """A tax larger than the statement's wall time (clock skew, partial
    timing) floors at zero rather than crediting cash with negative work."""
    magics, _shell, _backend = magics_fixture
    magics._update_session_stats(
        [{"status": CacheStatus.COMPUTED, "execution_time": 1.0, "cash_tax": 5.0, "code": "x = f()"}],
        cell_total_time=1.2,
    )
    data = _stats_json(magics, capsys)
    assert data["total_compute_time"] == pytest.approx(0.0)
    assert data["total_overhead"] == pytest.approx(1.2)


def test_a_verified_saving_is_credited_at_the_users_cost(magics_fixture, capsys):
    """The baseline a later hit is credited against is the user's code time,
    not the time cash spent keying it: crediting the tax would pay cash for
    its own overhead."""
    magics, _shell, _backend = magics_fixture
    magics._update_session_stats(
        [{"status": CacheStatus.COMPUTED, "execution_time": 9.8, "cash_tax": 4.0, "code": "m = fit(big)"}],
        cell_total_time=10.0,
    )
    magics._update_session_stats(
        [{"status": CacheStatus.RESTORED, "saved_time": 9.8, "execution_time": 0.0, "code": "m = fit(big)"}],
        cell_total_time=0.3,
    )
    data = _stats_json(magics, capsys)
    assert data["total_verified_saved"] == pytest.approx(5.8)
