"""Idle joblib workers a statement leaves behind do not stop later
statements being served.

joblib's default backend keeps its worker processes (and a resource tracker)
alive after ``Parallel(n_jobs=2)(...)`` or a scikit-learn ``n_jobs=`` call
returns, waiting for the next job. They ran nothing of the statement's any
more, yet counted as a process it left running: no statement after it was
served while they lived. In the E10 notebook ``lda.fit_transform`` (about
100 s) ran again after every restart for that reason. A worker of a kept
pool with no work pending, and a resource tracker, now count as finished.
"""

from __future__ import annotations

import time

import pytest

from cash.backends.persistence_policy import PersistencePolicy
from tests._cell_driver import run_cash_cell

pytest.importorskip("joblib")
pytest.importorskip("psutil")

# ``os.system`` first, in the same statement: a process start cash hears of
# on every platform (loky starts its workers through one it hears of on
# Windows only).
PARALLEL = (
    "import os\nfrom joblib import Parallel, delayed\n"
    "r = (os.system('exit 0'), Parallel(n_jobs=2)(delayed(abs)(-i) for i in range(4)))[1]"
)
LATER = "y = sum(r) + 1"


def _reasons(magics) -> list[str]:
    statements = magics._last_cell_metrics.get("statements", [])
    return [reason for s in statements for reason in s.get("uncacheable_reasons") or ()]


@pytest.fixture
def fresh_pool():
    from joblib.externals.loky import get_reusable_executor

    get_reusable_executor().shutdown(wait=True)
    yield
    get_reusable_executor().shutdown(wait=True)


def test_a_statement_after_a_parallel_call_is_served(cash_magics, mock_shell, fresh_pool, monkeypatch):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    run_cash_cell(cash_magics, PARALLEL)
    assert mock_shell.user_ns["r"] == [0, 1, 2, 3]
    run_cash_cell(cash_magics, LATER)
    run_cash_cell(cash_magics, LATER)
    assert mock_shell.user_ns["y"] == 7
    assert not any("has not finished" in reason for reason in _reasons(cash_magics)), _reasons(cash_magics)
    statuses = [str(s.get("status")) for s in cash_magics._last_cell_metrics.get("statements", [])]
    assert statuses == ["RESTORED"], statuses


def test_idle_workers_and_trackers_are_known_and_busy_ones_are_not(fresh_pool):
    import psutil
    from joblib import Parallel, delayed
    from joblib.externals.loky import reusable_executor

    from cash.notebook.write_observer import idle_helper_pids

    Parallel(n_jobs=2)(delayed(abs)(-i) for i in range(4))
    executor = reusable_executor._executor
    workers = set(executor._processes)
    assert workers and workers <= idle_helper_pids()
    trackers = {child.pid for child in psutil.Process().children() if "resource_tracker" in " ".join(child.cmdline())}
    assert trackers <= idle_helper_pids()
    # Work pending: its workers run the user's work.
    future = executor.submit(time.sleep, 1.0)
    assert not set(executor._processes) & idle_helper_pids()
    future.result()
