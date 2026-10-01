"""A skipped upstream statement is not counted as a step in "step x/N".

The progress badge counts the upstream statements that will run ahead of the
cell's own. Upstream rows carry ``CacheStatus.SKIPPED`` (a member), and the
count compared with the string ``"SKIPPED"``; ``CacheStatus`` is a plain
``Enum``, so the comparison never matched and every skipped row inflated N.
"""

from __future__ import annotations

import warnings

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell


def _row(status):
    return {"code": "u = 1", "status": status, "is_upstream": True, "saved_time": 0.0, "total_time": 0.0}


def test_skipped_upstream_rows_do_not_count(cash_magics, monkeypatch):
    executor = cash_magics._cell_executor
    build = executor._build_pre_execution_notifications
    upstream = [_row(CacheStatus.SKIPPED), _row(CacheStatus.SKIPPED), _row(CacheStatus.RESTORED)]
    monkeypatch.setattr(
        executor, "_build_pre_execution_notifications", lambda raw, pre, up: build(raw, pre, list(upstream))
    )
    seen = []
    arm = executor._badges.arm_progress

    def spy(metrics, display_id, step, total, code):
        seen.append((step, total))
        return arm(metrics, display_id, step, total, code)

    monkeypatch.setattr(executor._badges, "arm_progress", spy)
    cash_magics.cash_on("")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # no notebook around the cell
        run_cash_cell(cash_magics, "a = 1\nb = a + 1")

    # One upstream statement that ran (the restored one), then the cell's two.
    assert seen == [(2, 3), (3, 3)]
