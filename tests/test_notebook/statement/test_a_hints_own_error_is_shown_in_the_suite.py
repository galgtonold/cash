"""An error inside the per-item rerun hint is logged at warning in the suite.

A hint never breaks a statement, so its errors are a debug line. In a test
that waits for a hint, a debug line is invisible: the macOS 3.10 failure of
test_a_block_changed_in_the_middle_is_named showed only "no hint". The suite
logs them as warnings, which pytest shows with the failing test.
"""

from __future__ import annotations

import logging

from cash.notebook import per_item_reruns
from tests._cell_driver import run_cash_cell


def test_the_error_is_a_warning_and_the_statement_still_runs(cash_magics, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(per_item_reruns.PerItemReruns, "after_run", broken)
    monkeypatch.setattr(per_item_reruns, "MIN_COST_S", 0.0)
    monkeypatch.setattr("cash.notebook.statement.processor.PER_ITEM_MIN_COST_S", 0.0)
    with caplog.at_level(logging.WARNING, logger="cash.notebook.statement.processor"):
        run_cash_cell(cash_magics, "import numpy as np\nimgs = np.zeros((300, 4))\nclean = imgs * 2")

    assert cash_magics.shell.user_ns["clean"].shape == (300, 4)
    found = [r for r in caplog.records if "per-item rerun" in r.getMessage()]
    assert found and found[0].levelno == logging.WARNING, caplog.records
    assert "boom" in caplog.text
