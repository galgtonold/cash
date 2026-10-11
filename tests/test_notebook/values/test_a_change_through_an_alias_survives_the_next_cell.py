"""A change made through one name is part of the value the other name holds.

``log = history; log.append('start')`` changes ``history``. When a later cell
changes ``history`` itself, the check for a stale value compares its content
with what the cells above left; that record has to include the change made
through ``log``, or the list is rebuilt from ``history = []``, losing
``'start'`` and leaving ``log`` on the old list.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

CASES = {
    "a statement": "history.append('step')",
    "a loop": "for i in range(2):\n    history.append(i)",
}


@pytest.mark.parametrize("name", list(CASES))
def test_run_all_keeps_the_change_made_through_the_alias(cash_magics, name):
    cells = ["history = []\nlog = history", "log.append('start')", CASES[name], "ok = log is history"]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    ns = cash_magics.shell.user_ns
    assert ns["history"][0] == "start"
    assert ns["ok"] is True
