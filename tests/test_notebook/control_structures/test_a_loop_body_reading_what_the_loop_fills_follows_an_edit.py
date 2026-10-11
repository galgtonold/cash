"""A statement of a loop body that reads what the loop fills is keyed on it.

``loss = score(history, epoch)`` reads the list ``history.append(loss * lr)``
fills on the passes before it. Each pass's ``append`` moves ``history`` on for
the passes after it, so after ``lr`` is edited the later passes are not served
the losses the old ``lr`` gave. When the loop ends, ``history`` gets the
lineage the loop gives it as a whole, as before.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

SETUP = "def score(history, epoch):\n    return sum(history) + epoch\nlr = {lr}"
SHAPES = {
    "append": "history = []\nfor epoch in range(4):\n    loss = score(history, epoch)\n    history.append(loss * lr)",
    "a helper": (
        "history = []\nfor epoch in range(4):\n    loss = score(history, epoch)\n    keep(history, loss * lr)"
    ),
}
HELPER = "\ndef keep(acc, v):\n    acc.append(v)"


@pytest.fixture
def store_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0


@pytest.mark.usefixtures("store_everything")
@pytest.mark.parametrize("shape", list(SHAPES))
def test_an_edit_above_reaches_every_pass(cash_magics, mock_shell, shape):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cells = [SETUP.format(lr=1) + HELPER, SHAPES[shape]]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["history"] == [0, 1, 3, 7]

    cells[0] = SETUP.format(lr=2) + HELPER
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    assert mock_shell.user_ns["history"] == [0, 2, 8, 26]


@pytest.mark.usefixtures("store_everything")
def test_an_unchanged_loop_restores_its_reads(cash_magics, mock_shell):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cells = [SETUP.format(lr=1) + HELPER, SHAPES["append"]]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    lineage_after_first = cash_magics.tracking_state.variable_lineage["history"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    assert mock_shell.user_ns["history"] == [0, 1, 3, 7]
    assert cash_magics.tracking_state.variable_lineage["history"] == lineage_after_first
