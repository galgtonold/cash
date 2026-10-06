"""A cell run with an edit the saved notebook lacks is not undone below it.

Edit ``x = 1`` to ``x = 2``, run it without saving, run the next cell: the
upstream check read ``x = 1`` from the file, saw ``x`` ahead of it and re-ran
the saved code over what the user had just run, with no warning. With the
cell's id, cash reads the cell as it ran; without one it cannot tell where the
code belongs, and stops until the notebook is saved.
"""

from __future__ import annotations

import pytest

from cash.exceptions import UpstreamStateError
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SLOW = f"import time\ndef slow(v):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return v"
SAVED = [SLOW, "x = 1", "y = slow(x * 10)"]
IDS = ["c1", "c2", "c3"]


def _run_all(magics, **ids):
    for i, cell in enumerate(SAVED):
        run_cash_cell(magics, cell, cells=SAVED, **({"cell_ids": IDS, "cell_id": IDS[i]} if ids else {}))


def test_with_cell_ids_the_cell_is_read_as_it_ran(cash_magics, mock_shell):
    _run_all(cash_magics, ids=True)
    run_cash_cell(cash_magics, "x = 2", cells=SAVED, cell_ids=IDS, cell_id="c2")
    run_cash_cell(cash_magics, SAVED[2], cells=SAVED, cell_ids=IDS, cell_id="c3")

    assert (mock_shell.user_ns["x"], mock_shell.user_ns["y"]) == (2, 20)


def test_once_saved_the_file_is_read_again(cash_magics, mock_shell):
    _run_all(cash_magics, ids=True)
    run_cash_cell(cash_magics, "x = 2", cells=SAVED, cell_ids=IDS, cell_id="c2")
    # Edited again and saved: the file is what the cell says now.
    saved = [SLOW, "x = 3", SAVED[2]]
    run_cash_cell(cash_magics, saved[2], cells=saved, cell_ids=IDS, cell_id="c3")

    assert (mock_shell.user_ns["x"], mock_shell.user_ns["y"]) == (3, 30)


def test_without_cell_ids_cash_stops_and_says_so(cash_magics, mock_shell):
    _run_all(cash_magics)
    run_cash_cell(cash_magics, "x = 2", cells=SAVED)
    with pytest.raises(UpstreamStateError, match="Save the notebook"):
        run_cash_cell(cash_magics, SAVED[2], cells=SAVED)

    assert mock_shell.user_ns["x"] == 2


def test_without_cell_ids_saving_lets_the_next_run_through(cash_magics, mock_shell):
    _run_all(cash_magics)
    run_cash_cell(cash_magics, "x = 2", cells=SAVED)
    saved = [SLOW, "x = 2", SAVED[2]]
    run_cash_cell(cash_magics, saved[2], cells=saved)

    assert (mock_shell.user_ns["x"], mock_shell.user_ns["y"]) == (2, 20)
