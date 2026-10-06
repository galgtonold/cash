"""``del x`` holds for the cells below it, however costly ``x`` was.

A cell reading a name missing from the namespace had it put back from the
entry this session stored it under, before the upstream check ran: a cell
below ``del x`` got ``x`` again instead of a ``NameError``, but only when
``x`` had been worth storing.
"""

from __future__ import annotations

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SLOW = f"import time\ndef slow(v):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return v\nx = slow(1)"
READER = "try:\n    y = x + 1\nexcept NameError:\n    y = 'NameError'"


def test_a_reader_below_the_del_gets_a_name_error(cash_magics, mock_shell):
    cells = [SLOW, "del x", READER]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    assert mock_shell.user_ns["y"] == "NameError"
    assert "x" not in mock_shell.user_ns


def test_a_reader_above_the_del_still_gets_its_value(cash_magics, mock_shell):
    cells = [SLOW, READER, "del x"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    run_cash_cell(cash_magics, READER, cells=cells)

    assert mock_shell.user_ns["y"] == 2
