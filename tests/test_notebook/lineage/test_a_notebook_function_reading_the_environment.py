"""A statement calling a function defined in a cell is keyed on the environment that function reads.

``def mode(): return os.environ.get("MODE")`` in a cell, ``MODE`` set in the
next and ``x = mode() * 2`` below: the environment a called function read
was folded only for functions of the user's modules, a cell's own being
left to "the notebook". Nothing else keyed it, so editing the cell that
sets ``MODE`` and running it and the caller, or the whole notebook, served
the first mode with no warning.
"""

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

DEF = (
    "import os, time\n"
    "def _mode():\n    return os.environ.get('CASH_UT_NB_MODE', 'none')\n"
    f"def mode():\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return _mode()\n"
)


@pytest.fixture(autouse=True)
def _no_mode(monkeypatch):
    monkeypatch.delenv("CASH_UT_NB_MODE", raising=False)


@pytest.mark.parametrize("reader", ["x = mode()", "x = mode() * 2"], ids=["statement", "call_in_an_expression"])
def test_editing_the_variable_and_running_again_recomputes(cash_magics, mock_shell, reader):
    cells = [DEF, "os.environ['CASH_UT_NB_MODE'] = 'a'", reader]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    first = mock_shell.user_ns["x"]
    assert first.startswith("a")

    for mode in "bc":
        cells[1] = f"os.environ['CASH_UT_NB_MODE'] = '{mode}'"
        run_cash_cell(cash_magics, cells[1], cells=cells)
        run_cash_cell(cash_magics, cells[2], cells=cells)
        assert mock_shell.user_ns["x"] == first.replace("a", mode)
