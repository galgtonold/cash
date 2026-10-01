"""A bare method call in a loop body changes its receiver's lineage.

``buf.write(...)`` is a method no rule lists as mutating. A loop body is not
classified statement by statement, so unless the loop counts the receiver as
changed, ``buf`` keeps its pre-loop lineage and ``text = buf.getvalue()``
after an edited loop hits the entry the old loop left. Every statement here is
worth storing (a zero cost floor), as each one is on a busy machine.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

CELL = (
    "import io\n"
    "buf = io.StringIO()\n"
    "for i in range({n}):\n"
    "    buf.write(f'line{{i}}\\n')\n"
    "text = buf.getvalue()\n"
    "line_count = text.count('\\n')\n"
)


@pytest.fixture
def store_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0


@pytest.mark.usefixtures("store_everything")
def test_an_edited_loop_is_not_served_the_old_receivers_value(cash_magics, mock_shell):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    run_cash_cell(cash_magics, CELL.format(n=3))
    assert mock_shell.user_ns["line_count"] == 3

    run_cash_cell(cash_magics, CELL.format(n=5))

    assert mock_shell.user_ns["line_count"] == 5


@pytest.mark.usefixtures("store_everything")
def test_an_unchanged_loop_still_restores_what_follows_it(cash_magics, mock_shell):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    run_cash_cell(cash_magics, CELL.format(n=3))
    run_cash_cell(cash_magics, CELL.format(n=3))

    assert mock_shell.user_ns["line_count"] == 3
    assert mock_shell.user_ns["text"] == "line0\nline1\nline2\n"
