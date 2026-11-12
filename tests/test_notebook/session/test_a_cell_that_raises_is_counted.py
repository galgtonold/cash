"""A cell that raises is counted with the statements it ran.

The statements before the error ran, and their time was spent: an upstream
fit re-run to rebuild what the cell reads, or a slow statement of the cell
itself. ``%cash_status`` showed the cell before instead, and ``%cash_stats``
counted none of it, so that time was invisible.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from cash.notebook.cache_status import CacheStatus


def _through_the_hook(magics, code: str) -> None:
    """Run *code* as the ``%cash_on`` hook does, IPython's own run_cell
    standing by for the stand-in raise."""
    magics._original_run_cell = MagicMock()
    magics.shell.showtraceback = MagicMock()
    magics._execute_cell(code)


def test_the_statements_before_the_error_are_shown_and_counted(cash_magics):
    cash_magics.cash_on("")
    _through_the_hook(cash_magics, "z = 0")
    before = cash_magics.session.stats["cells_executed"]

    started = time.perf_counter()
    _through_the_hook(cash_magics, "import time\nx = (time.sleep(0.2), 1)[1]\ny = x / z")
    took = time.perf_counter() - started

    calls = [c.args[0] for c in cash_magics._original_run_cell.call_args_list if c.args]
    assert any("raise __cash_exception__" in c for c in calls), "control: the cell raised"
    last = cash_magics.cash_status("dict")["last_cell"]
    rows = {row["code"]: row for row in last["statements"]}
    assert "x = (time.sleep(0.2), 1)[1]" in rows, rows
    assert rows["x = (time.sleep(0.2), 1)[1]"]["status"] == CacheStatus.COMPUTED
    assert rows["x = (time.sleep(0.2), 1)[1]"]["execution_time"] >= 0.2
    assert rows["y = x / z"]["status"] == CacheStatus.ERROR
    assert 0.2 <= last["total_time"] <= took
    assert cash_magics.session.stats["cells_executed"] == before + 1

    # Raised before any statement ran: nothing of the cell before is
    # counted again.
    cash_magics._cell_executor._prepare_cell = MagicMock(side_effect=RuntimeError("before the statements"))
    _through_the_hook(cash_magics, "c = 2")
    assert cash_magics.cash_status("dict")["last_cell"] == last
    assert cash_magics.session.stats["cells_executed"] == before + 1
