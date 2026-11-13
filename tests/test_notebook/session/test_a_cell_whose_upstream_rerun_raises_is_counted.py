"""A cell whose upstream re-run raises is counted with what that re-run ran.

The cell below an edit re-runs the statements above that it reads; when one
of them raises, the cell fails with that error. The statements re-run before
it, and the one that raised, ran all the same, but ``%cash_status`` still
showed the cell before and ``%cash_stats`` counted none of it, so a run of
the notebook that stopped there lost its upstream rows.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from cash.notebook.cache_status import CacheStatus


def _through_the_hook(magics, code: str, cells: list[str]) -> None:
    """Run *code* as the ``%cash_on`` hook does, in a notebook of *cells*."""
    magics._original_run_cell = MagicMock()
    magics.shell.showtraceback = MagicMock()
    with (
        patch("cash.notebook.upstream.checker.get_notebook_cells", side_effect=lambda _path=None: list(cells)),
        patch("cash.notebook.upstream.checker.get_notebook_cells_with_ids", return_value=None),
    ):
        magics._execute_cell(code)


def test_the_statements_re_run_above_are_shown_and_counted(cash_magics, cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_magics.cash_on("")
    cells = ["a = 1\nzero = 1", "b = a * 2\nc = 1 / zero", "d = b + c"]
    for cell in cells:
        _through_the_hook(cash_magics, cell, cells)
    before = cash_magics.session.stats["cells_executed"]

    cells[0] = "a = 5\nzero = 0"
    _through_the_hook(cash_magics, cells[2], cells)

    calls = [c.args[0] for c in cash_magics._original_run_cell.call_args_list if c.args]
    assert any("UpstreamStateError" in c for c in calls), "control: the cell raised the upstream failure"
    rows = {row["code"]: row for row in cash_magics.cash_status("dict")["last_cell"]["statements"]}
    assert rows["b = a * 2"]["status"] == CacheStatus.COMPUTED
    assert rows["b = a * 2"]["is_upstream"]
    assert rows["c = 1 / zero"]["status"] == CacheStatus.ERROR
    assert "d = b + c" not in rows
    assert cash_magics.session.stats["cells_executed"] == before + 1

    # A cell that runs through afterwards counts no row of it again.
    cells[0] = "a = 5\nzero = 1"
    _through_the_hook(cash_magics, cells[0], cells)
    rows = {row["code"]: row for row in cash_magics.cash_status("dict")["last_cell"]["statements"]}
    assert "c = 1 / zero" not in rows
