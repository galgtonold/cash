"""Run one notebook cell through cash's pipeline, for unit tests.

``run_cash_cell(magics, code)`` does what the ``%cash_on`` run_cell hook does
with a cell: :meth:`CellExecutor.execute_cell` (upstream resolution, the AST
parse, then every statement through the statement pipeline), followed by the
finaliser body (session stats, provenance, the final badge).

It leaves out the two things only a live IPython kernel needs. It passes no
``original_run_cell``, so an exception from the pipeline reaches the test
instead of being re-run through the shell's ``run_cell``, and it does not hand
IPython the trailing ``"pass"`` cell that keeps the execution count in step.

In a kernel the upstream check reads the notebook's cells from the saved
.ipynb. Unit tests have none, so by default a cell has no notebook around it;
pass ``cells=`` to give it one.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from cash.notebook.ipython.cell_executor import EarlyReturn, PipelineSyntaxError

__all__ = ["run_cash_cell"]


def run_cash_cell(magics: Any, code: str, *, ttl: int | None = None, cells: list[str] | None = None) -> None:
    """Run *code* as one cell under ``%cash_on`` (``%cash_on ttl=N`` with *ttl*).

    *ttl* applies to this cell only; the magics' own TTL is not changed.
    *cells*, when given, are the notebook's code cells in order, as the upstream
    check would read them from the .ipynb (*code* is normally one of them).
    """
    if cells is None:
        _run(magics, code, ttl)
        return
    with (
        patch("cash.notebook.upstream.checker.get_notebook_cells", side_effect=lambda _path=None: list(cells)),
        patch("cash.notebook.upstream.checker.get_notebook_cells_with_ids", return_value=None),
    ):
        _run(magics, code, ttl)


def _run(magics: Any, code: str, ttl: int | None) -> None:
    result = magics._cell_executor.execute_cell(code, ttl=ttl, cell_id=magics.resolve_cell_id())
    if isinstance(result, (EarlyReturn, PipelineSyntaxError)):
        return
    magics._finalize_cell_body(code, result)
