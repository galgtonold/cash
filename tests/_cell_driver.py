"""Run one notebook cell through cash's pipeline, for unit tests.

``run_cash_cell(magics, code)`` does what the ``%cash_on`` run_cell hook does
with a cell: :meth:`CellExecutor.execute_cell` (upstream resolution, the AST
parse, then every statement through the statement pipeline), followed by the
finaliser body (session stats, provenance, the final badge).

It leaves out the two things only a live IPython kernel needs. It passes no
``original_run_cell``, so an exception from the pipeline reaches the test
instead of being re-run through the shell's ``run_cell``, and it does not hand
IPython the trailing ``"pass"`` cell that keeps the execution count in step.
"""

from __future__ import annotations

from typing import Any

from cash.notebook.ipython.cell_executor import EarlyReturn, PipelineSyntaxError

__all__ = ["run_cash_cell"]


def run_cash_cell(magics: Any, code: str, *, ttl: int | None = None) -> None:
    """Run *code* as one cell under ``%cash_on`` (``%cash_on ttl=N`` with *ttl*).

    *ttl* applies to this cell only; the magics' own TTL is restored after it.
    """
    saved_ttl = magics.global_ttl
    magics.global_ttl = ttl
    try:
        result = magics._cell_executor.execute_cell(code)
        if isinstance(result, (EarlyReturn, PipelineSyntaxError)):
            return
        magics._finalize_cell_body(
            code,
            result.all_metrics,
            result.buffered_outputs,
            result.badge_display_id,
            result.hook_start,
            result.timing_breakdown,
            result.badge_render_time,
        )
    finally:
        magics.global_ttl = saved_ttl
