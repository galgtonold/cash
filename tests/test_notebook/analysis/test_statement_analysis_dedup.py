"""Regression test: ``analyze_statement`` must be computed at most once per
``process_statement`` call.

Before this fix, the same ``StatementAnalysis`` was computed twice — once in
the pre-execution cacheability decision and again in ``_post_execute``. Both
passed identical ``(code, tree)`` args, so three AST visitors ran twice per
statement on the hot path.
"""

from __future__ import annotations

from unittest.mock import patch

from tests._cell_driver import run_cash_cell


def test_analyze_statement_called_once_per_processed_statement(cash_magics):
    """A single cacheable assignment must trigger exactly one analyze_statement
    call, not two.

    We patch at the call site (statement_processor) rather than at the source
    (cacheability) so that internal cacheability tests are unaffected.
    """

    real_analyze = __import__("cash.notebook.statement.processor", fromlist=["analyze_statement"]).analyze_statement

    with patch(
        "cash.notebook.statement.processor.analyze_statement",
        wraps=real_analyze,
    ) as spy:
        run_cash_cell(cash_magics, "y = 1 + 2")

    # The cell runs one statement (`y = 1 + 2`). Pre-fix this was 2 calls; the
    # de-dup contract is exactly 1 per processed statement.
    assert spy.call_count == 1, (
        f"Expected exactly 1 analyze_statement call, got {spy.call_count}. calls: {spy.call_args_list}"
    )
