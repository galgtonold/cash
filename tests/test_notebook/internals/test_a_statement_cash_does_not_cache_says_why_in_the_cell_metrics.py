"""``%cash_status``'s last-cell metrics carry why a statement was not
cached (``uncacheable_reasons``), as its badge says: before, a refused
statement showed only COMPUTED there, and a refusal read as no refusal.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell


def _row(magics, starts):
    [row] = [r for r in magics._last_cell_metrics["statements"] if r["code"].startswith(starts)]
    return row


@pytest.mark.parametrize(
    ("cell", "statement", "reason"),
    [
        ("import time\nv = (time.sleep(0.02), a[2:])[1]", "v = ", "Live-alias object 'v'"),
        ("import time\ng = (time.sleep(0.02), (x for x in rows))[1]", "g = ", "'g' is consumed as it is read"),
    ],
    ids=["a view of a variable", "a generator"],
)
def test_a_refused_statement_says_why(cash_magics, cell, statement, reason):
    run_cash_cell(cash_magics, "import numpy as np\na = np.arange(10.0)")
    run_cash_cell(cash_magics, "rows = [1, 2, 3]")
    run_cash_cell(cash_magics, cell)
    reasons = _row(cash_magics, statement).get("uncacheable_reasons")
    assert reasons and reason in reasons[0], reasons


def test_a_stored_statement_has_no_reason(cash_magics):
    run_cash_cell(cash_magics, "import time\ntotal = (time.sleep(0.02), sum(range(10)))[1]")
    assert "uncacheable_reasons" not in _row(cash_magics, "total = ")
