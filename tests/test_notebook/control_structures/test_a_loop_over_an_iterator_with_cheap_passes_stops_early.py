"""A loop over an iterator whose first passes were measured cheap runs the rest
as one unit after a handful of passes.

``for part in chunks(df, 20):`` and ``for m in re.finditer(...):`` ran 125
passes of a one-line body through the per-statement machinery before the
rest went as one unit, the count the static rule asks for (1 s of assumed
bookkeeping): 1.1 s and 0.6 s against 8 ms and 3 ms plain. A body whose work
is most of its passes' time still runs them one by one.
"""

from __future__ import annotations

import pytest

from cash.notebook.control_structures import single_unit_policy
from cash.notebook.statement import StatementProcessor
from tests._cell_driver import run_cash_cell

CHUNKS = (
    "def chunks(rows, size):\n    for start in range(0, len(rows), size):\n        yield rows[start:start + size]\n"
)


@pytest.fixture
def statements(monkeypatch):
    """How many statements went through the statement processor."""
    seen = []
    real = StatementProcessor.process_statement

    def counting(self, *a, **k):
        seen.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(StatementProcessor, "process_statement", counting)
    return seen


def test_cheap_passes_hand_the_rest_to_one_unit(cash_magics, statements):
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, "rows = list(range(4000))\n" + CHUNKS)
    statements.clear()
    run_cash_cell(cash_magics, "sizes = []\nfor part in chunks(rows, 20):\n    sizes.append(len(part))")
    assert ns["sizes"] == [20] * 200
    assert ns["part"] == list(range(3980, 4000))
    # The first passes, the rest's unit, and `sizes = []`: 125 passes before.
    assert len(statements) <= single_unit_policy.CHEAP_PASSES + 3, len(statements)


def test_a_finditer_loop_counts_as_python_counts(cash_magics, statements):
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, "import re\ntext = 'fix it ' * 300")
    statements.clear()
    run_cash_cell(cash_magics, "n = 0\nfor m in re.finditer('fix', text):\n    n += 1")
    assert ns["n"] == 300 and ns["m"].start() == 7 * 299
    assert len(statements) <= single_unit_policy.CHEAP_PASSES + 3, len(statements)


def test_passes_whose_work_is_most_of_their_time_stay_one_by_one(cash_magics, statements):
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, "import time\nrows = list(range(600))\n" + CHUNKS)
    statements.clear()
    run_cash_cell(
        cash_magics, "sizes = []\nfor part in chunks(rows, 20):\n    sizes.append([len(part), time.sleep(0.05)][0])"
    )
    assert ns["sizes"] == [20] * 30
    assert len(statements) >= 30, len(statements)


def test_a_short_cheap_loop_keeps_its_first_passes_and_runs_the_rest_as_one_unit(cash_magics, statements):
    """Under the static rule's 125 passes too: 40 passes keep the first five
    one by one, and the other 35 go as one unit."""
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, "rows = list(range(40))")
    statements.clear()
    run_cash_cell(cash_magics, "it = iter(rows)\nseen = []\nfor r in it:\n    seen.append(r)")
    assert ns["seen"] == list(range(40))
    # `it = ...`, `seen = []`, five passes, and the rest's unit.
    assert len(statements) == 2 + single_unit_policy.CHEAP_PASSES + 1, len(statements)
