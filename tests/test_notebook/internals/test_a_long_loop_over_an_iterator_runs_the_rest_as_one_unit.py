"""A long loop over an iterator runs most of its passes as one unit.

Every pass of ``for (name, act) in parsed:`` over a ``filter`` of 87,464
items went through the per-statement machinery, about 1 ms a pass: 21.9 s
against 0.19 s plain. The loop now runs its first passes one by one, until it
is known to be long enough to run as one unit, and the rest as one unit.
Pins the work, not the behaviour: a refactor may expect this to fail.
"""

from __future__ import annotations

from cash.notebook.statement import StatementProcessor
from tests._cell_driver import run_cash_cell
from tests._work_counts import method_calls

LOOP = "for (name, act) in it:\n    if (len(act) == 0):\n        pass\n    m = len(act)"


def _statements(magics, header: str) -> int:
    run_cash_cell(magics, "data = [('u%d' % i, [1] * (i % 3)) for i in range(5000)]")
    with method_calls(StatementProcessor, "process_statement") as statements:
        run_cash_cell(magics, header + "\n" + LOOP)
    assert magics.shell.user_ns["name"] == "u4999"
    return statements.calls


def test_a_long_loop_over_a_named_iterator(cash_magics):
    # 63 passes one by one (the count from which a two-statement body runs
    # as one unit) and the rest as one statement: 6,667 before.
    assert _statements(cash_magics, "it = filter(lambda x: True, data)") <= 120


def test_a_long_loop_over_a_named_generator(cash_magics):
    assert _statements(cash_magics, "it = (x for x in data)") <= 120


def test_a_loop_over_map_or_filter_in_its_header_is_sized(cash_magics):
    run_cash_cell(cash_magics, "data = [('u%d' % i, [1] * (i % 3)) for i in range(5000)]")
    for header in ("filter(lambda x: True, data)", "map(tuple, data)", "iter(data)"):
        with method_calls(StatementProcessor, "process_statement") as statements:
            run_cash_cell(cash_magics, f"for (name, act) in {header}:\n    if (len(act) == 0):\n        pass")
        assert statements.calls <= 2, header
        assert cash_magics.shell.user_ns["name"] == "u4999"


def test_a_short_loop_over_an_iterator_stays_per_pass(cash_magics):
    """Below the count, every pass keeps its own entries, as before."""
    run_cash_cell(cash_magics, "data = [('u%d' % i, [1] * (i % 3)) for i in range(40)]")
    with method_calls(StatementProcessor, "process_statement") as statements:
        run_cash_cell(cash_magics, "it = filter(lambda x: True, data)\n" + LOOP)
    assert statements.calls >= 40  # `m = len(act)`, once a pass
