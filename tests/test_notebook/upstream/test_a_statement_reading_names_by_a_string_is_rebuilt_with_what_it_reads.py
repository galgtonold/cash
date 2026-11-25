"""A statement that looks a variable up by a string is re-run, after a
restart, with the variables above it rebuilt first.

``y = eval('K * 10')``, or ``y = f(10)`` where ``f`` reads
``globals()['K']``, names no ``K`` in its text. Re-run to rebuild ``y`` for
a cell below, it ran before ``K = 2`` was rebuilt and stopped on a NameError
or a KeyError, where Restart & Run All runs through.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

CASES = {
    "eval in the statement": ["K = 2", "y = eval('K * 10')"],
    "globals() in a function it calls": ["K = 2\ndef f(x):\n    return x * globals()['K']", "y = f(10)"],
    "globals() in a helper of the function it calls": [
        "K = 2\ndef look():\n    return globals()['K']\ndef f(x):\n    return x * look()",
        "y = f(10)",
    ],
}


def _restart(magics) -> None:
    """What a kernel restart leaves: no session state, none of the notebook's names."""
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    user_ns = magics.shell.user_ns
    for name in [n for n in user_ns if not n.startswith("_") and n not in ("In", "Out", "get_ipython", "exit", "quit")]:
        del user_ns[name]


@pytest.mark.filterwarnings("ignore::cash.exceptions.CashCacheIneffectiveWarning")
@pytest.mark.parametrize("case", sorted(CASES))
def test_a_reader_below_gets_the_value_after_a_restart(cash_magics, case):
    cells = [*CASES[case], "out = y + 1"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["out"] == 21, "control: the top-to-bottom run"

    _restart(cash_magics)
    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert cash_magics.shell.user_ns["out"] == 21


def test_a_statement_that_names_what_it_reads_rebuilds_only_that(cash_magics):
    """Control: without a lookup by a string, a variable above that the
    statement does not read is left alone."""
    cells = ["K = 2", "unused = [1]", "y = K * 10", "out = y + 1"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    _restart(cash_magics)

    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert cash_magics.shell.user_ns["out"] == 21
    assert "unused" not in cash_magics.shell.user_ns
