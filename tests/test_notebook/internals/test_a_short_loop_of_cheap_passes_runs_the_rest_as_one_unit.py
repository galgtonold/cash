"""A loop of known length too short to run as one unit from the start runs
the rest as one unit once its first passes measured cheap.

A loop of at most 50 passes kept one entry per statement per pass, whatever
its passes cost: E11's MNIST build, ten passes of five statements, took
0.67 s against 0.06 s plain, and a 40-epoch training loop of six statements
1 s against 2 ms. Its first passes now go one by one, as before; when their
own work was a small share of their time (`passes_were_cheap`), the rest
runs as one unit that draws from the same iterator, as Python draws. A loop
whose passes do real work keeps its entries. Pins the work, not the
behaviour: a refactor may expect this to fail.
"""

from __future__ import annotations

import pytest

from cash.notebook.control_structures import single_unit_policy
from cash.notebook.statement import StatementProcessor
from tests._cell_driver import run_cash_cell
from tests._work_counts import method_calls

LOOP = "seen = []\ntotal = 0\nfor i in {header}:\n    y = i * 2\n    total += y\n    seen.append(y)"


@pytest.fixture
def measured_cheap(monkeypatch):
    """Passes measured cheap after five, and the rest worth a unit: the
    measure is wall time, which a busy machine moves."""
    monkeypatch.setattr(single_unit_policy, "passes_were_cheap", lambda work_shares: len(work_shares) >= 5)
    monkeypatch.setattr(single_unit_policy, "rest_worth_one_unit", lambda overheads, remaining: remaining > 0)


def _statements(magics, header: str) -> int:
    with method_calls(StatementProcessor, "process_statement") as statements:
        run_cash_cell(magics, LOOP.format(header=header))
    return statements.calls


def test_a_loop_of_40_cheap_passes_runs_most_of_them_as_one(cash_magics, measured_cheap):
    # 2 seeds, 5 passes of 3 statements, then one unit: 122 before.
    assert _statements(cash_magics, "range(40)") <= 20
    ns = cash_magics.shell.user_ns
    assert ns["seen"] == [i * 2 for i in range(40)]
    assert (ns["total"], ns["i"], ns["y"]) == (1560, 39, 78)


def test_the_rest_is_drawn_from_the_same_iterator(cash_magics, measured_cheap):
    # A list the body grows is iterated as Python iterates it: the unit
    # goes on from where the passes left the list's own iterator.
    run_cash_cell(cash_magics, "items = list(range(8))")
    run_cash_cell(
        cash_magics, "out = []\nfor v in items:\n    out.append(v)\n    if v < 3:\n        items.append(v + 100)"
    )
    assert cash_magics.shell.user_ns["out"] == [0, 1, 2, 3, 4, 5, 6, 7, 100, 101, 102]


def test_a_rerun_gives_the_same_values(cash_magics, measured_cheap):
    for _ in range(2):
        _statements(cash_magics, "range(12)")
        ns = cash_magics.shell.user_ns
        assert ns["seen"] == [i * 2 for i in range(12)]
        assert ns["total"] == 132


def test_passes_that_do_real_work_keep_their_entries(cash_magics, monkeypatch):
    monkeypatch.setattr(single_unit_policy, "passes_were_cheap", lambda work_shares: False)
    assert _statements(cash_magics, "range(40)") >= 120
    assert cash_magics.shell.user_ns["seen"] == [i * 2 for i in range(40)]


def test_a_loop_of_five_passes_or_fewer_stays_per_pass(cash_magics, monkeypatch):
    monkeypatch.setattr(single_unit_policy, "passes_were_cheap", lambda work_shares: True)
    monkeypatch.setattr(single_unit_policy, "rest_worth_one_unit", lambda overheads, remaining: True)
    assert _statements(cash_magics, "range(5)") >= 15


def test_the_rest_goes_as_one_unit_only_when_its_bookkeeping_is_worth_it():
    # Two passes of 2 ms left: a unit's fingerprint of what it may change
    # (a 4 MB array) costs more than they would.
    assert not single_unit_policy.rest_worth_one_unit([0.002] * 5, 2)
    assert single_unit_policy.rest_worth_one_unit([0.03] * 5, 5)
    assert not single_unit_policy.rest_worth_one_unit([0.03] * 5, 0)
