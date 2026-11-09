"""A holder a statement left as it was keeps no other holder of the same
lineage, after a restart either.

``c`` and ``o`` are bound by one loop and have one lineage; ``o`` holds
``customers[1]``, ``c`` the last customer. ``customers = add_rate(customers)``
is stored with both (they hold objects of its output), leaves ``c`` as it
was and changes ``o``. The run keeps ``c`` and moves ``o`` on; the alias
that keeps ``c`` was the lineage's, not ``c``'s, so the simulation and the
restore after a restart kept ``o`` too: ``o`` came out of a Run All with
another lineage than the run gave it, and where ``out = revenue([o])`` had
run before the change, it was served the total from before the rate.
"""

from __future__ import annotations

import pytest

from cash.analysis import analysis_budget
from cash.notebook.ipython.magics import CashMagics
from cash.notebook.lineage_formula import held_lineage, keep_held_lineage
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SETUP = f"""
import os, time

class Customer:
    def __init__(self, cid):
        self.cid = cid
        self.rules = []

class Order:
    def __init__(self, customer, amount):
        self.customer = customer
        self.amount = amount
    def total(self):
        return self.amount + sum(self.amount * r for r in self.customer.rules)

def revenue(orders):
    fd = os.open(RUNS, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
    os.write(fd, b"x")
    os.close(fd)
    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})
    return sum(o.total() for o in orders)

def slow_rate():
    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})
    return 1.0

def add_rate(cs):
    cs[0].rules.append(slow_rate())
    return cs

customers = [Customer(i) for i in range(20)]
orders = []
for k in range(200):
    c = customers[k % 20]
    o = Order(customers[(k + 1) % 20], 12.0)
    orders.append(o)
"""

CHANGE = "customers = add_rate(customers)"
READ = "out = revenue([o])"
#: ``o`` holds ``customers[0]`` once the rate is added: 12.0 plus 12.0 * 1.0.
WITH_RATE = 24.0


@pytest.fixture(autouse=True)
def _no_budget_trip(monkeypatch):
    """About which lineage each holder takes, not the analysis budget
    (`test_a_statement_past_its_analysis_budget_runs_uncached_and_keys_its_outputs_by_value`)."""
    monkeypatch.setattr(analysis_budget, "ANALYSIS_BUDGET_FLOOR_S", 60.0)


def test_a_kept_holder_keeps_only_its_own_name():
    keep_held_lineage("c", "L", "key")

    assert held_lineage("c", "L", "key") == "L"
    assert held_lineage("o", "L", "key") != "L"
    assert held_lineage("o", "L", "key") == held_lineage("c", "L", "key", resolve=False)


@pytest.mark.parametrize("read_first", [True, False], ids=["read before the change", "run all only"])
def test_a_restart_and_run_all_reads_the_changed_holder(cash_magics, mock_shell, cash_instance, tmp_path, read_first):
    mock_shell.user_ns["__name__"] = "__main__"
    setup = f"RUNS = {str(tmp_path / 'runs.log')!r}\n" + SETUP
    cells = [setup, CHANGE, READ]
    run_cash_cell(cash_magics, setup, cells=cells)
    if read_first:
        run_cash_cell(cash_magics, READ, cells=[setup, READ])
        assert mock_shell.user_ns["out"] == 12.0
    run_cash_cell(cash_magics, CHANGE, cells=cells)
    run_cash_cell(cash_magics, READ, cells=cells)
    assert mock_shell.user_ns["out"] == WITH_RATE
    # The lineage the run moved ``o`` on to, which a Run All gives it too.
    moved = cash_magics.tracking_state.variable_lineage["o"]
    assert moved != cash_magics.tracking_state.variable_lineage["c"]

    for _ in range(2):
        mock_shell.user_ns.clear()
        mock_shell.user_ns["__name__"] = "__main__"
        restarted = CashMagics(mock_shell, cash_instance)
        for cell in cells:
            run_cash_cell(restarted, cell, cells=cells)
        assert mock_shell.user_ns["out"] == WITH_RATE
        assert restarted.tracking_state.variable_lineage["o"] == moved
