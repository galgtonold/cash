"""A function held deep in a graph of the notebook's own objects still keys
the statements that run it, and a change made through a second name for
part of the graph still reaches the first.

Orders hold their customer, whose list holds the customer's other orders,
and the customer holds the pricing rules ``total`` applies to each of its
orders; one rule reads ``scale``. ``revenue(orders)`` calls ``o.total()``
on each order. The walk that looks for what those methods run stops at what
an earlier walk for the same question looked through
(`test_a_graph_of_the_users_objects_is_walked_once_per_statement`): the
rule is found wherever it sits, also when another order's walk reached its
customer first.
"""

from __future__ import annotations

import pytest

from cash.analysis import analysis_budget
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SETUP = f"""
import os
import time

class Product:
    def __init__(self, price):
        self.price = price

class Customer:
    def __init__(self, cid):
        self.cid = cid
        self.orders = []
        self.rules = []

class LineItem:
    def __init__(self, product, qty):
        self.product = product
        self.qty = qty
    def amount(self):
        return self.product.price * self.qty

class Order:
    def __init__(self, customer, items):
        self.customer = customer
        self.items = items
    def total(self):
        base = sum(i.amount() for i in self.items)
        return base + sum(rule(base) for rule in self.customer.rules)

def revenue(orders):
    fd = os.open(RUNS, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
    os.write(fd, b"x")
    os.close(fd)
    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})
    return sum(o.total() for o in orders)

product = Product(2.0)
customers = [Customer(i) for i in range(20)]
orders = []
for k in range(200):
    c = customers[k % 20]
    o = Order(c, [LineItem(product, 1 + j) for j in range(3)])
    c.orders.append(o)
    orders.append(o)
scale = 2
"""


@pytest.fixture(autouse=True)
def _no_budget_trip(monkeypatch):
    """These tests are about what keys the readers, not the analysis budget
    (`test_a_statement_past_its_analysis_budget_runs_uncached_and_keys_its_outputs_by_value`).
    The analysis here takes about 10 ms, but the budget is wall time, and a
    loaded CI worker once spent the
    0.5 s floor on it: the statement ran uncached and ran again."""
    monkeypatch.setattr(analysis_budget, "ANALYSIS_BUDGET_FLOOR_S", 60.0)


#: Plain total of the 200 orders: each order is 2 * (1 + 2 + 3).
PLAIN = 200 * 12.0


def _runs(path) -> int:
    return len(path.read_bytes()) if path.exists() else 0


@pytest.mark.parametrize(
    "where",
    ["customers[0]", "customers[-1]"],
    ids=["reached first through another order", "of the last customer"],
)
def test_a_rule_deep_in_the_orders_keys_their_readers(cash_magics, mock_shell, tmp_path, where):
    mock_shell.user_ns["__name__"] = "__main__"
    runs = tmp_path / "runs.log"
    # The customer's 10 orders, 12.0 each, each add the rule's share.
    run_cash_cell(cash_magics, f"RUNS = {str(runs)!r}\n" + SETUP)
    run_cash_cell(cash_magics, f"{where}.rules.append(lambda v: v * scale)")
    run_cash_cell(cash_magics, "out = revenue(orders)")
    assert mock_shell.user_ns["out"] == PLAIN + 10 * 12.0 * 2
    assert _runs(runs) == 1
    # Positive control: nothing changed, so the call is served.
    run_cash_cell(cash_magics, "out = revenue(orders)")
    assert _runs(runs) == 1
    run_cash_cell(cash_magics, "scale = 3")
    run_cash_cell(cash_magics, "out = revenue(orders)")
    assert mock_shell.user_ns["out"] == PLAIN + 10 * 12.0 * 3
    assert _runs(runs) == 2


def test_a_change_through_a_second_name_reaches_the_first(cash_magics, mock_shell, tmp_path):
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, f"RUNS = {str(tmp_path / 'runs.log')!r}\n" + SETUP)
    run_cash_cell(cash_magics, "big = [o for o in orders if o.customer.cid == 3]")
    run_cash_cell(cash_magics, "out = revenue(orders)")
    assert mock_shell.user_ns["out"] == PLAIN
    before = cash_magics.tracking_state.variable_lineage["orders"]
    run_cash_cell(cash_magics, "big[0].items[0].qty = 100")
    assert cash_magics.tracking_state.variable_lineage["orders"] != before
    run_cash_cell(cash_magics, "out = revenue(orders)")
    assert mock_shell.user_ns["out"] == PLAIN + 2.0 * 99
