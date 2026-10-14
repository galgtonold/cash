"""A graph of the notebook's own objects is walked once per question, not
once per object in it.

Orders that hold their customer, whose list holds the customer's other
orders: ``revenue(orders)`` calls ``o.total()`` on each order, and
``total`` calls ``amount()`` on what the order holds. Each order's method
was followed by walking everything the order reaches -- its customer, the
customer's other orders and their items -- so the graph was opened once
for each order it holds (25,000 orders: 21 s of analysis for a 0.05 s
statement). A walk now stops at what an earlier walk of the same question
looked through, and the several analyses of one statement before it runs
share their answers. Every object is still opened: what a method of one of
them reaches is found (the tests next to
`test_a_function_held_deep_in_the_users_objects_keys_its_readers`).
"""

from __future__ import annotations

import ast
import linecache

import pytest

import cash.analysis.handed_callables as hc
from cash.analysis.handed_callables import handed_callable_values, handed_callables, new_analysis_round
from tests._cell_driver import run_cash_cell

N_PRODUCTS = 50
N_CUSTOMERS = 300
N_ORDERS = 1500
ITEMS = 3
#: The objects of the user's classes the graph holds, without its lists.

OBJECTS = N_PRODUCTS + N_CUSTOMERS + N_ORDERS * (1 + ITEMS)

SETUP = f"""
class Product:
    def __init__(self, pid, price):
        self.pid = pid
        self.price = price

class Customer:
    def __init__(self, cid):
        self.cid = cid
        self.orders = []

class LineItem:
    def __init__(self, product, qty):
        self.product = product
        self.qty = qty
    def amount(self):
        return self.product.price * self.qty

class Order:
    def __init__(self, oid, customer, items):
        self.oid = oid
        self.customer = customer
        self.items = items
    def total(self):
        return sum(i.amount() for i in self.items)

def revenue(orders):
    out = {{}}
    for o in orders:
        out[o.customer.cid] = out.get(o.customer.cid, 0.0) + o.total()
    return out

products = [Product(i, 1.0 + i % 7) for i in range({N_PRODUCTS})]
customers = [Customer(i) for i in range({N_CUSTOMERS})]
orders = []
for k in range({N_ORDERS}):
    c = customers[k % {N_CUSTOMERS}]
    o = Order(k, c, [LineItem(products[(k + j) % {N_PRODUCTS}], 1 + j) for j in range({ITEMS})])
    c.orders.append(o)
    orders.append(o)
"""


def _namespace(code: str = SETUP) -> dict:
    filename = f"<orders-graph-{abs(hash(code))}>"
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(code, filename, "exec"), namespace)
    return namespace


@pytest.fixture
def opened(monkeypatch):
    """How many containers and objects the walks for callables open."""
    count = [0]
    original = hc._items

    def counting(container):
        count[0] += 1
        return original(container)

    monkeypatch.setattr(hc, "_items", counting)
    return count


def test_one_reading_opens_the_graph_once_per_question(opened):
    namespace = _namespace()
    new_analysis_round()
    found = handed_callables(ast.parse("out = revenue(orders)"), namespace)
    assert any("def total" in source for source in found.sources)
    assert any("def amount" in source for source in found.sources)
    # Twice: for the methods ``revenue`` calls on what it is handed, and for
    # those ``total`` calls on what each order holds. Before, each of the
    # 1,500 orders opened its customer's share of the graph again (55,000).
    assert opened[0] < 3.5 * OBJECTS, opened[0]


def test_the_analyses_before_a_statement_runs_share_one_walk(opened):
    namespace = _namespace()
    call = ast.parse("out = revenue(orders)").body[0].value
    new_analysis_round()
    with hc.answers_shared():
        first = handed_callable_values(call, namespace)
        for _ in range(3):  # what the other analyses before the statement runs ask
            assert len(handed_callable_values(call, namespace)) == len(first)
        assert opened[0] < 2 * OBJECTS, opened[0]
        # Code ran: asked afresh.
        before = opened[0]
        new_analysis_round()
        handed_callable_values(call, namespace)
        assert opened[0] > before


def test_nothing_is_kept_past_the_statement(opened):
    """An answer holds the user's objects: kept, it would read as one more
    holder to the share check after the statement."""
    namespace = _namespace()
    call = ast.parse("out = revenue(orders)").body[0].value
    new_analysis_round()
    with hc.answers_shared():
        handed_callable_values(call, namespace)
    before = opened[0]
    handed_callable_values(call, namespace)
    assert opened[0] > before
    assert not hc._ANSWERED_THIS_ROUND


def test_a_statement_over_the_graph_opens_it_a_few_times_not_once_per_object(cash_magics, mock_shell, opened):
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, SETUP)
    opened[0] = 0
    run_cash_cell(cash_magics, "out = revenue(orders)")
    assert len(mock_shell.user_ns["out"]) == N_CUSTOMERS
    # Once per question in each of the cell's check, the statement's
    # analysis and the call itself (about 43 times the objects before).
    assert opened[0] < 15 * OBJECTS, opened[0]


RULES = SETUP.replace("        self.orders = []\n", "        self.orders = []\n        self.rules = []\n").replace(
    "        return sum(i.amount() for i in self.items)\n",
    "        base = sum(i.amount() for i in self.items)\n        return base + sum(rule(base) for rule in self.customer.rules)\n",
)


@pytest.mark.parametrize(
    "where",
    ["customers[0].rules", "customers[-1].rules", "orders[-1].items[-1].__dict__"],
    ids=[
        "on a customer the first order's walk reaches",
        "on the last customer",
        "an item's own amount on the last order",
    ],
)
def test_a_function_held_deep_is_still_found(where):
    namespace = _namespace(RULES + "\ndef deep_rule(v):\n    return v * 2\n")
    holder = eval(where, namespace)
    if isinstance(holder, list):
        holder.append(namespace["deep_rule"])
    else:
        holder["amount"] = namespace["deep_rule"]
    new_analysis_round()
    found = handed_callables(ast.parse("out = revenue(orders)"), namespace)
    assert any("def deep_rule" in source for source in found.sources), where
