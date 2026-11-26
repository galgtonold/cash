"""A value cash cannot read counts as changed when a lambda or a
comprehension of the statement calls a method on it, also when the
statement keeps its result.

An object holding a lock has no content to fingerprint before and after a
statement. ``rows = fetch(conn)`` and ``r = f([conn])`` hand it to a call
that returns an answer: it is taken as only read, or every statement over
a connection would run every time. ``rows = [c.fetch() for c in conns]``
and ``r = list(map(lambda c: c.read(), handles))`` work on each item
themselves: taken as only read, the change went unseen, the statement was
served from the cache without making it, and a reader below was served its
old result.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

SETUP = (
    "import threading\n"
    "class Conn:\n"
    "    def __init__(self):\n"
    "        self.n = 0\n"
    "        self.lock = threading.Lock()\n"
    "    def fetch(self):\n"
    "        self.n += 1\n"
    "        return self.n\n"
    "def reads(*conns):\n"
    "    return sum(c.n for c in conns)\n"
    "def count(conns):\n"
    "    return len(conns)\n"
    "def peek(c):\n"
    "    return c.n\n"
    "def run(c, handlers):\n"
    "    return handlers['k'](c)\n"
    "conn = Conn()\n"
    "conns = [conn]\n"
    "state = {'conn': conn}\n"
    "by_name = {'a': conn}\n"
)

#: Each keeps its result and calls ``fetch`` on the one connection once.
WORKED_ON = {
    "a lambda given to map": "r = list(map(lambda c: c.fetch(), conns))",
    "a lambda given to map over a list written out": "r = list(map(lambda c: c.fetch(), [conn]))",
    "a lambda reaching the object through an item": "r = list(map(lambda s: s['conn'].fetch(), [state]))",
    "a lambda given as a keyword": "r = sorted(conns, key=lambda c: c.fetch())",
    "a lambda called on the spot": "r = (lambda c: c.fetch())(conn)",
    "a lambda handed on inside a dict": "r = run(conn, {'k': lambda c: c.fetch()})",
    "a lambda calling a method on a notebook name": "r = list(map(lambda _: conn.fetch(), range(1)))",
    "a comprehension inside a lambda": "r = (lambda cs: [c.fetch() for c in cs])(conns)",
}

#: The same from a comprehension, which is watched item by item by another
#: rule too (`comprehension_watch_candidates`): kept here so both forms
#: stay together.
COMPREHENSIONS = {
    "a list comprehension": "rows = [c.fetch() for c in conns]",
    "a dict comprehension": "rows = {k: c.fetch() for k, c in by_name.items()}",
    "a generator given to a call": "rows = list(c.fetch() for c in conns)",
}

#: Handed to code that returns an answer: taken as read, stored.
ONLY_HANDED_ON = {
    "a call handed the object": "rows = peek(conn)",
    "a call handed a list of it": "r = count([conn])",
    "a call handed the list": "r = count(conns)",
    "a function called per item in a comprehension": "r = [peek(c) for c in conns]",
    "a function given to map": "r = list(map(peek, conns))",
    "a lambda that calls no method on its item": "r = list(map(lambda c: peek(c), conns))",
    "a comprehension that reads an attribute": "r = [c.n for c in conns]",
}


@pytest.fixture
def notebook(cash_magics, cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    run_cash_cell(cash_magics, SETUP)
    return cash_magics


def _statuses(magics) -> list[str]:
    return [str(s["status"]) for s in magics._last_cell_metrics["statements"]]


@pytest.mark.parametrize("case", sorted(WORKED_ON))
def test_the_statement_runs_every_time(notebook, case):
    statement = WORKED_ON[case]
    for run in (1, 2, 3):
        run_cash_cell(notebook, statement)
        assert notebook.shell.user_ns["conn"].n == run, f"served without the call: {_statuses(notebook)}"


@pytest.mark.parametrize("case", sorted(COMPREHENSIONS))
def test_a_comprehension_calling_a_method_on_each_item_runs_every_time(notebook, case):
    statement = COMPREHENSIONS[case]
    for run in (1, 2, 3):
        run_cash_cell(notebook, statement)
        assert notebook.shell.user_ns["conn"].n == run, f"served without the call: {_statuses(notebook)}"


@pytest.mark.parametrize("case", sorted(WORKED_ON))
def test_a_reader_below_sees_the_change(notebook, case):
    statement = WORKED_ON[case]
    run_cash_cell(notebook, statement)
    run_cash_cell(notebook, "out = reads(conn, *conns)")
    assert notebook.shell.user_ns["out"] == 2, "control: the first run"

    run_cash_cell(notebook, statement + "  # again")
    run_cash_cell(notebook, "out = reads(conn, *conns)")

    assert notebook.shell.user_ns["out"] == 4


@pytest.mark.parametrize("case", sorted(ONLY_HANDED_ON))
def test_a_statement_that_only_hands_the_value_on_is_still_stored(notebook, case):
    statement = ONLY_HANDED_ON[case]
    run_cash_cell(notebook, statement)
    run_cash_cell(notebook, statement)
    assert _statuses(notebook) == ["RESTORED"]


def test_a_readable_value_worked_on_by_a_lambda_is_judged_by_what_it_holds(notebook):
    """Control: a list of plain values has a fingerprint, so a lambda
    calling a method on each item that changes none of them is stored."""
    run_cash_cell(notebook, "lines = [' a', ' b']")
    statement = "r = list(map(lambda x: x.strip(), lines))"
    run_cash_cell(notebook, statement)
    run_cash_cell(notebook, statement)
    assert _statuses(notebook) == ["RESTORED"]
