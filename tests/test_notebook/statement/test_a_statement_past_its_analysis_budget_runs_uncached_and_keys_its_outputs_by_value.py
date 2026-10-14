"""A statement whose analysis runs past its budget runs uncached, and what
it binds is keyed by its value.

Working out how to key a statement reads what its calls can run through the
objects they are handed. Over a big graph of the notebook's own objects that
reading can cost more than the statement, every run: the analysis now has a
budget (four times what the statement last took, at least half a second),
checked as the walks go. Past it the statement runs every time, and its
outputs carry a digest of their value, as a ``# @cash:no-cache``
statement's do: a cell below is served only when what it reads is the same.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import itertools
import linecache
import os
import warnings

import pytest

import cash.analysis.analysis_budget as ab
import cash.analysis.handed_callables as hc
from cash.analysis.analysis_budget import (
    ANALYSIS_BUDGET_REASON,
    AnalysisBudget,
    AnalysisBudgetExceeded,
    analysis_budget,
)
from cash.analysis.handed_callables import handed_callables, new_analysis_round
from cash.analysis.mutation_effects import NotebookSources, cell_effects, statement_effects
from cash.notebook.cache_key import statement_source_hash
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SETUP = f"""
import os
import time

class Item:
    def __init__(self, v):
        self.v = v
    def value(self):
        return self.v

def total(xs):
    return sum(i.value() for i in xs)

def slow(x):
    fd = os.open(RUNS, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
    os.write(fd, b"x")
    os.close(fd)
    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})
    return x * 2

items = [Item(i) for i in range(3)]
"""


@contextlib.contextmanager
def spent_budget():
    """Every analysis budget opened below is spent before it starts."""
    floor, factor = ab.ANALYSIS_BUDGET_FLOOR_S, ab.ANALYSIS_BUDGET_FACTOR
    ab.ANALYSIS_BUDGET_FLOOR_S, ab.ANALYSIS_BUDGET_FACTOR = 0.0, 0.0
    try:
        yield
    finally:
        ab.ANALYSIS_BUDGET_FLOOR_S, ab.ANALYSIS_BUDGET_FACTOR = floor, factor


def _runs(path) -> int:
    return len(path.read_bytes()) if path.exists() else 0


def test_a_change_the_lineage_cannot_see_reaches_the_cell_below(cash_magics, mock_shell, tmp_path):
    mock_shell.user_ns["__name__"] = "__main__"
    runs = tmp_path / "runs.log"
    run_cash_cell(cash_magics, f"RUNS = {str(runs)!r}\n" + SETUP)
    with spent_budget():
        run_cash_cell(cash_magics, "x = total(items)")
    assert mock_shell.user_ns["x"] == 3
    assert cash_magics.tracking_state.no_cache_values, "the output is keyed by its value"
    run_cash_cell(cash_magics, "y = slow(x)")
    assert mock_shell.user_ns["y"] == 6
    assert _runs(runs) == 1

    # The producer runs again over the same objects, unchanged: the cell
    # below reads the same value and is served.
    with spent_budget():
        run_cash_cell(cash_magics, "x = total(items)")
    run_cash_cell(cash_magics, "y = slow(x)")
    assert _runs(runs) == 1

    # A change no lineage sees (made from outside the notebook's cells): the
    # producer, which runs every time, binds a new value, and the cell below
    # reads it.
    mock_shell.user_ns["items"][0].v = 100
    with spent_budget():
        run_cash_cell(cash_magics, "x = total(items)")
    assert mock_shell.user_ns["x"] == 103
    run_cash_cell(cash_magics, "y = slow(x)")
    assert mock_shell.user_ns["y"] == 206
    assert _runs(runs) == 2


def test_within_its_budget_the_same_statement_is_cached_by_its_lineage(cash_magics, mock_shell, tmp_path):
    """Positive control: with its budget the producer is an ordinary cached
    statement, whose output is keyed by lineage, not by value."""
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, f"RUNS = {str(tmp_path / 'runs.log')!r}\n" + SETUP)
    cash_magics.tracking_state.no_cache_values.clear()
    run_cash_cell(cash_magics, "x = total(items)")
    assert not cash_magics.tracking_state.no_cache_values


def test_the_hint_comes_once_and_only_for_a_slow_statement(cash_magics, mock_shell, tmp_path):
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, f"RUNS = {str(tmp_path / 'runs.log')!r}\n" + SETUP)
    run_cash_cell(cash_magics, "def slow_total(xs):\n    time.sleep(0.15)\n    return sum(i.value() for i in xs)\n")

    def hints(code: str) -> list[str]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with spent_budget():
                run_cash_cell(cash_magics, code)
        return [str(w.message) for w in caught if "NOTEBOOK-ANALYSIS-BUDGET" in str(w.message)]

    assert hints("fast = total(items)") == []
    first = hints("slow_out = slow_total(items)")
    assert len(first) == 1, first
    assert hints("slow_out = slow_total(items)") == []


def test_what_it_changes_of_the_process_is_seen(cash_magics, mock_shell, tmp_path, monkeypatch):
    """Past the budget, what its calls were handed is not read, so a change
    to the environment made through it is watched for as it runs."""
    monkeypatch.delenv("CASH_TEST_BUDGET_MODE", raising=False)
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, f"RUNS = {str(tmp_path / 'runs.log')!r}\n" + SETUP)
    run_cash_cell(
        cash_magics,
        "class Setter:\n    def value(self):\n        os.environ['CASH_TEST_BUDGET_MODE'] = 'b'\n        return 0\n",
    )
    run_cash_cell(cash_magics, "setters = [Setter()]")
    with spent_budget():
        run_cash_cell(cash_magics, "z = total(setters)")
    assert os.environ.get("CASH_TEST_BUDGET_MODE") == "b"
    assert statement_source_hash("z = total(setters)") in cash_magics.tracking_state.process_state_ran


# ---------------------------------------------------------------------------
# The analyses: stopped as they go, each with its "could not tell" answer
# ---------------------------------------------------------------------------

GRAPH = """
class Order:
    def __init__(self, items):
        self.items = items
    def total(self):
        return sum(i.amount() for i in self.items)

class Line:
    def __init__(self, q):
        self.q = q
    def amount(self):
        return self.q

def revenue(orders):
    global counter
    counter += 1
    return sum(o.total() for o in orders)

counter = 0
orders = [Order([Line(j) for j in range(3)]) for _ in range(10000)]
"""


def _namespace() -> dict:
    filename = "<budget-graph>"
    linecache.cache[filename] = (len(GRAPH), None, GRAPH.splitlines(True), filename)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(GRAPH, filename, "exec"), namespace)
    return namespace


def test_the_walk_stops_at_the_budget_not_after_it(monkeypatch):
    """The clock is read as the walk goes: a budget spent after a few reads
    stops it there, long before the graph is looked through."""
    namespace = _namespace()
    opened = [0]
    original = hc._items

    def counting(container):
        opened[0] += 1
        return original(container)

    monkeypatch.setattr(hc, "_items", counting)
    tree = ast.parse("out = revenue(orders)")
    new_analysis_round()
    handed_callables(tree, namespace)
    full = opened[0]
    assert full > 10000

    clock = itertools.count()
    monkeypatch.setattr(ab, "_perf_counter", lambda: float(next(clock)))
    opened[0] = 0
    new_analysis_round()
    with pytest.raises(AnalysisBudgetExceeded), analysis_budget(AnalysisBudget(2.5)):
        handed_callables(tree, namespace)
    assert opened[0] < full / 10, (opened[0], full)


def test_past_the_budget_a_global_a_named_function_writes_is_still_an_output():
    namespace = _namespace()
    budget = AnalysisBudget(0.0)
    budget.tripped = True
    with analysis_budget(budget):
        effects = statement_effects(
            "out = revenue(orders)",
            ast.parse("out = revenue(orders)"),
            namespace=namespace,
            resolve_source=lambda name: inspect.getsource(namespace[name]) if name == "revenue" else None,
        )
    assert effects.unanalysed == (ANALYSIS_BUDGET_REASON,)
    assert "counter" in effects.outputs
    assert "out" in effects.outputs


def test_past_the_budget_the_cell_check_takes_every_notebook_function_as_run():
    """The cell-wide check cannot say what the cell's calls were handed: any
    function of the notebook may be among it, and what it writes counts."""
    namespace = _namespace()
    namespace["hits"] = 0
    cells = [GRAPH, "def bump():\n    global hits\n    hits = 1\n"]
    code = "out = sum(o.total() for o in orders)"
    within = cell_effects(code, NotebookSources(lambda: cells, code), namespace)
    assert "hits" not in within.mutated
    budget = AnalysisBudget(0.0)
    budget.tripped = True
    with analysis_budget(budget):
        past = cell_effects(code, NotebookSources(lambda: cells, code), namespace)
    assert "hits" in past.mutated
