"""A value that takes too long to read is not read again: it counts as changed.

The checks around a statement read a value that is not a frame or a big
collection in full: its hash after it is bound (the session hash), and before
and after a statement that may change it. An openpyxl workbook of 400,000
cells pickles in about 3 s, and a cell that loads, formats and saves one was
read after each of its statements, 15-40 s a run next to 11 s of its own
work. Past `CHANGE_CHECK_BUDGET_S` the value is remembered as costly: its
lineage stands for its content, a statement that may change it is taken to
change it, and nothing reads it again. Treating it as changed costs re-runs,
never a stale value; a key never takes this shortcut.
"""

from __future__ import annotations

import time

import pytest

from cash import value_hash
from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

#: A pickle of a `Book` takes this long, past the budget the tests set.
READ_S = 0.3
BUDGET_S = 0.1

#: How many times a `Book` was pickled.
READS: list[int] = []


def _book(rows):
    book = Book.__new__(Book)
    book.rows = rows
    return book


class Book:
    """Pickles in `READ_S`, as a big openpyxl workbook takes seconds."""

    def __init__(self):
        self.rows = [0]

    def __reduce__(self):
        READS.append(1)
        time.sleep(READ_S)
        return (_book, (list(self.rows),))


SETUP = f"""
import time

def slow(x):
    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})
    return x

def touch(book):
    book.rows.append(1)
"""

CELLS = [SETUP, "book = slow(Book())", "touch(book)\ntouch(book)\nbook.rows.append(5)", "n = slow(len(book.rows))"]


@pytest.fixture
def small_budget(monkeypatch, cash_magics):
    monkeypatch.setattr(value_hash, "CHANGE_CHECK_BUDGET_S", BUDGET_S, raising=False)
    cash_magics.shell.user_ns.update(Book=Book, _book=_book)
    READS.clear()


def test_a_budgeted_hash_is_the_full_hash_within_its_budget():
    value = {"a": [1, 2, (3, "x")], "b": "y" * 1000}
    assert value_hash.budgeted_hash(value, seconds=60) == value_hash.compute_hash(value)
    assert not value_hash.is_costly(value)


def test_a_value_past_the_budget_is_remembered_as_costly():
    value = Book()
    assert value_hash.budgeted_hash(value, seconds=BUDGET_S) is None
    assert value_hash.is_costly(value)
    started = time.perf_counter()
    assert value_hash.budgeted_hash(value, seconds=BUDGET_S) is None
    assert time.perf_counter() - started < READ_S
    # A key is never taken under the budget: the full hash still reads it.
    assert value_hash.compute_hash(value) == value_hash.compute_hash(Book())
    assert not value_hash.is_costly(Book())


def test_a_costly_value_changed_in_place_reads_as_changed(cash_magics, statement_processor, small_budget):
    """Run All twice: the change ``touch`` makes reaches ``n`` both times,
    as in plain Python, and the call that makes it is never served. (A
    control: it holds without the budget too, by reading the book.)"""
    user_ns = cash_magics.shell.user_ns
    for run in range(2):
        for cell in CELLS:
            run_cash_cell(cash_magics, cell, cells=CELLS)
        assert user_ns["n"] == 4, f"Run All {run + 1}"
        assert user_ns["book"].rows == [0, 1, 1, 5], f"Run All {run + 1}"
    metrics = statement_processor.process_statement("touch(book)")
    assert metrics["status"] == CacheStatus.COMPUTED
    assert "book" in " ".join(metrics.get("uncacheable_reasons") or []), metrics.get("uncacheable_reasons")


def test_a_costly_value_is_read_at_most_once_a_run(cash_magics, small_budget):
    """Each Run All binds a new book: it is read until the budget runs out
    once (and pickled to be stored), never after each statement."""
    for cell in CELLS:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    first = len(READS)
    for cell in CELLS[1:]:
        run_cash_cell(cash_magics, cell, cells=CELLS)
    assert cash_magics.shell.user_ns["n"] == 4
    assert first <= 2, first
    assert len(READS) - first <= 2, len(READS) - first
