"""A change check runs on past its budget for a value it knows the size of,
when the value is on course to finish soon: it is not a coin flip.

A networkx graph of 350,000 edges (an 11 MB pickle) hashes in 0.3-0.55 s,
right at a change check's 0.5 s budget (`value_hash.budgeted_hash`). After
a restart, one run in a few cut its check short: the graph was remembered
as too costly to read, counted as changed, and every statement built on it
ran again (195 s of a 615 s replay). The same graph read in full matched
its earlier content, and all of it would have been served from the cache.

A check now knows what the value pickled to last time (`known_size`: its
last full digest, or the entry it was restored from, which records it as
``output_sizes``). Past its budget, a value whose pickle so far projects it
done within `CHANGE_CHECK_RUN_ON` budgets runs on, for up to
`CHANGE_CHECK_STRETCH` budgets in all. A value nothing is known of, or one
clearly too big to read (an openpyxl workbook, a 2 GB object), stops at the
budget as before and costs no more. When a cut does make a value count as
changed, the miss below it says so.

The clock here is a fake one that each part of a value moves on as it is
pickled, so nothing depends on how fast the machine is.
"""

from __future__ import annotations

import time

import pytest

from cash import stored_pickle, value_hash
from cash.core import Cash
from cash.notebook.cache_status import CacheStatus
from cash.notebook.ipython.magics import CashMagics
from cash.notebook.restored_var import apply_restored_var
from cash.notebook.statement._metadata import StatementCacheMetadata
from cash.notebook.statement.input_change import input_change_reason
from cash.notebook.tracking_state import TrackingState
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S, MockShell

#: The bytes of one part, enough that each is written to the digest's
#: sink on its own (and so checked against the deadline).
PART = 1 << 17


class Clock:
    """The time the checks read: moved on only by pickling a `Part`."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.step = 0.0
        self.parts = 0

    def __call__(self) -> float:
        return self.now


CLOCK = Clock()


def _part(data):
    return data


class Part:
    """One part of a `Slab`: pickling it takes `CLOCK.step` seconds."""

    def __init__(self, i: int) -> None:
        self.data = bytes([i % 251]) * PART

    def __reduce__(self):
        CLOCK.now += CLOCK.step
        CLOCK.parts += 1
        return (_part, (self.data,))


class Slab:
    """A value that pickles in ``n * CLOCK.step`` seconds, ``n * PART`` bytes."""

    def __init__(self, n: int) -> None:
        self.parts = [Part(i) for i in range(n)]


@pytest.fixture
def clock(monkeypatch):
    CLOCK.now, CLOCK.step, CLOCK.parts = 1000.0, 0.0, 0
    monkeypatch.setattr(value_hash, "perf_counter", CLOCK)
    monkeypatch.setattr(stored_pickle, "perf_counter", CLOCK)
    monkeypatch.setattr(value_hash, "CHANGE_CHECK_BUDGET_S", 0.5)
    return CLOCK


def _size_of(n: int) -> int:
    """What a `Slab` of *n* parts pickles to, read in full."""
    _digest, size = stored_pickle.sized_object_digest(Slab(n))
    return size


def _check(clock: Clock, n: int, step: float, known: int | None) -> tuple[str | None, int, Slab]:
    """A budgeted hash of a new `Slab` of *n* parts, *step* seconds each,
    its size known to be *known* bytes (None: not known); and how many
    parts it pickled."""
    slab = Slab(n)
    if known is not None:
        value_hash.note_size(slab, known)
    clock.step, clock.parts = step, 0
    found = value_hash.budgeted_hash(slab)
    return found, clock.parts, slab


def test_a_value_nothing_is_known_of_is_cut_at_the_budget(clock):
    """As before: no size known, so no telling how far along it is."""
    found, parts, slab = _check(clock, 20, 0.1, known=None)
    assert found is None
    assert value_hash.is_costly(slab)
    assert value_hash.costly_after(slab) == pytest.approx(0.5)
    assert parts <= 6, "read past its budget"


def test_a_value_known_to_be_close_is_read_in_full(clock):
    """0.7 s against a 0.5 s budget: at the budget it is 70% through, so on
    course for 0.7 s, within two budgets. It runs on, gets its real digest,
    and is not costly."""
    size = _size_of(10)
    found, parts, slab = _check(clock, 10, 0.07, known=size)
    assert found is not None
    assert found == value_hash.compute_hash(Slab(10))
    assert parts == 10
    assert not value_hash.is_costly(slab)


def test_a_clear_overrun_is_cut_at_the_budget_however_well_known(clock):
    """4 s against a 0.5 s budget: at the budget it is an eighth through,
    projected far past two budgets, and stops there, as before."""
    size = _size_of(40)
    found, parts, slab = _check(clock, 40, 0.1, known=size)
    assert found is None
    assert value_hash.is_costly(slab)
    assert parts <= 6, "a clear overrun read past its budget"


def test_running_on_ends_at_the_stretch(clock):
    """A value that grew far past what it was known to pickle to looks on
    course at the budget, runs on, and is cut at `CHANGE_CHECK_STRETCH`
    budgets all the same."""
    found, parts, slab = _check(clock, 40, 0.07, known=_size_of(10))
    assert found is None
    assert value_hash.is_costly(slab)
    stretch = value_hash.CHANGE_CHECK_BUDGET_S * value_hash.CHANGE_CHECK_STRETCH
    assert parts <= int(stretch / 0.07) + 2
    assert value_hash.costly_after(slab) == pytest.approx(stretch)


def test_a_value_past_what_it_was_known_to_be_is_cut_at_the_budget(clock):
    """At the budget it has pickled more than it ever did: how far along it
    is is not known, so it stops there."""
    found, parts, _slab = _check(clock, 20, 0.1, known=_size_of(3))
    assert found is None
    assert parts <= 6


def test_a_full_digest_teaches_the_size(clock):
    slab = Slab(10)
    assert value_hash.known_size(slab) is None
    _digest, size = stored_pickle.sized_object_digest(slab)
    value_hash.compute_hash(slab)
    assert value_hash.known_size(slab) == size
    small = Slab(1)
    value_hash.compute_hash(small)
    assert value_hash.known_size(small) is None, "a small value is not worth remembering"


def test_a_restored_value_knows_its_size(clock):
    """Restored, a value knows what it pickled to when it was stored, as in
    the process that stored it."""
    state = TrackingState()
    slab = Slab(10)
    metadata = StatementCacheMetadata(output_lineages={"s": "l1"}, output_sizes={"s": 1_300_000})
    apply_restored_var(state, "s", slab, metadata)
    assert value_hash.known_size(slab) == 1_300_000


SETUP = f"""
import time

def slow(x):
    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})
    return x
"""


def test_a_stored_value_records_its_size_for_the_next_kernel(tmp_path, monkeypatch):
    """The entry records what the output pickled to; a kernel that restores
    it knows the size before it checks it."""
    monkeypatch.setattr(stored_pickle, "perf_counter", time.perf_counter)
    cells = [SETUP, "s = slow(Slab(12))"]

    def kernel() -> CashMagics:
        magics = CashMagics(MockShell(), Cash(cache_dir=str(tmp_path), register_magic=False))
        magics.cash_on("")
        magics.cash_persist("on")
        magics.badges.mode = "off"
        magics.shell.user_ns.update(Slab=Slab, Part=Part, _part=_part)
        return magics

    first = kernel()
    for cell in cells:
        run_cash_cell(first, cell, cells=cells)
    size = value_hash.known_size(first.shell.user_ns["s"])
    assert size is not None and size >= 12 * PART

    second = kernel()
    for cell in cells:
        run_cash_cell(second, cell, cells=cells)
    restored = second.shell.user_ns["s"]
    assert restored is not first.shell.user_ns["s"]
    assert value_hash.known_size(restored) == size


# -- when a cut does count a value as changed, the miss says so --------------

#: A pickle of a `Book` takes this long, past the budget the test sets.
READ_S = 0.3
BUDGET_S = 0.1


def _book(rows):
    book = Book.__new__(Book)
    book.rows = rows
    return book


class Book:
    """Pickles in `READ_S`, as a big openpyxl workbook takes seconds."""

    def __init__(self):
        self.rows = [0]

    def __reduce__(self):
        time.sleep(READ_S)
        return (_book, (list(self.rows),))


def test_a_miss_names_the_check_that_gave_up(cash_magics, statement_processor, monkeypatch):
    """``show(book)`` keeps nothing and hands on a book too costly to read:
    the book counts as changed, and the statement below it that misses for
    that says it was a check that gave up, not a change seen."""
    monkeypatch.setattr(value_hash, "CHANGE_CHECK_BUDGET_S", BUDGET_S)
    user_ns = cash_magics.shell.user_ns
    user_ns.update(Book=Book, _book=_book)
    cells = [
        SETUP + "\ndef show(book):\n    return len(book.rows)\n",
        "book = slow(Book())",
        "n = slow(len(book.rows))",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert value_hash.is_costly(user_ns["book"])

    shown = statement_processor.process_statement("show(book)")
    reasons = " ".join(shown.get("uncacheable_reasons") or [])
    assert "book" in reasons and "too costly to read" in reasons, reasons

    again = statement_processor.process_statement("n = slow(len(book.rows))")
    assert again["status"] == CacheStatus.COMPUTED
    reason = again.get("miss_reason") or ""
    assert reason.startswith("input changed: book"), reason
    assert "gave up after 0.1s" in reason, reason


def test_a_miss_after_a_real_change_names_no_check():
    state = TrackingState()
    state.lineage.record("df", "new")
    state.executed_input_lineages["out"] = {"df": "old"}
    assert input_change_reason(state, ["df"], ["out"]) == "input changed: df"
    state.budget_cut_lineages["df"] = ("older", 0.5)
    assert input_change_reason(state, ["df"], ["out"]) == "input changed: df", "a cut since moved on says nothing"
    state.budget_cut_lineages["df"] = ("new", 0.5)
    assert "gave up after 0.5s" in input_change_reason(state, ["df"], ["out"])
