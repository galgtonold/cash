"""A function that a statement run every time hands on is keyed by what it
runs and holds, not by its identity.

``a = bump()`` changes the cell ``peek`` shares, so it runs every time and
hands ``peek`` on, keyed by value. A function hashed by identity, remade by
``make_counter()`` in each Run All, was new each time, so ``p = peek()``
below ran again in every Run All though ``n`` was the same.
"""

from cash.notebook.lineage_formula import no_cache_value_digest
from tests._cell_driver import run_cash_cell

MAKE = (
    "import time\n"
    "def make_counter():\n    n = 0\n    def bump():\n        nonlocal n\n        n += 1\n        return n\n"
    "    def peek():\n        time.sleep(0.05)\n        return n\n    return bump, peek\n"
)


def _make_counter():
    n = 0

    def bump():
        nonlocal n
        n += 1
        return n

    def peek():
        return n

    return bump, peek


def test_the_same_closure_values_give_the_same_digest():
    first, second = _make_counter(), _make_counter()
    assert no_cache_value_digest(first[1]) == no_cache_value_digest(second[1])
    second[0]()
    assert no_cache_value_digest(first[1]) != no_cache_value_digest(second[1])


def test_the_reader_is_served_in_a_later_run_all(cash_magics):
    cells = [MAKE, "bump, peek = make_counter()", "a = bump()", "p = peek()"]
    for _ in range(3):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    statuses = [str(s.get("status")).split(".")[-1] for s in cash_magics._last_cell_metrics.get("statements", [])]
    assert statuses == ["RESTORED"], statuses
    assert cash_magics.shell.user_ns["p"] == 1
