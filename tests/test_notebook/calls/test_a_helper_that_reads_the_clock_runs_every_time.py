"""A statement calling a function of yours that reads the clock or a UUID.

``s = stamp(1)``, with ``stamp`` doing some work and returning
``time.time()``: the statement's text reads no clock, so it was cached and
the first time came back on every run, without a word. It now runs every
time and KEY-AMBIENT-READ says why, once.
"""

from __future__ import annotations

import warnings

import pytest

from cash.exceptions import CashImpurityWarning
from tests._cell_driver import run_cash_cell

_IMPORTS = "import time\nimport uuid"
_STAMP = "def stamp(x):\n    time.sleep(0.2)\n    return x, str(uuid.uuid4())"
_OUTER = "def label(x):\n    return {'stamp': stamp(x)}"
_TIMED = (
    "def timed(x):\n    t0 = time.perf_counter()\n    y = x * 2\n"
    "    time.sleep(0.2)\n    print('took', time.perf_counter() - t0)\n    return y"
)


def _run_twice(magics, cells, statement):
    cells = [_IMPORTS, *cells, statement]
    values = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cell in cells:
            run_cash_cell(magics, cell, cells=cells)
        values.append(magics.shell.user_ns["s"])
        run_cash_cell(magics, statement, cells=cells)
        values.append(magics.shell.user_ns["s"])
    found = [str(w.message) for w in caught if issubclass(w.category, CashImpurityWarning)]
    return values, [m for m in found if "KEY-AMBIENT-READ" in m]


@pytest.mark.parametrize(
    ("cells", "statement", "names"),
    [
        ([_STAMP], "s = stamp(1)", "stamp()"),
        ([_STAMP, _OUTER], "s = label(1)", "stamp()"),
    ],
    ids=["direct", "through-a-helper"],
)
def test_a_helper_reading_a_uuid_runs_every_time(full_warning_text, cash_magics, cells, statement, names):
    values, found = _run_twice(cash_magics, cells, statement)
    assert values[0] != values[1]
    assert len(found) == 1, found
    assert names in found[0]
    assert "uuid.uuid4()" in found[0]


def test_a_clock_read_only_logged_is_cached(full_warning_text, cash_magics):
    values, found = _run_twice(cash_magics, [_TIMED], "s = timed(3)")
    assert found == []


def test_assume_safe_keeps_the_first_value(full_warning_text, cash_magics):
    values, found = _run_twice(cash_magics, [_STAMP], "s = stamp(1)  # @cash:assume-safe")
    assert values[0] == values[1]
    assert found == []
