"""Without a record of what ran, the reason names a method that could run.

With no trace (Python 3.10 and 3.11, ``CASH_TRACE=0``, a debugger holding
``sys.monitoring``) the objects a statement reads are looked through for a
special method that writes outside them, and a value too big to look
through counts every class alive with one. The reason said such a method
"ran in this statement" when nothing showed it did.
"""

from __future__ import annotations

import pytest

from cash.notebook import shadow
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S as F

HEAD = (
    "import time, types\n"
    "LOG = []\n"
    "class Vec:\n"
    "    def __add__(self, other):\n"
    "        LOG.append(other)\n"
    "        return 1\n"
)


@pytest.fixture
def no_trace(monkeypatch):
    monkeypatch.setenv("CASH_TRACE", "0")
    monkeypatch.setattr(shadow, "SHADOW", None)


def _reasons(statement_processor, code):
    return " ".join(statement_processor.process_statement(code)["uncacheable_reasons"])


def test_a_method_reached_could_run(no_trace, cash_magics, statement_processor):
    run_cash_cell(cash_magics, HEAD + "import numpy as np\narr = np.empty(1, dtype=object)\narr[0] = Vec()")
    reasons = _reasons(statement_processor, f"w = (arr + 1, time.sleep({F}))[0]")
    assert "Vec.__add__() could run in this statement (no record of what runs" in reasons, reasons
    assert "ran in this statement" not in reasons


def test_a_value_too_big_to_look_through_says_so(no_trace, cash_magics, statement_processor):
    run_cash_cell(cash_magics, HEAD + "v = Vec()\nbig = [types.SimpleNamespace(i=i) for i in range(300_000)]")
    reasons = _reasons(statement_processor, f"w = (len(big), time.sleep({F}))[0]")
    assert "Vec.__add__() could run in this statement: a value it reads is too big to look through" in reasons, reasons
    assert "ran in this statement" not in reasons
