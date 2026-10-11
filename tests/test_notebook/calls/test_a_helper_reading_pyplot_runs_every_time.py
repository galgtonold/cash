"""A helper of the user's that reads pyplot's current figure runs every time.

``plt.gca()`` written in a statement is display: it reads pyplot's current
figure, which no key holds. ``n = nlines()``, where ``nlines`` returns
``len(plt.gca().lines)``, changes no figure, so the check of pyplot's state
before and after it saw nothing, and the old drawing's answer was served
after the drawing cell above was edited.
"""

from __future__ import annotations

import time

import pytest

from cash.notebook.call_interception import CallSite
from cash.purity import pure
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import gcf


def nlines():
    return len(plt.gca().lines)


def summary():
    return {"lines": nlines()}  # pyplot is one call further down


def title_of_current():
    return gcf().axes[0].get_title()


def total(values):
    return sum(values)


@pure
def promised():
    return len(plt.gca().lines)


def test_a_helper_calling_pyplot_is_found():
    from cash.analysis.namespace_effects import user_callee_using_pyplot

    assert user_callee_using_pyplot(nlines) == "nlines"
    assert user_callee_using_pyplot(title_of_current) == "title_of_current"


def test_pyplot_is_found_through_another_user_function():
    from cash.analysis.namespace_effects import user_callee_using_pyplot

    assert user_callee_using_pyplot(summary) == "nlines"


def test_a_helper_without_pyplot_or_promised_pure_is_not():
    from cash.analysis.namespace_effects import user_callee_using_pyplot

    assert user_callee_using_pyplot(total) is None
    assert user_callee_using_pyplot(promised) is None


HELPER = (
    "import time\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
    f"def nlines():\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return len(plt.gca().lines)"
)


def test_a_statement_calling_it_reads_the_new_drawing(cash_magics):
    one = "plt.figure()\nplt.plot([1, 2])"
    two = "plt.figure()\nplt.plot([1, 2])\nplt.plot([2, 1])"
    cells = [HELPER, one, "n = nlines()"]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    try:
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert cash_magics.shell.user_ns["n"] == 1
        cells[1] = two
        for cell in cells[1:]:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert cash_magics.shell.user_ns["n"] == 2
    finally:
        plt.close("all")


def test_a_call_unit_does_not_store_it(call_unit_harness):
    calls = []
    note = calls.append

    def lines_now():
        note(1)
        time.sleep(0.05)
        return len(plt.gca().lines)

    try:
        plt.figure()
        plt.gca()  # the axes exist before the call: it changes no figure
        unit = call_unit_harness(lineage={}, user_ns={"lines_now": lines_now})
        wrapped = unit.wrap(
            lines_now, CallSite(source="lines_now()", free_names=frozenset({"lines_now"}), occurrence_index=0)
        )
        assert wrapped() == 0
        plt.plot([1, 2])
        assert wrapped() == 1
        assert calls == [1, 1]
    finally:
        plt.close("all")
