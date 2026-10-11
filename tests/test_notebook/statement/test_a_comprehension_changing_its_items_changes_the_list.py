"""A comprehension or ``map`` that changes the items it walks changes the list.

``[r.update(tax=1) for r in records]`` and ``list(map(lambda r:
r.update(tax=1), records))`` change what ``records`` holds, as handing each
``r`` to a function of the user's does. Watched like that call's argument,
``records`` moves on when it changed, and a cell reading it after the
statement is edited is not served the old items.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

SETUP = "def summarize(recs):\n    return sorted(set(k for r in recs for k in r))\ndef add(r):\n    r['FIELD'] = 1"
SHAPES = {
    "a method on the comprehension's item": "[r.update(FIELD=1) for r in records]",
    "a lambda handed to map": "list(map(lambda r: r.update(FIELD=1), records))",
    "a function of the user's": "_ = [add(r) for r in records]",
}


@pytest.fixture
def store_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0


@pytest.mark.usefixtures("store_everything")
@pytest.mark.parametrize("shape", list(SHAPES))
def test_an_edit_reaches_the_reader(cash_magics, mock_shell, shape):
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"

    def run(field: str) -> None:
        cells = [
            SETUP.replace("FIELD", field),
            "records = [{'p': 1}, {'p': 2}]",
            SHAPES[shape].replace("FIELD", field),
            "fields = summarize(records)",
        ]
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)

    run("tax")
    assert mock_shell.user_ns["fields"] == ["p", "tax"]
    run("vat")
    assert mock_shell.user_ns["fields"] == ["p", "vat"]
