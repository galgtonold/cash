"""Putting what the check above a cell restored and re-ran in notebook order
reads each cell's text once, not the whole notebook after every edit.

The order of every statement in the notebook was worked out per notebook
version: an edited cell is a new version, so the next check unparsed every
statement of every cell again. A participant who edits and runs one cell
after another paid that on every run, more the longer the notebook. It is now
worked out per cell text, and not at all for a check with nothing to order.

Counted in ``ast.unparse`` calls, not timed.
"""

from __future__ import annotations

import ast
from unittest.mock import patch

from cash.notebook.upstream.replay import StatementReplay


def _unparses(call) -> int:
    real = ast.unparse
    count = 0

    def counting(*args, **kwargs):
        nonlocal count
        count += 1
        return real(*args, **kwargs)

    with patch.object(ast, "unparse", counting):
        call()
    return count


def _notebook(n: int, tag: str) -> list[str]:
    return [f"{tag}{i} = {i}\nfor k in range(2):\n    {tag}{i} += k" for i in range(n)]


def test_an_edit_reorders_by_reading_the_edited_cell_alone():
    cells = _notebook(60, "ord")
    metrics = [{"code": "ord5 = 5"}, {"code": "ord1 = 1"}]
    ordered = StatementReplay.in_notebook_order(metrics, cells)
    assert [m["code"] for m in ordered] == ["ord1 = 1", "ord5 = 5"]

    edited = list(cells)
    edited[30] = "ord30 = 300\nfor k in range(2):\n    ord30 += k"
    unparsed = _unparses(lambda: StatementReplay.in_notebook_order(metrics, edited))

    # The edited cell's three statements; every cell's was 180.
    assert unparsed <= 3, f"{unparsed} statements unparsed after one cell of 60 changed"
    assert [m["code"] for m in StatementReplay.in_notebook_order(metrics, edited)] == ["ord1 = 1", "ord5 = 5"]


def test_a_check_with_nothing_to_order_reads_no_cell():
    edited = _notebook(40, "none")
    edited[3] = "none3 = 30\nfor k in range(2):\n    none3 += k"

    unparsed = _unparses(lambda: StatementReplay.in_notebook_order([{"code": "none1 = 1"}], edited))

    assert unparsed == 0
