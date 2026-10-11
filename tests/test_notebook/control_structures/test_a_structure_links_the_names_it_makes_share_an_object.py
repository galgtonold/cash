"""A loop, branch or try block links the names it makes share an object.

A body statement records no shared-object edge: it runs once per pass, and
the upstream simulation treats the structure as one statement. When the
structure ends, what it changed or bound is linked to every other variable it
shares an object with, so ``for name, rows in pairs: splits[name] = rows``
then a change of ``train_rows`` inside another block moves ``splits``.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

SPLITS_LOOP = "splits = {}\nfor name, rows in [('train', train_rows), ('test', test_rows)]:\n    splits[name] = rows"
SPLITS_TRY = "splits = {}\ntry:\n    splits['train'] = train_rows\nexcept KeyError:\n    pass"
SPLITS_IF = "splits = {}\nif train_rows:\n    splits['train'] = train_rows"
SPLITS_NESTED = (
    "splits = {}\nfor k in range(2):\n    for name, rows in [('train', train_rows), ('test', test_rows)]:\n"
    "        splits[name] = rows"
)
CHANGES = {
    "a loop": "for x in [10]:\n    train_rows.append(x)",
    "an if": "if train_rows:\n    train_rows.extend([10])",
    "a try": "try:\n    train_rows.extend([10])\nexcept KeyError:\n    pass",
}


@pytest.mark.parametrize(
    "make", [SPLITS_LOOP, SPLITS_TRY, SPLITS_IF, SPLITS_NESTED], ids=["loop", "try", "if", "nested loop"]
)
@pytest.mark.parametrize("change", list(CHANGES))
def test_a_change_in_a_block_moves_what_a_block_made_hold_it(cash_magics, make, change):
    cells = ["train_rows = [1, 2]\ntest_rows = [3]\nunrelated = [1, 2]", make, CHANGES[change]]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for cell in cells[:2]:
        run_cash_cell(cash_magics, cell, cells=cells)
    lineage = cash_magics.tracking_state.variable_lineage
    before, control = lineage["splits"], lineage["unrelated"]
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert lineage["splits"] != before
    assert lineage["unrelated"] == control


def test_a_block_holding_nothing_shared_links_nothing(cash_magics):
    """Control: the dict holds copies, so changing the list leaves it."""
    cells = [
        "train_rows = [1, 2]",
        "splits = {}\nfor name in ['train']:\n    splits[name] = list(train_rows)",
        "for x in [10]:\n    train_rows.append(x)",
    ]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for cell in cells[:2]:
        run_cash_cell(cash_magics, cell, cells=cells)
    before = cash_magics.tracking_state.variable_lineage["splits"]
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert cash_magics.tracking_state.variable_lineage["splits"] == before
