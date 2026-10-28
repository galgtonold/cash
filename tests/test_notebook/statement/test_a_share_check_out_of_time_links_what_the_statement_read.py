"""A statement whose share check runs out of time is linked to what it read.

``box = Box(BIG)`` with ``box`` holding ``BIG`` inside a pandas object: when
the check of what ``box`` shares ran out of time, the variables holding
``box`` were looked for up the garbage collector's referrers, which finds
what holds ``box``, never what ``box`` holds. ``box`` and ``BIG`` were left
unlinked, and after ``BIG[0].id`` was edited, ``y = box.total()`` was served
from before. What the statement read is linked now, as when that search is
cut short itself."""

from __future__ import annotations

import pytest

from cash.notebook.shared_objects import WalkBudgetExceeded
from cash.notebook.statement import processor as processor_module
from tests._cell_driver import run_cash_cell

MAKE = (
    "import pandas as pd\nclass Row:\n    def __init__(self, i):\n        self.id = i\n"
    "BIG = [Row(i) for i in range(50)]\n"
    "class Box:\n    def __init__(self, rows):\n        self.h = pd.Series([rows, 1], dtype=object)\n"
    "    def total(self):\n        return sum(r.id for r in self.h[0])"
)


@pytest.fixture
def out_of_time_for_box(monkeypatch):
    """The share check of ``box = Box(BIG)`` runs out of time, both the one
    of a stored statement and the one of its links."""
    refusal, group = processor_module.unrestorable_output_reason, processor_module.share_group

    def refuse(outputs, *args, **kwargs):
        if "box" in outputs:
            raise WalkBudgetExceeded
        return refusal(outputs, *args, **kwargs)

    def share(names, *args, **kwargs):
        if "box" in names:
            raise WalkBudgetExceeded
        return group(names, *args, **kwargs)

    monkeypatch.setattr(processor_module, "unrestorable_output_reason", refuse)
    monkeypatch.setattr(processor_module, "share_group", share)


def test_a_change_to_what_it_read_moves_the_reader_on(cash_magics, cash_instance, out_of_time_for_box):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    for edit in (-5, -6, -7):
        cells = [MAKE, "box = Box(BIG)", f"BIG[0].id = {edit}", "y = box.total()"]
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert cash_magics.shell.user_ns["y"] == sum(range(1, 50)) + edit
