"""What cash holds itself is not another holder of a statement's outputs.

The share check refuses a statement whose outputs something else holds too,
and finds that something by searching every object the garbage collector
tracks (`_holders_from_referrers`): 0.1-0.7 s a search next to torch and
MNIST on E11. Two of cash's own references sent it there:

* the digests a statement took of its arguments before it ran, handed to
  its call (`DigestHandoff.held`): ``ds = Dataset(file_list, labels)`` was
  searched for, and refused, since the search names no variable for them;
* the tuple ``enumerate`` fills again for each item, which still holds the
  last one after the loop (`_iterator_parts`).
"""

from __future__ import annotations

import pytest

from cash.notebook import shared_objects
from tests._cell_driver import run_cash_cell

PLAIN = "class Plain:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y"
ITEMS = (
    "class Items:\n    def __getitem__(self, i):\n        if i >= 3:\n            raise IndexError\n"
    "        return ([i] * 3, i)\n    def __len__(self):\n        return 3\nds = Items()"
)


@pytest.fixture
def searched(monkeypatch, mock_shell):
    mock_shell.user_ns["__name__"] = "__main__"
    found: list[int] = []
    real = shared_objects._holders_from_referrers
    monkeypatch.setattr(shared_objects, "_holders_from_referrers", lambda *a, **k: found.append(1) or real(*a, **k))
    return found


def test_the_digests_handed_to_a_call_are_no_holder(cash_magics, statement_processor, searched):
    run_cash_cell(cash_magics, PLAIN)
    run_cash_cell(cash_magics, "l1 = [[1], [2]]\nl2 = [[3]]")
    searched.clear()
    metrics = statement_processor.process_statement("pl = Plain(l1, l2)")
    assert searched == []
    reasons = " ".join(metrics.get("uncacheable_reasons") or [])
    assert "hold" not in reasons, reasons


def test_the_tuple_enumerate_fills_is_no_holder(cash_magics, searched):
    run_cash_cell(cash_magics, ITEMS)
    searched.clear()
    run_cash_cell(cash_magics, "out = []\nfor i, example in enumerate(ds):\n    out.append(example[1])")
    assert searched == []
