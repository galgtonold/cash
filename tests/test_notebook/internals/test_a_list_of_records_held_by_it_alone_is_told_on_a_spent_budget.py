"""A list of records held by it alone is told apart on a spent budget.

``for p in points: total += p.x`` over 200,000 records: the share check of
the loop tells at C speed that each record but ``p`` is held by the list
alone, one step for the whole list. The budget was asked before that step,
once the list's size was known, and a cheap loop had spent its 0.1 s on the
checks before it, so the loop was refused as too slow to check instead of
finding that ``p`` shares with ``points``. The budget is asked before a
step per child, which a list left with many children to walk still takes.
"""

from __future__ import annotations

import pytest

from cash.notebook.shared_objects import WalkBudgetExceeded, share_group, walk_budget

#: The limits as shipped, not as tests/test_notebook pins them.
pytestmark = pytest.mark.real_time_limits

RECORDS = 20_000


class Point:
    def __init__(self, x: int) -> None:
        self.x = x


def test_the_holder_of_a_record_is_found_with_the_budget_spent():
    points = [Point(i) for i in range(RECORDS)]
    ns = {"points": points, "p": points[-1]}
    with walk_budget(0.0):
        holders, shared = share_group(["p"], {"p": ns["p"]}, ns)
    assert (holders, shared) == ({}, {"p"})


def test_a_list_left_with_many_children_to_walk_asks_the_budget():
    """The positive control: records that hold more than values are
    walked one at a time, and the spent budget stops that walk."""
    points = [Point([i]) for i in range(RECORDS)]
    ns = {"points": points, "p": points[-1]}
    with pytest.raises(WalkBudgetExceeded), walk_budget(0.0):
        share_group(["p"], {"p": ns["p"]}, ns)
