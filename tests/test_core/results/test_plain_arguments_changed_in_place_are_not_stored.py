"""A plain list or tuple changed in place by the body is never stored.

The check that the body left a plain argument alone compares the identities
of what it holds instead of hashing it again (`_plain_data.identity_changed`).
Each change below must still read as a change, at any depth, and a call
that makes one is not stored, so the warm run makes it again.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.core]


def _sort(rows):
    rows.sort(reverse=True)


def _append(rows):
    rows.append(0)


def _replace_item(rows):
    rows[3] = rows[3] + 0.5


def _swap_equal_items(rows):
    # The same values in other objects: -0.0 equals 0.0 and True equals 1,
    # but they pickle apart, so they are a change.
    rows[0] = -0.0
    rows[1] = True


def _nested_list(rows):
    rows[2][1] = "changed"


def _list_under_tuple(rows):
    rows[1][0].append("x")


def _shrink_and_regrow(rows):
    last = rows.pop()
    rows.insert(0, last)


CASES = [
    (_sort, lambda: [5, 3, 9, 1, 7]),
    (_append, lambda: [1, 2, 3, 4]),
    (_replace_item, lambda: [1.0, 2.0, 3.0, 4.0]),
    (_swap_equal_items, lambda: [0.0, 1, 2, 3]),
    (_nested_list, lambda: [[1, "a"], [2, "b"], [3, "c"]]),
    (_list_under_tuple, lambda: ((["a"], 1), ([["b"]], 2))),
    (_shrink_and_regrow, lambda: [1, 2, 3]),
]


@pytest.mark.parametrize("change, make", CASES, ids=[c.__name__.strip("_") for c, _ in CASES])
def test_a_change_in_place_is_seen_and_made_again_on_the_warm_run(disk_cash, change, make):
    def work(rows):
        change(rows)
        return 1

    f = disk_cash.cache(work)
    rows = make()
    f(rows)
    without_cash = repr(rows)
    rows = make()
    f(rows)
    assert repr(rows) == without_cash, "the warm run skipped the change"
    outcome = next(o for k, o in disk_cash._misses.outcomes.items() if "work" in k)
    assert "in place" in (outcome.get("not_stored") or ""), outcome


def test_a_plain_list_beside_a_changed_dict_is_not_stored(disk_cash):
    """A list proved unchanged by identity does not vouch for a dict beside it."""
    runs = []

    @disk_cash.cache
    def work(rows, seen):
        runs.append(1)
        seen["n"] = seen.get("n", 0) + 1
        return len(rows)

    work([1, 2, 3], {})
    work([1, 2, 3], {})
    assert len(runs) == 2


def test_plain_data_only_read_is_stored(disk_cash):
    """Control."""
    runs = []

    @disk_cash.cache
    def work(rows, labels):
        runs.append(1)
        return sum(r[0] for r in rows) + len(labels)

    work([(i, [i]) for i in range(1000)], ("a", "b", frozenset({1})))
    work([(i, [i]) for i in range(1000)], ("a", "b", frozenset({1})))
    assert len(runs) == 1
