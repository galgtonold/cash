"""The warning for code that can clear a helper memo names the memo whose
counters moved, not the first memo cash watches.

E16's ``held_karp_tsp`` clears and refills ``shortest_segment``; the
statement calling it was refused as "It may empty the helper memo
random_cities", a memo it never touched.
"""

from __future__ import annotations

import functools

from cash.notebook import helper_memos


def test_the_memo_that_moved_is_named():
    @functools.cache
    def untouched(n):
        return n

    @functools.cache
    def cleared(n):
        return n + 1

    ns = {"untouched": untouched, "cleared": cleared}
    untouched(1)
    cleared(1)
    before = helper_memos.watched(ns)
    assert [where[-1] for where, *_ in before][0] == "untouched"

    cleared.cache_clear()
    cleared(2)
    verdict = helper_memos.used(before, clears=True)

    assert isinstance(verdict, tuple), verdict
    assert "helper memo cleared " in verdict[1], verdict[1]
