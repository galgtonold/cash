"""The warning for code that can clear a helper memo names the memo the
statement cleared (its counters went back or it dropped a key), else one
whose counters moved, and none when none moved: never just the first memo
cash watches.

E16's ``held_karp_tsp`` clears and refills ``shortest_segment``; the
statement calling it was refused as "It may empty the helper memo
random_cities", a memo it never touched.
"""

from __future__ import annotations

import functools

import pytest

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


def _three():
    @functools.cache
    def first(n):
        return n

    @functools.cache
    def second(n):
        return n * 2

    @functools.cache
    def third(n):
        return n + 1

    for memo in (first, second, third):
        memo(1)
    return {"first": first, "second": second, "third": third}


def test_the_memo_cleared_is_named_over_one_only_filled_or_hit():
    for touch in ("fill", "hit"):
        ns = _three()
        before = helper_memos.watched(ns)
        ns["second"](5 if touch == "fill" else 1)
        ns["third"].cache_clear()
        ns["third"](7)
        verdict = helper_memos.used(before, clears=True)
        assert isinstance(verdict, tuple), verdict
        assert "helper memo third " in verdict[1], (touch, verdict[1])


def test_no_memo_is_named_when_none_moved():
    """Reading the counters only (``cache_info``), or a clear and refill
    that leaves the same counts and the same small-int key and answer:
    nothing tells which memo, so none is named."""
    ns = _three()
    before = helper_memos.watched(ns)
    ns["third"].cache_info()
    verdict = helper_memos.used(before, clears=True)
    assert isinstance(verdict, tuple), verdict
    assert "It may empty a helper memo or read" in verdict[1], verdict[1]
    assert "first" not in verdict[1], verdict[1]


@pytest.mark.parametrize(
    "code",
    [
        "second(1); third.cache_clear(); third(1); third(2)",
        "first(1); third.cache_clear(); third(1); third(1)",
    ],
)
def test_the_memo_whose_clear_the_code_names_is_named_when_it_only_grew(code):
    """Cleared and filled again with the same keys and more, the memo's
    counters only grew, as a memo only hit did: the one whose
    ``cache_clear`` the code names is named."""
    ns = _three()
    before = helper_memos.watched(ns)
    exec(code, ns)
    verdict = helper_memos.used(before, clears=True, ns=ns, site=(code, None))
    assert isinstance(verdict, tuple), verdict
    assert "helper memo third " in verdict[1], verdict[1]


def test_no_memo_is_named_when_several_only_grew_and_none_is_named():
    ns = _three()
    before = helper_memos.watched(ns)
    ns["first"](2)
    ns["second"](2)
    verdict = helper_memos.used(before, clears=True)
    assert isinstance(verdict, tuple), verdict
    assert "It may empty a helper memo or read" in verdict[1], verdict[1]
