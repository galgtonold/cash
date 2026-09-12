"""A cached step that changes a big or frozen list in place is not stored.

Round 20 (r20s2): ``rows.sort()`` inside a cached step, on a million parsed
rows, was stored -- the mutation check re-hashes the arguments and skips any
that took over 50 ms to hash for the key -- so the warm run skipped the sort
and a later order-reading step differed from the program without cash. With a
``frozen=True`` producer it was wrong at a thousand rows: a frozen argument was
never checked at all, and a step that rewrote a field in every row made later
cached steps, keyed by the producer's identity, serve pre-rewrite results.

Plain lists and tuples are now compared by the identities of what they hold,
level by level, whatever their size.
"""
from __future__ import annotations

import time

import pytest

from cash import Cash

pytestmark = [pytest.mark.core]


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)


def _tail(rows):
    return [r[1] for r in rows[-3:]]


@pytest.mark.parametrize("n, frozen", [(300_000, False), (1_000, True)])
def test_sorting_the_argument_in_place_is_not_stored(c, n, frozen):
    @c.cache(frozen=frozen)
    def load(n):
        time.sleep(0.12)
        return [((i * 7919) % n, i) for i in range(n)]       # "file order" is i order

    @c.cache
    def top(rows):
        time.sleep(0.12)
        rows.sort(reverse=True)                              # the accident
        return rows[:3]

    rows = load(n)
    top(rows)
    without_cash = _tail(rows)                                # what the program sees
    rows = load(n)
    top(rows)
    assert _tail(rows) == without_cash, "the warm run skipped the in-place sort"
    outcome = next(o for k, o in c._store_outcomes.items() if "top" in k)
    assert "in place" in (outcome.get("not_stored") or ""), outcome


def test_rewriting_a_field_of_a_frozen_result_invalidates_its_later_consumers(c):
    @c.cache(frozen=True)
    def load(n):
        time.sleep(0.12)
        return [[i, "/api/items" if i % 2 else "/login"] for i in range(n)]

    @c.cache
    def per_path(rows):
        time.sleep(0.12)
        out: dict = {}
        for r in rows:
            out[r[1]] = out.get(r[1], 0) + 1
        return out

    @c.cache
    def normalise(rows):
        time.sleep(0.12)
        for r in rows:
            r[1] = r[1].removeprefix("/api")                 # a field of every row
        return len(rows)

    rows = load(100)
    assert per_path(rows) == {"/login": 50, "/api/items": 50}
    rows = load(100)
    normalise(rows)                                           # the step the user adds
    assert per_path(rows) == {"/login": 50, "/items": 50}, \
        "served the counts of the rows before the rewrite"


def _sorts(rows):
    rows.sort()
    return rows[:1]


def _rewrites(rows):
    for i, r in enumerate(rows):
        r[0] = i
    return len(rows)


@pytest.mark.parametrize("fn, says", [
    (_sorts, "changes the argument 'rows' in place"),
    (_rewrites, "changes an element of the argument 'rows' in place"),
])
def test_the_static_finding_says_the_argument_is_changed(c, fn, says):
    """Round 20 (r20s2 F14): both read as the label a local list's `.sort()`
    gets ("write method", "subscript mutation"), the same as the false alarms
    beside them, so the one that mattered was not read."""
    import warnings
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        c.cache(fn)([[2], [1]])
    text = "\n".join(str(w.message) for w in rec)
    assert says in text, text
    assert "return a modified copy" in text


def test_a_big_list_that_is_only_read_still_stores(c):
    """Control: identity is compared, not content -- an untouched argument of
    any size keeps caching."""
    runs = []

    @c.cache
    def total(rows):
        runs.append(1)
        time.sleep(0.12)
        return sum(r[0] for r in rows)

    rows = [(i, str(i)) for i in range(300_000)]
    total(rows)
    total(rows)
    assert len(runs) == 1
