"""A cached step that changes a big or frozen list in place is not stored.

``rows.sort()`` inside a cached step, on a million parsed
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

pytestmark = [pytest.mark.core]


def _tail(rows):
    return [r[1] for r in rows[-3:]]


@pytest.mark.parametrize("n, frozen", [(300_000, False), (1_000, True)])
def test_sorting_the_argument_in_place_is_not_stored(disk_cash, n, frozen):
    @disk_cash.cache(frozen=frozen)
    def load(n):
        time.sleep(0.12)
        return [((i * 7919) % n, i) for i in range(n)]  # "file order" is i order

    @disk_cash.cache
    def top(rows):
        time.sleep(0.12)
        rows.sort(reverse=True)  # the accident
        return rows[:3]

    rows = load(n)
    top(rows)
    without_cash = _tail(rows)  # what the program sees
    rows = load(n)
    top(rows)
    assert _tail(rows) == without_cash, "the warm run skipped the in-place sort"
    outcome = next(o for k, o in disk_cash._misses.outcomes.items() if "top" in k)
    assert "in place" in (outcome.get("not_stored") or ""), outcome


def test_rewriting_a_field_of_a_frozen_result_invalidates_its_later_consumers(disk_cash):
    @disk_cash.cache(frozen=True)
    def load(n):
        time.sleep(0.12)
        return [[i, "/api/items" if i % 2 else "/login"] for i in range(n)]

    @disk_cash.cache
    def per_path(rows):
        time.sleep(0.12)
        out: dict = {}
        for r in rows:
            out[r[1]] = out.get(r[1], 0) + 1
        return out

    @disk_cash.cache
    def normalise(rows):
        time.sleep(0.12)
        for r in rows:
            r[1] = r[1].removeprefix("/api")  # a field of every row
        return len(rows)

    rows = load(100)
    assert per_path(rows) == {"/login": 50, "/api/items": 50}
    rows = load(100)
    normalise(rows)  # the step the user adds
    assert per_path(rows) == {"/login": 50, "/items": 50}, "served the counts of the rows before the rewrite"


def _sorts(rows):
    rows.sort()
    return rows[:1]


def _rewrites(rows):
    for i, r in enumerate(rows):
        r[0] = i
    return len(rows)


@pytest.mark.parametrize(
    "fn, says",
    [
        (_sorts, "changes the argument 'rows' in place"),
        (_rewrites, "changes an element of the argument 'rows' in place"),
    ],
)
def test_the_static_finding_says_the_argument_is_changed(disk_cash, fn, says):
    """Both read as the label a local list's `.sort()`
    gets ("write method", "subscript mutation"), the same as the false alarms
    beside them, so the one that mattered was not read."""
    import warnings

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        disk_cash.cache(fn)([[2], [1]])
    text = "\n".join(str(w.message) for w in rec)
    assert says in text, text
    assert "return a modified copy" in text


def test_a_big_list_that_is_only_read_still_stores(disk_cash):
    """Control: identity is compared, not content -- an untouched argument of
    any size keeps caching."""
    runs = []

    @disk_cash.cache
    def total(rows):
        runs.append(1)
        time.sleep(0.12)
        return sum(r[0] for r in rows)

    rows = [(i, str(i)) for i in range(300_000)]
    total(rows)
    total(rows)
    assert len(runs) == 1


def _scale_in_place(values, factor):
    import numpy as np

    np.multiply(values, factor, out=values)  # a library writing through out=


def test_a_big_array_changed_inside_a_library_is_not_stored(disk_cash, monkeypatch):
    """``normalise(values)`` on an array whose key hash took over 50 ms
    retired the argument check for the function: a library writing into the
    array was stored, and the warm run returned without scaling it. The
    budget is set to nothing so a small array stands in for a big one."""
    import numpy as np

    from cash.decorator import purity_checks

    monkeypatch.setattr(purity_checks, "MUTATION_CHECK_BUDGET_S", 0.0)

    @disk_cash.cache
    def normalise(values):
        time.sleep(0.12)
        _scale_in_place(values, 2.0)
        return float(values.sum())

    first = np.arange(10, dtype=float)
    normalise(first)
    second = np.arange(10, dtype=float)
    normalise(second)
    assert second.tolist() == first.tolist(), "the warm run skipped the in-place scale"


def test_a_big_array_only_read_still_stores(disk_cash, monkeypatch):
    """Control: an array the call leaves alone is stored however costly."""
    import numpy as np

    from cash.decorator import purity_checks

    monkeypatch.setattr(purity_checks, "MUTATION_CHECK_BUDGET_S", 0.0)
    runs = []

    @disk_cash.cache
    def total(values):
        runs.append(1)
        time.sleep(0.12)
        return float(values.sum())

    total(np.arange(10, dtype=float))
    total(np.arange(10, dtype=float))
    assert len(runs) == 1
