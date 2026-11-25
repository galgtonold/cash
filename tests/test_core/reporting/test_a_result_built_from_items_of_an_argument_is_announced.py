"""A result built from the items of a list or dict argument says that a
hit will stop sharing them with the argument.

``[r for r in records if ...]`` hands back the caller's own dicts in plain
Python, so marking them (``r["flagged"] = True``) changes ``records``. The
first cached call does that too; every hit returns copies, and the
caller's writes go nowhere. The same result built from a global list
warned CACHE-RESULT-SHARED; one built from an argument's items said
nothing.

The look costs nothing for a result made of new objects (no item of the
arguments is compared), every item of the arguments is compared when the
result holds an object something else holds too, and an argument too big
to search to its end ends in the warning, not in silence.
"""

from __future__ import annotations

import datetime
import warnings

import pytest

from cash.decorator import purity_checks


def _said(fn, *args):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        fn(*args)
    return [str(w.message) for w in seen if "CACHE-RESULT-SHARED" in str(w.message)]


def records():
    return [{"k": 1}, {"k": 2}, {"k": 1}]


def test_the_callers_writes_stop_reaching_the_argument_on_a_hit_and_cash_says_so(cash_instance):
    """The whole scenario: what a plain function does, what the cached one
    does, and the warning that stands for the difference."""

    def select_plain(rows, k):
        return [r for r in rows if r["k"] == k]

    select_cached = cash_instance.cache(select_plain)

    def flag_matching(select, rows):
        for r in select(rows, 1):
            r["flagged"] = True
        return sum(1 for r in rows if r.get("flagged"))

    assert [flag_matching(select_plain, records()) for _ in range(3)] == [2, 2, 2]
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        cached = [flag_matching(select_cached, records()) for _ in range(3)]
    said = [str(w.message) for w in seen if "CACHE-RESULT-SHARED" in str(w.message)]
    assert cached == [2, 0, 0], "a hit hands back copies: this is what the warning is about"
    assert said and "holds an item of the argument 'rows[0]'" in said[0], said


@pytest.mark.parametrize(
    ("body", "argument", "label"),
    [
        (lambda rows: [r for r in rows if r["k"] == 2], records, "holds an item of the argument 'rows[1]'"),
        (lambda rows: rows[0], records, "is an item of the argument 'rows[0]'"),
        (lambda rows: rows[-2:], records, "holds an item of the argument 'rows[1]'"),
        (lambda cfg: cfg["items"], lambda: {"n": 1, "items": [1, 2]}, "is an item of the argument 'cfg['items']'"),
        (lambda cfg: cfg["a"]["b"], lambda: {"a": {"b": [1]}}, "is an item of the argument 'cfg['a']['b']'"),
        (lambda cfg: {"picked": cfg["a"]["b"]}, lambda: {"a": {"b": [1]}}, "holds an item of the argument"),
        (lambda pair: pair[1], lambda: ({"a": 1}, {"b": 2}), "is an item of the argument 'pair[1]'"),
        (lambda rows: (len(rows), rows[1]), records, "holds an item of the argument 'rows[1]'"),
    ],
    ids=["a filter", "rows[0]", "rows[-2:]", "cfg[key]", "cfg[a][b]", "nested, held", "a tuple's item", "in a tuple"],
)
def test_each_shape_of_a_result_made_of_an_arguments_items_warns(cash_instance, body, argument, label):
    said = _said(cash_instance.cache(body), argument())
    assert said, "no warning for a result that shares an item of its argument"
    assert label in said[0], said[0]


def test_an_item_far_into_a_long_argument_is_found(cash_instance):
    @cash_instance.cache
    def last(rows):
        return rows[-1]

    said = _said(last, [{"i": i} for i in range(5000)])
    assert said and "rows[4999]" in said[0], said


@pytest.mark.parametrize(
    ("body", "argument"),
    [
        (lambda rows: [dict(r) for r in rows], records),
        (lambda rows: [{"k": r["k"]} for r in rows if r["k"] == 1], records),
        (lambda rows: {"n": len(rows), "ks": sorted({r["k"] for r in rows})}, records),
        (lambda rows: [r["day"] for r in rows], lambda: [{"day": datetime.date(2024, 1, d)} for d in (1, 2)]),
        (lambda rows: [r["at"] for r in rows], lambda: [{"at": (1, "a")}, {"at": (2, "b")}]),
        (lambda rows: rows[0]["name"], lambda: [{"name": "x" * 50}]),
    ],
    ids=["copies", "new dicts", "a summary", "dates passed through", "tuples of values", "a string"],
)
def test_a_result_of_new_objects_or_of_values_nothing_writes_through_says_nothing(cash_instance, body, argument):
    assert not _said(cash_instance.cache(body), argument())


def test_assume_safe_waives_it(cash_instance):
    @cash_instance.cache(assume_safe=True)
    def first(rows):
        return rows[0]

    assert not _said(first, records())


class TestTheCostOfLooking:
    """Counted in items of the arguments compared, not timed."""

    @pytest.fixture
    def compared(self, monkeypatch):
        """The containers of the arguments the search looked through."""
        looked: list[int] = []
        real = purity_checks._keyed

        def keyed(box):
            looked.append(len(box))
            return real(box)

        monkeypatch.setattr(purity_checks, "_keyed", keyed)
        held: list = []
        real_held = purity_checks._held_elsewhere

        def held_elsewhere(result, was_held):
            found = real_held(result, was_held)
            held.append(len(found))
            return found

        monkeypatch.setattr(purity_checks, "_held_elsewhere", held_elsewhere)
        return looked, held

    def test_a_result_of_new_objects_compares_no_item_of_a_big_argument(self, cash_instance, compared):
        looked, held = compared
        rows = [{"k": i % 7, "tags": [i]} for i in range(20_000)]

        @cash_instance.cache
        def summary(rows):
            return [{"k": r["k"], "n": len(r["tags"])} for r in rows[:300]]

        assert not _said(summary, rows)
        assert held == [0], "nothing in the result is held by anything else"
        assert looked == []

    def test_a_shared_item_at_the_front_ends_the_search_there(self, cash_instance, compared):
        looked, _held = compared
        rows = [{"k": i % 7, "tags": [i]} for i in range(20_000)]

        @cash_instance.cache
        def head(rows):
            return rows[:3]

        assert _said(head, rows)
        assert looked == [20_000], "the argument's own items once, and no record's"

    def test_an_argument_too_big_to_search_ends_in_the_warning(self, cash_instance, monkeypatch):
        monkeypatch.setattr(purity_checks, "SHARED_ARGUMENT_ITEMS", 1000)
        rows = [{"k": i, "tags": [i]} for i in range(600)]

        @cash_instance.cache
        def deepest(rows):
            return rows[-1]["tags"]  # 600 + 600 * 2 items in before it is reached

        said = _said(deepest, rows)
        assert said, "the search gave up without a word"
        assert "possibly an item of the argument 'rows'" in said[0], said[0]

    def test_within_the_bound_the_same_item_is_named(self, cash_instance):
        rows = [{"k": i, "tags": [i]} for i in range(600)]

        @cash_instance.cache
        def deepest(rows):
            return rows[-1]["tags"]

        said = _said(deepest, rows)
        assert said and "is an item of the argument 'rows[599]['tags']'" in said[0], said
