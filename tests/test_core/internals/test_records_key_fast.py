"""A list of records -- dataclasses, plain objects, namedtuples -- is keyed
without walking each record.

Each record was searched for a set and for frames, and by the decorator for
code its attributes hold: 15-25 us an element, 13x the same records as
dicts. Records holding JSON-like data are now recognised for the whole list
at once, and each gets the form the walk gave it, so the key is unchanged.
"""

from __future__ import annotations

import collections
import dataclasses
import datetime
import decimal

import pytest

from cash import Cash, canonical_form
from cash.content_hashers import BUILTIN_CONTENT

pytestmark = [pytest.mark.core]


@dataclasses.dataclass
class Rec:
    id: int
    name: str
    tags: list
    extra: dict = dataclasses.field(default_factory=dict)


class Plain:
    def __init__(self, i, when=None):
        self.id = i
        self.when = when or datetime.date(2024, 1, 1)
        self.amount = decimal.Decimal("1.5")
        self.nested = {"a": [i, (i, "x")], "b": None}


NT = collections.namedtuple("NT", "id name tags")


def _forms(value, monkeypatch, **kwargs):
    """The walk's form with records recognised, and without."""
    seen_fast: dict = {}
    fast = canonical_form.stable_key_repr(value, BUILTIN_CONTENT, seen=seen_fast, left=[], **kwargs)
    monkeypatch.setattr(canonical_form, "RECORDS_FROM", 10**9)
    seen_slow: dict = {}
    slow = canonical_form.stable_key_repr(value, BUILTIN_CONTENT, seen=seen_slow, left=[], **kwargs)
    monkeypatch.undo()
    return fast, slow, seen_fast, seen_slow


def _shared_tags():
    tags = ["shared"]
    return [NT(i, f"n{i}", tags if i % 3 == 0 else [i, {"k": i}]) for i in range(20)] + [None, 3]


CASES = {
    "dataclasses": lambda: [Rec(i, f"n{i}", [i, i + 1], {"k": [i]}) for i in range(20)],
    "plain objects with dates": lambda: [Plain(i) for i in range(20)] + ["x", 1.5],
    "namedtuples": lambda: [NT(i, f"n{i}", [i, (i, i)]) for i in range(20)],
    "namedtuples sharing a list": _shared_tags,
    "a tuple of records": lambda: tuple(Rec(i, "n", []) for i in range(20)),
}


@pytest.mark.parametrize("make", list(CASES.values()), ids=list(CASES))
def test_records_keep_the_form_the_walk_gives_them(make, monkeypatch):
    value = make()
    assert canonical_form.record_class(value if isinstance(value, (list, tuple)) else [], BUILTIN_CONTENT.family)
    fast, slow, seen_fast, seen_slow = _forms(value, monkeypatch)
    assert fast == slow
    assert {k: v[0] for k, v in seen_fast.items()} == {k: v[0] for k, v in seen_slow.items()}


def test_a_hook_still_sees_every_record(monkeypatch):
    def hook(value):
        return ("hooked", value.id) if isinstance(value, Rec) else canonical_form.NOT_HOOKED

    value = [Rec(i, "n", [i]) for i in range(20)]
    fast, slow, _, _ = _forms(value, monkeypatch, hook=hook)
    assert fast == slow and fast[2][0] == ("hooked", 0)


def test_what_is_not_a_record_list_takes_the_walk():
    family = BUILTIN_CONTENT.family
    assert canonical_form.record_class([Rec(i, "n", [{1, 2}]) for i in range(20)], family) is None  # a set
    assert canonical_form.record_class([Rec(i, "n", [object()]) for i in range(20)], family) is None
    assert canonical_form.record_class([Rec(1, "n", []), Plain(1)] * 10, family) is None  # two classes
    assert canonical_form.record_class([Plain(1, datetime.datetime(2024, 1, 1))] * 10, family) is None  # tzinfo

    @dataclasses.dataclass
    class Slotted:
        __slots__ = ("x",)
        x: int

    assert canonical_form.record_class([Slotted(1)] * 10, family) is None


def test_a_warm_hit_on_many_records_does_not_walk_them(tmp_path, monkeypatch):
    """Counted, not timed: no record is searched for a set or walked for code."""
    c = Cash(cache_dir=str(tmp_path / "cache"))
    rows = [Rec(i, f"n{i}", [i, i + 1]) for i in range(5_000)]

    @c.cache
    def count(rows):
        return len(rows)

    count(rows)
    calls = {"contains_set": 0, "carriers": 0}
    real = canonical_form.contains_set
    monkeypatch.setattr(
        canonical_form,
        "contains_set",
        lambda *a, **k: calls.__setitem__("contains_set", calls["contains_set"] + 1) or real(*a, **k),
    )
    real_walk = c._code_args._walk_carriers
    monkeypatch.setattr(
        c._code_args,
        "_walk_carriers",
        lambda *a, **k: calls.__setitem__("carriers", calls["carriers"] + 1) or real_walk(*a, **k),
    )
    assert count(rows) == 5_000
    assert count.cache_info()["hits"] == 1
    assert max(calls.values()) < 50, f"walked the records: {calls}"


def test_records_key_by_content_and_class(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "cache"))

    @c.cache
    def names(rows):
        return [r.name for r in rows]

    rows = [Rec(i, f"n{i}", [i]) for i in range(20)]
    assert names(rows) == [f"n{i}" for i in range(20)]
    edited = [Rec(i, f"n{i}", [i]) for i in range(20)]
    edited[10].name = "changed"
    assert names(edited)[10] == "changed"
    nts = [NT(i, f"n{i}", [i]) for i in range(20)]
    assert names(nts) == [f"n{i}" for i in range(20)]
    other = collections.namedtuple("Other", "id name tags")
    assert names([other(*nt) for nt in nts]) == names(nts)
    assert names.cache_info()["misses"] == 4


def test_the_record_class_is_the_code_they_carry(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "cache"))
    rows = [Rec(i, "n", []) for i in range(20)]
    assert Rec in list(c._code_args.iter_code_carriers(rows))
