"""Records of a plain class, and an object holding many records, are keyed by
their fields at C speed.

A list of a million small objects or dataclasses cost about 5 us an object
per hit: each was asked of the hook, left to pickle whole and pickled with a
memo entry per object. A method whose ``self.data`` held 100,000 records was
keyed by a search of all of them for a set and a pickle of the object whole,
0.77 s a call. Records whose ``__dict__`` holds values alone are now keyed
by their class and one pickle of those dicts; an object whose ``__dict__``
holds JSON-like data by one pickle of it (`canonical_form._fields_form`).

The key must still tell apart what it told apart: equal fields hit,
different ones miss, and one object held twice -- in the list, in another
argument, inside another object -- keys apart from two equal copies. These
tests pin that, and compare the key's verdicts with the way it was made
before (`fields_class` off).
"""

from __future__ import annotations

import copy
import copyreg
import dataclasses
import datetime
import decimal
import random

import pytest

from cash import Cash, canonical_form
from cash.content_hashers import BUILTIN_CONTENT

pytestmark = [pytest.mark.core]


class Point:
    def __init__(self, x, y, name):
        self.x, self.y, self.name = x, y, name


@dataclasses.dataclass
class DPoint:
    x: int
    y: float
    name: str


class When:
    def __init__(self, i, at):
        self.i, self.at = i, at


class Reduced(Point):
    def __reduce__(self):
        return (Point, (self.x, 0, ""))


class Stated(Point):
    def __getstate__(self):
        return {"x": self.x}


class Keyed(Point):
    def __cash_key__(self):
        return self.x


@dataclasses.dataclass
class Slotted:
    __slots__ = ("x",)
    x: int


class Registered(Point):
    pass


copyreg.pickle(Registered, lambda p: (Registered, (p.x, 0, "")))


class Model:
    def __init__(self, rows, name="m"):
        self.data = rows
        self.name = name


class Scorer(Model):
    def score(self, k):
        return sum(r["id"] for r in self.data) + k


def _points(n=50, cls=Point):
    return [cls(i, i * 2.0, f"p{i}") for i in range(n)]


def _rows(n=50):
    return [{"id": i, "tags": ["a", i]} for i in range(n)]


def _key(value) -> bytes:
    return canonical_form.canonical_bytes(((value,), {}), BUILTIN_CONTENT)


def _keys(*values) -> list[bytes]:
    return [canonical_form.canonical_bytes((tuple(v), {}), BUILTIN_CONTENT) for v in values]


@pytest.fixture
def old_way(monkeypatch):
    """Keys made as before: no class is keyed by its fields."""

    def off():
        monkeypatch.setattr(canonical_form, "fields_class", lambda t: False)

    return off


def _form(value):
    left: list = []
    form = canonical_form.stable_key_repr(value, BUILTIN_CONTENT, left=left, fielded=[])
    return form, left


# -- the work ------------------------------------------------------------------


@pytest.mark.parametrize("cls", [Point, DPoint])
def test_records_are_one_digest_not_an_object_each(cls):
    form, left = _form(_points(5_000, cls))

    assert form[2][0][:3] == ("__cash_fields__", "records", f"{cls.__module__}.{cls.__qualname__}")
    assert left == []
    assert len(_key(_points(5_000, cls))) < 400, "the key pickled the records"


def test_an_object_holding_many_records_is_one_digest():
    form, left = _form(Model(_rows(5_000)))

    assert form[:3] == ("__cash_fields__", "object", f"{Model.__module__}.{Model.__qualname__}")
    assert left == []
    assert len(_key(Model(_rows(5_000)))) < 400, "the key pickled the object"


def test_a_warm_hit_does_not_pickle_each_record(tmp_path, monkeypatch):
    c = Cash(cache_dir=str(tmp_path / "cache"))

    @c.cache
    def total(rows):
        return sum(p.y for p in rows)

    rows = _points(5_000)
    total(rows)
    states = []
    real = canonical_form._pickled_state
    monkeypatch.setattr(canonical_form, "_pickled_state", lambda v: states.append(v) or real(v))
    assert total(rows) == sum(p.y for p in rows)
    assert total.cache_info()["hits"] == 1
    assert states == []


def test_a_method_on_an_object_holding_records_is_not_walked_per_record(tmp_path, monkeypatch):
    """``self.data`` of records: neither the key nor the search for code an
    argument carries reads them one dict at a time."""
    c = Cash(cache_dir=str(tmp_path / "cache"))
    score = c.cache(Scorer.score)
    m = Scorer(_rows(5_000))
    score(m, 1)
    calls = {"carriers": 0, "contains_set": 0}
    real_walk = c._code_args._walk_carriers
    monkeypatch.setattr(
        c._code_args,
        "_walk_carriers",
        lambda *a, **k: calls.__setitem__("carriers", calls["carriers"] + 1) or real_walk(*a, **k),
    )
    real_set = canonical_form.contains_set
    monkeypatch.setattr(
        canonical_form,
        "contains_set",
        lambda *a, **k: calls.__setitem__("contains_set", calls["contains_set"] + 1) or real_set(*a, **k),
    )
    assert score(m, 1) == sum(range(5_000)) + 1
    assert score.cache_info()["hits"] == 1
    assert calls["carriers"] < 50, f"walked the records for code: {calls}"
    assert calls["contains_set"] == 0, "searched the object for a set"


# -- what is keyed apart -------------------------------------------------------


@pytest.mark.parametrize("cls", [Point, DPoint])
def test_equal_fields_key_equal_and_a_changed_field_keys_apart(cls):
    a, b = _points(50, cls), _points(50, cls)
    assert _key(a) == _key(b)
    b[30].name = "changed"
    assert _key(a) != _key(b)
    b[30].name = "p30"
    b[30].extra = 1  # one more field
    assert _key(a) != _key(b)
    assert _key(_points(50, cls)) != _key(tuple(_points(50, cls)))
    assert _key(_points(50, Point)) != _key(_points(50, DPoint))


def test_a_field_s_type_keys_apart():
    a, b = _points(50), _points(50)
    b[3].x = 3.0
    assert _key(a) != _key(b)
    b[3].x = True if a[3].x == 1 else 3
    assert _key(a) == _key(b)


@pytest.mark.parametrize(
    "shape",
    ["twice in the list", "also another argument", "inside an object left to pickle", "in two lists"],
)
def test_one_record_held_twice_keys_apart_from_two_copies(shape):
    def build(same: bool):
        rows = _points(20)
        other = rows[4] if same else copy.copy(rows[4])
        if shape == "twice in the list":
            return [rows + [other]]
        if shape == "also another argument":
            return [rows, other]
        if shape == "inside an object left to pickle":
            return [rows, {"held": [other, {1, 2}]}]
        return [rows, [other] + _points(10)]

    shared, copied = _keys(build(True), build(False))
    assert shared != copied


def test_two_objects_sharing_their_records_key_apart_from_two_copies():
    rows = _rows(50)
    shared, copied = _keys([Model(rows), Model(rows)], [Model(rows), Model(copy.deepcopy(rows))])
    assert shared != copied


def test_records_held_twice_inside_one_object_key_apart_from_copies():
    row = {"id": 1}
    twice = Model([row, row] + _rows(20))
    copies = Model([row, dict(row)] + _rows(20))
    assert _key(twice) != _key(copies)


def test_an_object_s_records_also_walked_elsewhere_key_apart():
    rows = _rows(50)
    shared, copied = _keys([Model(rows), [rows, {1}]], [Model(rows), [copy.deepcopy(rows), {1}]])
    assert shared != copied


def test_a_mutated_record_inside_an_object_keys_apart():
    m = Model(_rows(50))
    before = _key(m)
    m.data[7]["tags"].append("x")
    assert _key(m) != before
    m.data[7]["tags"].pop()
    assert _key(m) == before


# -- what is not keyed by its fields ------------------------------------------


@pytest.mark.parametrize("cls", [Reduced, Stated, Keyed, Registered])
def test_a_class_that_says_how_it_pickles_is_not_keyed_by_its_fields(cls):
    assert not canonical_form.fields_class(cls)
    form, _ = _form(_points(20, cls))
    assert "__cash_fields__" not in repr(form)[:200]


def test_slots_and_metaclasses_are_not_keyed_by_their_fields():
    class Meta(type):
        pass

    class WithMeta(metaclass=Meta):
        pass

    assert not canonical_form.fields_class(Slotted)
    assert not canonical_form.fields_class(WithMeta)
    assert not canonical_form.fields_class(int)
    assert canonical_form.fields_class(Point) and canonical_form.fields_class(DPoint)


def test_a_class_pickle_cannot_find_by_name_is_not_keyed_by_its_fields():
    class Local:
        def __init__(self, i):
            self.i = i

    rows = [Local(i) for i in range(20)]
    form, left = _form(rows)
    assert "__cash_fields__" not in repr(form)[:200]
    assert left, "left to pickle, as before"


def test_a_redefined_class_s_records_are_keyed_as_before():
    old = _points(20)
    global Point
    first = Point

    class Point:  # noqa: F811 - a cell defining the class again
        def __init__(self, x, y, name):
            self.x, self.y, self.name = x, y, name

    Point.__qualname__ = first.__qualname__
    try:
        form, _ = _form(old)
        assert "__cash_fields__" not in repr(form)[:200]
    finally:
        Point = first


def test_a_hook_still_keys_every_record():
    def hook(value):
        return ("hooked", value.x) if type(value) is Point else canonical_form.NOT_HOOKED

    a, b = _points(20), _points(20)
    b[5].name = "changed"
    keys = [canonical_form.canonical_bytes(((v,), {}), BUILTIN_CONTENT, hook=hook) for v in (a, b)]
    assert keys[0] == keys[1], "the hook says only x identifies a point"


@pytest.mark.parametrize(
    "value",
    [
        datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc),
        bytearray(b"x"),
        [1],
        {"a": 1},
        frozenset({1}),
    ],
    ids=["aware datetime", "bytearray", "list", "dict", "frozenset"],
)
def test_records_with_more_than_values_are_keyed_as_before(value):
    rows = [When(i, value) for i in range(20)]
    form, _ = _form(rows)
    assert "__cash_fields__" not in repr(form)[:200]


def test_naive_dates_and_decimals_are_values():
    rows = [When(i, datetime.datetime(2024, 1, 1, i)) for i in range(20)] + [When(1, decimal.Decimal("1.5"))]
    form, _ = _form(rows)
    assert form[2][0][0] == "__cash_fields__"


def test_registered_hashers_still_key_the_records(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "cache"))
    c.register_hasher(Point, lambda p: p.x)

    @c.cache
    def names(rows):
        return [p.name for p in rows]

    a, b = _points(20), _points(20)
    b[3].name = "other"
    assert names(a)[3] == "p3"
    assert names(b)[3] == "p3", "the hasher says x alone identifies a point"


# -- the same verdicts as before ----------------------------------------------


def _random_value(rng: random.Random):
    kind = rng.choice(["points", "dpoints", "model", "mixed"])
    if kind == "points":
        return _points(rng.randint(8, 30))
    if kind == "dpoints":
        return tuple(_points(rng.randint(8, 30), DPoint))
    if kind == "model":
        return Model(_rows(rng.randint(8, 30)), name=rng.choice(["a", "b"]))
    return [_points(10), {"k": rng.randint(0, 2)}]


def _variant(value, rng: random.Random):
    """*value* again: an equal copy, or one with a part changed or shared."""
    other = copy.deepcopy(value)
    how = rng.choice(["same", "field", "share", "share-across"])
    if how == "field":
        if isinstance(other, Model):
            other.data[rng.randrange(len(other.data))]["id"] = -1
        elif isinstance(other, list) and isinstance(other[0], list):
            other[0][rng.randrange(10)].x = -1
        else:
            other[rng.randrange(len(other))].y = -1.0
    elif how == "share" and isinstance(other, list) and not isinstance(other[0], list):
        other[1] = other[0]
    elif how == "share-across":
        return [other, other if rng.random() < 0.5 else copy.deepcopy(other)]
    return [other, None]


@pytest.mark.parametrize("seed", range(150))
def test_the_key_tells_apart_what_it_told_apart_before(seed, old_way):
    rng = random.Random(seed)
    value = _random_value(rng)
    other = _variant(value, rng)
    first = [value, None] if other[1] is None else [value, value if other[0] is other[1] else copy.deepcopy(value)]
    new = _keys(first, other)
    old_way()
    old = _keys(first, other)
    assert (new[0] == new[1]) == (old[0] == old[1])
