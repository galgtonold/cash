"""A cached function's result is the object the body built, with nothing added.

cash wrote its lineage tag (``_cash_lineage_src``, ``_producer``, ``_hash``)
into the ``__dict__`` of every result that took attributes, on the miss and
on every hit, and pickled it into the stored value: ``vars(ns)`` gained three
keys, ``SimpleNamespace(a=1) == result`` and a ``__dict__``-based ``__eq__``
turned False, and a function returning its argument tagged the caller's own
object. The tag is kept beside the value now (`cash.lineage_tag`).
"""

from __future__ import annotations

import enum
import pickle
import types

import pytest

from cash import Cash
from cash.lineage_tag import own_tag


class Point:
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return type(other) is Point and self.__dict__ == other.__dict__


class Color(enum.Enum):
    RED = 1


@pytest.fixture
def app(tmp_path):
    return Cash(cache_dir=str(tmp_path / "cache"))


def test_a_result_is_unchanged_on_miss_and_hit(app):
    @app.cache
    def make_ns(n):
        return types.SimpleNamespace(a=n)

    @app.cache
    def make_point(n):
        return Point(n)

    for _ in range(2):
        ns = make_ns(1)
        assert vars(ns) == {"a": 1}
        assert ns == types.SimpleNamespace(a=1)
        assert make_point(1) == Point(1)
        assert vars(make_point(1)) == {"x": 1}
    assert make_ns.cache_info()["hits"] >= 1


def test_the_stored_value_holds_no_tag(app):
    @app.cache
    def make_point(n):
        return Point(n)

    make_point(2)
    hit = make_point(2)
    assert vars(pickle.loads(pickle.dumps(hit))) == {"x": 2}


def test_a_returned_argument_is_left_alone(app):
    @app.cache
    def ident(o):
        return o

    mine = Point(5)
    ident(mine)
    assert vars(mine) == {"x": 5}


def test_an_enum_member_is_left_alone(app):
    @app.cache
    def pick():
        return Color.RED

    pick()
    pick()
    assert not [k for k in vars(Color.RED) if k.startswith("_cash")]


def test_the_result_still_carries_its_lineage(app):
    """Positive control: the tag exists, beside the value."""

    @app.cache
    def make_point(n):
        return Point(n)

    first = make_point(3)
    assert own_tag(first) is not None
    assert own_tag(make_point(3)) == own_tag(first)
