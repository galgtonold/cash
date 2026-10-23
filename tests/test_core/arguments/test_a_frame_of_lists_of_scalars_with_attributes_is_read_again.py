"""A frame of lists keeps its digest only while no item can carry a change.

The memo of a frame of lists (``agg(list)``) checked that each list held
the same objects, taken as immutable. A ``pd.Timestamp`` takes attributes,
and a subclass of ``int`` or ``datetime`` holds them too, and the key reads
them: after ``t.tag = 2`` a cached function given the frame was served what
it gave for ``t.tag = 1``. Such a list is read every time, and a list of
dates is trusted only while none of them has an attribute set.
"""

from __future__ import annotations

import datetime
import decimal

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from cash.content_hashers import builtin_hash
from cash.decorator.arg_hashing import cow_frame_digest


class Int(int):
    pass


class Float(float):
    pass


class Str(str):
    pass


class Dec(decimal.Decimal):
    pass


class Stamp(pd.Timestamp):
    pass


class When(datetime.datetime):
    pass


def _sub(kind):
    """A *kind* object with an attribute, in a frame of lists."""
    made = {
        "timestamp": lambda: pd.Timestamp("2021-01-01").isoformat(),
        "int": lambda: Int(3),
        "float": lambda: Float(1.5),
        "str": lambda: Str("a"),
        "decimal": lambda: Dec("1.5"),
        "datetime": lambda: When(2021, 1, 1),
    }[kind]()
    if kind == "timestamp":
        made = pd.Timestamp(made)
    made.tag = 1
    return made


KINDS = ["timestamp", "int", "float", "str", "decimal", "datetime"]


def _frame(item):
    dates = list(pd.date_range("2020-01-01", periods=50))
    return pd.DataFrame({"o": [dates + [item], dates[:3]]})


@pytest.mark.parametrize("kind", KINDS)
def test_an_attribute_set_on_an_item_changes_the_digest(kind):
    item = _sub(kind)
    frame = _frame(item)
    first = cow_frame_digest(frame)
    item.tag = 2
    assert cow_frame_digest(frame) != first
    assert cow_frame_digest(frame) == builtin_hash(frame)


def test_an_attribute_set_later_on_a_plain_timestamp_changes_the_digest():
    item = pd.Timestamp("2021-01-01T00:00:01")
    frame = _frame(item)
    first = cow_frame_digest(frame)
    assert cow_frame_digest(frame) == first
    item.tag = 1
    assert cow_frame_digest(frame) != first
    assert cow_frame_digest(frame) == builtin_hash(frame)


@pytest.mark.parametrize("kind", ["timestamp", "int"])
def test_a_cached_function_sees_the_attribute_change(kind, disk_cash):
    item = _sub(kind)
    frame = _frame(item)

    @disk_cash.cache
    def tags(f):
        return [getattr(x, "tag", None) for x in f["o"].iloc[0]]

    assert tags(frame)[-1] == 1
    item.tag = 2
    assert tags(frame)[-1] == 2
