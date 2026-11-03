"""``shape_witness`` (``cwalk.c``) fingerprints what a list holds for the
identity check's memo (`identity_holders._SHAPE_READS`): equal for the very
same objects nested the same way, whatever list holds them, and different
when any of them, at any depth, is another object, another type, another
length, or a str or int of another value. It holds nothing it read."""

from __future__ import annotations

import datetime
import sys

import pytest

from cash import _native

pytestmark = pytest.mark.skipif(_native.kernels is None, reason=f"no C walk kernels: {_native.REASON}")

W = getattr(_native.kernels, "shape_witness", None)
LEVELS = 8


def _sessions() -> list:
    t0 = datetime.datetime(2020, 1, 1)
    return [[(f"a{a}", t0 + datetime.timedelta(seconds=s + a)) for a in range(4)] for s in range(50)]


def test_the_same_objects_give_the_same_witness():
    cells = _sessions()
    first = W(cells, LEVELS)
    assert type(first) is tuple and len(first) == 2 and all(type(lane) is int for lane in first)
    assert W(cells, LEVELS) == first
    # Another list holding the very same sessions: the content is the same.
    assert W(list(cells), LEVELS) == first


@pytest.mark.parametrize(
    "change",
    [
        lambda cells: cells[7].append(("x", 1)),
        lambda cells: cells[7].pop(),
        lambda cells: cells.__setitem__(3, list(cells[3])),
        lambda cells: cells[3].__setitem__(1, tuple(cells[3][1])[:1] + (cells[3][1][1],)),
        lambda cells: cells[3].__setitem__(1, ("".join(["a", "1"]), cells[3][1][1])),
        lambda cells: cells[3].__setitem__(1, ("a1", 12345678)),
        lambda cells: cells.reverse(),
    ],
    ids=[
        "a record added",
        "a record removed",
        "a session copied",
        "a record copied",
        "an equal text built apart",
        "a time made a number",
        "the order",
    ],
)
def test_any_change_gives_another_witness(change):
    cells = _sessions()
    before = W(cells, LEVELS)
    change(cells)
    assert W(cells, LEVELS) != before


def test_ints_and_texts_are_told_by_value():
    assert W([10_000_000, "0x7f00aa12bc"], LEVELS) != W([10_000_008, "0x7f00aa12bc"], LEVELS)
    big = [1 << 70, -(1 << 70)]
    assert W(big, LEVELS) == W(big, LEVELS)


def test_nesting_past_the_levels_gives_none():
    deep: list = [1]
    for _ in range(LEVELS):
        deep = [deep]
    assert W([deep], LEVELS) is None
    assert W([[[1]]], 3) is not None
    assert W([[[[1]]]], 3) is None


def test_it_keeps_no_reference():
    cells = _sessions()
    counts = [sys.getrefcount(cell) for cell in cells]
    for _ in range(3):
        W(cells, LEVELS)
    assert [sys.getrefcount(cell) for cell in cells] == counts
