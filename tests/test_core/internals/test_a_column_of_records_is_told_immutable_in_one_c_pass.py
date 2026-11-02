"""Whether a column's cells are immutable all the way down
(`_plain_data.immutable_below`, the RAM tier's question before it shares a
frame's object column with its copy) is asked of the C kernel in one depth
first pass, and answers exactly as the Python read of each level does.

The Python read made a list of every level: for a frame's two million
``(action, time)`` records, 0.9 s a stored statement (student_8).
"""

from __future__ import annotations

import datetime
import decimal
import random

import pytest

from cash import _native, _plain_data

pytestmark = pytest.mark.skipif(_native.kernels is None, reason=f"no C walk kernels: {_native.REASON}")


def _nested(depth: int, leaf=1):
    value = leaf
    for _ in range(depth):
        value = (value,)
    return value


CASES = [
    [],
    (),
    [1, "a", (2, (3, None))],
    [("Action_1", datetime.datetime(2020, 1, 1)), ("Action_2", datetime.datetime(2020, 1, 2))],
    [1.5, b"x", 1j, decimal.Decimal(1), datetime.date(2020, 1, 1), datetime.time(1), datetime.timedelta(2)],
    ((1, 2), (3, (4, (5,)))),
    [[1]],
    [(1, [2])],
    [bytearray(b"x")],
    [(1, bytearray(b"x"))],
    [object()],
    [{1: 2}],
    [(1, {2})],
    [True, None, (False,)],
    "abc",
    {1: 2},
    *([_nested(depth)] for depth in range(13, 19)),
    *(_nested(depth) for depth in range(13, 19)),
]


def _python_answer(monkeypatch, value) -> bool:
    with monkeypatch.context() as patch:
        patch.setattr(_plain_data, "_VALUES_IN_TUPLES", None)
        return _plain_data.immutable_below(value)


@pytest.mark.parametrize("value", CASES, ids=[f"case{n}" for n in range(len(CASES))])
def test_the_c_answer_is_the_python_answer(monkeypatch, value):
    assert _plain_data.immutable_below(value) == _python_answer(monkeypatch, value)


def test_random_records_are_answered_alike(monkeypatch):
    rng = random.Random(7)
    leaves = [1, "s", 2.5, None, b"b", datetime.datetime(2021, 5, 1), bytearray(b"w"), [3], {4: 5}]
    for _ in range(300):
        items = [
            tuple(rng.choice(leaves[: rng.choice((6, 9))]) for _ in range(rng.randint(0, 3)))
            for _ in range(rng.randint(0, 6))
        ]
        if rng.random() < 0.3:
            items.append(tuple(items))
        assert _plain_data.immutable_below(items) == _python_answer(monkeypatch, items), items


def test_a_column_of_records_is_read_in_one_call(monkeypatch):
    calls = []
    kernel = _plain_data._VALUES_IN_TUPLES
    monkeypatch.setattr(_plain_data, "_VALUES_IN_TUPLES", lambda *a: calls.append(len(a[0])) or kernel(*a))
    t0 = datetime.datetime(2020, 1, 1)
    records = [(f"a{i % 9}", t0 + datetime.timedelta(seconds=i)) for i in range(10_000)]
    assert _plain_data.immutable_below(records)
    assert calls == [10_000]
