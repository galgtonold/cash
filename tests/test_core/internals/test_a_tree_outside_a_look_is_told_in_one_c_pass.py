"""Whether a value is JSON-like data (`_plain_data.is_tree`) is asked of the
C walk when no `one_look` is open, and answers exactly as the Python read
of each level (`tree_levels`) does.

The upstream simulation asks it of every live receiver and argument of every
statement above the cell (`receiver_is_identity_coupled`), outside any look:
`tree_levels` made a list of every level of a parsed log, 0.6 s a value
where the C walk takes 0.2 s (student_8).
"""

from __future__ import annotations

import datetime
import decimal

import pytest

from cash import _native, _plain_data

pytestmark = pytest.mark.skipif(_native.kernels is None, reason=f"no C walk kernels: {_native.REASON}")


def _nested(depth: int, leaf=1):
    value = leaf
    for _ in range(depth):
        value = [value]
    return value


class _Leaf:
    pass


CASES = [
    [],
    (),
    {},
    [1, "a", (2, (3, None))],
    [("Action_1", datetime.datetime(2020, 1, 1)), ("Action_2", datetime.datetime(2020, 1, 2))],
    [("u", [("a", datetime.datetime(2020, 1, 1))]), ("v", [])],
    [1.5, b"x", 1j, decimal.Decimal(1), datetime.date(2020, 1, 1), datetime.timedelta(2)],
    {"a": {"b": (1, 2)}, "c": [1, {"d": None}]},
    [{1: 2}],
    [{(1, 2): 3}],
    [{"a": [1]}],
    [set()],
    [bytearray(b"x")],
    [_Leaf()],
    [(1, _Leaf())],
    "abc",
    *(_nested(depth) for depth in range(13, 19)),
]

LEAVES = [None, (int, str), (int, str, tuple), (int, str, float, type(None), _Leaf)]


def _python_answer(monkeypatch, value, leaves) -> bool:
    with monkeypatch.context() as patch:
        patch.setattr(_plain_data, "_K", None)
        return _plain_data.is_tree(value, leaves)


@pytest.mark.parametrize("leaves", LEAVES, ids=[f"leaves{n}" for n in range(len(LEAVES))])
@pytest.mark.parametrize("value", CASES, ids=[f"case{n}" for n in range(len(CASES))])
def test_the_c_answer_is_the_python_answer(monkeypatch, value, leaves):
    assert _plain_data.is_tree(value, leaves) == _python_answer(monkeypatch, value, leaves)


def test_a_tree_outside_a_look_is_not_read_a_level_at_a_time_in_python(monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("tree_levels read the value in Python")

    monkeypatch.setattr(_plain_data, "tree_levels", refused)
    t0 = datetime.datetime(2020, 1, 1)
    records = [(f"u{n}", [(f"a{k}", t0) for k in range(5)]) for n in range(100)]
    assert _plain_data.is_tree(records)
    assert _plain_data.is_tree(records, (str, datetime.datetime))
    assert not _plain_data.is_tree([*records, _Leaf()])
