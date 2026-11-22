"""A library function's plain return costs no Python call while a
statement's lookups are watched (`lookups_seen`).

A library callee armed for PY_RETURN is never switched off, since the next
value it returns may stand by reference (``random.choice`` over a list that
holds a logger). In a simulation it returns millions of ints
(``random.randint`` in the pytudes notebook, about 2.8 s of a 66 s run). The C
extension's filter answers a value of exactly one of `lookups_seen._PLAIN`'s
classes itself, and hands every other value, a subclass of a plain class
included, to the Python callback with the same arguments.
"""

from __future__ import annotations

import logging
import sys

import pytest

from cash._native import cwatch
from cash.tracking import lookups_seen

needs_filter = pytest.mark.skipif(cwatch is None, reason="the C extension is not loaded")

#: One value of each class `_PLAIN` names.
SAMPLES = {
    type(None): None,
    bool: True,
    int: 7,
    float: 1.5,
    complex: 2j,
    str: "s",
    bytes: b"b",
    tuple: (1,),
    list: [1],
    dict: {1: 2},
    set: {1},
    frozenset: frozenset({1}),
}


class MyInt(int):
    def __reduce__(self):
        return "LOGGER"


class MyList(list):
    pass


@needs_filter
def test_the_filter_answers_each_plain_class_itself():
    assert set(SAMPLES) == set(lookups_seen._PLAIN)
    slow = []
    callback = cwatch.plain_return_filter(lambda *args: slow.append(args) or "slow")
    for value in SAMPLES.values():
        assert callback(None, 0, value) is None
    assert slow == []


@needs_filter
@pytest.mark.parametrize(
    "value",
    [MyInt(3), MyList([1]), logging.getLogger("cash_test_plain_return"), object(), sys],
    ids=["an-int-subclass", "a-list-subclass", "a-logger", "an-object", "a-module"],
)
def test_any_other_value_reaches_the_python_callback(value):
    slow = []
    callback = cwatch.plain_return_filter(lambda *args: slow.append(args) or "answer")
    assert callback("code", 12, value) == "answer"
    assert slow == [("code", 12, value)]


@needs_filter
def test_the_watch_s_return_callback_is_the_filter():
    callback = lookups_seen._return_callback()
    assert callback is not lookups_seen._on_return
    assert callback(None, 0, 5) is None


def test_without_the_extension_the_callback_is_the_python_one(monkeypatch):
    from cash import _native

    monkeypatch.setattr(_native, "cwatch", None)
    assert lookups_seen._return_callback() is lookups_seen._on_return
