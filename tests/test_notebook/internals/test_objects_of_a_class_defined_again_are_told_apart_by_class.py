"""Running a class cell again while many objects of the old class are
alive looks for the variables holding them, and asked two questions of
each object that its class answers: whether it is a class
(`_instances_of`) and whether it is a value (`is_value`, in
`referring_names`). Beside 1.5 million objects the class cell took
3.7-4.1 s, against 1.4-1.5 s asked once per class.
"""

from __future__ import annotations

import sys

from cash.notebook import shared_objects
from cash.notebook.statement import processor


class Rec:
    def __init__(self, i):
        self.i = i


class Number(int):
    """A value by its class (an ``int``), with an identity of its own."""


def _counting(monkeypatch, module, name):
    calls = [0]
    real = getattr(module, name)

    def counting(*args, **kwargs):
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, counting)
    return calls


def test_many_objects_of_one_class_are_asked_whether_they_are_values_once(monkeypatch):
    calls = _counting(monkeypatch, shared_objects, "is_value")
    records = [Rec(i) for i in range(2000)]
    numbers = [Number(i) for i in range(5)]
    kept = shared_objects._not_values([*records, *numbers], shared_objects.VALUE_TYPES)
    assert kept == records  # the numbers are values
    assert calls[0] == 2, calls


def test_the_holders_of_the_objects_are_still_found():
    records = [Rec(i) for i in range(2000)]
    numbers = [Number(i) for i in range(5)]
    names, complete = shared_objects.referring_names(
        [*records, *numbers], {"records": records, "numbers": numbers, "other": [1]}, 30.0
    )
    assert complete
    assert names == {"records"}


def test_the_process_streams_are_asked_about_for_themselves():
    value_types = shared_objects.VALUE_TYPES
    kept = shared_objects._not_values([sys.__stdout__, Rec(1), sys.__stdout__], value_types)
    expected = [obj for obj in (sys.__stdout__,) if not shared_objects.is_value(obj, value_types)]
    assert kept[: len(expected)] == expected
    assert sum(type(obj) is Rec for obj in kept) == 1


def test_objects_of_a_class_are_found_without_asking_each_whether_it_is_a_class(monkeypatch):
    calls = _counting(monkeypatch, processor, "instance_of")
    records = [Rec(i) for i in range(2000)]
    found = processor._instances_of([Rec])
    assert len(found) == 2000 and all(type(obj) is Rec for obj in found)
    assert calls[0] < 50, calls
    del records


def test_a_metaclass_defined_again_has_no_objects_to_look_for():
    class Meta(type):
        pass

    made = Meta("Made", (), {})
    assert processor._instances_of([Meta]) == []
    del made
