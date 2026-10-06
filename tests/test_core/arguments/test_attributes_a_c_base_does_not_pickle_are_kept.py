"""An attribute that pickle drops is still keyed on and stored.

``pickle`` stores an object as its class's reduce says. A C class with its own
reduce -- ``numpy.ndarray``, ``datetime.date`` -- describes its own data only,
so a Python subclass's attributes were lost: numpy's ``InfoArray`` example
keyed without its ``info`` (two arrays differing only there shared one entry)
and came back from disk without it.
"""

from __future__ import annotations

import datetime

import pytest

from cash import Cash
from cash.value_hash import compute_hash

np = pytest.importorskip("numpy")


class InfoArray(np.ndarray):
    """numpy's own subclassing example."""

    def __new__(cls, input_array, info=None):
        obj = np.asarray(input_array).view(cls)
        obj.info = info
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.info = getattr(obj, "info", None)


class Day(datetime.date):
    pass


def _day(label):
    day = Day(2024, 1, 2)
    day.label = label
    return day


def test_arrays_differing_only_in_an_attribute_key_apart(cash_instance):
    @cash_instance.cache
    def describe(a):
        return f"{a.sum()} {a.info}"

    assert describe(InfoArray([1, 2, 3], info="metres")) == "6 metres"
    assert describe(InfoArray([1, 2, 3], info="kilometres")) == "6 kilometres"
    assert describe(InfoArray([1, 2, 3], info="metres")) == "6 metres"
    assert describe.cache_info()["hits"] == 1


def test_one_held_by_a_container_keys_apart_too(cash_instance):
    @cash_instance.cache
    def labels(days):
        return [d.label for d in days["days"]]

    assert labels({"days": [_day("a")]}) == ["a"]
    assert labels({"days": [_day("b")]}) == ["b"]


def test_the_notebook_hash_tells_them_apart():
    assert compute_hash(InfoArray([1, 2], info="a")) != compute_hash(InfoArray([1, 2], info="b"))
    assert compute_hash(_day("a")) != compute_hash(_day("b"))
    assert compute_hash(InfoArray([1, 2], info="a")) == compute_hash(InfoArray([1, 2], info="a"))


def test_a_disk_hit_keeps_the_attribute(tmp_path):
    def make_cash():
        cash = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)

        @cash.cache
        def make():
            return InfoArray([1, 2, 3], info="metres"), _day("x")

        return make

    first = make_cash()()
    assert first[0].info == "metres"
    make = make_cash()  # a new process's view: the RAM tier is empty
    array, day = make()
    assert make.cache_info()["hits"] == 1
    assert type(array) is InfoArray and array.info == "metres" and array.tolist() == [1, 2, 3]
    assert type(day) is Day and day.label == "x" and day == datetime.date(2024, 1, 2)
