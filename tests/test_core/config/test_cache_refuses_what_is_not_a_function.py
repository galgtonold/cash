"""``@cash.cache`` refuses, when it is applied, what it cannot wrap.

Each of these used to fail later and far from the cause: a property read as
a bound method, a classmethod raised at its first call, a class stopped
being a class, and a number passed the ``functools.lru_cache(128)`` way
raised ``'int' object is not callable``. The order that works (the
descriptor above ``@cash.cache``) is pinned beside each refusal.
"""

from __future__ import annotations

import functools

import pytest


@pytest.mark.parametrize("descriptor", [property, classmethod, staticmethod])
def test_above_a_descriptor_it_names_the_order_that_works(disk_cash, descriptor):
    def body(*args):
        return 1

    name = descriptor.__name__
    with pytest.raises(TypeError, match=rf"put @{name} above @cash\.cache"):
        disk_cash.cache(descriptor(body))
    with pytest.raises(TypeError, match=rf"put @{name} above @cash\.cache"):
        disk_cash.cache(ttl=60)(descriptor(body))


@pytest.mark.filterwarnings("ignore::cash.CashCacheIneffectiveWarning")
def test_below_property_classmethod_and_staticmethod_it_works(disk_cash):
    """The order the messages name. ``self`` and ``cls`` of a class made
    inside a test cannot be keyed (hence the filter), so the hit is counted
    on the static method, which takes neither."""

    class Model:
        scale = 3

        @property
        @disk_cash.cache
        def size(self):
            return 7

        @classmethod
        @disk_cash.cache
        def scaled(cls, x):
            return cls.scale * x

        @staticmethod
        @disk_cash.cache
        def double(x):
            return 2 * x

    m = Model()
    assert (m.size, m.size) == (7, 7)
    assert (Model.scaled(2), Model.scaled(2)) == (6, 6)
    assert (Model.double(2), m.double(2)) == (4, 4)
    assert Model.__dict__["double"].__func__.cache_info()["hits"] == 1


@pytest.mark.parametrize("value", [128, "name", 2.5, True])
def test_a_positional_option_says_options_are_named(disk_cash, value):
    with pytest.raises(TypeError, match=r"options are keyword-only.*@cash\.cache\(ttl=") as raised:

        @disk_cash.cache(value)
        def f(x):
            return x

    assert repr(value) in str(raised.value)


def test_the_lru_cache_habit_is_told_there_is_no_maxsize(disk_cash):
    with pytest.raises(TypeError, match="no maxsize"):
        disk_cash.cache(128)


def test_a_class_is_refused_by_name(disk_cash):
    with pytest.raises(TypeError, match=r"functions, not classes: Kept would stop being a class"):

        @disk_cash.cache
        class Kept:
            def __init__(self, v):
                self.v = v


def test_other_callables_are_still_cached(disk_cash):
    """A lambda, a partial and an instance with ``__call__`` are callables
    cash wraps as before; only the shapes above are refused."""

    class Adder:
        def __call__(self, x):
            return x + 1

    def add(x, y):
        return x + y

    for target in (lambda x: x + 1, functools.partial(add, 1), Adder()):
        cached = disk_cash.cache(target)
        assert cached(1) == 2
