"""A library default the callee only reads in place is not hashed around
each miss, unless it holds a value of the user's: ``buf=[Vec(1)]`` with
``x + buf[0]`` runs ``Vec.__add__``, the user's own code, which may change
the default. Only the default's own type was checked, so a list holding
such a value was read as a list of plain values and left unwatched.
"""

from __future__ import annotations

import pytest

from cash.notebook import call_key

np = pytest.importorskip("numpy")


class Vec:
    def __init__(self, x):
        self.x = x

    def __add__(self, other):
        self.x += 1
        return self.x + other


VEC = [Vec(1)]
NESTED = {"k": (Vec(1),)}
PLAIN = [1.0, 2.0]
ARRAY = np.zeros(10)
OBJECTS = np.array([Vec(1)], dtype=object)


def f(x, buf=VEC):
    return buf[0] + x


def g(x, buf=NESTED):
    return buf["k"][0] + x


def h(x, buf=OBJECTS):
    return buf[0] + x


def reads_plain(x, buf=PLAIN):
    return buf[0] + x


def reads_array(x, buf=ARRAY):
    return buf[0] + x


@pytest.mark.parametrize("fn", [f, g, h])
def test_a_default_holding_a_users_value_is_watched(fn):
    value = fn.__defaults__[0]
    assert not call_key._only_reads(fn, "buf", value)


@pytest.mark.parametrize("fn", [reads_plain, reads_array])
def test_a_default_of_plain_values_read_in_place_is_not(fn):
    """The control: the fast path for plain values stays."""
    assert call_key._only_reads(fn, "buf", fn.__defaults__[0])
