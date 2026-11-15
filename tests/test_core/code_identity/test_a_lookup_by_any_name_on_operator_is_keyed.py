"""``getattr(operator, name)`` with a computed name is keyed as a plain call.

xclim picks its comparison with ``getattr(operator, f"__{op}__")``. Whatever
name it is handed, it finds one of the standard library's ``operator``
functions, which no code of the user's defines: the call is keyed by its
arguments, on every Python, called at once or bound to a local first. On
Python 3.12+ a call traced for another lookup checks ``operator``'s names
instead of refusing a lookup whose name the recorder could not read.
"""

from __future__ import annotations

import operator
import sys

import pytest

pytestmark = [pytest.mark.core]


def test_a_comparison_picked_by_name_is_stored_and_hit(disk_cash):
    @disk_cash.cache
    def compare(a, b, op):
        return getattr(operator, f"__{op}__")(a, b)

    assert [compare(1, 2, "lt"), compare(1, 2, "lt"), compare(1, 2, "gt")] == [True, True, False]
    info = compare.cache_info()
    assert info["hits"] == 1, info["miss_reasons"]


def test_a_comparison_bound_to_a_local_first_is_stored_and_hit(disk_cash):
    @disk_cash.cache
    def combine(op):
        fn = getattr(operator, f"__{op}__")
        return fn(3, 2)

    assert [combine("add"), combine("add"), combine("sub")] == [5, 5, 1]
    assert combine.cache_info()["hits"] == 1


class _Ops:
    double = staticmethod(lambda x: 2 * x)


def test_beside_a_lookup_on_a_class_of_the_users_it_is_stored(disk_cash):
    """A lookup on a user's class makes the call traced on 3.12+; the
    ``operator`` lookup in the same call is checked there, not refused.
    Before 3.12 there is no trace, and the lookup on the class runs the call
    uncached."""

    def pick(op):
        return getattr(operator, f"__{op}__")

    @disk_cash.cache
    def apply(kind, op, x):
        return pick(op)(getattr(_Ops, kind)(x), 1)

    assert [apply("double", "add", 3), apply("double", "add", 3)] == [7, 7]
    traced = sys.version_info >= (3, 12)
    assert apply.cache_info()["hits"] == (1 if traced else 0), apply.cache_info()["miss_reasons"]
