"""A class or function renamed through ``__name__`` alone is a new key.

Setting ``__name__`` leaves the qualname, the code, the source and the dict
as they were, and a key named a class or function by its qualname: a result
that read the name was served after the rename with the old name in it.
The name is now keyed when it is not the last part of the qualname, so no
key of a class or function not renamed that way moved.
"""

from __future__ import annotations

from threading import Thread

import pytest

from cash.analysis.helper_code import renamed

pytestmark = pytest.mark.core


class Mine:
    pass


def helper():
    return 1


def test_a_user_class_renamed(cash_instance):
    @cash_instance.cache
    def name_of(x):
        return (x, Mine.__name__)

    assert name_of(1) == (1, "Mine")
    Mine.__name__ = "Renamed"
    try:
        assert name_of(1) == (1, "Renamed")
    finally:
        Mine.__name__ = "Mine"
    assert name_of(1) == (1, "Mine")


def test_a_library_class_imported_by_name_renamed(cash_instance):
    @cash_instance.cache
    def name_of(x):
        return (x, Thread.__name__)

    assert name_of(1) == (1, "Thread")
    Thread.__name__ = "Renamed"
    try:
        assert name_of(1) == (1, "Renamed")
    finally:
        Thread.__name__ = "Thread"


def test_a_helper_function_renamed(cash_instance):
    @cash_instance.cache
    def name_of(x):
        return (x, helper.__name__, helper())

    assert name_of(1) == (1, "helper", 1)
    helper.__name__ = "renamed"
    try:
        assert name_of(1) == (1, "renamed", 1)
    finally:
        helper.__name__ = "helper"


def test_nothing_is_added_while_the_name_ends_the_qualname():
    assert renamed(Mine) == renamed(helper) == renamed(Thread) == ""

    class Inner:
        pass

    assert Inner.__qualname__.endswith(".Inner") and renamed(Inner) == ""
    Inner.__name__ = "Other"
    assert renamed(Inner) == "|name=Other"
