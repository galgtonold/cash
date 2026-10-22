"""The walk that looks for a writing special method without a record of
what ran (`writers_without_a_trace`) reads objects by type and Python's own
slots: no property (``dtype``, ``columns``) or method (``to_numpy``) of the
user's runs, and its steps are bounded however often a value repeats. Two
lambdas on one line are told apart by where each starts."""

from __future__ import annotations

import sys
import time

import pytest

from cash.notebook.statement.unseen_writes import SET_LATER, own_writes, writers_without_a_trace

RAN: list[str] = []


class Lazy:
    """A dataset wrapper whose attributes load on read."""

    @property
    def dtype(self):
        RAN.append("dtype")
        return None

    @property
    def dtypes(self):
        RAN.append("dtypes")
        return []

    @property
    def columns(self):
        RAN.append("columns")
        return []


class ArrayLike:
    dtype = None

    def to_numpy(self):
        RAN.append("to_numpy")
        return []

    def ravel(self):
        RAN.append("ravel")
        return []


def test_no_property_or_method_of_the_users_runs():
    import numpy as np

    ArrayLike.dtype = np.dtype(object)
    RAN.clear()
    ns = {"x": Lazy(), "lst": [Lazy(), ArrayLike()], "a": ArrayLike()}
    # Its properties write (``RAN.append``): found, read from their text.
    writers_without_a_trace("w = (type(x).__name__, lst, a)", ns)
    assert RAN == []


def test_a_grid_of_repeated_values_is_quick():
    ns = {"g": [[0] * 3000 for _ in range(3000)]}
    started = time.perf_counter()
    # Past what it reads, any live class of the user's with a writing
    # special method counts (this module's `Lazy`): only the time is asked.
    writers_without_a_trace("w = g[0][0]", ns)
    # 3.3 s when every item was opened; well under a second now, under load too.
    assert time.perf_counter() - started < 1.0


class Plain:
    pass


def test_each_value_taken_counts_toward_the_limit():
    # The same object a million times: one new object, a million steps.
    one = Plain()
    ns = {"g": [[one] * 1000 for _ in range(1000)]}
    started = time.perf_counter()
    # Past what it reads, any live class of the user's with a writing
    # special method counts (this module's `Lazy`): only the time is asked.
    writers_without_a_trace("w = g[0][0]", ns)
    # 3.3 s when every item was opened; well under a second now, under load too.
    assert time.perf_counter() - started < 1.0


calls: list = []
H = {"g": lambda o: 1, "f": lambda o: calls.append(1) or 1}
G = {"f": lambda o: calls.append(1) or 1, "g": lambda o: 1}


@pytest.mark.skipif(sys.version_info < (3, 11), reason="positions to tell them apart are there from 3.11")
def test_two_lambdas_on_one_line_are_read_each_as_itself():
    assert own_writes(H["f"].__code__) == {"calls"}
    assert own_writes(H["g"].__code__) == set()
    assert own_writes(G["f"].__code__) == {"calls"}
    assert own_writes(G["g"].__code__) == set()


@pytest.mark.skipif(sys.version_info >= (3, 11), reason="3.10 has no positions to tell them apart")
def test_two_lambdas_on_one_line_are_unreadable_before_3_11():
    assert own_writes(H["g"].__code__) == {"?"}


def test_a_method_set_on_a_class_later_is_said_to_be_unseeable():
    class Vec:
        pass

    Vec.__add__ = lambda self, o: 10
    found = writers_without_a_trace("w = v + 1", {"v": Vec()})
    assert [targets for _, targets in found] == [(SET_LATER,)]
