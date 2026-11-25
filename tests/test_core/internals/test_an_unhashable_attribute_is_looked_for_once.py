"""Naming the attribute that makes an argument unhashable costs one walk.

The walk hashes each attribute of the argument on its own. It is for the
warning, which is shown once per function and type; every later call that
cannot be keyed must not pay it again.
"""

from __future__ import annotations

import threading
import warnings

import pytest

from cash.decorator.arg_hashing import ArgHasher
from tests._work_counts import method_calls


class Model:
    def __init__(self):
        self.weights = list(range(1000))
        self._lock = threading.Lock()

    def predict(self, x):
        return x


def test_the_attributes_are_walked_once_not_on_every_uncached_call(disk_cash):
    predict = disk_cash.cache(Model.predict)
    model = Model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with method_calls(ArgHasher, "unhashable_attribute") as walks:
            assert predict(model, 0) == 0
            first = walks.calls
            for x in range(1, 5):
                assert predict(model, x) == x
    assert predict.cache_info()["hits"] == 0, "the control: every call ran uncached"
    # The argument, and the lock it holds, which has no attributes.
    assert first == 2
    assert walks.calls == first


@pytest.mark.parametrize("value", [object, len, threading, lambda: 1])
def test_a_class_function_or_module_is_not_walked(disk_cash, value):
    """Their ``__dict__`` is code and names, not the state of an argument."""
    with method_calls(ArgHasher, "hash_payload") as hashed:
        assert disk_cash._args.unhashable_attribute(value) is None
    assert hashed.calls == 0
