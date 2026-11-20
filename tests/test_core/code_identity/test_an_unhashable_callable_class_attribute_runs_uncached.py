"""A callable class attribute the key cannot hold runs the call uncached.

A class an argument carries is keyed by its code surface, class attributes
included. A callable attribute with no code of its own (a callable
instance, a ``functools.partial`` of a builtin) is keyed by its code and its
pickled content. When the content cannot be pickled (it holds a lock), the
state that decides what the attribute computes was left out of the key with
no word: a second process built the attribute differently and was served
the first process's result. The call now runs uncached with
KEY-UNHASHABLE-DEFAULT, as for any other value code reaching the call holds
that cannot be keyed.
"""

from __future__ import annotations

import warnings

import pytest

from cash.exceptions import CashCacheIneffectiveWarning
from tests._scripts import run_python

_SCRIPT = """
import functools
import operator
import sys
import threading

import cash


class Scaler:
    def __init__(self, n):
        self.n = n
        self._lock = threading.Lock()

    def __call__(self, x):
        return x * self.n

    def __mul__(self, x):
        return x * self.n


class Model:
    __getitem__ = {attribute}


@cash.cache
def run(m, x):
    return m[x]


print(run(Model(), 2))
"""

_ATTRIBUTES = {
    "callable instance": "Scaler(int(sys.argv[1]))",
    "partial of a builtin": "functools.partial(operator.mul, Scaler(int(sys.argv[1])))",
}


@pytest.mark.parametrize("attribute", list(_ATTRIBUTES.values()), ids=list(_ATTRIBUTES))
def test_a_second_process_building_the_attribute_differently_is_not_served(tmp_path, attribute):
    (tmp_path / "main.py").write_text(_SCRIPT.format(attribute=attribute), encoding="utf-8")
    first = run_python("main.py", "3", cwd=tmp_path)
    assert first.stdout.split() == ["6"]
    second = run_python("main.py", "4", cwd=tmp_path)
    assert second.stdout.split() == ["8"], "an attribute the key could not hold was served the old result"
    assert "KEY-UNHASHABLE-DEFAULT" in second.stderr, second.stderr


class _Locked:
    def __init__(self, n):
        import threading

        self.n = n
        self._lock = threading.Lock()

    def __call__(self, x):
        return x * self.n


class _Holder:
    __getitem__ = _Locked(3)


class _Plain:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        return x * self.n


class _PlainHolder:
    __getitem__ = _Plain(3)


def test_the_call_warns_with_the_attribute_named(cash_instance):
    @cash_instance.cache
    def run(m, x):
        return m[x]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert run(_Holder(), 2) == 6
        assert run(_Holder(), 2) == 6
    assert run.cache_info()["hits"] == 0
    said = [str(w.message) for w in caught if issubclass(w.category, CashCacheIneffectiveWarning)]
    assert any("KEY-UNHASHABLE-DEFAULT" in s and "_Holder.__getitem__" in s for s in said), said


def test_a_picklable_callable_attribute_still_caches(cash_instance):
    # The positive control: a callable attribute whose state pickles is keyed
    # by it and served from the cache.
    @cash_instance.cache
    def run(m, x):
        return m[x]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert run(_PlainHolder(), 2) == 6
        assert run(_PlainHolder(), 2) == 6
    assert run.cache_info()["hits"] == 1
    assert not [w for w in caught if "KEY-UNHASHABLE-DEFAULT" in str(w.message)]
