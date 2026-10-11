"""What a class or a callable that no other channel keys stands for in the
key: its ``module:qualname`` (`code_ref`), or for a callable object with no
name of its own, its value (`GlobalValues.code_ref_part`), which warns
KEY-UNHASHABLE-GLOBAL when it cannot be hashed.
"""

from __future__ import annotations

import math
import warnings
from unittest import mock

import pytest

from cash import Cash
from cash.decorator.key_values import code_ref

np = pytest.importorskip("numpy")


class _Base:
    @classmethod
    def build(cls):
        return cls()


class _Child(_Base):
    pass


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (sum, "builtins:sum"),
        (math.floor, "math:floor"),
        (np.float32, "numpy:float32"),
        (np.add, "numpy:add"),
        (ValueError, "builtins:ValueError"),
        (str.upper, "builtins:str.upper"),
    ],
)
def test_a_name_is_module_and_qualname(value, expected):
    assert code_ref(value) == expected


def test_an_inherited_classmethod_names_the_class_it_is_bound_to():
    assert code_ref(_Base.build) != code_ref(_Child.build)
    assert code_ref(_Child.build).endswith("_Child.build")


def test_what_has_no_name_has_no_ref():
    assert code_ref(mock.Mock()) is None
    assert code_ref(object()) is None


class _NoState:
    """A callable object with no name of its own."""

    __slots__ = ()

    def __call__(self, x):
        return x


class _Unpicklable:
    def __init__(self):
        self.lock = __import__("threading").Lock()

    def __call__(self, x):
        return x

    def __reduce_ex__(self, protocol):
        raise TypeError("no")


def test_an_unnamed_callable_is_keyed_by_its_value_or_warns(tmp_path, monkeypatch):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    values = c._values
    # Stand-ins for a library's objects: their code is not the user's.
    monkeypatch.setattr(_NoState, "__module__", "somelib.mod")
    monkeypatch.setattr(_Unpicklable, "__module__", "somelib.mod")
    monkeypatch.setattr("cash.decorator.global_values.own_code_is_user", lambda *_: False)
    first = values.code_ref_part(_NoState(), "app", "f", "OP")
    assert first is not None and first == values.code_ref_part(_NoState(), "app", "f", "OP")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert values.code_ref_part(_Unpicklable(), "app", "f", "OP") is None
    assert any("KEY-UNHASHABLE-GLOBAL" in str(w.message) for w in caught)
