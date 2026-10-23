"""An object whose ``__class__`` answers ``FunctionType``, ``type``, a
module, a method or ``np.ndarray`` passes ``isinstance`` for one without
having any of its parts (a function's ``__closure__``, a class's
``__qualname__``, an array's ``.base``). cash read those parts off it after
such a check, and the read's AttributeError failed the user's cell, where
plain Python runs it. It is read as the object it is, and one that is
callable is called as itself: its type's ``__call__`` runs, with the globals
that reads."""

from __future__ import annotations

import pytest

from cash.backends.persistence_policy import PersistencePolicy
from tests._cell_driver import run_cash_cell

MODULE = """
import types, numpy as np


def passing_for(kind):
    class Passes:
        def __init__(self, v):
            self.v = v

        @property
        def __class__(self):
            return kind

    return Passes


def calling_for(kind):
    class Calls(passing_for(kind)):
        def __call__(self, x):
            return self.v + x

    return Calls


AsFunc = passing_for(types.FunctionType)
AsArray = passing_for(np.ndarray)
AsType = passing_for(type)
AsModule = passing_for(types.ModuleType)
AsMethod = passing_for(types.MethodType)
CallsAsFunc = calling_for(types.FunctionType)
CallsAsType = calling_for(type)
CallsAsMethod = calling_for(types.MethodType)
"""
KINDS = ["AsFunc", "AsArray", "AsType", "AsModule", "AsMethod"]
SHAPES = {
    "bound": ["c = C(5)", "w = c.v + 1", "object.__setattr__(c, 'v', 9)", "w2 = c.v + 1"],
    "in_a_list": ["c = [C(5)]", "w = c[0].v + 1", "object.__setattr__(c[0], 'v', 9)", "w2 = c[0].v + 1"],
    "returned": ["def make(v):\n    return C(v)", "c = make(5)", "w = c.v + 1"],
}
CALLED = ["c = C(5)", "w = c(1)", "object.__setattr__(c, 'v', 9)", "w2 = c(1)"]


def _runs_as_plain_python(cash_magics, tmp_path, monkeypatch, cells, names):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    (tmp_path / "passing_kinds.py").write_text(MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    plain: dict = {}
    for cell in cells:
        exec(cell, plain)
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    ns = cash_magics.shell.user_ns
    names = [name for name in names if name in plain]
    assert {name: ns[name] for name in names} == {name: plain[name] for name in names}


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("kind", KINDS)
def test_an_object_passing_for_a_kind_runs_and_answers_as_plain_python(cash_magics, tmp_path, monkeypatch, kind, shape):
    cells = [f"from passing_kinds import {kind} as C", *SHAPES[shape]]
    _runs_as_plain_python(cash_magics, tmp_path, monkeypatch, cells, ("w", "w2"))


@pytest.mark.parametrize("kind", ["CallsAsFunc", "CallsAsType", "CallsAsMethod"])
def test_a_callable_passing_for_a_function_class_or_method_is_called_as_plain_python(
    cash_magics, tmp_path, monkeypatch, kind
):
    cells = [f"from passing_kinds import {kind} as C", *CALLED]
    _runs_as_plain_python(cash_magics, tmp_path, monkeypatch, cells, ("w", "w2"))


@pytest.mark.parametrize("kind", ["types.FunctionType", "type", "types.MethodType"])
def test_a_callable_of_the_users_passing_for_a_kind_answers_for_the_global_it_reads(
    cash_magics, tmp_path, monkeypatch, kind
):
    cells = [
        "import types\nG = 1",
        "class C:\n    @property\n    def __class__(self):\n"
        f"        return {kind}\n\n    def __call__(self, x):\n        return G + x",
        "c = C()",
        "w = c(1)",
        "G = 2",
        "w2 = c(1)",
    ]
    _runs_as_plain_python(cash_magics, tmp_path, monkeypatch, cells, ("w", "w2"))


def test_kinds_nested_as_isinstance_takes_them_are_read_as_isinstance_reads_them():
    from cash._user_attrs import instance_of

    assert instance_of(3, (str, (bytes, (int,))))
    assert instance_of(3, str | int)
    assert not instance_of(3.0, (str, (bytes, int)))
    assert instance_of(type, (int, (type,)))
