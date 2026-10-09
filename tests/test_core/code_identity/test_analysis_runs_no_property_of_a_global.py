"""Analysing a cached function runs no property of the values it names.

The call-graph analysis resolved a dotted call such as ``df.values.tolist()``
against the function's module with ``getattr`` on each part, even when ``df``
was the function's own parameter: a module-level table named ``df`` had its
``values`` property built (a copy of the whole table) in every new process
before the first call, and a property of any module-level object (a lazy
loader, a database handle) ran during analysis. The chain is now read
without running code, and a name the function binds itself is never the
global of that name. Every cached function the body calls is still found.
"""

from __future__ import annotations

import functools

import pytest

from cash.analysis.code_analyzer import CodeAnalyzer

pytestmark = pytest.mark.core


class _Counting:
    """A value whose property and ``__getattr__`` count every evaluation."""

    def __init__(self) -> None:
        self.evaluated: list[str] = []

    @property
    def values(self):
        self.evaluated.append("values")
        return self

    def tolist(self):
        return [1]

    def __getattr__(self, name):
        if not name.startswith("__"):
            self.evaluated.append(name)
        raise AttributeError(name)


df = _Counting()
loader = _Counting()


def shadowing(df):
    return df.values.tolist()


def comprehension_shadowing(rows):
    return [df.values.tolist() for df in rows]


def lambda_shadowing(rows):
    return list(map(lambda df: df.values.tolist(), rows))


def reads_the_global(n):
    return loader.values.tolist() + [n]


def test_a_parameter_named_like_a_global_runs_none_of_its_properties(cash_instance):
    df.evaluated.clear()
    cached = cash_instance.cache(shadowing)
    assert cached(_Counting()) == [1]
    assert cached(_Counting()) == [1]
    assert df.evaluated == []


@pytest.mark.parametrize("func", [comprehension_shadowing, lambda_shadowing], ids=["comprehension", "lambda"])
def test_a_local_of_an_inner_scope_named_like_a_global_runs_none_of_its_properties(cash_instance, func):
    df.evaluated.clear()
    cached = cash_instance.cache(func)
    assert cached([]) == []
    assert df.evaluated == []


def test_a_global_s_property_runs_only_when_the_body_runs(cash_instance):
    loader.evaluated.clear()
    cached = cash_instance.cache(reads_the_global)
    assert cached(2) == [1, 2]
    assert loader.evaluated == ["values"]  # the body's own read, none from analysis


# -- every cached callee is still found -------------------------------------


def target(n):
    return n


class Holder:
    @staticmethod
    def static(n):
        return target(n)

    @classmethod
    def klass(cls, n):
        return n

    def method(self, n):
        return n


class Slotted:
    __slots__ = ("fn",)

    def __init__(self, fn):
        self.fn = fn


class Box:
    def __init__(self, fn):
        self.fn = fn


holder = Holder()
box = Box(target)
slotted = Slotted(target)
partial_target = functools.partial(target, 1)


def calls_static():
    return Holder.static(1)


def calls_classmethod():
    return Holder.klass(1)


def calls_method_on_instance():
    return holder.method(1)


def calls_through_instance_dict():
    return box.fn(1)


def calls_through_slot():
    return slotted.fn(1)


def calls_first_iterable_of_shadowing_comprehension():
    return [target for target in target(range(2))]


def calls_through_default(fn=target):
    return fn(1)


def declares_global():
    global target
    return target(1)


def _known(*funcs):
    return {f"{f.__module__}.{f.__qualname__}": f for f in funcs}


@pytest.mark.parametrize(
    ("caller", "callee"),
    [
        (calls_static, Holder.static),
        (calls_classmethod, Holder.klass.__func__),
        (calls_method_on_instance, Holder.method),
        (calls_through_instance_dict, target),
        (calls_through_slot, target),
        (calls_first_iterable_of_shadowing_comprehension, target),
        (calls_through_default, target),
        (declares_global, target),
    ],
    ids=[
        "staticmethod",
        "classmethod",
        "method",
        "instance-dict",
        "slot",
        "comprehension-iterable",
        "default",
        "global-decl",
    ],
)
def test_a_cached_callee_reached_through_a_class_or_object_is_still_found(caller, callee):
    known = _known(callee)
    assert CodeAnalyzer.find_called_functions(caller, known, include_references=True) == set(known)


def uses_parameter_named_like_a_cached_function(target):
    return target(1)


def test_a_parameter_named_like_a_cached_function_is_not_that_function():
    known = _known(target)
    assert CodeAnalyzer.find_called_functions(uses_parameter_named_like_a_cached_function, known) == set()
