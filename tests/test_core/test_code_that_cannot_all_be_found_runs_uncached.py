"""When cash cannot find all the code a call runs, the call runs uncached.

Three places kept a partial answer and keyed it: the helper analysis, whose
failure was taken as "no helpers" (the function keyed by its own code alone),
the search for the functions a callable runs (``callable_layers``), which
stopped at an object it could not look into, and the search for the code an
argument's code names, which stopped four references down. Each now either
finds everything or runs the call uncached with KEY-HELPERS-UNWALKABLE.
"""

from __future__ import annotations

import importlib
import sys
import types
import typing
import warnings

import pytest

import cash.decorator.code_identity as code_identity
import cash.decorator.registry as registry
from cash import Cash
from cash.analysis.helper_code import callable_layers
from cash.exceptions import CashWarning

pytestmark = [pytest.mark.core]


def _uncached_twice(fn, *args) -> dict:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn(*args)
        fn(*args)
    info = fn.cache_info()
    assert info["hits"] == 0, info
    return {"warnings": [str(w.message) for w in caught if issubclass(w.category, CashWarning)], "info": info}


def _plus_one(x):
    return x + 1


def test_a_failed_helper_analysis_runs_the_call_uncached(tmp_path, monkeypatch):
    class _Broken:
        def analyze(self, func):
            raise OSError("the source moved")

    monkeypatch.setattr(registry, "get_analyzer", lambda: _Broken())
    c = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
    seen = _uncached_twice(c.cache(_plus_one), 1)
    assert any("KEY-HELPERS-UNWALKABLE" in m and "the source moved" in m for m in seen["warnings"]), seen


class _Opaque:
    """A callable whose ``__wrapped__`` cannot be read."""

    @property
    def __wrapped__(self):
        raise RuntimeError("not now")

    def __call__(self, x):
        return x


def test_a_callable_that_cannot_be_looked_into_raises():
    from cash.analysis.helper_code import UnwalkableLayers

    with pytest.raises(UnwalkableLayers, match="not now"):
        callable_layers(_Opaque())


opaque_helper = _Opaque()


def _through_opaque(x):
    return opaque_helper(x)


def test_a_function_calling_such_a_callable_runs_uncached(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
    seen = _uncached_twice(c.cache(_through_opaque), 1)
    assert any("KEY-HELPERS-UNWALKABLE" in m and "not now" in m for m in seen["warnings"]), seen


def _class_name(cls):
    return cls.__name__


ENDLESS = """
import sys

_module = sys.modules[__name__]


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)

    def made():
        return getattr(_module, "next_one")()

    made.__qualname__ = made.__name__ = name
    return made


class Model:
    def go(self):
        return getattr(_module, "next_one")()
"""


def test_argument_code_whose_references_never_end_runs_uncached(tmp_path, monkeypatch):
    """Each lookup makes a new function that looks one up: the reference walk
    from an argument's code cannot finish, and a key of the part it saw
    (four references, before) is not a key of the code that runs."""
    monkeypatch.setattr(code_identity, "MAX_CODE_REF_TARGETS", 40)  # the real bound only costs time
    monkeypatch.syspath_prepend(str(tmp_path))
    name = f"_endless_refs_{tmp_path.name}"
    monkeypatch.delitem(sys.modules, name, raising=False)
    (tmp_path / f"{name}.py").write_text(ENDLESS, encoding="utf-8")
    module = importlib.import_module(name)

    c = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
    seen = _uncached_twice(c.cache(_class_name), module.Model)
    assert any("KEY-HELPERS-UNWALKABLE" in m for m in seen["warnings"]), seen


ANNOTATED = """
class Weird:
    def __getattr__(self, name):
        if name == "__forward_arg__":
            raise RuntimeError("not a forward reference")
        raise AttributeError(name)


class B:
    def value(self):
        return {value}


class A:
    first: Weird()
    b: "B"
"""


def _hint_builder(cls):
    return typing.get_type_hints(cls)["b"]().value()


def _annotated_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    # Compiled on its own: `exec` would otherwise inherit this file's
    # `from __future__ import annotations` and turn every hint into a string.
    exec(compile(ANNOTATED.format(value=1), name, "exec", dont_inherit=True), module.__dict__)
    return module


def test_an_annotation_that_cannot_be_followed_is_not_skipped():
    """The first annotation raised, and the walk returned what it had found
    so far -- nothing -- so ``B``, named only by ``A``'s hint, was left out."""
    from cash._annotation_refs import annotation_referents

    name = "_cash_test_annotation_refs"
    try:
        module = _annotated_module(name)
        with pytest.raises(RuntimeError, match="not a forward reference"):
            annotation_referents(module.A)
    finally:
        sys.modules.pop(name, None)


def test_a_class_whose_annotations_cannot_be_followed_runs_uncached():
    name = "_cash_test_annotation_refs_call"
    try:
        module = _annotated_module(name)
        seen = _uncached_twice(Cash().cache(_hint_builder), module.A)
        assert any("not a forward reference" in m for m in seen["warnings"]), seen
    finally:
        sys.modules.pop(name, None)
