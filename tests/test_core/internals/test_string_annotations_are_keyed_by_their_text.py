"""A string annotation (``typing.ForwardRef``) is keyed by its text.

``Optional["Cfg"]`` holds a ``ForwardRef``, and so does every field of a
``TypedDict`` defined in a module with ``from __future__ import
annotations`` (on 3.14, read through its ``__annotate__``). Its pickle
holds the code object it compiled from the text, which no pickle writes:
the key could not be built and a function reading one (scanpy's
``highly_variable_genes`` reads a ``TypedDict``'s fields) ran uncached
with KEY-UNHASHABLE-CAPTURE. Keyed by its text, the call hits, and an edit
to the fields misses.
"""

from __future__ import annotations

import importlib
import sys
import typing
import warnings

import pytest

from cash import Cash

pytestmark = [pytest.mark.core]

SCHEMA = """\
from __future__ import annotations

from typing import Literal, TypedDict


class Opts(TypedDict):
    flavor: Literal["a", "b"]
    n: {n_type}
"""

CALLER = """\
import typing

import {schema}


def fields(x, opts: {schema}.Opts | None = None):
    return (x, sorted(typing.get_type_hints({schema}.Opts)))


def raw(x):
    return (x, {schema}.Opts.__annotations__["n"])
"""


def _modules(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(tmp_path))
    schema = f"_typeddict_schema_{tmp_path.name}"
    caller = f"_typeddict_caller_{tmp_path.name}"
    for name in (schema, caller):
        monkeypatch.delitem(sys.modules, name, raising=False)
    (tmp_path / f"{schema}.py").write_text(SCHEMA.format(n_type="int"), encoding="utf-8")
    (tmp_path / f"{caller}.py").write_text(CALLER.format(schema=schema), encoding="utf-8")
    return schema, importlib.import_module(caller)


@pytest.mark.parametrize("which", ["fields", "raw"])
def test_a_typeddict_with_string_annotations_keys_hits_and_misses_on_an_edit(tmp_path, monkeypatch, which):
    schema, caller = _modules(tmp_path, monkeypatch)
    c = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
    f = c.cache(getattr(caller, which))

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no KEY-UNHASHABLE-CAPTURE
        first = f(1)
        assert f(1) == first
        assert f.cache_info()["hits"] == 1

        (tmp_path / f"{schema}.py").write_text(SCHEMA.format(n_type="float"), encoding="utf-8")
        importlib.reload(sys.modules[schema])
        f(1)
        assert f.cache_info()["hits"] == 1  # the edited fields missed
        f(1)
        assert f.cache_info()["hits"] == 2


def test_a_string_annotation_in_a_default_keys_by_its_text(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
    runs = []

    def make(hint):
        def described(x, hint=hint):
            runs.append(x)  # @cash:assume-safe
            return (x, repr(hint))

        return c.cache(described)

    f = make(typing.Optional[typing.ForwardRef("Cfg")])  # what Optional["Cfg"] holds
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no KEY-UNHASHABLE-DEFAULT
        assert f(1) == f(1)
        assert len(runs) == 1
        g = make(typing.Optional[typing.ForwardRef("Other")])
        g(1)
        assert len(runs) == 2  # another text, another key
