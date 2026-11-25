"""The modules a piece of code imports are read in one place for both keys.

The decorator (which folds the versions of the libraries a body imports) and
the notebook (which keys a statement on the modules of the user's a helper
imports) each scanned the bytecode for imports with their own loop. A fix
for code with more than 255 names, where an ``EXTENDED_ARG`` prefix stands
between the import and the two constants before it, reached only the
decorator's: the notebook's read a relative import as absolute and lost the
from-list. Both now ask `cash.analysis.bytecode.imports_of`.
"""

from __future__ import annotations

import inspect
import types

import pytest

from cash.decorator import library_versions
from cash.notebook import cache_key


def _imports_of(code):
    from cash.analysis.bytecode import imports_of

    return imports_of(code)


def _function_code(filler: str) -> types.CodeType:
    source = (
        "def f():\n"
        f"{filler}\n"
        "    from .helper import scale\n"
        "    from sklearn import linear_model\n"
        "    from .. import sibling\n"
        "    import os.path\n"
        "    import numpy as np\n"
        "    from collections import abc, deque\n"
        "    return scale, linear_model, sibling, os, np\n"
    )
    namespace: dict = {}
    exec(compile(source, "<imports>", "exec"), namespace)
    return namespace["f"].__code__


FILLERS = {
    "a few names": "\n".join(f"    v{i}" for i in range(10)),
    "more than 255 names": "\n".join(f"    v{i}" for i in range(300)),
    "more than 255 constants": "\n".join(f"    x = {i + 1000}.5" for i in range(300)),
}

IMPORTS = [
    (1, "helper", ("scale",)),
    (0, "sklearn", ("linear_model",)),
    (2, "", ("sibling",)),
    (0, "os.path", ()),
    (0, "numpy", ()),
    (0, "collections", ("abc", "deque")),
]


@pytest.mark.parametrize("filler", list(FILLERS))
def test_each_import_is_read_with_its_level_and_names(filler):
    assert _imports_of(_function_code(FILLERS[filler])) == IMPORTS


@pytest.mark.parametrize("filler", list(FILLERS))
def test_the_notebook_resolves_a_relative_import_in_a_long_body(filler):
    found = sorted(cache_key._imports_of(_function_code(FILLERS[filler]), "mypkg.sub"))
    assert found == [
        "collections",
        "collections.abc",
        "collections.deque",
        "mypkg",
        "mypkg.sibling",
        "mypkg.sub.helper",
        "mypkg.sub.helper.scale",
        "numpy",
        "os.path",
        "sklearn",
        "sklearn.linear_model",
    ]


@pytest.mark.parametrize("filler", list(FILLERS))
def test_the_decorator_takes_the_absolute_ones(filler):
    found = library_versions._imported_modules(_function_code(FILLERS[filler]))
    assert found == ("sklearn", "os.path", "numpy", "collections")


def test_code_that_imports_nothing_gives_nothing():
    assert _imports_of((lambda rows: [len(r) for r in rows]).__code__) == []
    assert _imports_of(compile("x = __import__('json')", "<s>", "exec")) == []


@pytest.mark.parametrize("module", [cache_key, library_versions], ids=["notebook", "decorator"])
def test_neither_key_scans_for_imports_itself(module):
    """A second loop over ``IMPORT_NAME`` is how the two drifted apart."""
    source = inspect.getsource(module)
    assert "IMPORT_NAME" not in source
    assert "imports_of" in source
