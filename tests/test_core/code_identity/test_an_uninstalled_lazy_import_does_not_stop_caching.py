"""A function whose module lazily imports an optional dependency that is not
installed still caches.

Lazy-import helpers (``optional = lazy.load("optional")``) bind the name to
a module object at import time; when the package is missing, every
attribute read on that object raises ModuleNotFoundError. A call through it
cannot run, so it reaches no code to key. cash read those attributes while
walking the function's helpers, globals, libraries and random generators,
and the ImportError ended the key build: the function never cached, even on
the branch that does not touch the optional package.
"""

from __future__ import annotations

import types
import warnings

import pytest

from cash import CashCacheIneffectiveWarning, CashImpurityWarning
from tests._scripts import run_python


class _DelayedImportErrorModule(types.ModuleType):
    """The shape a lazy loader gives a package that is not installed."""

    def __getattr__(self, name):
        raise ModuleNotFoundError("No module named 'optional_pkg'")


optional_pkg = _DelayedImportErrorModule("optional_pkg")


def _fast(x):
    return [v * 2 for v in x]


def resample(x, fancy=False):
    if fancy:
        return optional_pkg.resample(x)
    return _fast(x)


def test_a_function_that_may_call_an_uninstalled_module_caches(disk_cash):
    f = disk_cash.cache(resample)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CashCacheIneffectiveWarning)
        warnings.simplefilter("ignore", CashImpurityWarning)
        assert f([1, 2]) == [2, 4]
        assert f.explain([1, 2]).reason == "hit"
        assert f([1, 2]) == [2, 4]


def test_the_call_through_it_still_raises_the_import_error(disk_cash):
    f = disk_cash.cache(resample)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ModuleNotFoundError):
            f([1, 2], fancy=True)


MODULE = """
import types

class _DelayedImportErrorModule(types.ModuleType):
    def __getattr__(self, name):
        raise ModuleNotFoundError("No module named 'optpkg'")

try:
    import optpkg
except ImportError:
    optpkg = _DelayedImportErrorModule("optpkg")

import cash

@cash.cache
def resample(x, fancy=False):
    if fancy:
        return optpkg.resample(x)
    return [v * 2 for v in x]
"""

SCRIPT = """
import os, sys
sys.path.insert(0, os.getcwd())
import mod
print(mod.resample.explain([1, 2]).reason)
print(mod.resample([1, 2]))
"""


def test_installing_the_package_changes_the_key(tmp_path):
    """What the name reaches once it resolves is keyed: the entry stored
    while the package was missing is not served once it is installed."""
    (tmp_path / "mod.py").write_text(MODULE, encoding="utf-8")
    (tmp_path / "main.py").write_text(SCRIPT, encoding="utf-8")
    assert run_python("main.py", cwd=tmp_path).stdout.splitlines()[0] == "no_entry"
    assert run_python("main.py", cwd=tmp_path).stdout.splitlines()[0] == "hit"

    (tmp_path / "optpkg.py").write_text("def resample(x):\n    return x[::-1]\n", encoding="utf-8")
    assert run_python("main.py", cwd=tmp_path).stdout.splitlines()[0] != "hit"
