"""A redefined cached function always runs its own body, never an old version's.

The decorator pins a function's own-source identity per function object. That
pin used to be keyed on ``id(func)`` alone and never dropped, so once a
redefined function died, a later definition could get its address, find the
old pin and key its calls by the OLD source: re-running a notebook cell or
reloading a module then served the previous version's result as a hit (about
two in three of 60 redefinitions in practice).
"""

from __future__ import annotations

import gc
import importlib
import linecache
import os
import sys
import types
import warnings

import pytest

from cash import Cash

ROUNDS = 50


@pytest.fixture
def quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_a_rerun_cell_gets_its_own_result(tmp_path, quiet):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    module = types.ModuleType("redefined_cells_under_test")
    sys.modules[module.__name__] = module
    module.c = c
    wrong = []
    try:
        for i in range(ROUNDS):
            src = f"@c.cache\ndef train(x):\n    return x + {i}\n"
            # ipykernel names each cell run's code this way.
            fname = str(tmp_path / f"ipykernel_1234/{1000 + i}.py")
            linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
            exec(compile(src, fname, "exec"), module.__dict__)
            got = module.train(1)
            if got != 1 + i:
                wrong.append((i, got))
            gc.collect()
    finally:
        sys.modules.pop(module.__name__, None)
    assert wrong == []


def test_a_reloaded_module_gets_its_own_result(tmp_path, quiet, monkeypatch):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    holder = types.ModuleType("redefined_holder_under_test")
    holder.c = c
    monkeypatch.setitem(sys.modules, holder.__name__, holder)
    monkeypatch.syspath_prepend(str(tmp_path))
    path = tmp_path / "redefined_mod_under_test.py"
    wrong = []
    module = None
    try:
        for i in range(ROUNDS):
            path.write_text(
                f"from redefined_holder_under_test import c\n@c.cache\ndef f(x):\n    return x + {i}\n",
                encoding="utf-8",
            )
            # A distinct mtime per version, so the import system recompiles.
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + (i + 1) * 10**9))
            if module is None:
                module = importlib.import_module("redefined_mod_under_test")
            else:
                module = importlib.reload(module)
            got = module.f(1)
            if got != 1 + i:
                wrong.append((i, got))
            gc.collect()
    finally:
        sys.modules.pop("redefined_mod_under_test", None)
    assert wrong == []


def test_a_dead_functions_pin_is_dropped(tmp_path, quiet):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    for i in range(ROUNDS):
        ns: dict = {}
        exec(f"def g(x):\n    return x * {i}\n", ns)
        wrapped = c.cache(ns["g"])
        assert wrapped(2) == 2 * i
        del wrapped, ns
        gc.collect()
    # Pins follow their functions out: only the live registration (the
    # registry holds the last definition) keeps one.
    assert len(c._code._own_pins) <= 1
