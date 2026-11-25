"""A cached call made while another thread is inside ``import numpy`` (or
pandas, polars, pyarrow) runs as it does without that thread.

An import puts the module into ``sys.modules`` before the module's code
runs. cash asks whether a library is loaded, never imports one, and took
the module it found there: one without ``ndarray`` or ``DataFrame`` yet,
so the call failed with AttributeError from inside cash although its
function uses no library at all. A worker thread's first task importing a
library is ordinary.

The library here is a stand-in whose import stops at its first line until
the test lets it finish: the state a real library is in when its import
has just begun, held for as long as the test needs.
"""

from __future__ import annotations

import sys
import textwrap
import threading
import types

import pytest

from tests._scripts import run_python

SCRIPT = """
import dataclasses, datetime, decimal, random, sys, threading
sys.path.insert(0, "libs")
import cash

begun, finish = threading.Event(), threading.Event()
sys._import_gate = (begun, finish)

@dataclasses.dataclass
class Point:
    x: float
    tags: list

class Box:
    def __init__(self, held):
        self.held = held

@cash.cache
def double(x):
    return {"x": x, "twice": x * 2}

@cash.cache
def rows(n):
    return [{"i": i, "s": str(i), "f": i / 3} for i in range(n)]

@cash.cache
def boxed(point, when):
    return Box([Point(point.x * 2, list(point.tags)), when, decimal.Decimal("1.5"), {1, 2}, (1, "a"), b"xx"])

@cash.cache
def drawn(n):
    return [random.random() for _ in range(n)]

@cash.cache
def table(n):
    return {"words": {str(i): [i, str(i)] for i in range(n)}, "log": []}

double(1)
table(20000)
assert LIBRARY not in sys.modules

worker = threading.Thread(target=__import__, args=(LIBRARY,))
worker.start()
assert begun.wait(20)
assert LIBRARY in sys.modules and not hasattr(sys.modules[LIBRARY], "done")
try:
    assert double(1) == {"x": 1, "twice": 2}            # a hit
    assert double(2) == {"x": 2, "twice": 4}            # a miss, stored
    assert double(2) == {"x": 2, "twice": 4}
    assert double.cache_info()["hits"] >= 2, double.cache_info()
    assert len(rows(3)) == 3 and len(rows(4000)) == 4000
    assert boxed(Point(1.0, ["a"]), datetime.datetime(2020, 1, 1)).held[0] == Point(2.0, ["a"])
    assert len(drawn(3)) == 3
    assert table(20000)["log"] == [] and table(20000)["log"] == []
    double.explain(3)
    assert not hasattr(sys.modules[LIBRARY], "done"), "the import was still under way for all of it"
finally:
    finish.set()
    worker.join()
print("OK")
"""

GATED = """
import sys
begun, finish = sys._import_gate
begun.set()
assert finish.wait(60)
done = True
"""


@pytest.mark.parametrize("library", ["numpy", "pandas", "torch"])
def test_a_cached_call_runs_while_another_thread_imports_a_library(tmp_path, library):
    package = tmp_path / "libs" / library
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(textwrap.dedent(GATED), encoding="utf-8")
    script = tmp_path / "main.py"
    script.write_text(f"LIBRARY = {library!r}\n" + textwrap.dedent(SCRIPT), encoding="utf-8")

    done = run_python(script, cwd=tmp_path)

    assert done.returncode == 0, done.stderr[-3000:]
    assert done.stdout.strip().endswith("OK"), done.stdout + done.stderr[-3000:]


def test_a_module_is_loaded_only_once_its_import_has_finished(tmp_path, monkeypatch):
    from cash.loaded_modules import loaded_module

    (tmp_path / "gated_library.py").write_text(textwrap.dedent(GATED), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    begun, finish = threading.Event(), threading.Event()
    monkeypatch.setattr(sys, "_import_gate", (begun, finish), raising=False)
    worker = threading.Thread(target=__import__, args=("gated_library",))
    worker.start()
    try:
        assert begun.wait(20)
        assert "gated_library" in sys.modules, "the import system already lists it"
        assert loaded_module("gated_library") is None
    finally:
        finish.set()
        worker.join()
        half = sys.modules.pop("gated_library", None)
    assert half is not None and half.done
    monkeypatch.setitem(sys.modules, "gated_library", half)
    assert loaded_module("gated_library") is half


def test_a_module_that_is_absent_or_has_no_spec_is_answered_plainly(monkeypatch):
    from cash.loaded_modules import loaded_module

    assert loaded_module("no_such_library_anywhere") is None
    assert loaded_module("sys") is sys
    bare = types.ModuleType("made_by_hand")  # __spec__ is None, as for a script's __main__
    monkeypatch.setitem(sys.modules, "made_by_hand", bare)
    assert loaded_module("made_by_hand") is bare
