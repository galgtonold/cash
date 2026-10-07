"""A cold process reads each function's and class's source once per file version.

The analysis and the key ask for the same source many times on a cached
function's first call. On Python before 3.13 each ``inspect.getsource`` of a
class parses its whole module: in a 3000-line module three of them cost about
150 ms. The whole-module scan for mutated globals re-read and re-hashed the
module for every helper it met. Both are now memoised on the file's
``(mtime, size)``, once the file has settled -- so an edit is always read.
"""

from __future__ import annotations

import importlib
import inspect
import linecache
import os
import sys
import time

import pytest

from cash import source_reading
from cash.analysis import mutable_globals

pytestmark = [pytest.mark.core]

MODULE = """\
import functools

COUNTER = []


def plain(x):
    return x + {step}


def bump():
    COUNTER.append(1)


@functools.total_ordering
class Decorated:
    def __eq__(self, other):
        return True

    def __lt__(self, other):
        return False


class Outer:
    class Inner:
        def method(self):
            return {step}

    def method(self):
        return lambda: {step}


if True:
    class Twice:
        first = True
else:
    class Twice:
        first = False


def factory():
    class Local:
        value = {step}

    return Local
"""


def _load(tmp_path, monkeypatch, step: int = 1, age: float = 60.0):
    monkeypatch.syspath_prepend(str(tmp_path))
    name = f"_source_memo_{tmp_path.name}"
    monkeypatch.delitem(sys.modules, name, raising=False)
    path = tmp_path / f"{name}.py"
    path.write_text(MODULE.format(step=step), encoding="utf-8")
    settled = time.time() - age
    os.utime(path, (settled, settled))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    return importlib.import_module(name), path


def _objects(module):
    return [
        module.plain,
        module.bump,
        module.Decorated,
        module.Decorated.__lt__,
        module.Outer,
        module.Outer.Inner,
        module.Outer.Inner.method,
        module.Outer().method,
        module.Outer().method(),
        module.Twice,
        module.factory(),
        module.plain.__code__,
    ]


def test_the_memo_answers_what_inspect_answers(tmp_path, monkeypatch):
    module, _ = _load(tmp_path, monkeypatch)
    for obj in _objects(module):
        expected = inspect.getsourcelines(obj)
        assert source_reading.getsourcelines(obj) == expected, obj
        assert source_reading.getsourcelines(obj) == expected, obj  # and from the memo
        assert source_reading.getsource(obj) == inspect.getsource(obj), obj


def test_a_settled_file_is_read_once(tmp_path, monkeypatch):
    module, _ = _load(tmp_path, monkeypatch)
    calls = []
    real = inspect.getsourcelines
    monkeypatch.setattr(inspect, "getsourcelines", lambda obj: calls.append(obj) or real(obj))
    parses = []
    real_starts = source_reading._class_starts
    monkeypatch.setattr(source_reading, "_class_starts", lambda tree: parses.append(1) or real_starts(tree))

    for _ in range(3):
        source_reading.getsource(module.plain)
        source_reading.getsource(module.Decorated)
        source_reading.getsource(module.Outer)
        source_reading.getsource(module.Outer.Inner)

    assert calls == [module.plain] or (sys.version_info >= (3, 13) and len(calls) == 4)
    if sys.version_info < (3, 13):
        assert len(parses) == 1, "the module was parsed for each class"


def test_an_edit_is_read(tmp_path, monkeypatch):
    module, path = _load(tmp_path, monkeypatch, step=1)
    assert "x + 1" in source_reading.getsource(module.plain)
    assert "value = 1" in source_reading.getsource(module.factory())

    path.write_text(MODULE.format(step=2), encoding="utf-8")  # same size
    edited = time.time() - 30
    os.utime(path, (edited, edited))

    assert "x + 2" in source_reading.getsource(module.plain)
    assert "value = 2" in source_reading.getsource(module.factory())
    assert source_reading.getsource(module.plain) == inspect.getsource(module.plain)


def test_a_file_saved_just_now_is_not_memoised(tmp_path, monkeypatch):
    """Two saves inside one mtime tick keep the stat, so what was read from a
    file saved moments ago is not memoised on it (`stat_has_settled`)."""
    module, _ = _load(tmp_path, monkeypatch, age=0)
    calls = []
    real = inspect.getsourcelines
    monkeypatch.setattr(inspect, "getsourcelines", lambda obj: calls.append(obj) or real(obj))
    source_reading.getsource(module.plain)
    source_reading.getsource(module.plain)
    assert calls == [module.plain, module.plain]


def test_source_linecache_holds_by_hand_is_not_memoised(tmp_path, monkeypatch):
    """A notebook or ``exec`` puts its text into ``linecache`` under a name;
    that text can change while any file of that name does not."""
    module, path = _load(tmp_path, monkeypatch)
    fake = MODULE.format(step=7)
    monkeypatch.setitem(linecache.cache, str(path), (len(fake), None, fake.splitlines(True), str(path)))
    assert "x + 7" in source_reading.getsource(module.plain)
    fake = MODULE.format(step=8)
    monkeypatch.setitem(linecache.cache, str(path), (len(fake), None, fake.splitlines(True), str(path)))
    assert "x + 8" in source_reading.getsource(module.plain)


def test_the_module_mutation_scan_is_memoised_per_file_version(tmp_path, monkeypatch):
    module, path = _load(tmp_path, monkeypatch)
    reads = []
    real = inspect.getsource
    monkeypatch.setattr(inspect, "getsource", lambda obj: reads.append(obj) or real(obj))

    assert "COUNTER" in mutable_globals.module_modified_globals(module)
    assert "COUNTER" in mutable_globals.module_modified_globals(module)
    assert reads == [module], "the module was read again for the same file version"

    path.write_text(MODULE.format(step=1).replace("COUNTER.append(1)", "return 1"), encoding="utf-8")
    edited = time.time() - 30
    os.utime(path, (edited, edited))
    assert "COUNTER" not in mutable_globals.module_modified_globals(module)


def _held_cell(name: str, source: str) -> dict:
    """Compile *source* the way a notebook cell is: its lines held by
    ``linecache`` under a name with no file behind it."""
    linecache.cache[name] = (len(source), None, source.splitlines(True), name)
    namespace: dict = {}
    exec(compile(source, name, "exec"), namespace)
    return namespace


def test_a_cells_function_is_read_once_while_its_lines_are_held(monkeypatch):
    name = "<cash-test-held-source>"
    try:
        fn = _held_cell(name, "def f(x):\n    return x + 1\n")["f"]
        assert source_reading.getsource(fn) == "def f(x):\n    return x + 1\n"
        calls = []
        real = inspect.getsourcelines
        monkeypatch.setattr(inspect, "getsourcelines", lambda obj: calls.append(obj) or real(obj))
        assert source_reading.getsource(fn) == "def f(x):\n    return x + 1\n"
        assert calls == []
        # The cell run again with other text: a new entry, read again.
        fn2 = _held_cell(name, "def f(x):\n    return x + 2\n")["f"]
        assert source_reading.getsource(fn2) == "def f(x):\n    return x + 2\n"
        # The old function's lines are the entry's now, as inspect answers too.
        assert source_reading.getsource(fn) == "def f(x):\n    return x + 2\n"
        assert len(calls) == 1
    finally:
        linecache.cache.pop(name, None)
