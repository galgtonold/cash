"""A library constant read through its module is part of the key.

Only a user module's ``mod.ATTR`` was keyed: ``x * math.pi`` was served
``2 * 3.14159...`` after ``math.pi = 3.0``, a function reading ``sys.argv``
kept its result after ``sys.argv.append(...)`` and in a run with other
arguments, and ``threading.Thread.__name__`` set to another name was served
the old one. A library's plain data read through its module (numbers,
strings, and lists, tuples and dicts of them) is now keyed by value, and a
class or function of it by its name when it was renamed. Its code is keyed by
its name as before, so a function reading only a library's functions, classes
and modules keeps its key.
"""

from __future__ import annotations

import math
import sys
import threading

import pytest

from cash import Cash
from cash.decorator import hit_memo
from cash.decorator.module_attrs import _AttrReader, _library_attr_parts, written_module_attrs
from tests._scripts import run_python

RUNS = {"n": 0}


def times_pi(x):
    RUNS["n"] += 1
    return x * math.pi


def the_arguments(x):
    RUNS["n"] += 1
    return (x, tuple(sys.argv[1:]))


def thread_name(x):
    RUNS["n"] += 1
    return (x, threading.Thread.__name__)


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def _call_until_served(f, *args):
    """Call *f* until it is served (the fast hit path too, where there is one)."""
    for _ in range(6):
        before = RUNS["n"]
        result = f(*args)
    assert RUNS["n"] == before, "the repeated call was not served"
    return result


def test_a_library_constant_set_to_another_value(c):
    f = c.cache(times_pi)
    assert _call_until_served(f, 2) == 2 * math.pi
    old = math.pi
    math.pi = 3.0
    try:
        assert f(2) == 6.0
        assert _call_until_served(f, 2) == 6.0
    finally:
        math.pi = old
    assert f(2) == 2 * old


def test_a_library_list_changed_in_place(c, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    f = c.cache(the_arguments)
    assert _call_until_served(f, 1) == (1, ())
    sys.argv.append("--fast")
    assert f(1) == (1, ("--fast",))


def test_a_library_class_renamed_through_its_module(c):
    f = c.cache(thread_name)
    assert _call_until_served(f, 1) == (1, "Thread")
    threading.Thread.__name__ = "Renamed"
    try:
        assert f(1) == (1, "Renamed")
    finally:
        threading.Thread.__name__ = "Thread"
    assert f(1) == (1, "Thread")


SCRIPT = """\
import sys, warnings; warnings.simplefilter('ignore')
import cash
RAN = [0]

@cash.cache
def the_arguments():
    RAN[0] += 1
    return list(sys.argv[1:])

print(the_arguments(), RAN[0])
"""


def test_a_script_run_with_other_arguments_runs_again(tmp_path):
    (tmp_path / "main.py").write_text(SCRIPT, encoding="utf-8")
    first = run_python("main.py", "a", cwd=tmp_path).stdout.strip()
    again = run_python("main.py", "a", cwd=tmp_path).stdout.strip()
    other = run_python("main.py", "b", cwd=tmp_path).stdout.strip()
    assert first == "['a'] 1"
    assert again == "['a'] 0", "the same arguments: served"
    assert other == "['b'] 1", "other arguments: computed again"


@pytest.mark.parametrize(
    ("mod", "attr"),
    [(math, "sqrt"), (threading, "Thread"), (sys, "stdout"), (sys, "modules"), (math, "no_such_name")],
)
def test_a_librarys_code_and_objects_add_nothing(c, mod, attr):
    """What keeps every other key as it was: a function, an unrenamed class,
    an object and a module table add no part."""
    reader = _AttrReader(times_pi, "f", None, frozenset(), None, None, None)
    assert _library_attr_parts(c._values, reader, mod, mod.__name__, attr) == []


@pytest.mark.skipif(not hit_memo.ENABLED, reason="the fast hit path needs the C extension")
def test_the_fast_hit_path_sees_the_constant_set_again(c):
    f = c.cache(times_pi)
    before = hit_memo.status()["fast_hits"]
    _call_until_served(f, 3)
    assert hit_memo.status()["fast_hits"] > before, "the memo never served it"
    old = math.pi
    math.pi = 3.0
    try:
        assert f(3) == 9.0
    finally:
        math.pi = old


class Mine:
    pass


def mine_name(x):
    RUNS["n"] += 1
    return (x, Mine.__name__, Mine.__qualname__)


@pytest.mark.skipif(not hit_memo.ENABLED, reason="the fast hit path needs the C extension")
@pytest.mark.parametrize("attr", ["__name__", "__qualname__"])
def test_the_fast_hit_path_sees_a_class_renamed(c, attr):
    """Setting ``__name__`` or ``__qualname__`` moves no version tag and
    tells no type watcher: the fast hit path compares the names too."""
    f = c.cache(mine_name)
    before = hit_memo.status()["fast_hits"]
    first = _call_until_served(f, 1)
    assert hit_memo.status()["fast_hits"] > before, "the memo never served it"
    old = getattr(Mine, attr)
    setattr(Mine, attr, "Renamed")
    try:
        assert f(1) != first
        assert "Renamed" in f(1)
    finally:
        setattr(Mine, attr, old)


def writes_them():
    sys.argv.append("ran")
    sys.argv[0] = "prog"
    math.tau = 6.0
    threading.TIMEOUT_MAX += 1
    return math.e


def test_what_the_code_writes_is_its_output_not_an_input():
    """A list a call fills (`builtins._CASH_TEST_CALLS.append(1)` in the
    module-instance tests) would move the key on every call: what the code
    rebinds or changes is left out, and what it only reads is kept."""
    written = written_module_attrs(writes_them.__code__)
    assert written == {("sys", "argv"), ("math", "tau"), ("threading", "TIMEOUT_MAX")}
