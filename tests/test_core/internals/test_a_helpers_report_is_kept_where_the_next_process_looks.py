"""A helper walk's report is kept where the next process looks it up.

A fresh process reads a cached function's purity report from the store
beside the results instead of walking its helpers again
(`cash.analysis.report_store`), and the answers worked out from a helper's
text alone (`cash.analysis.text_store`). Two things kept lmfit's from
ever being read back, so every process worked them out again:

* Both are written in batches and at exit. At exit no call runs, so
  the folder was the default instance's: with a ``Cash(cache_dir=...)``
  of the user's and a default cache folder some other script had made,
  the reports went there, while the next process looked up the reports
  of what a call reaches (the methods of its arguments, and the classes
  those make) in the user's folder. Each report now goes to the folder
  its own lookup read, and each kind of text answer to the folder its
  file was read from.
* A method with a private name (``def __set_expression``) is held by its
  class under its mangled name. A report whose walk visited one was never
  stored: no name led to it.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from cash._active import default_cash, set_default_cash
from cash.analysis import report_store, scan_store, text_store
from tests._scripts import run_python

_MODULE = """
LIMIT = 3


class Box:
    def __init__(self, x):
        self.x = x

    def __grow(self, by):
        return self.x + by + LIMIT

    def grown(self, by):
        return self.__grow(by)


class Runner:
    def run(self, x):
        return Box(x).grown(2) * 2


def top(runner, x):
    return runner.run(x)
"""

_SCRIPT = """
import json, sys
import cash
from cash.analysis import report_store

report_store._READ_COST = -1.0  # every walk is worth keeping
import mod

c = cash.Cash(cache_dir=sys.argv[1])
f = c.cache(mod.top)
assert f(mod.Runner(), 2) == 14
print(json.dumps(report_store.status()))
"""


@pytest.fixture
def _no_default_instance():
    """No default instance an earlier test left behind: its cache dir would
    stand in for the ``CASH_CACHE_DIR`` the test sets."""
    earlier = default_cash()
    set_default_cash(None)
    yield
    set_default_cash(earlier)


def _run(tmp_path, user_cache, default_cache):
    out = run_python("main.py", str(user_cache), cwd=tmp_path, cache_dir=default_cache)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_next_process_reads_the_report_a_users_cash_kept(tmp_path):
    (tmp_path / "mod.py").write_text(textwrap.dedent(_MODULE), encoding="utf-8")
    (tmp_path / "main.py").write_text(textwrap.dedent(_SCRIPT), encoding="utf-8")
    default_cache = tmp_path / "default-cache"
    default_cache.mkdir()
    user_cache = tmp_path / "user-cache"
    first = _run(tmp_path, user_cache, default_cache)
    assert first["loaded"] == 0
    assert first["refused"] == 0, f"a report could not be kept: {first}"
    second = _run(tmp_path, user_cache, default_cache)
    assert second["loaded"] >= 1, f"no report was read back: {second}"


def test_a_private_method_is_found_by_its_mangled_name():
    class _Owner:
        def __hidden(self):
            return 1

    method = _Owner._Owner__hidden
    assert report_store._chain(method.__qualname__)[-1] == "_Owner__hidden"
    assert report_store._chain("A.B.__x") == ("A", "B", "_B__x")
    assert report_store._chain("A.__dunder__") == ("A", "__dunder__")
    assert report_store._chain("__lead.__x") == ("__lead", "_lead__x")
    assert report_store._chain("___.__x") == ("___", "__x"), "a class named only by underscores mangles nothing"


def test_at_exit_a_report_goes_to_the_folder_its_lookup_read(tmp_path, monkeypatch, _no_default_instance):
    default_cache = tmp_path / "default-cache"
    default_cache.mkdir()
    monkeypatch.setenv("CASH_CACHE_DIR", str(default_cache))
    monkeypatch.delenv("CASH_ANALYSIS_STORE", raising=False)
    default = scan_store.store_dir()
    (tmp_path / "user-cache").mkdir()
    used = tmp_path / "user-cache" / "analysis" / default.name
    kept, moved = ("m", ("kept",), "f.py:1"), ("m", ("moved",), "f.py:2")
    gone = tmp_path / "gone-cache" / "analysis" / default.name
    monkeypatch.setattr(report_store, "_PENDING", [(kept, {"text": "a"}, used), (moved, {"text": "b"}, gone)])
    report_store.flush(at_exit=True)
    assert report_store._entries(report_store._file(kept, used), kept) == [{"text": "a"}]
    assert not report_store._file(kept, default).exists()
    assert report_store._entries(report_store._file(moved, default), moved) == [{"text": "b"}], (
        "a report whose cache folder was removed was not kept in the one there is"
    )
    assert not (tmp_path / "gone-cache").exists(), "the store made a cache folder of its own"


def test_at_exit_text_answers_go_to_the_folder_their_kind_was_read_from(tmp_path, monkeypatch, _no_default_instance):
    monkeypatch.delenv("CASH_ANALYSIS_STORE", raising=False)
    text_store.forget()
    user_cache, default_cache = tmp_path / "user-cache", tmp_path / "default-cache"
    user_cache.mkdir()
    default_cache.mkdir()
    monkeypatch.setenv("CASH_CACHE_DIR", str(user_cache))
    used = scan_store.store_dir()

    def ask(compute):
        return text_store.kept("probe-kind", "x = 1\n", "", compute, list, tuple)

    try:
        assert ask(lambda: ("x",)) == ("x",)
        monkeypatch.setenv("CASH_CACHE_DIR", str(default_cache))
        text_store.flush(at_exit=True)
        text_store.forget()
        monkeypatch.setenv("CASH_CACHE_DIR", str(user_cache))
        assert scan_store.store_dir() == used
        assert ask(lambda: pytest.fail("worked out again: the answer was not kept where it was read")) == ("x",)
    finally:
        text_store.forget()
