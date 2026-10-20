"""A class change cash's window must not pass for cash's own write.

1. While a write is armed, a type event on the armed thread for the owner
   or a subclass is passed over. A class change that writes no dict
   (``__bases__``, and so the MRO) made on that thread inside the window
   would then have been taken for cash's write, and the memo served a
   result computed under the old bases. Code that can run on cash's thread
   there: a signal handler between the bytecodes of ``Patches._set``, a
   ``__del__`` or weakref callback run by a collection. A world type's bases
   and MRO are held from when it was watched, and the settle re-tags only
   types that still have them; otherwise the epoch moves. The same change
   from another thread, or a dict write on the same thread, is a change too.

2. Renaming a class the key read (``threading.Thread.__name__ =
   "Renamed"``) writes no dict and CPython sends no type event: each world
   type's name and qualname are held, and a memo declines once they move.
"""

from __future__ import annotations

import threading

import pytest

from cash import Cash
from cash._native import cwatch as _cw
from cash.decorator import hit_memo

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not hit_memo.ENABLED, reason="the fast hit path needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]


class Small:
    setting = "small"


class Large:
    setting = "large"


class Worker(threading.Thread, Small):
    pass


RUNS = {"n": 0}


def setting_of_worker(x):
    RUNS["n"] += 1
    return (x, Worker.setting)


def name_of_thread(x):
    RUNS["n"] += 1
    return (x, threading.Thread.__name__)


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)


def served_fast(f, *args):
    before_hits, before_runs = hit_memo.status()["fast_hits"], RUNS["n"]
    result = f(*args)
    return result, hit_memo.status()["fast_hits"] > before_hits and RUNS["n"] == before_runs


def warm(f, *args):
    for _ in range(6):
        result, fast = served_fast(f, *args)
        if fast:
            return result
    raise AssertionError(f"{f.__name__}{args} was never served from the memo")


@pytest.fixture
def wrapper(monkeypatch):
    start = threading.Thread.__dict__["start"]
    monkeypatch.setattr(threading.Thread, "start", start)  # put back after the test

    def wrapped(self, *a, **k):
        return start(self, *a, **k)

    return wrapped


def test_bases_changed_outside_any_window(c):
    f = c.cache(setting_of_worker)
    assert warm(f, 1) == (1, "small")
    Worker.__bases__ = (threading.Thread, Large)
    try:
        assert f(1) == (1, "large")
    finally:
        Worker.__bases__ = (threading.Thread, Small)


def test_bases_changed_on_the_patching_thread_inside_the_window(c, wrapper):
    f = c.cache(setting_of_worker)
    assert warm(f, 1) == (1, "small")
    _cw.expect_set(threading.Thread, "start", wrapper)
    try:
        Worker.__bases__ = (threading.Thread, Large)  # not the armed write
        threading.Thread.start = wrapper
    finally:
        _cw.expect_clear()
    try:
        assert f(1) == (1, "large")
    finally:
        Worker.__bases__ = (threading.Thread, Small)


def test_bases_changed_by_another_thread_inside_the_window(c, wrapper):
    f = c.cache(setting_of_worker)
    assert warm(f, 1) == (1, "small")

    def change():
        Worker.__bases__ = (threading.Thread, Large)

    _cw.expect_set(threading.Thread, "start", wrapper)
    try:
        t = threading.Thread(target=change)
        t.start()
        t.join()
        threading.Thread.start = wrapper
    finally:
        _cw.expect_clear()
    try:
        assert f(1) == (1, "large")
    finally:
        Worker.__bases__ = (threading.Thread, Small)


def test_subclass_dict_written_on_the_patching_thread_inside_the_window(c, wrapper):
    f = c.cache(setting_of_worker)
    assert warm(f, 1) == (1, "small")
    _cw.expect_set(threading.Thread, "start", wrapper)
    try:
        Worker.setting = "own"
        threading.Thread.start = wrapper
    finally:
        _cw.expect_clear()
    try:
        assert f(1) == (1, "own")
    finally:
        del Worker.setting


def test_a_rename_stops_the_memo_serving(c):
    # The ordinary key names a class by its qualname and dict, so what a
    # renamed class answers is the ordinary path's to decide; the memo must
    # at least not answer in its place.
    f = c.cache(name_of_thread)
    warm(f, 1)
    threading.Thread.__name__ = "Renamed"
    try:
        assert served_fast(f, 1)[1] is False
    finally:
        threading.Thread.__name__ = "Thread"
