"""Cash putting its reader wrappers in and taking them out is no change for a
remembered hit; anything else written meanwhile still is.

Every miss opens a tracker, which sets cash's wrappers on ``threading.Thread``,
``pathlib.Path``, ``os`` and the rest and puts the originals back after
(``io_watch.Patches``). Once a remembered function's key read one of those
(say ``threading.Thread``), each of those writes moved the memo's epoch, so a
miss of any other function stopped the memo serving: a test that started a
thread left the next file's memo tests failing. Now exactly the write cash is
making, on the thread making it, with the value it sets, is passed over;
a user's thread writing in that window still moves the epoch.
"""

from __future__ import annotations

import os
import threading

import pytest

from cash import Cash
from cash._native import cwatch as _cw
from cash.decorator import hit_memo
from cash.tracking import reader_patches

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not hit_memo.ENABLED, reason="the fast hit path needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

RUNS = {"n": 0}


def names_the_thread_class(x):
    RUNS["n"] += 1
    return (x, threading.Thread.__name__, os.stat.__name__)


def starts_a_thread(x):
    out = []
    t = threading.Thread(target=out.append, args=(x,))
    t.start()
    t.join()
    return out[0]


def plus_one(x):
    # No count: a write to RUNS is a change for the memo of the others.
    return x + 1


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


def test_a_miss_of_another_function_keeps_the_memo_serving(c):
    f, g = c.cache(names_the_thread_class), c.cache(plus_one)
    warm(f, 1)
    for x in range(3):
        g(x)  # a miss: cash's wrappers go in and come out
        assert served_fast(f, 1) == ((1, "Thread", "stat"), True)


def test_a_function_that_starts_a_thread_leaves_the_next_memo_serving(c):
    # The two tests, in one process, that left CI red: the first starts a
    # thread under its tracker, the second's memo must serve after it.
    starter = c.cache(starts_a_thread)
    assert starter(5) == 5
    f, g = c.cache(names_the_thread_class), c.cache(plus_one)
    warm(f, 1)
    g(1)
    assert served_fast(f, 1)[1]


def _writes_while_cash_patches(monkeypatch, write):
    """Run *write* on a thread of the user's while cash has armed its first
    write and not yet made it."""
    real = reader_patches._patches._expect
    done = []

    def expect(*args):
        real(*args)
        if not done:
            t = threading.Thread(target=write)
            t.start()
            t.join()
            done.append(args[:2])

    monkeypatch.setattr(reader_patches._patches, "_expect", expect)
    return done


def test_a_user_thread_changing_thread_start_while_cash_patches_still_counts(c, monkeypatch):
    f, g = c.cache(names_the_thread_class), c.cache(plus_one)
    start = threading.Thread.start
    monkeypatch.setattr(threading.Thread, "start", start)  # put back after the test
    warm(f, 1)

    def users_start(self, *args, **kwargs):
        return start(self, *args, **kwargs)

    done = _writes_while_cash_patches(monkeypatch, lambda: setattr(threading.Thread, "start", users_start))
    g(1)
    assert done, "cash armed no write during the miss"
    assert served_fast(f, 1)[1] is False


def test_a_user_thread_changing_os_stat_while_cash_patches_still_counts(c, monkeypatch):
    f, g = c.cache(names_the_thread_class), c.cache(plus_one)
    stat = os.stat
    monkeypatch.setattr(os, "stat", stat)
    warm(f, 1)

    def users_stat(*args, **kwargs):
        return stat(*args, **kwargs)

    done = _writes_while_cash_patches(monkeypatch, lambda: setattr(os, "stat", users_stat))
    g(1)
    assert done
    assert served_fast(f, 1)[1] is False


def test_the_value_cash_sets_written_by_another_thread_still_counts(c, monkeypatch):
    # Same owner, name and value as armed, but not the thread that armed it.
    f = c.cache(names_the_thread_class)
    start = threading.Thread.__dict__["start"]
    monkeypatch.setattr(threading.Thread, "start", start)
    warm(f, 1)

    def wrapper(self, *args, **kwargs):
        return start(self, *args, **kwargs)

    _cw.expect_set(threading.Thread, "start", wrapper)
    try:
        t = threading.Thread(target=setattr, args=(threading.Thread, "start", wrapper))
        t.start()
        t.join()
    finally:
        _cw.expect_clear()
    assert served_fast(f, 1)[1] is False


def test_another_write_on_the_patching_thread_still_counts(c, monkeypatch):
    f = c.cache(names_the_thread_class)
    join = threading.Thread.__dict__["join"]
    monkeypatch.setattr(threading.Thread, "join", join)
    warm(f, 1)

    def users_join(self, *args, **kwargs):
        return join(self, *args, **kwargs)

    _cw.expect_set(threading.Thread, "start", threading.Thread.__dict__["start"])
    try:
        threading.Thread.join = users_join  # armed for `start`, not `join`
    finally:
        _cw.expect_clear()
    assert served_fast(f, 1)[1] is False


def test_another_value_for_the_armed_name_still_counts(c, monkeypatch):
    f = c.cache(names_the_thread_class)
    stat = os.stat
    monkeypatch.setattr(os, "stat", stat)
    warm(f, 1)

    def users_stat(*args, **kwargs):
        return stat(*args, **kwargs)

    _cw.expect_set(os, "stat", stat)
    try:
        os.stat = users_stat
    finally:
        _cw.expect_clear()
    assert served_fast(f, 1)[1] is False


def test_another_value_for_an_armed_class_attribute_still_counts(c, monkeypatch):
    f = c.cache(names_the_thread_class)
    start = threading.Thread.__dict__["start"]
    monkeypatch.setattr(threading.Thread, "start", start)
    warm(f, 1)

    def users_start(self, *args, **kwargs):
        return start(self, *args, **kwargs)

    _cw.expect_set(threading.Thread, "start", start)
    try:
        threading.Thread.start = users_start
    finally:
        _cw.expect_clear()
    assert served_fast(f, 1)[1] is False


def test_the_armed_write_itself_is_passed_over(c, monkeypatch):
    f = c.cache(names_the_thread_class)
    stat, start = os.stat, threading.Thread.__dict__["start"]
    monkeypatch.setattr(os, "stat", stat)
    monkeypatch.setattr(threading.Thread, "start", start)
    warm(f, 1)
    for owner, name, value in ((os, "stat", stat), (threading.Thread, "start", start)):
        for v in (lambda *a, **k: value(*a, **k), value):  # a wrapper, then the original back
            _cw.expect_set(owner, name, v)
            try:
                setattr(owner, name, v)
            finally:
                _cw.expect_clear()
    assert served_fast(f, 1)[1] is True
