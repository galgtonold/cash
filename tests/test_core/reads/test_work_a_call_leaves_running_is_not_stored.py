"""A call that returns while work it handed to a thread or pool still runs is not stored.

The entry's file fingerprints are taken after the body returns. Work the body
submitted and did not wait for runs on: a file it reads from then on is in
no fingerprint, and what it puts in the result after the store is not in the
entry. A body that filled its result from a file in a ``pool.submit`` it never
waited for was stored, and the next call served the file's old content (or a
result the work had not filled yet) after the file changed. Such a call is now
returned but not stored (``STORE-WORK-RUNNING``); work it waited for, or that
was cancelled before it ran, stores as before.
"""

from __future__ import annotations

import os
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = pytest.mark.core


def _files(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("A1", encoding="utf-8")
    b.write_text("B1", encoding="utf-8")
    return a, b


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _fill_when_told(out, path, go, done):
    """Put *path*'s content in *out* once the file *go* exists, then make *done*."""
    deadline = time.monotonic() + 10
    while not os.path.exists(go) and time.monotonic() < deadline:
        time.sleep(0.001)
    out["b"] = _read(path)
    with open(done, "w", encoding="utf-8"):
        pass


def _wait_for(path):
    deadline = time.monotonic() + 10
    while not os.path.exists(path):
        assert time.monotonic() < deadline, "the work never finished"
        time.sleep(0.001)


def test_a_file_read_by_work_left_running_is_not_served_stale(disk_cash, tmp_path):
    a, b = _files(tmp_path)
    go, done = str(tmp_path / "go"), str(tmp_path / "done")
    runs = []

    @disk_cash.cache(assume_safe=True)
    def load(pa, pb, go, done):
        runs.append(1)
        out = {"a": _read(pa)}
        ThreadPoolExecutor(1).submit(_fill_when_told, out, pb, go, done)  # never waited for
        return out

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load(str(a), str(b), go, done)
    open(go, "w", encoding="utf-8").close()
    _wait_for(done)

    b.write_text("B2", encoding="utf-8")
    os.remove(go)
    os.remove(done)
    result = load(str(a), str(b), go, done)
    open(go, "w", encoding="utf-8").close()
    _wait_for(done)
    assert len(runs) == 2, "the entry stored while its work still ran was served"
    assert result == {"a": "A1", "b": "B2"}
    assert any("STORE-WORK-RUNNING" in str(w.message) for w in caught), [str(w.message) for w in caught]


def test_work_the_call_waited_for_is_stored(disk_cash, tmp_path):
    a, b = _files(tmp_path)
    runs = []

    @disk_cash.cache(assume_safe=True)
    def load(pa, pb):
        runs.append(1)
        pool = ThreadPoolExecutor(2)
        future = pool.submit(_read, pb)
        value = _read(pa) + future.result()
        pool.shutdown(wait=False)
        return value

    assert load(str(a), str(b)) == "A1B1"
    assert load(str(a), str(b)) == "A1B1"
    assert len(runs) == 1
    b.write_text("B2", encoding="utf-8")
    assert load(str(a), str(b)) == "A1B2", "a file read on the pool is an input"


def test_a_pool_the_call_closes_is_stored(disk_cash, tmp_path):
    a, b = _files(tmp_path)
    runs = []

    @disk_cash.cache(assume_safe=True)
    def load(*paths):
        runs.append(1)
        with ThreadPoolExecutor(2) as pool:
            return "".join(pool.map(_read, paths))

    assert load(str(a), str(b)) == "A1B1"
    assert load(str(a), str(b)) == "A1B1"
    assert len(runs) == 1


def test_work_cancelled_before_it_ran_does_not_hold_the_store(disk_cash, tmp_path):
    a, _b = _files(tmp_path)
    runs = []

    @disk_cash.cache(assume_safe=True)
    def load(pa):
        runs.append(1)
        pool = ThreadPoolExecutor(1)
        release = threading.Event()
        first = pool.submit(release.wait, 10)
        second = pool.submit(_read, pa)
        assert second.cancel()
        release.set()
        first.result()
        pool.shutdown(wait=False)
        return _read(pa)

    assert load(str(a)) == "A1"
    assert load(str(a)) == "A1"
    assert len(runs) == 1


def test_a_thread_the_call_does_not_join_holds_the_store(disk_cash, tmp_path):
    a, b = _files(tmp_path)
    go, done = str(tmp_path / "go"), str(tmp_path / "done")
    runs = []

    @disk_cash.cache(assume_safe=True)
    def load(pa, pb, go, done):
        runs.append(1)
        out = {"a": _read(pa)}
        threading.Thread(target=lambda: _fill_when_told(out, pb, go, done)).start()  # never joined
        return out

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load(str(a), str(b), go, done)
    open(go, "w", encoding="utf-8").close()
    _wait_for(done)

    b.write_text("B2", encoding="utf-8")
    os.remove(go)
    os.remove(done)
    result = load(str(a), str(b), go, done)
    open(go, "w", encoding="utf-8").close()
    _wait_for(done)
    assert len(runs) == 2, "the entry stored while its thread still ran was served"
    assert result == {"a": "A1", "b": "B2"}
    assert any("STORE-WORK-RUNNING" in str(w.message) for w in caught), [str(w.message) for w in caught]


def test_a_thread_the_call_joins_is_stored(disk_cash, tmp_path):
    a, b = _files(tmp_path)
    runs = []

    @disk_cash.cache(assume_safe=True)
    def load(pa, pb):
        runs.append(1)
        out = []
        t = threading.Thread(target=lambda: out.append(_read(pb)))
        t.start()
        t.join()
        return _read(pa) + out[0]

    assert load(str(a), str(b)) == "A1B1"
    assert load(str(a), str(b)) == "A1B1"
    assert len(runs) == 1
    b.write_text("B2", encoding="utf-8")
    assert load(str(a), str(b)) == "A1B2", "a file read on the thread is an input"


def _start_a_thread_that_cannot_start():
    """Start a thread while the OS refuses new threads."""

    def refuse(*args, **kwargs):
        raise RuntimeError("can't start new thread")

    names = [n for n in ("_start_joinable_thread", "_start_new_thread") if hasattr(threading, n)]
    saved = {n: getattr(threading, n) for n in names}
    for name in names:
        setattr(threading, name, refuse)
    try:
        threading.Thread(target=lambda: None).start()
    except RuntimeError:
        pass
    else:
        raise AssertionError("the thread started")
    finally:
        for name, value in saved.items():
            setattr(threading, name, value)


def test_a_thread_that_fails_to_start_is_not_counted(disk_cash, tmp_path):
    a, _b = _files(tmp_path)
    runs = []

    @disk_cash.cache(assume_safe=True)
    def load(pa):
        runs.append(1)
        _start_a_thread_that_cannot_start()
        return _read(pa)

    assert load(str(a)) == "A1"
    assert load(str(a)) == "A1"
    assert len(runs) == 1
