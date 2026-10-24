"""A hit is not remembered for a list another thread changed while it ran.

The memo remembers an ordinary hit under the token the watchers give its
list. It read that token after the call: a thread that changed the list
and called the function in between had its call give the list a new
token, and the first call's result, built for the old content, was filed
under it. The next call was served the old result. The token is now read
before the key is built, and a hit is remembered only if it is still the
same afterwards.

The key of a list of plain values is a digest of its content, so it comes
back when the content is put back: a hit keyed from the content in between
was filed under it. Each arm of a watched value has a number never handed
out again (`cwatch.keyed`), and that is compared too.
"""

from __future__ import annotations

import queue
import threading

import pytest

from cash import Cash
from cash.decorator import hit_memo

pytestmark = [
    pytest.mark.skipif(not hit_memo.ENABLED, reason="the fast hit path needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]


class Rec:
    def __init__(self, x):
        self.x = x


def _records():
    return [Rec(i) for i in range(500)]


def _set_records(lst, value):
    lst[5].x = value


def _ints():
    return list(range(500))


def _set_ints(lst, value):
    lst[5] = value


def _sum_records(lst):
    return sum(p.x for p in lst)


def _many_ints():
    return list(range(5000))  # past what the memo key holds by value: watched


KINDS = {
    "records": (_records, _set_records, _sum_records),
    "ints": (_ints, _set_ints, sum),
    "many ints": (_many_ints, _set_ints, sum),
}


class Worker:
    """A thread that calls on request: started before the memo is warm, as
    a thread started later moves the epochs the memo is keyed by."""

    def __init__(self):
        self.jobs: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while (job := self.jobs.get()) is not None:
            fn, done = job
            done.append(fn())

    def call(self, fn) -> list:
        done: list = []
        self.jobs.put((fn, done))
        return done

    def stop(self):
        self.jobs.put(None)
        self.thread.join(10)


@pytest.mark.parametrize("kind", KINDS.values(), ids=KINDS.keys())
def test_a_hit_paused_while_another_thread_changes_the_list(tmp_path, watching, monkeypatch, kind):
    make, set_item, truth = kind

    def total(lst):
        return truth(lst)

    worker = Worker()
    f = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False).cache(total)
    lst = make()
    watching["lst"] = lst
    changed = make()
    set_item(changed, 9)
    f(changed)
    f(changed)  # the changed content's entry, from an equal copy
    for _ in range(8):
        f(lst)  # the first content's entry, and the memo warm
    before = hit_memo.status()["fast_hits"]
    assert f(lst) == truth(lst)
    assert hit_memo.status()["fast_hits"] > before, "the list never took the fast path"
    hit_memo.bump()  # the next call takes the ordinary path, and offers

    paused, resume = threading.Event(), threading.Event()
    real = hit_memo.HitMemo._entry

    def entry(self, call, hit):
        if threading.current_thread() is worker.thread and not paused.is_set():
            paused.set()
            assert resume.wait(10)
        return real(self, call, hit)

    monkeypatch.setattr(hit_memo.HitMemo, "_entry", entry)
    try:
        result = worker.call(lambda: f(lst))
        assert paused.wait(10), "the worker's hit was never offered"
        set_item(lst, 9)
        assert f(lst) == truth(lst)  # an ordinary hit of the changed content
    finally:
        resume.set()
        worker.stop()
    assert result == [truth(make())]

    assert f(lst) == truth(lst)
    assert f(lst) == truth(lst)


@pytest.mark.parametrize("kind", KINDS.values(), ids=KINDS.keys())
def test_a_hit_keyed_while_the_list_was_changed_and_put_back(tmp_path, watching, monkeypatch, kind):
    make, set_item, truth = kind

    def total(lst):
        return truth(lst)

    keyed, keyed_on = threading.Event(), threading.Event()
    offered, offer_on = threading.Event(), threading.Event()
    real_epochs, real_entry = hit_memo.epochs, hit_memo.HitMemo._entry
    pausing: list = []

    def epochs(spec, args, kwargs):
        found = real_epochs(spec, args, kwargs)
        if pausing and threading.current_thread() is pausing[0] and not keyed.is_set():
            keyed.set()
            assert keyed_on.wait(10)
        return found

    def entry(self, call, hit):
        if pausing and threading.current_thread() is pausing[0] and not offered.is_set():
            offered.set()
            assert offer_on.wait(10)
        return real_entry(self, call, hit)

    # Patched before the memo is warm: a module's attribute set moves the
    # world epoch the memo is keyed by.
    monkeypatch.setattr(hit_memo, "epochs", epochs)
    monkeypatch.setattr(hit_memo.HitMemo, "_entry", entry)
    worker = Worker()
    f = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False).cache(total)
    lst = make()
    watching["lst"] = lst
    changed = make()
    set_item(changed, 9)
    f(changed)
    f(changed)  # the changed content's entry, from an equal copy
    for _ in range(8):
        f(lst)  # the first content's entry, and the memo warm
    hit_memo.bump()

    pausing.append(worker.thread)
    try:
        result = worker.call(lambda: f(lst))
        assert keyed.wait(10), "the worker never took the epochs"
        set_item(lst, 9)  # the worker's key is built from this content
        keyed_on.set()
        assert offered.wait(10), "the worker's hit was never offered"
        set_item(lst, 5)  # and put back before it offers
        assert f(lst) == truth(lst)
    finally:
        keyed_on.set()
        offer_on.set()
        worker.stop()
    assert result == [truth(changed)]

    assert f(lst) == truth(lst)
    assert f(lst) == truth(lst)


@pytest.mark.parametrize("kind", KINDS.values(), ids=KINDS.keys())
def test_a_hit_keyed_while_short_threads_changed_and_put_back_the_list(watching, monkeypatch, kind):
    """As above with the call in this thread, a memory cache (no thread of
    cash's own) and each write made by a thread that has ended by the
    offer, which the count of threads is made to show: the copy check runs
    whoever is alive."""
    make, set_item, truth = kind

    def total(lst):
        return truth(lst)

    real_epochs, real_entry = hit_memo.epochs, hit_memo.HitMemo._entry
    armed: list = []
    lst = make()

    def write(value):
        writer = threading.Thread(target=set_item, args=(lst, value))
        writer.start()
        writer.join(10)

    def epochs(spec, args, kwargs):
        found = real_epochs(spec, args, kwargs)
        if armed == ["epochs"]:
            armed[:] = ["entry"]
            write(9)  # the key is built from this content
        return found

    def entry(self, call, hit):
        if armed == ["entry"]:
            armed.clear()
            write(5)  # and put back before the offer
        return real_entry(self, call, hit)

    monkeypatch.setattr(hit_memo, "epochs", epochs)
    monkeypatch.setattr(hit_memo.HitMemo, "_entry", entry)
    monkeypatch.setattr(threading, "active_count", lambda: 1)
    f = Cash(backend="memory", register_magic=False).cache(total)
    watching["lst"] = lst
    changed = make()
    set_item(changed, 9)
    f(changed)
    f(changed)
    for _ in range(8):
        f(lst)
    hit_memo.bump()

    armed.append("epochs")
    assert f(lst) == truth(changed)  # served for the content it was keyed by
    assert not armed
    assert f(lst) == truth(lst)
    assert f(lst) == truth(lst)
