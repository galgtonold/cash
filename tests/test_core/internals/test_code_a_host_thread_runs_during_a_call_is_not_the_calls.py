"""Code a host's own thread runs while a cached call is in progress is not
the call's.

A host registers the threads it runs beside the program that change none
of its values (`trace_recorder.HOST_THREADS`): the test runner's stall
watchdog, a notebook harness's kernel threads. `other_thread_alive` left
them out, but the recorder of a call's trace did not: the watchdog wakes
every few seconds, and when it woke in the middle of a miss its code
(``_StallWatchdog.poll_interval``, the ``min`` and ``max`` it reads) went
into the call's trace. The entry was then checked on what the watchdog
read, and the hit memo's scan of that trace refused it (a list the scan
cannot watch), so the call was never served from the fast path while the
world stood still: a test waiting for its fast hit ran out of time, now
and then, under load.

Here a registered host thread runs code of this module, reading a list,
in the middle of a miss. The entry is not checked on what it read and the
hit is served from the memo. A thread the body starts is still the call's.
"""

from __future__ import annotations

import os
import sys
import threading
import weakref

import pytest

from cash import Cash
from cash.decorator import hit_memo
from cash.tracking import trace_recorder

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(sys.version_info < (3, 12), reason="the recorder runs on sys.monitoring, Python 3.12 or newer"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

#: What the host thread reads: a list no watcher sees changed.
POLLED: list[int] = [1]
#: Body runs.
RUNS = {"n": 0}
#: The pipes a body wakes the watchdog through: (ask, done), each (read, write).
PIPES: dict[str, int] = {}


class Watchdog:
    """A host's thread: polls when woken, as the stall watchdog does when
    its timer runs out."""

    @property
    def poll_interval(self) -> float:
        return min(5.0, max(0.05, len(POLLED) / 4.0))

    def run(self, ask: int, done: int) -> None:
        while os.read(ask, 1) == b"a":
            self.poll_interval  # the poll's own code, run on this thread
            os.write(done, b"d")


def wake_the_watchdog():
    """The watchdog's timer runs out mid-call: it polls while the body waits."""
    os.write(PIPES["ask_w"], b"a")
    os.read(PIPES["done_r"], 1)


@pytest.fixture
def watchdog(monkeypatch):
    ask_r, ask_w = os.pipe()
    done_r, done_w = os.pipe()
    PIPES.update(ask_w=ask_w, done_r=done_r)
    thread = threading.Thread(target=Watchdog().run, args=(ask_r, done_w), name="host-watchdog", daemon=True)
    thread.start()
    hosts = weakref.WeakSet(trace_recorder.HOST_THREADS)
    hosts.add(thread)
    monkeypatch.setattr(trace_recorder, "HOST_THREADS", hosts)
    yield thread
    os.write(ask_w, b"q")
    thread.join(10)
    assert not thread.is_alive()
    for fd in (ask_r, ask_w, done_r, done_w):
        os.close(fd)
    PIPES.clear()


@pytest.fixture
def c():
    return Cash(backend="memory", register_magic=False)


def _polled_mid_call(x):
    RUNS["n"] += 1
    wake_the_watchdog()
    return x * 2


def test_what_the_host_thread_read_is_not_checked_and_the_call_hits(c, watchdog):
    RUNS["n"] = 0
    f = c.cache(_polled_mid_call)
    try:
        assert f(3) == 6
        POLLED.append(2)  # what the host thread read moves
        assert f(3) == 6
        assert RUNS["n"] == 1, f.cache_info()["miss_reasons"]
    finally:
        del POLLED[1:]


@pytest.mark.skipif(not hit_memo.ENABLED, reason="the fast hit path needs the C extension")
def test_the_hit_is_served_from_the_memo(c, watchdog):
    RUNS["n"] = 0
    f = c.cache(_polled_mid_call)
    assert f(2) == 4
    served = 0
    for _ in range(20):
        before = hit_memo.status()["fast_hits"]
        assert f(2) == 4
        served += hit_memo.status()["fast_hits"] > before
    assert RUNS["n"] == 1
    assert served, "a hit after a miss a host thread ran code in was never served from the memo"


def read_polled(x):
    return x + len(POLLED)


def _handed(x):
    RUNS["n"] += 1
    out = []
    worker = threading.Thread(target=lambda: out.append(read_polled(x)))
    worker.start()
    worker.join()
    return out[0]


def test_work_the_body_hands_to_a_thread_of_its_own_is_still_the_calls(c, watchdog):
    RUNS["n"] = 0
    f = c.cache(_handed)
    try:
        assert f(1) == 2
        assert f(1) == 2
        assert RUNS["n"] == 1
        POLLED.append(2)
        assert f(1) == 3
        assert RUNS["n"] == 2
    finally:
        del POLLED[1:]
