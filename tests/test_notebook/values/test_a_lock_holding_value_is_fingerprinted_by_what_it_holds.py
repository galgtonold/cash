"""A lock-holding object, or a value holding one, is fingerprinted by what
it holds: it pickles to nothing a change moves, so ``H['e'].set()``,
``B.q.put(1)``, ``L.acquire()`` or ``MQ.put(1)`` in a callee was taken for
no change, and the statement was served with the change lost."""

from __future__ import annotations

import asyncio
import multiprocessing
import queue
import threading
import time

import pytest

from cash.mutation_fingerprint import mutation_fingerprint


class Box:
    pass


def _make(kind):
    if kind == "event_in_a_dict":
        held = {"e": threading.Event()}
        return held, lambda: held["e"].set()
    if kind == "queue_on_a_class":
        Box.q = queue.Queue()
        return Box, lambda: Box.q.put(1)
    if kind == "queue_on_an_object":
        box = Box()
        box.q = queue.Queue()
        return box, lambda: box.q.put(1)
    if kind == "event_two_levels_down":
        held = [{"e": threading.Event()}]
        return held, lambda: held[0]["e"].set()
    if kind == "lock":
        lock = threading.Lock()
        return lock, lambda: lock.acquire(blocking=False)
    if kind == "rlock":
        lock = threading.RLock()
        return lock, lambda: lock.acquire(blocking=False)
    if kind == "semaphore":
        sem = threading.Semaphore(3)
        return sem, lambda: sem.acquire(blocking=False)
    if kind == "condition":
        cond = threading.Condition()
        return cond, lambda: cond.acquire()
    if kind == "asyncio_queue":
        aq = asyncio.Queue()
        return aq, lambda: aq.put_nowait(1)
    if kind == "process_queue":
        mq = multiprocessing.Queue()
        return mq, lambda: (mq.put(1), time.sleep(0.2))
    raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind",
    [
        "event_in_a_dict",
        "queue_on_a_class",
        "queue_on_an_object",
        "event_two_levels_down",
        "lock",
        "rlock",
        "semaphore",
        "condition",
        "asyncio_queue",
        "process_queue",
    ],
)
def test_a_change_moves_the_fingerprint(kind):
    value, change = _make(kind)
    before = mutation_fingerprint(value)
    assert before is not None
    if kind != "process_queue":  # macOS cannot count it: new every time there
        assert mutation_fingerprint(value) == before
    change()
    assert mutation_fingerprint(value) != before


def test_a_value_holding_none_is_read_as_before():
    assert mutation_fingerprint({"a": [1, 2]}) == mutation_fingerprint({"a": [1, 2]})
    assert mutation_fingerprint({"a": [1, 2]}) != mutation_fingerprint({"a": [1, 3]})
