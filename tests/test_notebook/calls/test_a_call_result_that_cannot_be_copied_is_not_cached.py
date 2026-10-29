"""An intercepted call whose result holds a lock is never served by
reference.

The RAM tier kept a value it could not copy -- one holding a lock or a
connection -- as the object itself, unless the entry required a copy. The
call cache did not ask, so ``st = build()`` run again handed back the very
object of the first run, with every change made to it since, where plain
Python builds a fresh one. A free lock is now stored as a new free lock, so
the hit is a fresh copy, as a plain run builds.
"""

from __future__ import annotations

import threading
import time

import pytest

import cash
from cash.notebook.call_interception import CallSite
from tests._call_cache import make_call_cache
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


@pytest.fixture
def call_cache(tmp_path):
    return make_call_cache(cash.Cash(cache_dir=str(tmp_path / "cc")))


class Store:
    def __init__(self):
        self.lock = threading.Lock()
        self.rows = [1, 2]


class Node:
    def __init__(self, v, nxt):
        self.v = v
        self.next = nxt


def build_store():
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    return Store()


def build_chain():
    time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
    head = None
    for i in range(3000):
        head = Node(i, head)
    return head


def _resolved(call_cache, build):
    call_cache.set_sites([CallSite(source="build()", free_names=frozenset({build.__name__}), occurrence_index=0)])
    return call_cache.resolve(build)


def test_a_second_call_builds_a_fresh_object(call_cache):
    cached = _resolved(call_cache, build_store)
    first = cached()
    first.lock.acquire()
    first.rows.append(3)
    second = cached()
    third = cached()
    assert first is not second and second is not third
    for made in (second, third):
        assert made.lock is not first.lock and not made.lock.locked()
        assert made.rows == [1, 2] and made.rows is not first.rows
    assert [e["cache_hit"] for e in call_cache.drain_call_log()] == [False, True, True]


def test_a_chain_too_deep_for_deepcopy_is_served_as_a_full_copy(call_cache):
    """Control: the RAM tier copies a chain deepcopy cannot, so a hit is
    served -- a fresh copy, whole.

    Also where this Python's stack is too small for pickle to walk the chain
    (3.10 and 3.11, Windows): a chain too deep for the pickler is pickled
    flat, and copied without recursion."""
    cached = _resolved(call_cache, build_chain)
    first = cached()
    second = cached()
    assert first is not second and first.next is not second.next
    length, node = 0, second
    while node is not None:
        length, node = length + 1, node.next
    assert length == 3000
    assert [e["cache_hit"] for e in call_cache.drain_call_log()] == [False, True]
