"""An intercepted call whose result cannot be copied is not served by
reference.

The RAM tier keeps a value it cannot copy -- one holding a lock or a
connection -- as the object itself, unless the entry requires a copy. The call cache did not ask, so
``st = build()`` run again handed back the very object of the first run,
with every change made to it since, where plain Python builds a fresh one.
"""

from __future__ import annotations

import pickle
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
    second = cached()
    third = cached()
    assert first is not second and second is not third
    assert not any(e["cache_hit"] for e in call_cache.drain_call_log())


def test_a_chain_too_deep_for_deepcopy_is_served_as_a_full_copy(call_cache):
    """Control: the RAM tier copies by a pickle round trip, which takes a
    chain deepcopy cannot, so a hit is served -- a fresh copy, whole.

    Where this Python's stack is too small for pickle to walk the chain too
    (3.14 on Windows), no copy can be made, so nothing is cached."""
    cached = _resolved(call_cache, build_chain)
    first = cached()
    second = cached()
    try:
        pickle.dumps(first)
    except RecursionError:
        assert [e["cache_hit"] for e in call_cache.drain_call_log()] == [False, False]
        assert first is not second
        return
    assert first is not second and first.next is not second.next
    length, node = 0, second
    while node is not None:
        length, node = length + 1, node.next
    assert length == 3000
    assert [e["cache_hit"] for e in call_cache.drain_call_log()] == [False, True]
