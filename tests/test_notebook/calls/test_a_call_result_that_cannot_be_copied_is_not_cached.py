"""An intercepted call whose result cannot be copied is not served by
reference.

The RAM tier keeps a value it cannot deep-copy -- one holding a lock or a
connection, a chain of objects too deep for deepcopy -- as the object
itself, unless the entry requires a copy. The call cache did not ask, so
``st = build()`` run again handed back the very object of the first run,
with every change made to it since, where plain Python builds a fresh one.
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


@pytest.mark.parametrize("build", [build_store, build_chain], ids=["a lock", "a deep chain"])
def test_a_second_call_builds_a_fresh_object(call_cache, build):
    call_cache.set_sites([CallSite(source="build()", free_names=frozenset({build.__name__}), occurrence_index=0)])
    cached = call_cache.resolve(build)
    first = cached()
    second = cached()
    third = cached()
    assert first is not second and second is not third
    assert not any(e["cache_hit"] for e in call_cache.drain_call_log())
