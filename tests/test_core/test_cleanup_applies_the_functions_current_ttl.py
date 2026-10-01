"""``Cash.cleanup`` removes what a call would no longer serve.

A call judges an entry by its function's ``ttl=`` as it stands now; cleanup
judged it by the ttl the entry was written with, so after ``ttl=`` was
lowered it kept entries every call already treated as expired.
"""

from __future__ import annotations

import time

import cash
from cash.backends.memory_backend import InMemoryBackend


def test_lowering_a_functions_ttl_lets_cleanup_remove_its_entries():
    c = cash.Cash(backend=InMemoryBackend())

    @c.cache(ttl=1000)
    def f(x):
        return x + 1

    f(1)
    ((meta, _),) = c.backend._store.values()
    meta["created_at"] = meta["timestamp"] = time.time() - 100
    assert c.cleanup() == 0, "control: under ttl=1000 the entry is fresh"

    @c.cache(ttl=10)
    def f(x):  # noqa: F811 - the same function with a lower ttl
        return x + 1

    assert c.cleanup() == 1
