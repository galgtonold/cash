"""The per-call event log stays bounded when nothing drains it.

Every cached call appends an event to ``Cash._decorator_call_log`` for the
notebook badge, and only the notebook drains it. A script or a service never
does, so the log grew by one dict per call for as long as the process lived:
5,000 calls left 5,000 entries. ``cache_info()`` also rescanned that log after
every call to find the call's own entry, so its counts depended on the log
still holding it.
"""

from __future__ import annotations

import cash.core as core
from cash import Cash
from cash.backends.memory_backend import InMemoryBackend


def test_the_log_keeps_only_the_most_recent_calls(monkeypatch):
    monkeypatch.setattr(core, "_CALL_LOG_MAX", 50, raising=False)
    c = Cash(backend=InMemoryBackend(), register_magic=False)

    @c.cache
    def double(x):
        return 2 * x

    for _ in range(200):
        double(3)

    assert len(c._decorator_call_log) <= 50
    info = double.cache_info()
    assert (info["hits"], info["misses"]) == (199, 1)
