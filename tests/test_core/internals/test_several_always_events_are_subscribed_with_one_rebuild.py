"""Several events delivered always are subscribed with one rebuild.

The write queues' child-process watch subscribes a dozen events, each of
which rebuilt both dispatch tables and told the C hook twice: part of what
the first call of every process paid. They are now subscribed together.
"""

from __future__ import annotations

from cash.tracking import io_watch


def test_one_rebuild_subscribes_every_event_named(monkeypatch):
    rebuilds = []
    real = io_watch._rebuild
    monkeypatch.setattr(io_watch, "_rebuild", lambda: rebuilds.append(1) or real())
    seen = []

    def consumer(args):
        seen.append(args)

    events = ("cash.test.one", "cash.test.two")
    io_watch.subscribe_always(events, consumer)
    try:
        assert len(rebuilds) == 1
        for event in events:
            assert consumer in io_watch._always[event]
            assert consumer in io_watch._held_table[event]
            assert consumer in io_watch._idle_table[event]
    finally:
        with io_watch._lock:
            for event in events:
                io_watch._always.pop(event, None)
            real()
