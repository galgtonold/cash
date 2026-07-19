"""Durability is a property of the kernel, not of one Cash instance.

A notebook routinely holds more than one: the ``%cash_on`` instance, plus any
``Cash(...)`` the user builds in a cell — and decorator writes go to the latter.
Anything that flushes "the pending writes" therefore has to reach every queue in
the process. Flushing only the magic's own backend drains nothing that matters,
which is exactly how the first attempt at this failed (CAS-209).
"""
from __future__ import annotations

import gc
import threading

from cash.backends._base import PendingWrites, all_pending_writes
from cash.backends.file_backend import FileBackend


class TestRegistry:
    def test_a_new_queue_is_visible(self):
        pw = PendingWrites()
        try:
            assert pw in all_pending_writes()
        finally:
            pw.shutdown(wait=True)

    def test_a_shutdown_queue_drops_out(self):
        pw = PendingWrites()
        pw.shutdown(wait=True)
        assert pw not in all_pending_writes()
        assert pw.is_shutdown() is True

    def test_the_registry_holds_queues_weakly(self):
        """Registering strongly would pin every backend and its thread pool."""
        pw = PendingWrites()
        assert pw in all_pending_writes()
        del pw
        gc.collect()
        # Nothing to assert about a specific object once it is gone; the
        # contract is that collection is possible at all, which the WeakSet
        # allows and a plain set would not.
        assert all(q is not None for q in all_pending_writes())

    def test_queues_from_separate_backends_are_all_reachable(self, tmp_path):
        """The multi-instance case the flush exists for."""
        a = FileBackend(cache_dir=str(tmp_path / "a"))
        b = FileBackend(cache_dir=str(tmp_path / "b"))
        live = all_pending_writes()
        assert a._writes in live and b._writes in live


class TestDrainingEveryQueue:
    def test_draining_all_queues_settles_writes_from_every_backend(self, tmp_path, monkeypatch):
        """What the post_run_cell handler does, exercised directly."""
        started = threading.Event()
        real = FileBackend._do_set_sync

        def slow(self, *a, **kw):
            started.set()
            import time
            time.sleep(0.4)
            return real(self, *a, **kw)

        monkeypatch.setattr(FileBackend, "_do_set_sync", slow)

        a = FileBackend(cache_dir=str(tmp_path / "a"))
        b = FileBackend(cache_dir=str(tmp_path / "b"))
        a.set("ka", "va")
        b.set("kb", "vb")
        assert started.wait(timeout=10)

        for queue in all_pending_writes():
            queue.wait_all()

        assert a._writes.pending_count() == 0
        assert b._writes.pending_count() == 0
        # And the values are readable, i.e. actually on disk.
        assert a.get("ka")[1] == "va"
        assert b.get("kb")[1] == "vb"
