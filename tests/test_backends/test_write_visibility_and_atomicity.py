"""The durability boundary is the cache DIRECTORY, not the backend instance.

Two ``FileBackend`` objects over one directory are two views of one store.
Writes are asynchronous, and ``get()`` used to wait only on its OWN queue, so a
second instance reading a directory the first was still writing saw a
half-populated cache and reported clean misses. On the CI runners only 2-3 of 19
entries had landed when the second instance started reading; the same window let
a reader open a data file mid-write and crash with ``EOFError: Ran out of
input`` instead of degrading to a miss.

These tests force that window deterministically rather than relying on timing,
so they fail on every platform when the behaviour regresses. The original
regression was invisible on Windows and only showed up on Linux and macOS.
"""
from __future__ import annotations

import os
import pickle
import threading
import time

import pytest

from cash.backends._base import PendingWrites
from cash.backends.file_backend import FileBackend


SLOW = 0.5  # long enough that an unsynchronised reader loses the race every time


@pytest.fixture
def slow_writes(monkeypatch):
    """Make every background disk write take SLOW seconds."""
    real = FileBackend._do_set_sync

    def slow(self, *a, **kw):
        time.sleep(SLOW)
        return real(self, *a, **kw)

    monkeypatch.setattr(FileBackend, "_do_set_sync", slow)


class TestCrossInstanceVisibility:
    def test_second_instance_waits_for_first_instances_pending_write(
        self, tmp_path, slow_writes,
    ):
        """A sibling backend must observe a write that is still in flight."""
        cache_dir = str(tmp_path / "c")
        writer = FileBackend(cache_dir=cache_dir)
        writer.set("k", {"payload": 42})

        # The write is still in the writer's queue here; nothing is on disk yet.
        reader = FileBackend(cache_dir=cache_dir)
        metadata, value = reader.get("k")

        assert value == {"payload": 42}, (
            "a second FileBackend on the same directory missed an entry the "
            "first was still writing - get() is waiting only on its own queue"
        )
        assert metadata is not None

    def test_sibling_visibility_survives_path_spelling(self, tmp_path, slow_writes):
        """Sibling lookup is by resolved directory, not by the string given.

        macOS hands out ``/var/...`` temp paths that resolve to ``/private/var/...``;
        a plain string key would treat the two spellings as different caches.
        """
        cache_dir = str(tmp_path / "c")
        os.makedirs(cache_dir, exist_ok=True)
        writer = FileBackend(cache_dir=cache_dir)
        writer.set("k", "v")

        reader = FileBackend(cache_dir=os.path.join(cache_dir, ".", ""))
        assert reader.get("k")[1] == "v"

    def test_registry_holds_writers_weakly(self, tmp_path):
        """A collected queue must drop out of the registry.

        Registering strongly would pin every backend (and its thread pool) for
        the life of the process. This exercises the registry directly rather
        than through FileBackend, because a live backend keeps its own flusher
        thread running and so is legitimately not collectable — that lifecycle
        is a separate concern from the weak-reference contract under test.
        """
        import gc

        from cash.backends.file_backend import _register_writer, _sibling_writers

        cache_dir = str(tmp_path / "c")
        os.makedirs(cache_dir, exist_ok=True)
        anchor = PendingWrites()
        doomed = PendingWrites()
        _register_writer(cache_dir, anchor)
        _register_writer(cache_dir, doomed)
        assert doomed in _sibling_writers(cache_dir, anchor)

        del doomed
        gc.collect()
        assert _sibling_writers(cache_dir, anchor) == []


class TestWorkerThreadMarker:
    def test_marker_is_set_inside_a_task_and_clear_outside(self):
        """The guard that stops a write worker blocking on a sibling queue."""
        seen = {}
        pw = PendingWrites()
        try:
            assert PendingWrites.in_worker_thread() is False
            pw.submit("k", lambda: seen.update(inside=PendingWrites.in_worker_thread()))
            pw.wait("k")
            assert seen["inside"] is True
            assert PendingWrites.in_worker_thread() is False
        finally:
            pw.shutdown(wait=True)

    def test_reader_in_worker_thread_does_not_deadlock(self, tmp_path, slow_writes):
        """Two backends over one directory, each reading from its worker.

        If a worker waited on a sibling's queue, these could block on each
        other forever. Guarded by ``PendingWrites.in_worker_thread()``.
        """
        cache_dir = str(tmp_path / "c")
        a = FileBackend(cache_dir=cache_dir)
        b = FileBackend(cache_dir=cache_dir)
        a.set("ka", "va")
        b.set("kb", "vb")

        done = threading.Event()

        def cross_read():
            a.get("kb")
            b.get("ka")
            done.set()

        t = threading.Thread(target=cross_read, daemon=True)
        t.start()
        assert done.wait(timeout=30), "cross-instance read deadlocked"


def test_pickle_roundtrip_still_works_through_the_atomic_path(tmp_path):
    """Guard against the atomic path corrupting ordinary writes."""
    backend = FileBackend(cache_dir=str(tmp_path / "c"))
    payload = {"a": [1, 2, 3], "b": "x" * 10_000}
    backend.set("k", payload)
    assert backend.get("k")[1] == payload
    assert pickle.loads(pickle.dumps(payload)) == payload
