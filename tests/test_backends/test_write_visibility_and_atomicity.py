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


class TestAtomicWrites:
    """An entry is never half-written: the payload goes to a temp file and the
    target appears only at the rename.

    A plain ``open(path,'wb')`` is visible the instant it is created, while it
    still holds zero or half its bytes, so a concurrent reader can pass
    ``get()``'s ``exists`` check and unpickle a truncated file — the macOS
    ``EOFError: Ran out of input`` failures.
    """

    def test_target_is_untouched_when_the_rename_fails(self, tmp_path, monkeypatch):
        backend = FileBackend(cache_dir=str(tmp_path / "c"))
        backend._ensure_initialized()
        target = os.path.join(backend.cache_dir, "probe.data")

        monkeypatch.setattr(os, "replace",
                            lambda *a, **kw: (_ for _ in ()).throw(OSError("boom")))
        with pytest.raises(OSError):
            backend._atomic_write(target, b"x" * 4096, use_gzip=False)

        assert not os.path.exists(target), "payload was streamed into the live path"
        leftovers = [p for p in os.listdir(backend.cache_dir) if p.endswith(".part")]
        assert leftovers == [], f"partial write left behind: {leftovers}"

    def test_existing_entry_survives_a_failed_rewrite(self, tmp_path, monkeypatch):
        backend = FileBackend(cache_dir=str(tmp_path / "c"))
        backend._ensure_initialized()
        target = os.path.join(backend.cache_dir, "probe.data")
        backend._atomic_write(target, b"original", use_gzip=False)

        monkeypatch.setattr(os, "replace",
                            lambda *a, **kw: (_ for _ in ()).throw(OSError("boom")))
        with pytest.raises(OSError):
            backend._atomic_write(target, b"replacement", use_gzip=False)

        with open(target, "rb") as f:
            assert f.read() == b"original"

    def test_temp_files_are_invisible_to_the_entry_globs(self, tmp_path):
        """A stray ``.part`` must never be mistaken for a cache entry."""
        backend = FileBackend(cache_dir=str(tmp_path / "c"))
        backend.set("k", "v")
        assert backend.get("k")[1] == "v"
        import glob as _glob
        stray = os.path.join(backend.cache_dir, ".tmp-orphan.part")
        with open(stray, "wb") as f:
            f.write(b"junk")
        for ext in ("*.meta", "*.data"):
            assert stray not in _glob.glob(os.path.join(backend.cache_dir, ext))
        assert backend.get("k")[1] == "v"


class TestUnreadableEntryDegradesToMiss:
    """An entry that cannot be read is an entry that is absent.

    Writes are deliberately NOT atomic (see the note in _write_cache_files), so
    a concurrent reader can still open a data file between its creation and its
    bytes landing, and a cache directory can hold a partial file left by a
    killed process or a full disk. ``EOFError`` subclasses Exception directly —
    not OSError, not ValueError — so it escaped ``get()``'s handler and reached
    the user as "Ran out of input" instead of a recompute. That is what
    test_registry_identity and test_tiered hit on the macOS runners.

    Zero length, not half: a half-written pickle raises UnpicklingError, which
    was already caught. Only an empty file produces the actual CI signature.
    """

    def test_empty_data_file_is_a_miss(self, tmp_path):
        backend = FileBackend(cache_dir=str(tmp_path / "c"))
        backend.set("k", {"big": list(range(1000))})
        assert backend.get("k")[1] is not None  # settles the write

        _, data_path = backend._get_paths("k")
        with open(data_path, "wb"):
            pass
        backend._metadata_cache.pop("k", None)

        metadata, value = backend.get("k")
        assert value is None and metadata is None

    def test_empty_metadata_file_is_a_miss(self, tmp_path):
        backend = FileBackend(cache_dir=str(tmp_path / "c"))
        backend.set("k", "v")
        assert backend.get("k")[1] == "v"

        meta_path, _ = backend._get_paths("k")
        with open(meta_path, "wb"):
            pass
        backend._metadata_cache.pop("k", None)

        assert backend.get("k") == (None, None)

    def test_a_miss_lets_the_caller_recompute_and_rewrite(self, tmp_path):
        """Degrading must leave the cache usable, not poisoned."""
        backend = FileBackend(cache_dir=str(tmp_path / "c"))
        backend.set("k", "original")
        assert backend.get("k")[1] == "original"

        _, data_path = backend._get_paths("k")
        with open(data_path, "wb"):
            pass
        backend._metadata_cache.pop("k", None)
        assert backend.get("k") == (None, None)

        backend.set("k", "recomputed")
        assert backend.get("k")[1] == "recomputed"


def test_pickle_roundtrip_is_unaffected(tmp_path):
    """Guard against the write path corrupting ordinary values."""
    backend = FileBackend(cache_dir=str(tmp_path / "c"))
    payload = {"a": [1, 2, 3], "b": "x" * 10_000}
    backend.set("k", payload)
    assert backend.get("k")[1] == payload
    assert pickle.loads(pickle.dumps(payload)) == payload
