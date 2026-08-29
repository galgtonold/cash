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
from cash.backends.entry_format import ENTRY_SUFFIX, pack_entry, read_entry


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
        target = os.path.join(backend.cache_dir, f"probe{ENTRY_SUFFIX}")

        monkeypatch.setattr(os, "replace",
                            lambda *a, **kw: (_ for _ in ()).throw(OSError("boom")))
        with pytest.raises(OSError):
            backend._atomic_write(target, b"x" * 4096)

        assert not os.path.exists(target), "payload was streamed into the live path"
        leftovers = [p for p in os.listdir(backend.cache_dir) if p.endswith(".part")]
        assert leftovers == [], f"partial write left behind: {leftovers}"

    def test_existing_entry_survives_a_failed_rewrite(self, tmp_path, monkeypatch):
        backend = FileBackend(cache_dir=str(tmp_path / "c"))
        backend._ensure_initialized()
        target = os.path.join(backend.cache_dir, f"probe{ENTRY_SUFFIX}")
        backend._atomic_write(target, b"original")

        monkeypatch.setattr(os, "replace",
                            lambda *a, **kw: (_ for _ in ()).throw(OSError("boom")))
        with pytest.raises(OSError):
            backend._atomic_write(target, b"replacement")

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
        assert stray not in _glob.glob(
            os.path.join(backend.cache_dir, f"*{ENTRY_SUFFIX}"))
        assert backend.get("k")[1] == "v"


class TestUnreadableEntryDegradesToMiss:
    """An entry that cannot be read is an entry that is absent.

    A cache directory can hold a partial file left by a killed process or a
    full disk. ``EOFError`` subclasses Exception directly -- not OSError, not
    ValueError -- so it escaped ``get()``'s handler and reached the user as
    "Ran out of input" instead of a recompute. That is what
    test_registry_identity and test_tiered hit on the macOS runners.

    Two shapes, now that an entry is one file: a zero-length file, which loses
    the header too, and a file cut back to its header and metadata, which
    reads as a valid entry whose payload is empty. A half-written pickle
    raises UnpicklingError, which was always caught; only these produce the CI
    signature.
    """

    def test_empty_entry_file_is_a_miss(self, tmp_path):
        backend = FileBackend(cache_dir=str(tmp_path / "c"))
        backend.set("k", {"big": list(range(1000))})
        assert backend.get("k")[1] is not None  # settles the write

        with open(backend._get_path("k"), "wb"):
            pass
        backend._metadata_cache.pop("k", None)

        metadata, value = backend.get("k")
        assert value is None and metadata is None

    def test_an_entry_truncated_to_its_metadata_is_a_miss(self, tmp_path):
        """The header and metadata survive; the payload does not.

        The arm the single-file layout added: the entry still parses, so
        nothing upstream reports a problem, and the empty payload has to be
        caught where the value is deserialized.
        """
        backend = FileBackend(cache_dir=str(tmp_path / "c"))
        backend.set("k", {"big": list(range(1000))})
        assert backend.get("k")[1] is not None

        path = backend._get_path("k")
        _meta, payload = read_entry(path, with_payload=True)
        assert payload, "fixture is wrong: there was no payload to remove"
        with open(path, "r+b") as fh:
            fh.truncate(os.path.getsize(path) - len(payload))
        backend._metadata_cache.pop("k", None)

        assert backend.get("k") == (None, None)

    def test_a_miss_lets_the_caller_recompute_and_rewrite(self, tmp_path):
        """Degrading must leave the cache usable, not poisoned."""
        backend = FileBackend(cache_dir=str(tmp_path / "c"))
        backend.set("k", "original")
        assert backend.get("k")[1] == "original"

        with open(backend._get_path("k"), "wb"):
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


class TestAFailedWriteDoesNotDestroyWhatWasThere:
    """What writing through a temp file is actually FOR.

    The destination is untouched until the rename, so a write that fails --
    a full disk, a denied replace, a killed process -- leaves whatever was
    cached before exactly where it was. That is the guarantee; "no torn
    reads" is a consequence of it.

    It was not pinned by anything. Making ``_atomic_write`` write straight to
    the destination failed only two tests about the replace RETRY and three
    about file-dependency tracking; nothing asserted the guarantee itself, so
    it could have been optimised away silently.
    """

    @pytest.mark.expects_failed_writes
    def test_a_failed_rewrite_leaves_the_previous_value_readable(
        self, tmp_path, monkeypatch,
    ):
        backend = FileBackend(cache_dir=str(tmp_path / "c"), flush_interval=0)
        backend.set("k", "v1")
        backend._writes.wait_all()
        assert backend.get("k")[1] == "v1"

        def denied(tmp, dest):
            raise OSError("disk full")

        monkeypatch.setattr(FileBackend, "_replace_with_retry", staticmethod(denied))

        backend.set("k", "v2")
        backend._writes.wait_all()          # this write fails
        backend._metadata_cache.pop("k", None)   # force a read from disk

        assert backend.get("k")[1] == "v1", (
            "a failed rewrite destroyed the value that was already cached -- "
            "the destination was never overwritten, so there was nothing to "
            "clean up and the old entry should still be there"
        )

    @pytest.mark.expects_failed_writes
    def test_a_failed_first_write_leaves_no_entry_behind(self, tmp_path, monkeypatch):
        """The control: with nothing to preserve, a failure leaves nothing.

        Without this, simply never cleaning up would pass the arm above while
        leaving half-written files in the directory.
        """
        backend = FileBackend(cache_dir=str(tmp_path / "c"), flush_interval=0)

        def denied(tmp, dest):
            raise OSError("disk full")

        monkeypatch.setattr(FileBackend, "_replace_with_retry", staticmethod(denied))

        backend.set("fresh", "v1")
        backend._writes.wait_all()

        assert backend.get("fresh") == (None, None)
        leftovers = [f for f in os.listdir(backend.cache_dir)
                     if f.endswith(ENTRY_SUFFIX) or f.endswith(".part")]
        assert leftovers == [], f"partial files left behind: {leftovers}"

    @pytest.mark.expects_failed_writes
    def test_a_write_that_dies_halfway_leaves_the_previous_value_readable(
        self, tmp_path, monkeypatch,
    ):
        """The disk fills up mid-write. The entry that was already there survives.

        This is the arm that pins ATOMICITY rather than the cleanup around it:
        it fails the payload write itself, so it distinguishes writing through
        a temp file from writing straight to the destination. The sibling arm
        above fails the rename, which a direct write would never reach.

        Not a concurrency test, and deliberately so. The in-process race is
        already impossible -- ``_wait_for_writes`` makes a reader wait for any
        live write to the same directory -- so what remains is a writer that
        stops halfway: a full disk, a killed kernel, a denied replace. That
        needs no threads and no timing to reproduce.
        """
        import cash.backends.file_backend as fb

        backend = FileBackend(cache_dir=str(tmp_path / "c"), flush_interval=0)
        backend.set("k", "v1")
        backend._writes.wait_all()
        assert backend.get("k")[1] == "v1"

        real_open = open
        armed = {"on": False}

        class _DiesHalfway:
            def __init__(self, fh):
                self._fh = fh

            def write(self, data):
                self._fh.write(data[:len(data) // 2])
                raise OSError("No space left on device")

            def __getattr__(self, name):
                return getattr(self._fh, name)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return self._fh.__exit__(*exc)

        def failing_open(path, mode="r", *a, **kw):
            fh = real_open(path, mode, *a, **kw)
            if armed["on"] and "w" in mode and "b" in mode:
                return _DiesHalfway(fh)
            return fh

        monkeypatch.setattr(fb, "open", failing_open, raising=False)
        armed["on"] = True
        backend.set("k", "v2")
        backend._writes.wait_all()
        armed["on"] = False

        backend._metadata_cache.pop("k", None)   # force a read from disk
        assert backend.get("k")[1] == "v1", (
            "a write that died halfway destroyed the value that was already "
            "cached; the destination must not be touched until the swap"
        )
