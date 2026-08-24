"""Per-backend asynchronous write semantics.

Each "slow" backend (File, SQLite, Redis, S3) serializes on the calling
thread, then writes in its own background worker. RAM stays sync. The
contract is:

- ``set()`` returns before the actual storage write finishes.
- ``set()`` captures the *current* state of the value on the calling
  thread — mutating the value afterwards does NOT affect what gets
  cached.
- A ``get()`` that races a still-in-flight write for the same key waits
  for it, so we never see stale-or-missing reads.
- ``delete()`` drains any pending write for that key first.
- Background write failures are stored on the future and surface on the
  next ``get()``/``delete()`` for the same key.
- ``shutdown()`` blocks until every in-flight write is done.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.sqlite_backend import SQLiteBackend
from cash.backends.tiered_backend import TieredBackend


# ---------------------------------------------------------------------------
# Backend factory: parametrise across every slow backend that should be async
# ---------------------------------------------------------------------------

@pytest.fixture(params=["file", "sqlite", "redis"])
def slow_backend(request, tmp_path):
    """Yield a freshly-constructed slow backend for each test."""
    kind = request.param
    if kind == "file":
        b = FileBackend(str(tmp_path / "fb"), flush_interval=0)
    elif kind == "sqlite":
        b = SQLiteBackend(str(tmp_path / "c.db"))
    elif kind == "redis":
        fakeredis = pytest.importorskip("fakeredis")
        _redis = pytest.importorskip("redis")
        with patch.object(_redis, "Redis", fakeredis.FakeStrictRedis):
            from cash.backends.redis_backend import RedisBackend
            b = RedisBackend(prefix=f"cash:async:{kind}:")
    try:
        yield b
    finally:
        b.shutdown()


# ---------------------------------------------------------------------------
# Core contract
# ---------------------------------------------------------------------------

class TestSerializeOnCallingThread:
    """The value's state at the moment of set() must be captured before
    we return. Mutating it afterwards cannot corrupt the cached copy."""

    def test_post_set_mutation_does_not_corrupt_cache(self, slow_backend):
        value = {"counter": 0, "items": ["a", "b", "c"]}
        slow_backend.set("k", value)
        # Mutate aggressively after set() returns.
        value["counter"] = 9999
        value["items"].append("MUTATED")
        value["new_key"] = "evil"
        # The cached version should still be what we set, not the mutation.
        _, restored = slow_backend.get("k")
        assert restored == {"counter": 0, "items": ["a", "b", "c"]}


class TestSetReturnsBeforeWriteCompletes:
    """set() must not block for the duration of the write."""

    def test_set_returns_fast_for_slow_backend(self, slow_backend, monkeypatch):
        # Patch the slow backend's internal sync writer to sleep, so we can
        # tell whether set() actually waited for it.
        original_write = slow_backend._do_set_sync
        def slow_write(*args, **kwargs):
            # Generous: the window only has to outlast however long a loaded
            # runner takes to get back to the calling thread after set().
            time.sleep(2.0)
            return original_write(*args, **kwargs)
        monkeypatch.setattr(slow_backend, "_do_set_sync", slow_write)

        slow_backend.set("k", "value")

        # Assert the PROPERTY, not a stopwatch: if set() returned without
        # waiting, the write it scheduled is necessarily still in flight.
        #
        # This used to assert `elapsed < 0.1`, which measures the runner as
        # much as the code. It failed on contended Windows jobs at 0.109s and
        # again at 0.519s while the behaviour was perfectly correct — a shared
        # 2-4 vCPU box can stall a thread for longer than the thing being
        # timed. The pending-count check only fails if set() genuinely blocks
        # until the write completes, which is the regression worth catching,
        # and it stays valid however slow the machine is.
        assert slow_backend._writes.pending_count() > 0, (
            "set() returned only after its write finished — it should schedule "
            "the write and return"
        )


class TestReadAfterWriteConsistency:
    """A get() that races a pending write for the same key must wait."""

    def test_immediate_get_returns_just_set_value(self, slow_backend, monkeypatch):
        original_write = slow_backend._do_set_sync
        def slow_write(*args, **kwargs):
            time.sleep(0.3)
            return original_write(*args, **kwargs)
        monkeypatch.setattr(slow_backend, "_do_set_sync", slow_write)

        slow_backend.set("k", "the-value")
        _, restored = slow_backend.get("k")  # races the slow write
        assert restored == "the-value"

    def test_two_rapid_sets_same_key_serialize(self, slow_backend, monkeypatch):
        original_write = slow_backend._do_set_sync
        def slow_write(*args, **kwargs):
            time.sleep(0.15)
            return original_write(*args, **kwargs)
        monkeypatch.setattr(slow_backend, "_do_set_sync", slow_write)

        slow_backend.set("k", "first")
        slow_backend.set("k", "second")
        _, restored = slow_backend.get("k")
        assert restored == "second"


class TestDeleteDrainsPending:
    """delete() must wait for any pending write for the key before deleting,
    otherwise the write could fire after the delete and leave a ghost entry."""

    def test_delete_after_set_actually_deletes(self, slow_backend, monkeypatch):
        original_write = slow_backend._do_set_sync
        def slow_write(*args, **kwargs):
            time.sleep(0.3)
            return original_write(*args, **kwargs)
        monkeypatch.setattr(slow_backend, "_do_set_sync", slow_write)

        slow_backend.set("k", "value")
        slow_backend.delete("k")
        _, restored = slow_backend.get("k")
        assert restored is None


class TestShutdownWaitsForPending:
    """shutdown() must block until every in-flight write completes."""

    def test_shutdown_blocks_until_writes_finish(self, slow_backend, monkeypatch):
        write_finished = []
        original_write = slow_backend._do_set_sync
        def slow_write(*args, **kwargs):
            time.sleep(0.3)
            r = original_write(*args, **kwargs)
            write_finished.append(time.perf_counter())
            return r
        monkeypatch.setattr(slow_backend, "_do_set_sync", slow_write)

        slow_backend.set("k", "value")
        slow_backend.shutdown()
        shutdown_returned_at = time.perf_counter()
        # The write should have finished BEFORE shutdown returned.
        assert write_finished
        assert write_finished[0] <= shutdown_returned_at


# Induces write failures deliberately -- see the discarded-write guard in the
# root conftest.
@pytest.mark.expects_failed_writes
class TestFailureSurface:
    """Background write failures must surface on the next get() for that key.

    As a WARNING, not an exception. The value was computed successfully, and a
    cache that destroys it because its own write failed is worse than no cache
    -- which is the policy the decorator path has always followed via
    ``CashCacheStoreFailedWarning``. See ``PendingWrites.wait``.
    """

    def test_failed_write_warns_on_subsequent_get(self, slow_backend, monkeypatch):
        from cash.exceptions import CashCacheStoreFailedWarning

        def boom(*args, **kwargs):
            raise RuntimeError("simulated backend write failure")
        monkeypatch.setattr(slow_backend, "_do_set_sync", boom)

        slow_backend.set("bad-key", "value")  # returns; failure is in flight
        with pytest.warns(CashCacheStoreFailedWarning, match="simulated backend write failure"):
            slow_backend.get("bad-key")

    def test_a_failed_write_does_not_raise_into_the_caller(self, slow_backend, monkeypatch):
        """The regression that matters: get() must still return.

        Raising here killed a notebook cell before its variable was bound, so
        a cache write failure destroyed a computation that had succeeded.
        """
        def boom(*args, **kwargs):
            raise RuntimeError("simulated backend write failure")
        monkeypatch.setattr(slow_backend, "_do_set_sync", boom)

        slow_backend.set("bad-key", "value")
        with pytest.warns(Warning):
            result = slow_backend.get("bad-key")
        assert result is not None, "get() should still answer (as a miss)"


# ---------------------------------------------------------------------------
# RAM stays synchronous
# ---------------------------------------------------------------------------

class TestRAMStaysSync:
    """InMemoryBackend should NOT use an executor — RAM is too fast for the
    overhead to be worth it."""

    def test_no_background_executor_attribute(self):
        b = InMemoryBackend()
        # The marker the slow-backend mixin sets — RAM shouldn't have it.
        assert not hasattr(b, "_writes")


# ---------------------------------------------------------------------------
# Tiered backend cell-finish time
# ---------------------------------------------------------------------------

class TestTieredShutdownPropagates:
    """A script that exits via atexit calls Cash.shutdown → backend.shutdown.
    For TieredBackend (the default), that MUST cascade into every tier so
    each tier's PendingWrites is drained — otherwise a Python script can
    exit before the slow async write to S3/Redis lands, losing data."""

    def test_tiered_shutdown_drains_each_tier(self, tmp_path, monkeypatch):
        import time as _time
        ram = InMemoryBackend()
        disk = FileBackend(str(tmp_path / "fb"), flush_interval=0)
        tiered = TieredBackend([ram, disk], promotion_policy=lambda e, s: True)

        # Make the disk write slow so the test can prove shutdown waited.
        original = disk._do_set_sync
        write_done = []
        def slow(*args, **kwargs):
            _time.sleep(0.3)
            r = original(*args, **kwargs)
            write_done.append(_time.perf_counter())
            return r
        monkeypatch.setattr(disk, "_do_set_sync", slow)

        meta = {"execution_time": 5.0, "size": 100}
        tiered.set("k", "v", meta)
        tiered.shutdown()
        shutdown_returned_at = _time.perf_counter()

        assert write_done, "disk write must have completed"
        assert write_done[0] <= shutdown_returned_at, (
            "TieredBackend.shutdown returned before its disk tier's write finished"
        )


class TestTieredCellFinishTime:
    """A TieredBackend.set() with a slow async tier should return quickly —
    the cell-execution path is no longer blocked on the slow write."""

    def test_tiered_set_returns_fast_even_with_slow_disk(self, tmp_path, monkeypatch):
        ram = InMemoryBackend()
        disk = FileBackend(str(tmp_path / "fb"), flush_interval=0)
        tiered = TieredBackend([ram, disk], promotion_policy=lambda e, s: True)

        # Make the disk write take half a second.
        original = disk._do_set_sync
        def slow(*args, **kwargs):
            time.sleep(0.5)
            return original(*args, **kwargs)
        monkeypatch.setattr(disk, "_do_set_sync", slow)

        t0 = time.perf_counter()
        meta = {"execution_time": 5.0, "size": 100}  # promote past RAM
        tiered.set("k", "v", meta)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.1, f"tiered set() blocked for {elapsed:.3f}s"

        # Disk write happens in background but is still durable
        disk.shutdown()
        _, restored = disk.get("k")
        assert restored == "v"
