"""Cache-cap safety behaviors (CAS-142, second half of the ticket).

Two guards keep a too-small cap from silently making cash *slower* than no
cache (the write-and-evict treadmill from the friction log):

* **oversize refusal** — a single object larger than half a persistent
  tier's cap is skipped (kept in RAM) rather than written-then-evicted, and
  the tiered backend warns once with an actionable message;
* **evict-after-write** — if the disk backend evicts an entry within a couple
  of writes of storing it, the cache can't retain the working set, so it
  warns once/session.

Plus the headline acceptance test: two medium objects that together bust the
*old* 1 GiB cap both persist and restore under the new adaptive disk cap — the
treadmill is gone. The baseline half of that test reproduces the treadmill on
a small cap to prove the assertion discriminates.
"""
from __future__ import annotations

import pytest

from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend
from cash.exceptions import CashCacheIneffectiveWarning


# ---------------------------------------------------------------------------
# Oversize refusal
# ---------------------------------------------------------------------------

class TestOversizeRefusal:
    def test_object_over_half_cap_is_refused_and_warns_once(self, tmp_path):
        disk = FileBackend(str(tmp_path / "c"), max_size_bytes=8000, flush_interval=0)
        tiered = TieredBackend(
            [InMemoryBackend(), disk],
            promotion_policy=lambda exec_t, size: True,  # clear the compute floor
        )
        big = "x" * 5000  # > 4000 = half the 8000-byte cap → must be refused
        meta = {"execution_time": 2.0, "size": len(big)}

        with pytest.warns(CashCacheIneffectiveWarning) as rec:
            tiered.set("big", big, meta)
            # A second, distinct oversize object must NOT emit a second warning.
            tiered.set("big2", "y" * 6000, {"execution_time": 2.0, "size": 6000})

        # Refused: it never reached disk, only RAM.
        assert meta["storage"] == ["RAM"]
        assert disk.get("big") == (None, None)
        assert disk.get("big2") == (None, None)

        oversize = [w for w in rec if issubclass(w.category, CashCacheIneffectiveWarning)]
        assert len(oversize) == 1, "warn once/session, not per object"
        assert "max_cache_size" in str(oversize[0].message)  # actionable
        disk.shutdown()

    def test_object_under_half_cap_still_persists(self, tmp_path):
        disk = FileBackend(str(tmp_path / "c"), max_size_bytes=8000, flush_interval=0)
        tiered = TieredBackend(
            [InMemoryBackend(), disk],
            promotion_policy=lambda exec_t, size: True,
        )
        ok = "x" * 3000  # < 4000 half-cap → persists normally
        meta = {"execution_time": 2.0, "size": len(ok)}
        tiered.set("ok", ok, meta)
        disk._writes.wait_all()
        assert "DISK" in meta["storage"]
        _, restored = disk.get("ok")
        assert restored == ok
        disk.shutdown()


# ---------------------------------------------------------------------------
# Evict-after-write
# ---------------------------------------------------------------------------

class TestEvictAfterWrite:
    def test_warns_once_when_recent_entry_evicted(self, tmp_path):
        # Bare file backend (no tiered refusal) with a cap too small to hold
        # three medium entries: writing the third evicts the first, which was
        # written only two writes ago → treadmill signal.
        backend = FileBackend(str(tmp_path / "c"), max_size_bytes=8000, flush_interval=0)
        val = "x" * 3000  # under half-cap so it is written, not refused

        with pytest.warns(CashCacheIneffectiveWarning) as rec:
            backend.set("a", val)
            backend.set("b", val)
            backend.set("c", val)   # forces eviction of a recently-written entry
            backend.set("d", val)   # keeps churning — must NOT warn again
            backend._writes.wait_all()

        churn = [w for w in rec if issubclass(w.category, CashCacheIneffectiveWarning)]
        assert len(churn) == 1, "deduped to once per session"
        assert "max_cache_size" in str(churn[0].message)  # actionable
        backend.shutdown()

    def test_no_warning_when_cache_is_roomy(self, tmp_path, recwarn):
        # A generous cap never evicts these tiny entries → no warning.
        backend = FileBackend(str(tmp_path / "c"), max_size_bytes=10 * 1024 ** 2, flush_interval=0)
        for i in range(10):
            backend.set(f"k{i}", "x" * 100)
        backend._writes.wait_all()
        assert not [w for w in recwarn if issubclass(w.category, CashCacheIneffectiveWarning)]
        backend.shutdown()


# ---------------------------------------------------------------------------
# No-thrash acceptance — the treadmill is gone under the adaptive cap.
# ---------------------------------------------------------------------------

class TestNoThrashAcceptance:
    def test_baseline_small_cap_reproduces_treadmill(self, tmp_path):
        """DISCRIMINATOR: with an old-style small cap, two medium objects that
        together exceed it cannot both persist — the second evicts the first.
        This is the treadmill the fix removes; asserting it proves the fixed
        test below isn't vacuously green."""
        small_cap = 8000  # stands in for the old flat 1 GiB, scaled down
        backend = FileBackend(str(tmp_path / "c"), max_size_bytes=small_cap, flush_interval=0)
        frame = "x" * 5000  # each > half the cap; together (~10k) bust it

        backend.set("frame_a", frame)
        backend._writes.wait_all()
        backend.set("frame_b", frame)
        backend._writes.wait_all()

        a_present = backend.get("frame_a")[1] is not None
        b_present = backend.get("frame_b")[1] is not None
        assert not (a_present and b_present), (
            "baseline must NOT hold both — that's the treadmill we're removing"
        )
        backend.shutdown()

    def test_adaptive_cap_persists_both_frames(self, monkeypatch, tmp_path):
        """FIXED: the same two frames both persist and both restore once the
        disk tier is scaled to the machine instead of a flat 1 GiB."""
        from cash.backends import adaptive_caps
        # Big free disk → adaptive disk cap ≫ the two frames combined.
        monkeypatch.setattr(adaptive_caps, "_free_bytes_on_volume", lambda p: 500 * 1024 ** 3)
        from cash.backends.factory import build_backend_from_config
        from cash.config import CashConfig

        backend = build_backend_from_config(CashConfig(cache_dir=str(tmp_path / "c")))
        disk = backend.backends[1]
        assert disk._max_size_bytes > 1024 ** 3  # no longer the 1 GiB that thrashed

        frame = "x" * 5000
        # force_persist clears the compute floor deterministically; the frames
        # are far under the refusal threshold, so both reach disk.
        for key in ("frame_a", "frame_b"):
            backend.set(key, frame, {"execution_time": 5.0, "size": len(frame),
                                     "force_persist": True})
        disk._writes.wait_all()

        # Both persisted to disk (survive a restart) ...
        assert disk.get("frame_a")[1] == frame
        assert disk.get("frame_b")[1] == frame
        # ... and both restore through the tiered stack.
        assert backend.get("frame_a")[1] == frame
        assert backend.get("frame_b")[1] == frame
        backend.shutdown()
