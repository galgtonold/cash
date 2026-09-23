"""A tier's write error belongs to the call that hit it.

The errors used to be kept on the ``TieredBackend`` instance. Two overlapping
``set`` calls then reported each other's failures, and ``persist_from_memory``
raised ``AttributeError`` when a tier failed before any ``set`` had run.
"""

from __future__ import annotations

import threading

from cash.backends import InMemoryBackend, TieredBackend


class _FlakyTier(InMemoryBackend):
    """A persistent tier that fails for key "bad" and can hold key "slow"."""

    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def set(self, key, value, metadata=None, serializer=None):
        if key == "slow":
            self.entered.set()
            assert self.release.wait(10)
        if key == "bad":
            raise OSError("disk full")
        return super().set(key, value, metadata, serializer)


def test_overlapping_sets_keep_their_own_errors():
    tier = _FlakyTier()
    tiered = TieredBackend([InMemoryBackend(), tier])
    slow_meta = {"decorator_entry": True}
    writer = threading.Thread(target=tiered.set, args=("slow", 1, slow_meta))
    writer.start()
    try:
        assert tier.entered.wait(10)
        bad_meta = {"decorator_entry": True}
        tiered.set("bad", 2, bad_meta)
    finally:
        tier.release.set()
        writer.join(10)

    assert bad_meta["store_errors"] == ["_FlakyTier: OSError: disk full"]
    assert "store_errors" not in slow_meta, "the slow write reported the other call's failure"
    assert len(slow_meta["storage"]) == 2, "both tiers took the slow write"


def test_persist_from_memory_survives_a_failing_tier_before_any_set():
    ram = InMemoryBackend()
    tiered = TieredBackend([ram, _FlakyTier()])
    # Put the entry in RAM directly, so this TieredBackend has never run set().
    ram.set(
        "bad",
        list(range(1000)),
        {
            "execution_time": 0.02,
            "cost_model_family": "_GENERIC",
            "cost_model_type_name": "list",
            "cost_model_size_bytes": 8000,
            "force_persist": True,
        },
    )
    assert tiered.persist_from_memory("bad", rebuild_seconds=60.0) is False
