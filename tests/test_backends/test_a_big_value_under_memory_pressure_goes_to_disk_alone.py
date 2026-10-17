"""A big value stored while the machine's memory is nearly full goes to the
disk tier alone, and is not copied for it either.

The RAM tier's pressure check runs every few writes and sheds entries after
they were copied. With six notebooks at once on one Windows machine, a frame
of 460 MB was copied, pickled and written in 27-134 s for a cell of 0.6 s.
A value of 64 MB or more is now checked before the tier copies it: under
pressure the RAM tier refuses it, the copy for the background write is not
made, and the disk tier writes it from the value itself.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cash.backends import memory_backend
from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend

GB = 1_000_000_000
#: 80 MB: over the size the check starts at.
BIG = 10_000_000


@pytest.fixture
def machine(monkeypatch):
    """A 16 GB machine whose memory reading the test sets."""
    state = SimpleNamespace(percent=50.0, total=16 * GB, readings=0)

    def reading():
        state.readings += 1
        return SimpleNamespace(percent=state.percent, total=state.total)

    monkeypatch.setattr(memory_backend, "psutil", SimpleNamespace(virtual_memory=reading))
    return state


@pytest.fixture
def copies(monkeypatch):
    """Calls of the RAM tier's two copies of a value: its own, and the one
    for a background write."""
    seen = {"tier": 0, "write": 0}
    real_tier = InMemoryBackend._safe_deep_copy
    real_write = memory_backend._write_copy

    def tier(*a, **k):
        seen["tier"] += 1
        return real_tier(*a, **k)

    def write(*a, **k):
        seen["write"] += 1
        return real_write(*a, **k)

    monkeypatch.setattr(InMemoryBackend, "_safe_deep_copy", staticmethod(tier))
    monkeypatch.setattr(memory_backend, "_write_copy", write)
    return seen


def _store(tmp_path, value):
    tiers = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path / "c"), flush_interval=0)])
    metadata = {"execution_time": 5.0}
    tiers.set("k", value, metadata)
    return tiers, metadata


def test_under_pressure_the_value_is_written_to_disk_alone_without_a_copy(tmp_path, machine, copies):
    machine.percent = 85.0  # 80 MB more is past the 81% the tier sheds down to
    value = np.arange(BIG, dtype=np.float64)
    tiers, metadata = _store(tmp_path, value)

    assert metadata["storage"] == ["DISK"], metadata["storage"]
    assert tiers.backends[0].peek_metadata("k") is None
    assert copies == {"tier": 0, "write": 0}
    value[0] = -1.0  # the write was made from the value before the store returned
    _meta, read = tiers.get("k")
    tiers.shutdown()
    assert read[0] == 0.0 and read[-1] == BIG - 1


def test_with_room_the_ram_tier_keeps_it(tmp_path, machine, copies):
    machine.percent = 50.0
    tiers, metadata = _store(tmp_path, np.arange(BIG, dtype=np.float64))
    tiers.shutdown()
    assert metadata["storage"][0] == "RAM", metadata["storage"]


def test_a_small_value_takes_no_reading(machine):
    machine.percent = 99.0
    tier = InMemoryBackend(check_interval=1000)
    tier.set("k", np.arange(1000, dtype=np.float64), {"execution_time": 5.0})
    assert tier.peek_metadata("k") is not None
    assert machine.readings == 0


def test_a_failed_reading_keeps_the_value(monkeypatch):
    def broken():
        raise OSError("no reading")

    monkeypatch.setattr(memory_backend, "psutil", SimpleNamespace(virtual_memory=broken))
    tier = InMemoryBackend(check_interval=1000)
    tier.set("k", np.arange(BIG, dtype=np.float64), {"execution_time": 5.0})
    assert tier.peek_metadata("k") is not None
