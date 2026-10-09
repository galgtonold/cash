"""A store hands the persistent tiers the RAM tier's own copy, not the caller's value.

A miss copied its result twice before returning: once into the RAM tier,
once more for the disk write (which runs later, on another thread, and must
not see a change the caller makes meanwhile). The RAM tier's copy is
already one nothing changes, so a tier that takes such a value
(``takes_private_values``) serializes that in the background instead.
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest

from cash.backends import frame_sharing
from cash.backends._base import CacheBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend


class _Recording(CacheBackend):
    """A persistent tier that records what it was handed."""

    source_label = "DISK"
    cost_kind = "disk"
    takes_private_values = True

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def set(self, key, value, metadata=None, serializer=None, private=False):
        self.calls.append((key, value, private))

    def get(self, key):
        return None, None

    def get_metadata(self, key):
        return None

    def delete(self, key):
        pass

    def clear(self):
        pass

    def list_entries(self):
        return []

    def cleanup_expired(self, is_expired):
        return 0


def _tiers(ram: InMemoryBackend | None = None):
    disk = _Recording()
    return TieredBackend([ram or InMemoryBackend(), disk]), disk


_KEEP = {"execution_time": 5.0, "decorator_entry": True}


def test_an_array_goes_to_disk_as_the_ram_copy():
    tiers, disk = _tiers()
    value = np.arange(1000, dtype=float)
    tiers.set("k", value, dict(_KEEP))
    (_key, written, private), = disk.calls
    assert private and written is not value
    assert written is tiers.backends[0]._store["k"][1]
    value[0] = -1.0  # the caller's change after the call
    assert written[0] == 0.0


@pytest.mark.skipif(not frame_sharing.enabled(), reason="needs pandas copy-on-write")
def test_a_table_is_neither_copied_for_ram_nor_for_disk():
    tiers, disk = _tiers()
    df = pd.DataFrame({"x": np.arange(1000, dtype=float)})
    tiers.set("k", df, dict(_KEEP))
    (_key, written, private), = disk.calls
    assert private and np.shares_memory(written["x"].to_numpy(), df["x"].to_numpy())
    df.loc[0, "x"] = -1.0  # copy-on-write: the frozen data stays as it was
    assert written["x"].iloc[0] == 0.0


def test_big_records_go_to_disk_read_out_of_the_bytes_the_ram_tier_keeps():
    tiers, disk = _tiers()
    records = [{"id": i, "tags": [i, i + 1]} for i in range(10_000)]
    tiers.set("k", records, dict(_KEEP))
    (_key, written, private), = disk.calls
    assert private and written == records and written is not records


def test_a_value_the_ram_tier_keeps_by_reference_is_not_private():
    tiers, disk = _tiers()
    lock_holder = {"lock": threading.Lock(), "n": 1}  # cannot be copied: kept as it is
    tiers.set("k", lock_holder, {"execution_time": 5.0})
    (_key, written, private), = disk.calls
    assert not private and written is lock_holder


def test_a_value_the_ram_tier_refused_goes_to_disk_as_the_caller_gave_it():
    tiers, disk = _tiers(InMemoryBackend(max_size_bytes=1000))
    value = np.arange(10_000, dtype=float)
    tiers.set("k", value, dict(_KEEP))
    (_key, written, private), = disk.calls
    assert not private and written is value


def test_an_earlier_value_in_ram_is_never_written_for_a_later_store():
    ram = InMemoryBackend()
    tiers, disk = _tiers(ram)
    tiers.set("k", np.zeros(10), dict(_KEEP))

    def refuse(*args, **kwargs):
        raise RuntimeError("cannot copy")

    real_set, ram.set = ram.set, refuse
    try:
        tiers.set("k", np.ones(10), dict(_KEEP))
    finally:
        ram.set = real_set
    _key, written, private = disk.calls[-1]
    assert not private and written[0] == 1.0


def test_a_tier_that_does_not_take_private_values_gets_the_callers_value():
    ram = InMemoryBackend()
    disk = _Recording()
    disk.takes_private_values = False
    calls = []
    disk.set = lambda key, value, metadata=None, serializer=None: calls.append(value)
    tiers = TieredBackend([ram, disk])
    value = [{"id": i} for i in range(10_000)]
    tiers.set("k", value, dict(_KEEP))
    assert calls == [value] and calls[0] is value


# A notebook value: it carries a cost-model family, so `persist_from_memory` considers it.
_NOTEBOOK = {
    "execution_time": 0.02,
    "cost_model_family": "_GENERIC",
    "cost_model_type_name": "dict",
    "cost_model_size_bytes": 1000,
}


def test_a_later_persist_hands_the_ram_copy_as_private():
    tiers, disk = _tiers()
    value = {"rows": list(range(100))}
    tiers.set("k", value, dict(_NOTEBOOK))
    assert disk.calls == []  # cheap: kept in RAM only
    assert tiers.persist_from_memory("k", rebuild_seconds=5.0)
    (_key, written, private), = disk.calls
    assert private and written is not value and written == value


def test_a_later_persist_of_a_value_kept_by_reference_is_not_private():
    tiers, disk = _tiers()
    lock_holder = {"lock": threading.Lock(), "n": 1}  # cannot be copied: the caller still holds it
    tiers.set("k", lock_holder, dict(_NOTEBOOK))
    assert disk.calls == []
    assert tiers.persist_from_memory("k", rebuild_seconds=5.0)
    (_key, written, private), = disk.calls
    assert written is lock_holder and not private


def test_a_value_marked_no_private_copy_goes_to_disk_as_the_caller_gave_it():
    """A stored closed file copies back into a closed file, which does not
    pickle: the disk must get the stored form the caller handed over."""
    from cash.backends.memory_backend import NO_PRIVATE_COPY

    tiers, disk = _tiers()
    value = {"rows": list(range(100))}
    tiers.set("k", value, {**_KEEP, NO_PRIVATE_COPY: True})
    (_key, written, private), = disk.calls
    assert not private and written is value
