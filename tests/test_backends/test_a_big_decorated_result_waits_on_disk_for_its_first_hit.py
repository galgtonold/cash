"""A big result of a decorated function waits on disk for its first hit.

The RAM tier kept a copy of every result a decorated function stored, up
to its cap (a fifth of the machine's memory). Most are never asked for
again in the process that made them: a solar model's one-year runs held
3 GB in it against 0.35 GB plain. A result holding as many bytes of
arrays as a chunk the disk tier maps (`chunk_store.MAP_MIN_BYTES`) is now
written to disk alone, and the RAM tier keeps it from its first hit on. A
small one, records a read would unpickle one by one, a notebook entry, and
a result no disk tier takes go to the RAM tier as before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cash import Cash
from cash.backends.chunk_store import MAP_MIN_BYTES
from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import DISK_ALONE, TieredBackend

BIG = MAP_MIN_BYTES // 8 + 1000  # float64s: just over the size


def _decorated(**extra):
    return {"execution_time": 1.0, "decorator_entry": True, "copy_required": True, **extra}


@pytest.fixture
def tiers(tmp_path):
    ram = InMemoryBackend()
    disk = FileBackend(cache_dir=str(tmp_path / "cache"))
    yield ram, disk, TieredBackend([ram, disk])
    ram.clear()
    disk.clear()


def test_a_big_result_is_kept_in_ram_from_its_first_hit(tiers):
    ram, disk, tiered = tiers
    value = np.arange(BIG, dtype=float)
    metadata = _decorated()
    tiered.set("k", value, metadata)
    assert ram.peek_metadata("k") is None
    assert metadata["storage"] == [disk.source_label]

    meta, first = tiered.get("k")
    assert np.array_equal(first, value) and meta["source"] != ram.source_label
    assert ram.peek_metadata("k") is not None
    assert not meta.get(DISK_ALONE)

    first[0] = -1.0  # the caller's own: the kept copy is not changed
    meta, again = tiered.get("k")
    assert meta["source"] == ram.source_label and again[0] == 0.0


def test_a_small_result_goes_to_ram_at_once(tiers):
    ram, _disk, tiered = tiers
    tiered.set("k", np.arange(1000, dtype=float), _decorated())
    assert ram.peek_metadata("k") is not None


def test_big_records_go_to_ram_at_once(tiers):
    ram, _disk, tiered = tiers
    tiered.set("k", [{"id": i, "tags": [i, i + 1]} for i in range(20_000)], _decorated())
    assert ram.peek_metadata("k") is not None


def test_a_table_over_a_range_waits_on_disk(tiers):
    ram, _disk, tiered = tiers
    table = pd.DataFrame({"x": np.arange(BIG, dtype=float)})
    tiered.set("k", (table, {"rows": BIG}), _decorated())
    assert ram.peek_metadata("k") is None
    _meta, (back, info) = tiered.get("k")
    assert back.equals(table) and info == {"rows": BIG}


def test_a_notebook_entry_goes_to_ram_at_once(tiers):
    ram, _disk, tiered = tiers
    tiered.set("k", {"x": np.arange(BIG, dtype=float)}, {"execution_time": 1.0, "force_persist": True})
    assert ram.peek_metadata("k") is not None


def test_without_a_disk_tier_a_big_result_stays_in_ram():
    ram = InMemoryBackend()
    tiered = TieredBackend([ram])
    tiered.set("k", np.arange(BIG, dtype=float), _decorated())
    assert ram.peek_metadata("k") is not None


def test_a_decorated_function_serves_its_big_result_from_disk_then_ram(tmp_path):
    cash = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    ram = cash.backend.backends[0]
    calls = []

    @cash.cache
    def year(scale):
        calls.append(scale)  # @cash:assume-safe
        index = pd.date_range("2024-01-01", periods=BIG, freq="min", tz="UTC")
        return pd.Series(np.arange(BIG, dtype=float) * scale, index=index)

    stored = year(2.0)
    assert ram.entry_count() == 0
    first = year(2.0)
    assert ram.entry_count() == 1
    assert year(2.0).equals(stored) and first.equals(stored) and stored.index.freq == "min"
    assert calls == [2.0]
