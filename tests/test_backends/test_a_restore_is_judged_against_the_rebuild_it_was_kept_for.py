"""A restore is judged against what its entry was kept for.

``forest_reg = make_pipeline(preprocessing, ...)`` took 0.2 ms over a
``preprocessing`` kept in memory only, and was written to disk for the
0.15 s a restart would re-run to get it back (``rebuild_time``,
`TieredBackend.persist_from_memory`). Its first restore after the restart
took a few ms: no faster than its own 0.2 ms, so the disk tier deleted it,
and the next cell rebuilt the chain behind it. A restore, from disk or out
of the RAM tier's pickle bytes, now has to beat the rebuild the entry was
kept for; an entry kept for its own compute is judged as before.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from cash.backends import FileBackend, InMemoryBackend, TieredBackend, memory_backend
from cash.cost_model import GENERIC_FAMILY

#: What reading a `Slow` back takes, far over the entries' own compute and
#: far under the rebuild they are kept for.
READ_S = 0.02


class Slow:
    def __init__(self) -> None:
        # Big enough for the RAM tier to keep it as pickle bytes.
        self.design = np.ones(1 << 18)

    def __setstate__(self, state: dict) -> None:
        time.sleep(READ_S)
        self.__dict__.update(state)


@pytest.fixture(autouse=True)
def _readings():
    rates = memory_backend._ROUND_TRIP_RATES
    saved = dict(rates)
    yield
    rates.clear()
    rates.update(saved)


def _entry(rebuild_time: float | None) -> dict:
    """A notebook entry's metadata: 0.2 ms of compute, its biggest value a `Slow`."""
    metadata = {
        "execution_time": 0.0002,
        "cost_model_family": GENERIC_FAMILY,
        memory_backend.ROUND_TRIP_CLASS: memory_backend.class_key(Slow),
    }
    if rebuild_time is not None:
        metadata["rebuild_time"] = rebuild_time
    return metadata


@pytest.mark.parametrize(("rebuild_time", "kept"), [(0.5, True), (None, False)], ids=["kept_for_rebuild", "own_time"])
def test_the_rule(rebuild_time, kept):
    value = {"variables": {"b": Slow()}}
    metadata = _entry(rebuild_time)
    assert memory_backend.restore_too_slow(value, metadata, READ_S) is not kept
    assert memory_backend.restored_no_faster(value, metadata, READ_S) is not kept


@pytest.mark.parametrize(("rebuild_time", "kept"), [(0.5, True), (None, False)], ids=["kept_for_rebuild", "own_time"])
def test_a_restore_from_disk_keeps_an_entry_kept_for_its_rebuild(tmp_path, rebuild_time, kept):
    root = str(tmp_path / "c")
    disk = FileBackend(root, flush_interval=0)
    disk.set("k", {"variables": {"b": Slow()}}, _entry(rebuild_time))
    disk.shutdown()

    # A restart: an empty RAM tier over the same disk.
    disk = FileBackend(root, flush_interval=0)
    tiers = TieredBackend([InMemoryBackend(), disk])
    metadata, value = tiers.get("k")
    assert isinstance(value["variables"]["b"], Slow)
    disk.shutdown()
    assert (FileBackend(root).get_metadata("k") is not None) is kept


@pytest.mark.parametrize(("rebuild_time", "kept"), [(0.5, True), (None, False)], ids=["kept_for_rebuild", "own_time"])
def test_a_ram_hit_keeps_an_entry_kept_for_its_rebuild(rebuild_time, kept):
    ram = InMemoryBackend()
    # Over what the cost model prices reading it out at (~6 ms), so the tier
    # takes it; under what reading it out takes.
    metadata = {"execution_time": 0.01}
    if rebuild_time is not None:
        metadata["rebuild_time"] = rebuild_time
    ram.set("k", Slow(), metadata)
    assert isinstance(ram.get("k")[1], Slow)
    assert (ram.get_metadata("k") is not None) is kept
