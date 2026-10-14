"""An object kept as pickle bytes is unpickled when it is served, not before.

The RAM tier's copy of an object of a class of its own was a pickle round
trip on the store, and its unpickling (``__setstate__``: a statsmodels
result rebuilding its design matrices) was paid for every value stored,
read again or not. Now the store pickles it once; a hit unpickles a copy,
once per hit; and the disk tier writes the bytes the store made instead of
pickling the value a second time.

Counted by the class's own ``__getstate__`` and ``__setstate__`` calls.
"""

from __future__ import annotations

import numpy as np
import pytest

from cash.backends import FileBackend, InMemoryBackend, TieredBackend, memory_backend

DUMPS: list[int] = []
LOADS: list[int] = []


class Fit:
    def __init__(self) -> None:
        self.design = np.ones(1 << 18)

    def __getstate__(self) -> dict:
        DUMPS.append(1)
        return self.__dict__

    def __setstate__(self, state: dict) -> None:
        LOADS.append(1)
        self.__dict__.update(state)


@pytest.fixture(autouse=True)
def counts():
    rates = memory_backend._ROUND_TRIP_RATES
    saved = dict(rates)
    DUMPS.clear()
    LOADS.clear()
    yield
    rates.clear()
    rates.update(saved)


def test_a_value_never_served_is_never_unpickled():
    ram = InMemoryBackend()
    ram.set("k", Fit(), {"execution_time": 5.0})
    assert (len(DUMPS), len(LOADS)) == (1, 0)


def test_each_hit_unpickles_once():
    ram = InMemoryBackend()
    ram.set("k", Fit(), {"execution_time": 5.0})
    for hits in (1, 2, 3):
        assert isinstance(ram.get("k")[1], Fit)
        assert len(LOADS) == hits
    assert len(DUMPS) == 1, "a hit pickled the value again"


def test_the_disk_writes_the_bytes_the_store_made(tmp_path):
    """Pickled once for both tiers."""
    disk = FileBackend(str(tmp_path / "c"), flush_interval=0)
    tiers = TieredBackend([InMemoryBackend(), disk])
    metadata = {"execution_time": 5.0, "cost_model_family": None}
    tiers.set("k", Fit(), metadata)
    disk.shutdown()
    assert "DISK" in metadata["storage"]
    assert (len(DUMPS), len(LOADS)) == (1, 0)
    assert isinstance(FileBackend(str(tmp_path / "c")).get("k")[1], Fit)


def test_the_first_of_its_class_is_written_without_being_read_back(tmp_path):
    """Nothing says yet what reading one back takes: the cost model judges
    it, and its first restore is the reading (``restored_no_faster``)."""
    disk = FileBackend(str(tmp_path / "c"), flush_interval=0)
    tiers = TieredBackend([InMemoryBackend(), disk])
    metadata = {"execution_time": 5.0}
    tiers.set("a", Fit(), metadata)
    disk.shutdown()
    assert "DISK" in metadata["storage"]
    assert (len(DUMPS), len(LOADS)) == (1, 0)
