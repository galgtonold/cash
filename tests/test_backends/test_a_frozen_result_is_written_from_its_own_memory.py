"""A ``frozen=True`` result that goes to disk alone is written from its own
memory, and kept only when it still hashes as it did when stored.

Its big buffers were copied before the store returned, so a change made
after it could not reach the entry: 0.49 s of p08's cleaned 4M-row frame.
Its content digest is in hand already (``content_digest``), so it is now
pickled over its own memory and written in the background, and the writer
hashes it again once the bytes are on disk. A result changed in between is
dropped with STORE-RESULT-WRITTEN, and the next call computes it again.
"""

from __future__ import annotations

import threading
import warnings

import numpy as np
import pytest

from cash.backends import memory_backend
from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.serialization import PickleSerializer
from cash.backends.tiered_backend import TieredBackend
from cash.content_hashers import builtin_hash
from cash.exceptions import CashCacheStoreFailedWarning

pl = pytest.importorskip("polars")

ROWS = 1 << 18


@pytest.fixture(autouse=True)
def small_threshold(monkeypatch):
    monkeypatch.setattr(memory_backend, "DISK_ALONE_MIN_BYTES", 1 << 20)


@pytest.fixture
def tiers(tmp_path):
    made = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path / "c"), flush_interval=0)])
    yield made
    made.shutdown()


@pytest.fixture
def held_write(monkeypatch):
    """The writer waits at the start of each entry's files until released."""
    go = threading.Event()
    real = FileBackend._write_cache_files

    def waiting(self, *a, **k):
        go.wait(10)
        return real(self, *a, **k)

    monkeypatch.setattr(FileBackend, "_write_cache_files", waiting)
    return go


@pytest.fixture
def copies(monkeypatch):
    seen = []
    real = PickleSerializer.serialize_split

    def split(self, data, *, copy=True):
        seen.append(copy)
        return real(self, data, copy=copy)

    monkeypatch.setattr(PickleSerializer, "serialize_split", split)
    return seen


def _frame():
    return pl.DataFrame({"x": np.arange(ROWS, dtype=np.float64), "t": [f"row {i} past inline" for i in range(ROWS)]})


def _store(tiers, value):
    metadata = {"execution_time": 50.0, "content_digest": builtin_hash(value), "func_name": "clean"}
    tiers.set("k", value, metadata)
    return metadata


@pytest.mark.parametrize("make", [_frame, lambda: np.arange(ROWS, dtype=np.float64)])
def test_an_unchanged_result_is_written_without_a_copy(tiers, copies, make):
    value = make()
    metadata = _store(tiers, value)
    assert metadata["storage"] == ["DISK"]
    assert copies == [False]
    read = tiers.get("k")[1]
    assert builtin_hash(read) == builtin_hash(value)


def test_a_frame_changed_after_the_store_is_dropped(tiers, held_write):
    frame = _frame()
    _store(tiers, frame)
    frame[0, "x"] = -1.0  # after the store returned, before the write
    held_write.set()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert tiers.get("k") == (None, None)
    codes = [getattr(w.message, "code", None) for w in caught if issubclass(w.category, CashCacheStoreFailedWarning)]
    assert codes == ["STORE-RESULT-WRITTEN"]


def test_an_array_written_after_the_store_is_dropped(tiers, held_write):
    value = np.arange(ROWS, dtype=np.float64)
    _store(tiers, value)
    value[0] = -1.0  # into the memory the write reads
    held_write.set()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert tiers.get("k") == (None, None)


def test_without_a_digest_the_bytes_are_made_before_the_store_returns(tiers, copies):
    value = np.arange(ROWS, dtype=np.float64)
    tiers.set("k", value, {"execution_time": 50.0})
    value[0] = -1.0
    assert copies == [True]
    assert tiers.get("k")[1][0] == 0.0


def test_a_frozen_function_changed_right_after_return_is_computed_again(tmp_path, held_write):
    import cash

    c = cash.Cash(cache_dir=str(tmp_path / "cc"))
    runs = []

    @c.cache(frozen=True)
    def clean(n):
        runs.append(n)
        return _frame()

    first = clean(1)
    first[0, "x"] = -1.0  # breaks the frozen promise, before the write is done
    held_write.set()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        second = clean(1)
    assert second[0, "x"] == 0.0
    assert runs == [1, 1]
    assert "STORE-RESULT-WRITTEN" in [getattr(w.message, "code", None) for w in caught]
