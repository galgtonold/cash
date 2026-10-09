"""A value read from disk is handed over as it was read, not copied into RAM first.

The read-repair step copied every disk hit into the RAM tier before
returning it: a 400 MB table restored in 3.2 s instead of 1.3 s, and a
million records took longer than building them. The value just unpickled is
the caller's alone, so the RAM tier keeps it only when that costs no copy (a
pandas table it freezes where it is), and otherwise on the key's second read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cash.backends import frame_sharing
from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend


def _tiers(path) -> TieredBackend:
    return TieredBackend([InMemoryBackend(), FileBackend(str(path), flush_interval=0)])


def _written(path, key, value) -> None:
    writer = _tiers(path)
    writer.set(key, value, {"execution_time": 5.0, "decorator_entry": True})
    writer.shutdown()


def _counting_copies(monkeypatch) -> list:
    copies: list = []
    real_set = InMemoryBackend.set

    def counted(self, key, value, metadata=None, serializer=None):
        copies.append(key)
        return real_set(self, key, value, metadata, serializer)

    monkeypatch.setattr(InMemoryBackend, "set", counted)
    return copies


class TestAValueTheRamTierWouldCopy:
    @pytest.mark.parametrize(
        "value",
        [
            np.arange(1 << 18, dtype=float),  # 2 MB
            [{"id": i, "tags": [i, i + 1]} for i in range(5000)],
        ],
        ids=["array", "records"],
    )
    def test_is_kept_on_its_second_read_not_its_first(self, tmp_path, monkeypatch, value):
        _written(tmp_path, "k", value)
        reader = _tiers(tmp_path)
        copies = _counting_copies(monkeypatch)
        meta, first = reader.get("k")
        assert meta["source"] == "DISK" and copies == []
        assert reader.backends[0].get("k") == (None, None)
        meta, second = reader.get("k")
        assert meta["source"] == "DISK" and copies == ["k"]
        meta, third = reader.get("k")
        assert meta["source"] == "RAM"
        for got in (first, second, third):
            assert got is not value
        assert first is not second and second is not third

    def test_what_the_first_reader_does_to_it_never_reaches_a_later_read(self, tmp_path):
        _written(tmp_path, "k", np.arange(1 << 18, dtype=float))
        reader = _tiers(tmp_path)
        _meta, first = reader.get("k")
        first[0] = -1.0
        _meta, second = reader.get("k")  # from disk again, then kept
        assert second[0] == 0.0
        second[1] = -1.0
        _meta, third = reader.get("k")  # from RAM
        assert third[0] == 0.0 and third[1] == 1.0


@pytest.mark.skipif(not frame_sharing.enabled(), reason="needs pandas copy-on-write")
class TestATable:
    def _frame(self):
        return pd.DataFrame({"x": np.arange(100_000, dtype=float), "when": pd.date_range("2024", periods=100_000, freq="s")})

    def test_is_kept_on_its_first_read_without_a_copy(self, tmp_path):
        _written(tmp_path, "k", self._frame())
        reader = _tiers(tmp_path)
        meta, got = reader.get("k")
        assert meta["source"] == "DISK"
        _ram_meta, from_ram = reader.backends[0].get("k")
        assert np.shares_memory(got["x"].to_numpy(), from_ram["x"].to_numpy())

    def test_is_kept_inside_a_notebook_entry_too(self, tmp_path):
        _written(tmp_path, "k", {"variables": {"df": self._frame(), "n": 3}, "rng": (1, 2)})
        reader = _tiers(tmp_path)
        _meta, got = reader.get("k")
        _ram_meta, from_ram = reader.backends[0].get("k")
        assert np.shares_memory(got["variables"]["df"]["x"].to_numpy(), from_ram["variables"]["df"]["x"].to_numpy())

    def test_writing_the_restored_table_reaches_no_later_read(self, tmp_path):
        _written(tmp_path, "k", self._frame())
        reader = _tiers(tmp_path)
        _meta, got = reader.get("k")
        got.loc[0, "x"] = -1.0
        got["when"] = got["when"] + pd.Timedelta("1D")
        _meta, again = reader.get("k")
        pd.testing.assert_frame_equal(again, self._frame())
        again.iloc[1, 0] = -2.0  # a RAM hit is writable through pandas
        pd.testing.assert_frame_equal(reader.get("k")[1], self._frame())
