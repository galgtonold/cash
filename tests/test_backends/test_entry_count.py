"""``entry_count`` answers without reading the entries.

``%cash_on`` prints how many entries the cache holds, and used to get the
number from ``list_entries``, which opens every entry file to read its
metadata. Re-running r23s2's first cell (``%cash_on``) took 22.7 s for that on
a cache of a few thousand entries (round 23, 2026-09-14).
"""
from __future__ import annotations

import pytest

from cash.backends import file_backend
from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.tiered_backend import TieredBackend


def _fill(backend, n):
    for i in range(n):
        backend.set(f"k{i}", i, {"execution_time": 0.5})


def _no_entry_reads(monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("entry_count read an entry")
    monkeypatch.setattr(file_backend, "read_entry", refuse)


def test_a_file_cache_is_counted_without_opening_an_entry(tmp_path, monkeypatch):
    b = FileBackend(cache_dir=str(tmp_path))
    _fill(b, 5)
    assert len(b.list_entries()) == 5          # control: the listing agrees
    _no_entry_reads(monkeypatch)

    assert b.entry_count() == 5


def test_the_ram_tier_is_counted():
    b = InMemoryBackend()
    _fill(b, 3)

    assert b.entry_count() == 3


def test_tiers_are_counted_by_the_largest_not_summed(tmp_path, monkeypatch):
    """Every entry on disk, three of them also in RAM: 5, not 8."""
    ram, disk = InMemoryBackend(), FileBackend(cache_dir=str(tmp_path))
    _fill(disk, 5)
    _fill(ram, 3)
    _no_entry_reads(monkeypatch)

    assert TieredBackend([ram, disk]).entry_count() == 5


def test_an_empty_tier_list_counts_zero():
    assert TieredBackend([]).entry_count() == 0


@pytest.mark.parametrize("n", [0, 4])
def test_cash_on_counts_without_listing(cash_magics, capsys, monkeypatch, n):
    backend = cash_magics._cash_instance.backend
    _fill(backend, n)

    def refuse(*_a, **_k):
        raise AssertionError("%cash_on listed the cache")
    monkeypatch.setattr(type(backend), "list_entries", refuse)
    cash_magics.cash_on("")
    out = capsys.readouterr().out

    assert cash_magics._auto_cache_enabled is True
    if n:
        assert f"Found existing cache with {n} entries." in out
    else:
        assert "Found existing cache" not in out
