"""``%cash_on`` says how big the disk cache may grow; a cell that makes it
evict says so once.

The notebook half of ``tests/test_backends/test_the_disk_cap_is_said_out_loud.py``:
there the notices are log records, here they are printed into the cell, where a
notebook user looks. The shared ``cash_magics`` runs on an in-memory Cash,
which has no disk cap to show, so each test gives it a capped disk tier.
"""

from __future__ import annotations

import pytest

from cash.backends import FileBackend, InMemoryBackend, TieredBackend

MB = 1_000_000


@pytest.fixture
def disk_magics(cash_magics, cash_instance, tmp_path):
    """``cash_magics`` over RAM + a 5 MB disk tier: what %cash_on builds, capped small."""
    disk = FileBackend(str(tmp_path / "nbcache"), max_size_bytes=5 * MB, flush_interval=0)
    cash_instance.backend = TieredBackend([InMemoryBackend(), disk])
    yield cash_magics, disk
    disk.shutdown()


def test_cash_on_shows_the_cap_once(disk_magics, capsys):
    magics, disk = disk_magics
    magics.cash_on("")
    first = capsys.readouterr().out
    assert f"   Caching in {disk.cache_dir}, up to 5 MB (set by max_cache_size).\n" in first, first

    magics.cash_on("")  # the same kernel and folder: said already
    assert "Caching in" not in capsys.readouterr().out


def test_cash_on_with_caching_disabled_shows_no_cap_and_keeps_it_for_later(disk_magics, capsys):
    magics, disk = disk_magics
    magics._cash_instance.config.disable = True
    try:
        magics.cash_on("")
    finally:
        magics._cash_instance.config.disable = False
    out = capsys.readouterr().out
    assert "did nothing" in out and "Caching in" not in out
    # Declining used nothing up: turned on, it shows the cap.
    magics.cash_on("")
    assert f"Caching in {disk.cache_dir}, up to 5 MB" in capsys.readouterr().out


def test_the_cell_that_first_evicts_says_so_once(disk_magics, capsys):
    magics, disk = disk_magics

    def store(key):
        disk.set(key, b"x" * MB, {"execution_time": 1.0})
        magics._flush_pending_writes()  # what post_run_cell does after every cell

    for i in range(4):  # 4 MB and the headers: under the cap
        store(f"k{i}")
    assert "[cash]" not in capsys.readouterr().out

    store("k4")  # over it: the first eviction
    out = capsys.readouterr().out
    assert out.startswith(f"[cash] The cache in {disk.cache_dir} reached its 5 MB cap, so cash removed "), out

    for i in range(5, 10):
        store(f"k{i}")
    assert "[cash]" not in capsys.readouterr().out, "a later eviction was reported again"
