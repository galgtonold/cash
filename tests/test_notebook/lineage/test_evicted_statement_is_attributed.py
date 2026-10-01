"""A statement that recomputes because the disk cap evicted its value says so.

The notebook half of ``tests/test_backends/test_evicted_results_are_noted.py``:
the badge's miss reason names the eviction, and a recompute over the 2 s
"worth telling" floor warns ``CACHE-EVICTED-RECOMPUTE`` once per statement.
The shared ``cash_magics`` runs on an in-memory Cash, which evicts nothing to
disk, so these tests give it a disk tier.
"""

from __future__ import annotations

import os
import time
import warnings

import pytest

from cash.backends import FileBackend, InMemoryBackend, TieredBackend
from tests._cell_driver import run_cash_cell

MB = 1_000_000


def _evictions():
    """Imported per test: each fails, rather than the file failing to
    collect, on a tree without it."""
    from cash.backends.eviction_log import EvictionNote
    from cash.notebook.statement.evictions import EvictedRecomputes

    return EvictionNote, EvictedRecomputes


@pytest.fixture
def disk(tmp_path):
    b = FileBackend(str(tmp_path / "nbcache"), max_size_bytes=5 * MB, flush_interval=0)
    yield b
    b.shutdown()


def _note(disk, key, seconds=30.0):
    """Note *key* as evicted, the way the cap does (the folder exists by then)."""
    EvictionNote, _ = _evictions()
    os.makedirs(disk.cache_dir, exist_ok=True)
    disk.evictor.evictions.record([(disk._stem(key), EvictionNote(time.time(), seconds, MB))])


def _codes(caught):
    return [getattr(w.message, "code", "") for w in caught]


def test_a_noted_miss_is_attributed_and_warns_once_per_statement(disk):
    _, EvictedRecomputes = _evictions()
    _note(disk, "stmt:aaa")
    _note(disk, "stmt:bbb")
    evicted = EvictedRecomputes()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metrics: dict = {}
        assert evicted.attribute(metrics, disk, "stmt:aaa", "df = load()", 12.0)
        # The same statement again, under another key: already said.
        assert evicted.attribute({}, disk, "stmt:bbb", "df = load()", 12.0)
    assert metrics["miss_reason"] == "evicted to make room when the disk cache reached its size cap"
    assert _codes(caught) == ["CACHE-EVICTED-RECOMPUTE"], [str(w.message) for w in caught]
    text = str(caught[0].message)
    assert "the value of `df = load()` had been evicted from the disk cache to make room" in text
    assert "its 5 MB cap" in text and "took 12.0s" in text


def test_a_cheap_recompute_is_attributed_but_not_warned(disk):
    _, EvictedRecomputes = _evictions()
    _note(disk, "stmt:aaa")
    metrics: dict = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert EvictedRecomputes().attribute(metrics, disk, "stmt:aaa", "x = 1", 0.5)
    assert metrics["miss_reason"].startswith("evicted to make room")
    assert "CACHE-EVICTED-RECOMPUTE" not in _codes(caught)


def test_a_miss_without_a_note_is_left_alone(disk):
    _, EvictedRecomputes = _evictions()
    metrics: dict = {}
    assert not EvictedRecomputes().attribute(metrics, disk, "stmt:never", "x = 1", 30.0)
    assert metrics == {}


def test_the_processor_puts_the_eviction_on_the_badge(cash_magics, cash_instance, disk, monkeypatch):
    """Wired in: a statement whose key the cap noted misses with that reason."""
    cash_instance.backend = TieredBackend([InMemoryBackend(), disk])
    asked: list[str] = []
    real = disk.eviction_note
    monkeypatch.setattr(disk, "eviction_note", lambda key: asked.append(key) or real(key))

    run_cash_cell(cash_magics, "x = 21")  # too fast to store: it misses every time
    assert asked, "a miss did not ask about evictions"
    _note(disk, asked[-1])

    captured: list = []
    real_render = cash_magics.badges.render
    monkeypatch.setattr(
        cash_magics.badges, "render", lambda metrics, **kw: captured.append(list(metrics)) or real_render(metrics, **kw)
    )
    run_cash_cell(cash_magics, "x = 21")
    assert captured[-1][-1].get("miss_reason") == "evicted to make room when the disk cache reached its size cap"
