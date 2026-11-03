"""A statement lookup after a slow stat pass takes the pass's recent stats.

Before a trivial cell below a read of 38,000 files, the cell run stats every
file once (`run_memo.stats_this_run`) and the statement lookup takes those
stats only while they are at most ``_ANSWERS_LAST_S`` old. On Windows the
pass and the simulation took 3-5 s, and the lookup, dating the pass by its
oldest stat, stat-ed every file again, 7-15 s a cell. Dated a chunk at a
time, it takes again only the stats that are too old, and a second lookup
in the same cell only the ones that have lapsed since.
"""

from __future__ import annotations

import os
import time
import types

import pytest

from cash.notebook import run_memo
from cash.notebook.statement import freshness
from cash.notebook.statement.freshness import CacheFreshnessChecker
from cash.tracking import file_dep_snapshot
from cash.tracking.file_dep_snapshot import snapshot_file_deps

# The stats every check shares, where no folder is watched (`file_watch`).
pytestmark = pytest.mark.usefixtures("no_file_watch")

N = 40  # per half; under the size a whole set is kept for, so each file's answer counts


@pytest.fixture(autouse=True)
def _one_cell_run(monkeypatch):
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    run_memo._FILE_STATE_THIS_RUN.clear()
    file_dep_snapshot.begin_file_state_epoch()
    yield
    file_dep_snapshot.end_file_state_epoch()
    run_memo._FILE_STATE_THIS_RUN.clear()


@pytest.fixture
def clock(monkeypatch):
    """``time.monotonic`` moved on by hand: a slow pass without the wait."""
    real = time.monotonic
    offset = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: real() + offset[0])

    def advance(seconds: float) -> None:
        offset[0] += seconds

    return advance


@pytest.fixture
def halves(tmp_path) -> tuple[dict[str, dict], dict[str, dict]]:
    deps = []
    for half in range(2):
        paths = set()
        for i in range(N):
            p = tmp_path / f"h{half}img{i:03d}.npy"
            p.write_bytes(bytes([i % 251]) * 64)
            paths.add(str(p))
        deps.append(snapshot_file_deps(paths))
    return deps[0], deps[1]


@pytest.fixture
def stats_taken(monkeypatch) -> list[str]:
    taken: list[str] = []
    real = os.stat

    def counting(path, *args, **kwargs):
        taken.append(os.fspath(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", counting)
    return taken


class _Backend:
    def __init__(self, deps):
        self._meta = {"key": "stmt:x", "file_dependencies": deps}

    def get(self, key):
        return dict(self._meta, key=key), "value"


def _lookup(checker: CacheFreshnessChecker, key: str) -> bool:
    state = types.SimpleNamespace(executed_file_deps={}, variable_sources={})
    _, data, _ = checker.check_cache(state, key, None, epoch=file_dep_snapshot.file_state_epoch())
    return data is not None


def test_a_lookup_stats_again_only_the_part_of_the_pass_that_is_too_old(halves, clock, stats_taken):
    old, recent = halves
    window = freshness._ANSWERS_LAST_S
    run_memo.stats_this_run(old)  # the start of the cell run's pass
    clock(window * 0.75)
    run_memo.stats_this_run(recent)  # its end
    clock(window * 0.5)  # the simulation; the first half is now too old
    stats_taken.clear()

    assert _lookup(CacheFreshnessChecker(_Backend({**old, **recent})), "stmt:a")

    assert sorted(stats_taken) == sorted(old)


def test_a_second_lookup_stats_again_only_what_lapsed_since_the_first(halves, clock, stats_taken, monkeypatch):
    old, recent = halves
    listed: list[str] = []
    real_listings = file_dep_snapshot.stats_from_listings

    def listing(paths):
        found = real_listings(paths)
        listed.extend(found)
        return found

    monkeypatch.setattr(file_dep_snapshot, "stats_from_listings", listing)
    window = freshness._ANSWERS_LAST_S
    checker = CacheFreshnessChecker(_Backend({**old, **recent}))
    run_memo.stats_this_run(old)
    clock(window * 0.75)
    run_memo.stats_this_run(recent)
    clock(window * 0.5)
    assert _lookup(checker, "stmt:a")  # takes the first half again: now the newest
    clock(window * 0.6)  # the pass's second half lapses, the first lookup's own stats do not
    stats_taken.clear()

    assert _lookup(checker, "stmt:b")

    # No file of the first half stat-ed again, and each of the second half
    # taken again: stat-ed, or read from a listing of the folder, which on
    # Windows stands in for the stats of a lookup the run handed none
    # (`stats_from_listings`), one read of the folder for the lot.
    assert set(stats_taken) <= set(recent)
    assert sorted(set(stats_taken) | (set(listed) & set(recent))) == sorted(recent)
    assert len(stats_taken) == len(set(stats_taken))


def test_a_lookup_soon_after_the_pass_takes_every_stat_of_it(halves, clock, stats_taken):
    """The control: nothing too old, nothing taken again."""
    old, recent = halves
    run_memo.stats_this_run({**old, **recent})
    clock(freshness._ANSWERS_LAST_S * 0.5)
    stats_taken.clear()

    assert _lookup(CacheFreshnessChecker(_Backend({**old, **recent})), "stmt:a")

    assert stats_taken == []


def test_the_pass_dates_a_long_run_of_stats_a_chunk_at_a_time(tmp_path, monkeypatch):
    """A pass longer than a chunk dates its later stats by when their chunk began."""
    paths = []
    # Folders too small to list (`stats_from_listings`, Windows: one
    # listing takes a crowded folder's stats at once, under one date), so
    # each file is stat-ed in its turn.
    per_folder = file_dep_snapshot.LISTING_MIN_FILES - 1
    for i in range(3 * run_memo._STAT_CHUNK):
        folder = tmp_path / f"d{i // per_folder:03d}"
        folder.mkdir(exist_ok=True)
        p = folder / f"f{i:04d}"
        p.write_bytes(b"x")
        paths.append(str(p))
    real = time.monotonic
    ticks = [0.0]

    def ticking() -> float:
        ticks[0] += 1.0
        return real() + ticks[0]

    monkeypatch.setattr(time, "monotonic", ticking)
    run_memo.stats_this_run(paths)
    dates = run_memo.stat_dates_this_run()

    assert dates[paths[0]] < dates[paths[-1]]
    assert len(set(dates.values())) >= 3
    # The newest chunk alone is handed to a caller that takes only it.
    newest = max(dates.values())
    recent = run_memo.stats_taken_this_run(paths, newest)
    assert set(recent) == {p for p in paths if dates[p] >= newest}
    assert 0 < len(recent) < len(paths)
