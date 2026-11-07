"""A set of read files the folder watch cannot vouch for is not looked over each run.

Photos unpacked by hard links (a dataset cache beside the project) are files
the watch keeps no stat for. The upstream check offered the stats of all of
them to `run_memo.HeldPaths.keep` before each cell, and `keep` looked at
every path (`run_memo._vouched`) only to refuse the set: 0.3-1 s of a
trivial cell on Linux, seconds on Windows (E5, 13,000 LFW photos).

A refused set is now offered again only after a growing count of runs
(`HeldPaths.worth_keeping`); the keys, worked out afresh each run, still
follow each change, and a set the watch comes to vouch for is still held.
"""

from __future__ import annotations

import os
import sys

import pytest

from cash.notebook import run_memo
from cash.notebook.upstream import statement_lineage
from cash.tracking import file_dep_snapshot, file_watch
from cash.tracking.read_classification import stat_key

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="folders are not watched on macOS")

#: Past what is held at all (`statement_lineage._HOLD_MIN_FILES`).
FILES = 300

RUNS = 12


@pytest.fixture(autouse=True)
def _runs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CASH_FILE_WATCH", raising=False)
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    monkeypatch.setattr(statement_lineage, "_HELD_FILE_KEYS", run_memo.HeldPaths(kept=8))
    run_memo._FILE_STATE_THIS_RUN.clear()
    file_watch.reset()
    file_dep_snapshot.begin_file_state_epoch()
    yield
    file_dep_snapshot.end_file_state_epoch()
    run_memo._FILE_STATE_THIS_RUN.clear()
    file_watch.reset()


def _next_run() -> None:
    file_dep_snapshot.end_file_state_epoch()
    file_dep_snapshot.begin_file_state_epoch()


@pytest.fixture
def linked(tmp_path) -> list[str]:
    """Photos in the project, each with a second link in a cache beside it."""
    photos, cache = tmp_path / "photos", tmp_path / "cache"
    photos.mkdir()
    cache.mkdir()
    found = []
    for i in range(FILES):
        original = cache / f"p{i:03d}.pgm"
        original.write_text(f"photo {i:03d}", encoding="utf-8")
        p = photos / f"p{i:03d}.pgm"
        try:
            os.link(original, p)
        except (OSError, NotImplementedError):
            pytest.skip("no hard links here")
        found.append(f"photos/p{i:03d}.pgm" if i % 2 else str(p))
    return found


@pytest.fixture
def vouched(monkeypatch) -> list[int]:
    """Each set `keep` looked over (`run_memo._vouched`), by its size."""
    seen: list[int] = []
    real = run_memo._vouched

    def counting(watch, paths, *args, **kwargs):
        seen.append(len(paths))
        return real(watch, paths, *args, **kwargs)

    monkeypatch.setattr(run_memo, "_vouched", counting)
    return seen


def _keys(paths: list[str]) -> dict:
    present, keys = statement_lineage._present_file_keys(set(paths))
    assert present == frozenset(keys)
    _next_run()
    return dict(keys)


def _fresh_keys(paths: list[str]) -> dict:
    return {p: stat_key(os.stat(p)) for p in paths if os.path.exists(p)}


def test_a_refused_set_is_looked_over_a_few_times_not_each_run(linked, vouched):
    assert file_watch.watcher() is not None, "no folder watch: the count below would prove nothing"

    for _ in range(RUNS):
        assert _keys(linked) == _fresh_keys(linked)

    assert not statement_lineage._HELD_FILE_KEYS._entries, "the watch vouched for linked files"
    # The first run, then after 1, 2, 4 and 8 runs passed over: not 12.
    assert 1 <= len(vouched) <= 4, vouched


def test_a_change_among_them_is_seen_each_run(linked, vouched):
    for run in range(RUNS):
        p = linked[run * 7]
        st = os.stat(p)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(" EDITED")
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        if run == 5:
            os.unlink(linked[1])
        assert _keys(linked) == _fresh_keys(linked)


def test_a_set_the_watch_comes_to_vouch_for_is_held(linked, tmp_path, vouched):
    _keys(linked)
    assert not statement_lineage._HELD_FILE_KEYS._entries
    for p in (tmp_path / "cache").iterdir():
        p.unlink()
    file_watch.reset()
    _next_run()

    for _ in range(RUNS):
        assert _keys(linked) == _fresh_keys(linked)

    assert statement_lineage._HELD_FILE_KEYS._entries, "a set the watch vouches for was not held"
