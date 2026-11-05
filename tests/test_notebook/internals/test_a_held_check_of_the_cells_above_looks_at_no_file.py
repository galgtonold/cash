"""The check of the cells above costs what changed, not what they read.

Before each cell the upstream check asks whether each cached cell's files
still have the stats it recorded. Below a cell that read 25,000 files every
trivial cell went through each of them in Python, ~0.2 s a cell on Linux and
more on Windows. Under the folder watch the verdict is now held across cells
(`run_memo.stat_keys_as_recorded`): a later check looks at no file of a
folder that reported nothing, and only at the files of one that did.
"""

from __future__ import annotations

import collections
import os
import sys

import pytest

from cash.notebook import run_memo
from cash.tracking import file_dep_snapshot, file_watch
from cash.tracking.read_classification import stat_key

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="folders are not watched on macOS")

FOLDERS, FILES = 4, 50


@pytest.fixture(autouse=True)
def _runs(monkeypatch):
    monkeypatch.delenv("CASH_FILE_WATCH", raising=False)
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    run_memo._FILE_STATE_THIS_RUN.clear()
    run_memo._HELD_KEYS.clear()
    file_watch.reset()
    file_dep_snapshot.begin_file_state_epoch()
    yield
    file_dep_snapshot.end_file_state_epoch()
    run_memo._FILE_STATE_THIS_RUN.clear()
    run_memo._HELD_KEYS.clear()
    file_watch.reset()


def _ask(recorded: dict) -> bool:
    answer = run_memo.stat_keys_as_recorded(recorded)
    file_dep_snapshot.end_file_state_epoch()
    file_dep_snapshot.begin_file_state_epoch()
    return answer


@pytest.fixture
def recorded(tmp_path, monkeypatch) -> dict:
    monkeypatch.chdir(tmp_path)
    deps = {}
    for d in range(FOLDERS):
        folder = tmp_path / f"set{d}"
        folder.mkdir()
        for i in range(FILES):
            p = folder / f"f{i:02d}.txt"
            p.write_text(f"file {d} {i:02d}", encoding="utf-8")
            deps[f"set{d}/f{i:02d}.txt" if i % 2 else str(p)] = stat_key(os.stat(p))
    return deps


@pytest.fixture
def looks(monkeypatch, tmp_path) -> collections.Counter:
    """Each stat and lstat of a path under *tmp_path*, and each path the
    watch is asked about (``current``)."""
    seen: collections.Counter = collections.Counter()
    root = str(tmp_path)

    def counting(kind, real):
        def call(path, *args, **kwargs):
            if os.fspath(path).startswith(root) or not os.path.isabs(os.fspath(path)):
                seen[kind] += 1
            return real(path, *args, **kwargs)

        return call

    def start():
        monkeypatch.setattr(os, "stat", counting("stat", os.stat))
        monkeypatch.setattr(os, "lstat", counting("lstat", os.lstat))
        watch = file_watch.watcher()
        monkeypatch.setattr(watch, "current", counting("current", watch.current))

    seen.start = start  # type: ignore[attr-defined]
    return seen


def test_a_later_check_looks_at_no_file(recorded, looks):
    assert _ask(recorded) and _ask(recorded)
    looks.start()

    assert _ask(recorded)

    # Each watched folder's own stat before the run (`FileWatch.drain`), no file.
    assert looks["stat"] <= FOLDERS, looks
    assert looks["lstat"] == 0, looks
    assert looks["current"] == 0, looks


def test_a_change_in_one_folder_looks_again_only_at_its_files(recorded, looks, tmp_path):
    assert _ask(recorded) and _ask(recorded)
    (tmp_path / "set2" / "notes.ipynb").write_text("{}", encoding="utf-8")  # a save beside the files
    looks.start()

    assert _ask(recorded)
    assert looks["stat"] + looks["lstat"] <= FOLDERS + 2 * FILES, looks

    looks.clear()
    assert _ask(recorded)
    assert looks["stat"] <= FOLDERS and looks["lstat"] == 0 and looks["current"] == 0, looks
