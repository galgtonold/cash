"""A held verdict on the files of the cells above still sees each change.

Before each cell, the upstream check asks whether every file a cached cell
read still has the stat it recorded (`run_memo.stat_keys_as_recorded`).
Under the folder watch the verdict is kept across cells, and a later check
looks again only at the files of a folder that reported a change and at
those no folder vouches for. Each test changes a file in a way that verdict
must not survive, and checks the next ask says so. They pass without the
held verdict too: every file was stat-ed before every cell.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

from cash.notebook import run_memo
from cash.tracking import file_dep_snapshot, file_watch
from cash.tracking.read_classification import stat_key

FOLDERS, FILES = 3, 12

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="folders are not watched on macOS")


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
    """One cell's check of *recorded*, then the next cell run begins."""
    answer = run_memo.stat_keys_as_recorded(recorded)
    file_dep_snapshot.end_file_state_epoch()
    file_dep_snapshot.begin_file_state_epoch()
    return answer


@pytest.fixture
def recorded(tmp_path, monkeypatch) -> dict:
    """``{path: stat_key}`` of files in a few folders, half by a relative
    path, as a cell that read them records them; held by two checks."""
    monkeypatch.chdir(tmp_path)
    deps = {}
    for d in range(FOLDERS):
        folder = tmp_path / f"set{d}"
        folder.mkdir()
        for i in range(FILES):
            p = folder / f"f{i:02d}.txt"
            p.write_text(f"file {d} {i:02d}", encoding="utf-8")
            spelled = f"set{d}/f{i:02d}.txt" if i % 2 else str(p)
            deps[spelled] = stat_key(os.stat(p))
    assert _ask(deps) and _ask(deps)
    assert run_memo._HELD_KEYS, "no verdict was held: the tests below would prove nothing"
    return deps


def _same_size_edit(path) -> None:
    before = os.stat(path)
    text = open(path, encoding="utf-8").read()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text.upper())
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))  # mtime put back, as ``cp -p``


def test_an_unchanged_set_holds(recorded):
    assert _ask(recorded)
    assert _ask(dict(recorded)), "an equal dict built anew is answered too"


@pytest.mark.parametrize("name", ["set1/f03.txt", "set2/f04.txt"], ids=["relative", "absolute"])
def test_a_same_size_edit_is_seen(recorded, tmp_path, name):
    _same_size_edit(tmp_path / name)
    assert not _ask(recorded)


def test_a_deleted_file_is_seen(recorded, tmp_path):
    (tmp_path / "set0" / "f05.txt").unlink()
    assert not _ask(recorded)


def test_a_file_renamed_over_one_is_seen(recorded, tmp_path):
    other = tmp_path / "other.txt"
    other.write_text("file 0 07", encoding="utf-8")  # same size
    os.replace(other, tmp_path / "set0" / "f07.txt")
    assert not _ask(recorded)


def test_a_change_after_a_held_answer_on_a_changed_folder_is_seen(recorded, tmp_path):
    """A folder that reported a change has its files checked again and held
    again: a second change in it is seen too."""
    (tmp_path / "set1" / "notes.ipynb").write_text("{}", encoding="utf-8")  # a save beside the files
    assert _ask(recorded)
    _same_size_edit(tmp_path / "set1" / "f02.txt")
    assert not _ask(recorded)


def test_another_record_of_the_same_files_is_not_answered_by_the_held_one(recorded):
    other = dict(recorded)
    first = next(iter(other))
    size, mtime_ns, ctime_ns = other[first]
    other[first] = (size, mtime_ns - 1, ctime_ns)
    assert not _ask(other)
    assert _ask(recorded)


def test_a_file_edited_through_a_second_hard_link_is_seen(recorded, tmp_path):
    """A file with a second name is never vouched for by its folder: written
    through the other name, its folder reports nothing."""
    target = tmp_path / "set0" / "f00.txt"
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    try:
        os.link(target, outside / "alias.txt")
    except OSError:
        pytest.skip("cannot make a hard link here")
    file_watch.reset()  # the stats below are taken with the second name there
    deps = dict(recorded)
    deps[str(target)] = stat_key(os.stat(target))  # its link count moved its change time
    assert _ask(deps) and _ask(deps)
    with open(outside / "alias.txt", "w", encoding="utf-8") as fh:
        fh.write("FILE 0 00")
    assert not _ask(deps)


@pytest.mark.xfail(
    sys.platform.startswith("linux"),
    strict=True,
    reason="a second name made for a file after its stat was kept is reported to no watched folder",
)
def test_a_file_linked_and_edited_elsewhere_after_its_stat_was_kept_is_seen(recorded, tmp_path):
    """Known gap of the folder watch itself (`file_watch`): the link and the
    write through it are reported to the other folder only, so the kept
    stat of the first name stands."""
    target = tmp_path / "set0" / "f00.txt"
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    try:
        os.link(target, outside / "alias.txt")
    except OSError:
        pytest.skip("cannot make a hard link here")
    with open(outside / "alias.txt", "w", encoding="utf-8") as fh:
        fh.write("FILE 0 00")
    assert not _ask(recorded)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="no symlinks")
def test_an_edit_to_a_linked_files_target_is_seen(recorded, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    real = outside / "real.txt"
    real.write_text("real one", encoding="utf-8")
    link = tmp_path / "set2" / "link.txt"
    try:
        os.symlink(real, link)
    except OSError:
        pytest.skip("cannot make a symlink here")
    deps = dict(recorded)
    deps[str(link)] = stat_key(os.stat(link))
    assert _ask(deps) and _ask(deps)
    real.write_text("REAL ONE", encoding="utf-8")
    assert not _ask(deps)


def test_an_edit_after_the_watch_was_closed_is_seen(recorded, tmp_path):
    file_watch.reset()  # its folders report nothing from now on
    _same_size_edit(tmp_path / "set2" / "f06.txt")
    assert not _ask(recorded)


def test_relative_paths_in_another_working_directory_are_looked_at_there(recorded, tmp_path, monkeypatch):
    elsewhere = tmp_path / "copy"
    for d in range(FOLDERS):
        (elsewhere / f"set{d}").mkdir(parents=True)
        for i in range(FILES):
            (elsewhere / f"set{d}" / f"f{i:02d}.txt").write_text(f"other {d} {i:02d}!", encoding="utf-8")
    monkeypatch.chdir(elsewhere)
    assert not _ask(recorded)


def test_an_edit_made_by_a_statement_of_the_same_run_is_seen(recorded, tmp_path):
    """A statement of the cell run writes a file the cells above read: the
    next check in the run takes the change, as the run's stats do."""
    assert run_memo.stat_keys_as_recorded(recorded)
    time.sleep(0.01)
    (tmp_path / "set0" / "f09.txt").write_text("rewritten in the run", encoding="utf-8")
    run_memo.forget_file_state_this_run()  # what a statement that wrote files does
    assert not run_memo.stat_keys_as_recorded(recorded)
