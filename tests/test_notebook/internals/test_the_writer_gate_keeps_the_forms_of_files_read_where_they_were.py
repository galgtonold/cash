"""The writer gate keeps the forms of read files across cell runs.

Before every cell, the gate deciding whether a writer's output is read
indexes every file the cells above read: each path's absolute form and its
name. Below a loop over 50,000 documents that was 100,000 paths (a relative
and an absolute spelling of each), two absolute paths per path, before each
cell, for the same answer. A file found where it was recorded has forms that
depend on its path and the working folder alone, so they are kept while
both stay the same; a folder, a file that moved and a change of working
folder are indexed again.
"""

from __future__ import annotations

import os
import stat

import pytest

from cash._paths import resolve_file_dep_path
from cash.notebook import run_memo
from cash.notebook.upstream.file_writers import FileWriterScheduler
from cash.tracking import file_dep_snapshot

N = 40


@pytest.fixture(autouse=True)
def _one_cell_run(monkeypatch):
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    run_memo._FILE_STATE_THIS_RUN.clear()
    file_dep_snapshot.begin_file_state_epoch()
    yield
    file_dep_snapshot.end_file_state_epoch()


def _next_cell_run() -> None:
    file_dep_snapshot.end_file_state_epoch()
    file_dep_snapshot.begin_file_state_epoch()


def _tree(root) -> None:
    folder = root / "docs"
    folder.mkdir(parents=True)
    for i in range(N):
        (folder / f"{i}_7.txt").write_text(f"review {i}\n", encoding="utf-8")


def _reads(root) -> set[str]:
    """Both spellings of each document, as the tracker records a relative
    read, plus the folder and a file that is gone."""
    paths = {"docs", str(root / "docs"), str(root / "gone.txt")}
    for i in range(N):
        paths.add(os.path.join("docs", f"{i}_7.txt"))
        paths.add(str(root / "docs" / f"{i}_7.txt"))
    return paths


def _reference(paths) -> tuple[set[str], list[str], list[str]]:
    """The index built from scratch, path by path."""
    forms: set[str] = set()
    dirs: list[str] = []
    places: list[str] = []
    for rp in paths:
        try:
            is_dir = stat.S_ISDIR(os.stat(rp).st_mode)
            resolved = rp
        except OSError:
            resolved = resolve_file_dep_path(rp) or rp
            is_dir = resolved != rp and os.path.isdir(resolved)
        for candidate in {resolved, rp}:
            forms.add(os.path.normcase(os.path.abspath(candidate)))
            forms.add(os.path.normcase(os.path.basename(candidate)))
        places.append(os.path.normcase(os.path.abspath(resolved)))
        if is_dir:
            dirs.append(os.path.normcase(os.path.abspath(resolved)) + os.sep)
    return forms, sorted(dirs), sorted(places)


def _as_built(index) -> tuple[set[str], list[str], list[str]]:
    forms, dirs, places = index
    return set(forms), sorted(dirs), sorted(places)


@pytest.fixture
def abspaths(monkeypatch):
    """How many absolute paths the index takes."""
    calls: list[str] = []
    real = os.path.abspath

    def counting(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(os.path, "abspath", counting)
    return calls


def test_the_next_cell_takes_the_kept_forms(tmp_path, monkeypatch, abspaths):
    _tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    gate = FileWriterScheduler(None, None, None)
    first = _as_built(gate._read_path_index(_reads(tmp_path)))
    assert first == _reference(_reads(tmp_path))

    _next_cell_run()
    abspaths.clear()
    again = gate._read_path_index(_reads(tmp_path))  # a new set, read again

    assert len(abspaths) < N  # the folders and the missing file, not each document
    assert _as_built(again) == first


def test_a_cell_reading_part_of_them_takes_the_kept_forms(tmp_path, monkeypatch, abspaths):
    """A cell that needs other inputs reads another set of the same files."""
    _tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    gate = FileWriterScheduler(None, None, None)
    gate._read_path_index(_reads(tmp_path))

    _next_cell_run()
    abspaths.clear()
    part = {p for p in _reads(tmp_path) if not p.endswith("0_7.txt")}
    index = _as_built(gate._read_path_index(part))

    assert len(abspaths) < N
    assert index == _reference(part)


def test_another_working_folder_indexes_the_relative_reads_again(tmp_path, monkeypatch):
    """The same relative spellings name other files from another folder."""
    _tree(tmp_path)
    _tree(tmp_path / "copy")
    monkeypatch.chdir(tmp_path)
    gate = FileWriterScheduler(None, None, None)
    gate._read_path_index(_reads(tmp_path))

    _next_cell_run()
    monkeypatch.chdir(tmp_path / "copy")
    moved = _as_built(gate._read_path_index(_reads(tmp_path)))

    assert moved == _reference(_reads(tmp_path))
    assert os.path.normcase(str(tmp_path / "copy" / "docs" / "0_7.txt")) in moved[0]


def test_a_file_that_became_a_folder_or_went_is_indexed_again(tmp_path, monkeypatch):
    _tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    gate = FileWriterScheduler(None, None, None)
    gate._read_path_index(_reads(tmp_path))

    _next_cell_run()
    (tmp_path / "docs" / "0_7.txt").unlink()
    (tmp_path / "docs" / "0_7.txt").mkdir()
    (tmp_path / "docs" / "1_7.txt").unlink()
    changed = _as_built(gate._read_path_index(_reads(tmp_path)))

    assert changed == _reference(_reads(tmp_path))
    assert os.path.normcase(str(tmp_path / "docs" / "0_7.txt")) + os.sep in changed[1]
