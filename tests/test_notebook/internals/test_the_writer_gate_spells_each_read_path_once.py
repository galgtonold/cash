"""The writer gate spells each read path in its comparable forms once, not before every cell.

Each cell's check asks the gate with a new set of the same read paths, and
spelling each one (absolute, case-folded, its base name) cost more than the
stats: 0.2-0.5 s before every cell below a loop over 24,000 documents. The
forms of a path follow from the path, where this run's stat found it and the
working directory, so they are kept while those stay the same. Whether a
path is a folder is read from each run's stat.
"""

from __future__ import annotations

import os

import pytest

from cash.notebook import run_memo
from cash.notebook.upstream.file_writers import FileWriterScheduler
from cash.tracking import file_dep_snapshot

N = 30


@pytest.fixture(autouse=True)
def _no_run(monkeypatch):
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    run_memo._FILE_STATE_THIS_RUN.clear()
    yield
    run_memo._FILE_STATE_THIS_RUN.clear()


def _cell_run(gate: FileWriterScheduler, paths: set[str]):
    """The gate's index as one cell's check asks for it: a run of its own,
    and a new set of the paths."""
    file_dep_snapshot.begin_file_state_epoch()
    try:
        return gate._read_path_index(set(paths))
    finally:
        file_dep_snapshot.end_file_state_epoch()


def _reads(tmp_path) -> set[str]:
    folder = tmp_path / "docs"
    folder.mkdir()
    paths = {str(folder)}
    for i in range(N):
        p = folder / f"{i}_7.txt"
        p.write_text(f"review {i}\n", encoding="utf-8")
        paths.add(str(p))
    paths.add(str(tmp_path / "gone.txt"))
    return paths


@pytest.fixture
def spellings(monkeypatch):
    calls: list[str] = []
    real = os.path.abspath

    def counting(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(os.path, "abspath", counting)
    return calls


def _same(a, b) -> None:
    assert a[0] == b[0]
    assert sorted(a[1]) == sorted(b[1])
    assert sorted(a[2]) == sorted(b[2])


def test_the_next_cell_spells_no_path_again(tmp_path, spellings):
    paths = _reads(tmp_path)
    gate = FileWriterScheduler(None, None, None)
    first = _cell_run(gate, paths)
    assert len(spellings) >= len(paths)
    spellings.clear()

    second = _cell_run(gate, paths)

    assert spellings == []
    _same(second, first)
    _same(second, FileWriterScheduler(None, None, None)._read_path_index(set(paths)))


def test_a_file_that_became_a_folder_is_a_folder_next_cell(tmp_path):
    paths = _reads(tmp_path)
    out = tmp_path / "out"
    out.write_text("x", encoding="utf-8")
    paths.add(str(out))
    gate = FileWriterScheduler(None, None, None)
    assert os.path.normcase(str(out)) + os.sep not in _cell_run(gate, paths)[1]

    out.unlink()
    out.mkdir()

    assert os.path.normcase(str(out)) + os.sep in _cell_run(gate, paths)[1]


def test_a_relative_path_is_spelled_again_in_another_working_directory(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    for d in ("a", "b"):
        (tmp_path / d / "data.csv").write_text("1\n", encoding="utf-8")
    gate = FileWriterScheduler(None, None, None)
    monkeypatch.chdir(tmp_path / "a")
    in_a = _cell_run(gate, {"data.csv"})
    monkeypatch.chdir(tmp_path / "b")

    in_b = _cell_run(gate, {"data.csv"})

    assert os.path.normcase(str(tmp_path / "b" / "data.csv")) in in_b[0]
    assert os.path.normcase(str(tmp_path / "a" / "data.csv")) not in in_b[0]
    assert in_a[2] != in_b[2]
