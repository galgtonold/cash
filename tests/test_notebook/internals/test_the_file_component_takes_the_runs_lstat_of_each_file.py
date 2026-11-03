"""The lineage component of a statement's files takes the cell run's lstat of each.

Before a trivial cell below a read of a photo collection, the cell run
stat-ed each of 19,000 files once (`run_memo.stats_this_run`), and the file
component of the statement that read them then took an ``lstat`` of each
again to learn it is no link (`realpath_of_read_this_run`), 1.4 s on
Windows. The run's stat is now an ``lstat`` for anything that is no link,
and the component takes it; a link is stat-ed through, and resolved as
before.
"""

from __future__ import annotations

import os

import pytest

from cash.notebook import run_memo
from cash.notebook.run_memo import file_component_this_run, stats_this_run
from cash.notebook.statement.file_deps import compute_file_hash_component
from cash.tracking import file_dep_snapshot
from cash.tracking.file_dep_snapshot import realpath_of_read_this_run

N = 30


@pytest.fixture(autouse=True)
def _one_cell_run(monkeypatch):
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    run_memo._FILE_STATE_THIS_RUN.clear()
    file_dep_snapshot.begin_file_state_epoch()
    yield
    while file_dep_snapshot.file_state_epoch() is not None:
        file_dep_snapshot.end_file_state_epoch()
    run_memo._FILE_STATE_THIS_RUN.clear()


def _next_cell_run() -> None:
    file_dep_snapshot.end_file_state_epoch()
    file_dep_snapshot.begin_file_state_epoch()


@pytest.fixture
def photos(tmp_path) -> list[str]:
    folder = tmp_path / "photos"
    folder.mkdir()
    paths = []
    for i in range(N):
        p = folder / f"img{i:03d}.bin"
        p.write_bytes(bytes([i]) * 16)
        paths.append(str(p))
    return paths


@pytest.fixture
def linked(tmp_path) -> str:
    """A file read through a link in another folder."""
    target = tmp_path / "elsewhere" / "target.bin"
    target.parent.mkdir()
    target.write_bytes(b"the target" * 3)
    link = tmp_path / "photos_link.bin"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("no symlinks here")
    return str(link)


@pytest.fixture
def lstats_of_files(monkeypatch) -> list[str]:
    """The ``lstat`` calls made of a file (not a folder) by its name."""
    seen: list[str] = []
    real = os.lstat

    def counting(path, *args, **kwargs):
        name = os.path.basename(os.fspath(path))
        if name.endswith(".bin"):
            seen.append(os.fspath(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", counting)
    return seen


def test_the_component_after_the_runs_pass_takes_no_lstat_of_a_file(photos, lstats_of_files):
    stats_this_run(photos)  # the cell run's pass over the files above
    lstats_of_files.clear()

    file_component_this_run(set(photos))

    assert lstats_of_files == []


def test_the_component_is_the_one_taken_with_stats_of_its_own(photos, linked):
    files = {*photos, linked}
    stats_this_run(files)
    shared = file_component_this_run(set(files))
    _next_cell_run()

    assert compute_file_hash_component(set(files)) == shared


def test_a_link_is_stat_ed_through_and_resolved_to_its_target(linked):
    _resolved, st = stats_this_run([linked])[linked]

    assert (st.st_size, st.st_ino) == (os.stat(linked).st_size, os.stat(linked).st_ino)
    file_component_this_run({linked})
    assert realpath_of_read_this_run(linked)[0] == os.path.realpath(linked)
    assert os.path.realpath(linked) != linked


def test_a_relative_read_is_resolved_as_before(photos, monkeypatch, lstats_of_files):
    """The control: a relative spelling is stat-ed from the pass's working
    directory, and resolved on its own."""
    monkeypatch.chdir(os.path.dirname(photos[0]))
    names = {os.path.basename(p) for p in photos}
    stats_this_run(names)
    shared = file_component_this_run(set(names))
    _next_cell_run()

    assert compute_file_hash_component(set(names)) == shared
    assert {realpath_of_read_this_run(n)[0] for n in names} == {os.path.realpath(p) for p in photos}
