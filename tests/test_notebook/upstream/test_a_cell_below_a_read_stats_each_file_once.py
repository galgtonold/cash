"""A statement's file component is built from the stats the cell run took already.

The cell below a read of 5,000 small files stat-ed each of them five times:
once to find where it was recorded, once to check it had not moved, and once
more for each of three builds of the same file component, each looking for
the file again and taking a stat of its own. The component now takes the
run's stats, and is kept for the rest of the run as they are: the same
fields of the same files, so the same component.
"""

from __future__ import annotations

import os

import pytest

from cash.notebook import run_memo
from cash.notebook.run_memo import file_component_this_run, forget_file_state_this_run, stats_this_run
from cash.notebook.statement.file_deps import compute_file_hash_component
from cash.tracking import file_dep_snapshot

N = 30


@pytest.fixture(autouse=True)
def _one_cell_run(monkeypatch):
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    run_memo._FILE_STATE_THIS_RUN.clear()
    file_dep_snapshot.begin_file_state_epoch()
    yield
    file_dep_snapshot.end_file_state_epoch()


def _inputs(tmp_path) -> set[str]:
    paths = set()
    for i in range(N):
        p = tmp_path / f"doc{i:03d}.txt"
        p.write_text(f"document {i}\n", encoding="utf-8")
        paths.add(str(p))
    return paths


@pytest.fixture
def stats_of(monkeypatch, tmp_path):
    """The ``os.stat`` and ``os.lstat`` calls on the inputs and their folder."""
    calls: list[str] = []
    # Either spelling: a recorded path may use forward slashes on Windows.
    root = os.path.normcase(os.path.normpath(str(tmp_path)))

    def counting(real):
        def call(path, *a, **k):
            if os.path.normcase(os.path.normpath(os.fspath(path))).startswith(root):
                calls.append(str(path))
            return real(path, *a, **k)

        return call

    monkeypatch.setattr(os, "stat", counting(os.stat))
    monkeypatch.setattr(os, "lstat", counting(os.lstat))
    return calls


def test_the_component_is_the_one_built_from_scratch(tmp_path):
    paths = _inputs(tmp_path)
    expected = compute_file_hash_component(paths)
    stats_this_run(paths)

    assert file_component_this_run(paths) == expected
    assert file_component_this_run(paths) == expected  # kept for the run


def test_the_component_looks_at_each_file_once_in_a_run(tmp_path, stats_of):
    """Three lookups of the same statement's files: a look at each file and
    its folder, not one per lookup (3N + 1 before)."""
    paths = _inputs(tmp_path)
    stats_this_run(paths)  # the check that no input moved
    taken = len(stats_of)
    for _ in range(3):
        file_component_this_run(paths)

    assert len(stats_of) - taken <= N + 1


def test_a_write_in_the_run_gives_a_new_component(tmp_path):
    paths = _inputs(tmp_path)
    before = file_component_this_run(paths)
    target = sorted(paths)[0]
    with open(target, "a", encoding="utf-8") as f:
        f.write("more\n")
    forget_file_state_this_run()  # what a statement that writes files does

    after = file_component_this_run(paths)
    assert after != before
    assert after == compute_file_hash_component(paths)
