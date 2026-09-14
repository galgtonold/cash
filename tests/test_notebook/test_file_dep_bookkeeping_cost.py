"""File-dependency bookkeeping must cost each file once, not once per use.

Round 23's notebooks were slower with cash than without, and the profile put
nearly all of it in the same place (2026-09-14):

* r23s4 read 5,222 files into ``docs``. Every statement derived from it
  inherited all 5,222, and each save, each lookup and each upstream simulation
  re-hashed every one: 19-108 s of hashing per cell, against a notebook that
  runs in about 25 s uncached. The digest memo gave up at that scale -- 4,096
  entries, and a 5 s window shorter than one pass over the files.
* r23s2 read 1,312 CSVs in a loop. ``d = pd.read_csv(f)`` rebinds ``d`` each
  iteration, but its recorded files were MERGED, so iteration k snapshotted all
  k files so far: 865,265 hashes in one cell.
"""
from __future__ import annotations

import hashlib
import os
import types

import pytest

from cash.notebook import file_dep_snapshot
from cash.notebook.statement.file_deps import StatementFileDeps


def _state():
    return types.SimpleNamespace(executed_file_deps={}, executed_file_mtimes={})


def _aged(tmp_path, name, body=b"x" * 2048):
    path = tmp_path / name
    path.write_bytes(body)
    st = os.stat(path)
    os.utime(path, (st.st_atime - 3600, st.st_mtime - 3600))
    return str(path)


class _Frame:
    """Stands in for a DataFrame: not a scalar, so it inherits file deps."""


# --------------------------------------------------------------------------- #
# Rebinding replaces a variable's files; an in-place change adds to them.    #
# --------------------------------------------------------------------------- #

def test_a_rebound_variable_keeps_only_its_new_files(tmp_path):
    a, b = _aged(tmp_path, "a.csv"), _aged(tmp_path, "b.csv")
    state, deps = _state(), StatementFileDeps()
    deps.update_for_var(state, "d", {a}, set(), _Frame(), rebind=True)
    deps.update_for_var(state, "d", {b}, set(), _Frame(), rebind=True)
    assert state.executed_file_deps["d"] == {b}, "d = read(b) still depends on a"


def test_an_in_place_change_keeps_the_files_it_had(tmp_path):
    a, b = _aged(tmp_path, "a.csv"), _aged(tmp_path, "b.csv")
    state, deps = _state(), StatementFileDeps()
    deps.update_for_var(state, "df", {a}, set(), _Frame(), rebind=True)
    deps.update_for_var(state, "df", {b}, {"df"}, _Frame(), rebind=False)
    assert state.executed_file_deps["df"] == {a, b}


def test_a_restored_variable_carries_exactly_its_entry_files(tmp_path):
    a, b = _aged(tmp_path, "a.csv"), _aged(tmp_path, "b.csv")
    state, deps = _state(), StatementFileDeps()
    for path in (a, b):
        meta = types.SimpleNamespace(file_dependencies={path: {"mtime": 0.0, "size": 1}})
        deps.restore_from_metadata(state, {"d": object()}, meta)
    assert state.executed_file_deps["d"] == {b}


# --------------------------------------------------------------------------- #
# One digest per file per cell run.                                          #
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _no_cell_run_leaks(monkeypatch):
    """Each test starts and ends outside any cell run."""
    monkeypatch.setattr(file_dep_snapshot, "_HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)


@pytest.fixture
def count_hashes(monkeypatch):
    hashed: list[int] = []
    monkeypatch.setattr(file_dep_snapshot, "hashlib", types.SimpleNamespace(
        sha256=lambda *a: hashed.append(1) or hashlib.sha256(*a)))
    file_dep_snapshot._HASH_MEMO.clear()
    yield hashed
    file_dep_snapshot._HASH_MEMO.clear()


def test_one_cell_run_hashes_each_file_once(tmp_path, monkeypatch, count_hashes):
    """A pass over thousands of files outlasts the 5 s window; within one cell
    run a digest stays good however long the run takes."""
    path = _aged(tmp_path, "a.csv")
    clock = [1000.0]
    monkeypatch.setattr(file_dep_snapshot.time, "monotonic", lambda: clock[0])
    file_dep_snapshot.begin_file_state_epoch()
    file_dep_snapshot.file_content_hash(path)
    clock[0] += 600                                 # ten minutes into the same cell
    file_dep_snapshot.file_content_hash(path)
    assert len(count_hashes) == 1, "one cell run hashed an unchanged file twice"
    file_dep_snapshot.end_file_state_epoch()

    file_dep_snapshot.begin_file_state_epoch()      # the next cell looks again
    file_dep_snapshot.file_content_hash(path)
    assert len(count_hashes) == 2
    file_dep_snapshot.end_file_state_epoch()


def test_between_cell_runs_the_window_applies(tmp_path, monkeypatch, count_hashes):
    """A cell run's digests end with it. Left current until the next cell,
    anything running in between -- a thread the cell started, a cached
    function called from a callback -- reused them indefinitely, and so did
    every script-style test that ran after a notebook one in the same process."""
    path = _aged(tmp_path, "a.csv")
    clock = [1000.0]
    monkeypatch.setattr(file_dep_snapshot.time, "monotonic", lambda: clock[0])
    file_dep_snapshot.begin_file_state_epoch()
    file_dep_snapshot.file_content_hash(path)
    file_dep_snapshot.end_file_state_epoch()
    clock[0] += 60
    file_dep_snapshot.file_content_hash(path)
    assert len(count_hashes) == 2, "a digest from a finished cell run was reused a minute later"


def test_a_nested_cell_run_is_the_same_run(tmp_path, monkeypatch, count_hashes):
    """`%%cash` inside a hooked cell runs the pipeline again from within it."""
    path = _aged(tmp_path, "a.csv")
    clock = [1000.0]
    monkeypatch.setattr(file_dep_snapshot.time, "monotonic", lambda: clock[0])
    file_dep_snapshot.begin_file_state_epoch()
    file_dep_snapshot.file_content_hash(path)
    file_dep_snapshot.begin_file_state_epoch()
    clock[0] += 60
    file_dep_snapshot.file_content_hash(path)
    file_dep_snapshot.end_file_state_epoch()
    clock[0] += 60
    file_dep_snapshot.file_content_hash(path)       # still the outer run
    file_dep_snapshot.end_file_state_epoch()
    assert len(count_hashes) == 1


def test_outside_a_cell_run_the_window_still_bounds_reuse(tmp_path, monkeypatch, count_hashes):
    """Scripts never begin an epoch: the 5 s window is what they get."""
    path = _aged(tmp_path, "a.csv")
    monkeypatch.setattr(file_dep_snapshot, "_HASH_EPOCH", None)
    clock = [1000.0]
    monkeypatch.setattr(file_dep_snapshot.time, "monotonic", lambda: clock[0])
    file_dep_snapshot.file_content_hash(path)
    clock[0] += 60
    file_dep_snapshot.file_content_hash(path)
    assert len(count_hashes) == 2


def test_a_full_memo_keeps_memoizing(tmp_path, monkeypatch, count_hashes):
    """At 4,096 entries the memo stopped taking new ones, so file 4,097 on was
    hashed every time (r23s4: 5,222 files)."""
    monkeypatch.setattr(file_dep_snapshot, "_HASH_MEMO_MAX", 3)
    file_dep_snapshot.begin_file_state_epoch()
    paths = [_aged(tmp_path, f"f{i}.csv") for i in range(5)]
    for p in paths:
        file_dep_snapshot.file_content_hash(p)
    before = len(count_hashes)
    file_dep_snapshot.file_content_hash(paths[-1])
    assert len(count_hashes) == before, "the newest file was re-hashed once the memo was full"


def test_two_spellings_of_one_file_share_a_digest(tmp_path, monkeypatch, count_hashes):
    """A relative read is recorded as written and resolved (so a chdir is
    seen); both spellings name one file and were hashed once each."""
    _aged(tmp_path, "a.csv")
    monkeypatch.chdir(tmp_path)
    file_dep_snapshot.begin_file_state_epoch()
    rel = file_dep_snapshot.file_content_hash("a.csv")
    absolute = file_dep_snapshot.file_content_hash(str(tmp_path / "a.csv"))
    assert rel == absolute
    assert len(count_hashes) == 1


def test_without_a_file_identity_the_path_keeps_files_apart(tmp_path, count_hashes):
    """Where the filesystem reports no inode, two files with the same size and
    timestamps must not share a digest (CAS-108: two release copies laid down
    by one deploy did)."""
    a = _aged(tmp_path, "a.csv", b"x" * 2048)
    b = _aged(tmp_path, "b.csv", b"y" * 2048)
    fields = list(os.stat(a))
    fields[1] = 0                                    # st_ino
    same = os.stat_result(fields)
    file_dep_snapshot.begin_file_state_epoch()
    assert (file_dep_snapshot.file_content_hash(a, st=same)
            != file_dep_snapshot.file_content_hash(b, st=same))


def test_an_edit_between_cell_runs_is_seen(tmp_path, monkeypatch, count_hashes):
    """The control: a same-size edit with the mtime put back is caught by the
    next cell run once the 5 s window has passed -- the bound it had before."""
    path = _aged(tmp_path, "a.csv")
    clock = [1000.0]
    monkeypatch.setattr(file_dep_snapshot.time, "monotonic", lambda: clock[0])
    file_dep_snapshot.begin_file_state_epoch()
    first = file_dep_snapshot.file_content_hash(path)
    before = os.stat(path)
    with open(path, "r+b") as fh:
        fh.write(b"Z")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    file_dep_snapshot.end_file_state_epoch()
    clock[0] += 10
    file_dep_snapshot.begin_file_state_epoch()
    assert file_dep_snapshot.file_content_hash(path) != first
    file_dep_snapshot.end_file_state_epoch()
