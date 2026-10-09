"""File-dependency bookkeeping must cost each file once, not once per use.

Some real notebooks were slower with cash than without, and the profile put
nearly all of it in the same place:

* One read 5,222 files into ``docs``. Every statement derived from it
  inherited all 5,222, and each save, each lookup and each upstream simulation
  re-hashed every one: 19-108 s of hashing per cell, against a notebook that
  runs in about 25 s uncached. The digest memo gave up at that scale -- 4,096
  entries, and a 5 s reuse window shorter than one pass over the files.
* Another read 1,312 CSVs in a loop. ``d = pd.read_csv(f)`` rebinds ``d`` each
  iteration, but its recorded files were MERGED, so iteration k snapshotted all
  k files so far: 865,265 hashes in one cell.
"""

from __future__ import annotations

import os
import types

import pytest

from cash.notebook.restored_var import apply_restored_var
from cash.notebook.statement import StatementCacheMetadata
from cash.notebook.statement.file_deps import StatementFileDeps
from cash.notebook.tracking_state import TrackingState
from cash.tracking import digest_table, file_dep_snapshot


def _state():
    return types.SimpleNamespace(executed_file_deps={})


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
    state = TrackingState()
    for path in (a, b):
        meta = StatementCacheMetadata(file_dependencies={path: {"mtime": 0.0, "size": 1}})
        apply_restored_var(state, "d", object(), meta)
    assert state.executed_file_deps["d"] == {b}


# --------------------------------------------------------------------------- #
# One digest per file per cell run.                                          #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _no_cell_run_leaks(monkeypatch):
    """Each test starts and ends outside any cell run."""
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)


@pytest.fixture
def count_hashes(monkeypatch):
    hashed: list[int] = []
    real = file_dep_snapshot.BulkHasher
    monkeypatch.setattr(file_dep_snapshot, "BulkHasher", lambda: hashed.append(1) or real())
    # Only this process's memo: no cache directory's table from another test.
    monkeypatch.setattr(digest_table, "_TABLE", None)
    file_dep_snapshot._HASH_MEMO.clear()
    yield hashed
    file_dep_snapshot._HASH_MEMO.clear()


def test_an_unchanged_settled_file_is_hashed_once(tmp_path, monkeypatch, count_hashes):
    """A pass over thousands of files outlasts any time window; a settled
    file's digest stays good while its stat does, in one cell run, in the next
    and outside any."""
    path = _aged(tmp_path, "a.csv")
    clock = [1000.0]
    monkeypatch.setattr(file_dep_snapshot.time, "monotonic", lambda: clock[0])
    file_dep_snapshot.begin_file_state_epoch()
    file_dep_snapshot.file_content_hash(path)
    clock[0] += 600  # ten minutes into the same cell
    file_dep_snapshot.file_content_hash(path)
    file_dep_snapshot.end_file_state_epoch()
    file_dep_snapshot.begin_file_state_epoch()  # the next cell
    file_dep_snapshot.file_content_hash(path)
    file_dep_snapshot.end_file_state_epoch()
    clock[0] += 600
    file_dep_snapshot.file_content_hash(path)  # outside a cell run
    assert len(count_hashes) == 1, "an unchanged settled file was hashed again"


def test_a_touched_file_is_hashed_again(tmp_path, count_hashes):
    path = _aged(tmp_path, "a.csv")
    file_dep_snapshot.file_content_hash(path)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns - 10**9))
    file_dep_snapshot.file_content_hash(path)
    assert len(count_hashes) == 2


def test_between_cell_runs_a_listed_stat_is_reused_for_a_window_only(tmp_path, monkeypatch, count_hashes):
    """A stat with no file identity (``st_ino`` 0: a Windows directory
    listing's) can lag a file edited through another hard link: its digest is
    reused within a cell run or for a few seconds, never longer. A cell run's
    digests end with it: anything running in between -- a thread the cell
    started, a cached function called from a callback -- gets the window."""
    path = _aged(tmp_path, "a.csv")
    fields = list(os.stat(path))
    fields[1] = 0  # st_ino
    listed = os.stat_result(fields)
    clock = [1000.0]
    monkeypatch.setattr(file_dep_snapshot.time, "monotonic", lambda: clock[0])
    file_dep_snapshot.begin_file_state_epoch()
    file_dep_snapshot.file_content_hash(path, st=listed)
    clock[0] += 600
    file_dep_snapshot.file_content_hash(path, st=listed)
    assert len(count_hashes) == 1, "one cell run hashed an unchanged file twice"
    file_dep_snapshot.end_file_state_epoch()
    clock[0] += 60
    file_dep_snapshot.file_content_hash(path, st=listed)
    assert len(count_hashes) == 2, "a listed stat's digest was reused a minute after its cell run"


def test_a_nested_cell_run_is_the_same_run(tmp_path, monkeypatch, count_hashes):
    """A cell that calls `get_ipython().run_cell(...)` runs the pipeline again from within it."""
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
    file_dep_snapshot.file_content_hash(path)  # still the outer run
    file_dep_snapshot.end_file_state_epoch()
    assert len(count_hashes) == 1


def test_outside_a_cell_run_a_listed_stat_gets_the_window(tmp_path, monkeypatch, count_hashes):
    """Scripts never begin an epoch: a listed stat's digest lasts 5 s."""
    path = _aged(tmp_path, "a.csv")
    fields = list(os.stat(path))
    fields[1] = 0  # st_ino
    listed = os.stat_result(fields)
    clock = [1000.0]
    monkeypatch.setattr(file_dep_snapshot.time, "monotonic", lambda: clock[0])
    file_dep_snapshot.file_content_hash(path, st=listed)
    clock[0] += 1
    file_dep_snapshot.file_content_hash(path, st=listed)
    assert len(count_hashes) == 1
    clock[0] += 60
    file_dep_snapshot.file_content_hash(path, st=listed)
    assert len(count_hashes) == 2


def test_a_full_memo_keeps_memoizing(tmp_path, monkeypatch, count_hashes):
    """At 4,096 entries the memo stopped taking new ones, so file 4,097 on was
    hashed every time (one notebook had 5,222 files)."""
    monkeypatch.setattr(file_dep_snapshot._HASH_MEMO, "maxsize", 3)
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
    timestamps must not share a digest (two release copies laid down by one
    deploy did)."""
    a = _aged(tmp_path, "a.csv", b"x" * 2048)
    b = _aged(tmp_path, "b.csv", b"y" * 2048)
    fields = list(os.stat(a))
    fields[1] = 0  # st_ino
    same = os.stat_result(fields)
    file_dep_snapshot.begin_file_state_epoch()
    assert file_dep_snapshot.file_content_hash(a, st=same) != file_dep_snapshot.file_content_hash(b, st=same)


@pytest.mark.xfail(
    os.name == "nt",
    strict=True,
    reason="Windows: an edit that keeps the size and puts the mtime back is not seen once the file had settled -- a documented limitation (known-limitations: an edit that keeps size and timestamps); Linux and macOS catch it through the inode change time",
)
def test_an_edit_between_cell_runs_is_seen(tmp_path, monkeypatch, count_hashes):
    """The control: a same-size edit with the mtime put back is caught by the
    next cell run: the inode change time moved."""
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
