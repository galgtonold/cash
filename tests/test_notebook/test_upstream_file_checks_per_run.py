"""The upstream check answers each file once per cell run, not once per entry.

Every upstream entry of one real notebook depended on the same 5,222 documents, in two
spellings, and the check before each cell validated all of them for every
entry, then stat-ed them all again for their mtimes: 7-11 s before every cell
of a notebook that runs in 30 s uncached. An answer holds for the
rest of the cell run -- the trust whole entries already had -- until a
statement of the run writes files.
"""

from __future__ import annotations

import os

import pytest

from cash.notebook import run_memo
from cash.notebook.run_memo import forget_file_state_this_run
from cash.notebook.upstream.virtual_lineage import VirtualLineage
from cash.tracking import file_dep_snapshot
from cash.tracking.file_dep_snapshot import snapshot_file_deps

N = 30


@pytest.fixture(autouse=True)
def _one_cell_run(monkeypatch):
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    monkeypatch.setattr(file_dep_snapshot, "_HASH_MEMO_TTL_SECONDS", 0.0)
    file_dep_snapshot._HASH_MEMO.clear()
    run_memo._FILE_STATE_THIS_RUN.clear()
    run_memo._FRESH_ENTRY_VERDICTS.clear()
    file_dep_snapshot.begin_file_state_epoch()
    yield
    file_dep_snapshot.end_file_state_epoch()


def _inputs(tmp_path):
    paths = []
    for i in range(N):
        p = tmp_path / f"doc{i:03d}.txt"
        p.write_text(f"document {i}\n" * 20, encoding="utf-8")
        st = os.stat(p)
        os.utime(p, (st.st_atime - 3600, st.st_mtime - 3600))
        paths.append(str(p))
    return snapshot_file_deps(set(paths))


@pytest.fixture
def checks(monkeypatch):
    calls: list[str] = []
    real = file_dep_snapshot.file_dep_is_fresh

    def counting(path, *a, **k):
        calls.append(path)
        return real(path, *a, **k)

    monkeypatch.setattr(file_dep_snapshot, "file_dep_is_fresh", counting)
    return calls


def _edit_keeping_size_and_time(path):
    before = os.stat(path)
    with open(path, "r+b") as fh:
        fh.write(b"X")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def test_entries_sharing_files_check_them_once_per_run(tmp_path, checks):
    deps = _inputs(tmp_path)
    for key in ("stmt:a", "stmt:b", "stmt:c"):  # three entries, one set of files
        assert VirtualLineage._validate_file_freshness(deps, memo_key=key)
    assert len(checks) == N, f"{len(checks)} checks of {N} files for three entries"


@pytest.mark.xfail(
    os.name == "nt",
    strict=True,
    reason="Windows: an edit that keeps the size and puts the mtime back is not seen once the file had settled -- a documented limitation (known-limitations: an edit that keeps size and timestamps); Linux and macOS catch it through the inode change time",
)
def test_the_next_run_looks_again(tmp_path, checks):
    deps = _inputs(tmp_path)
    assert VirtualLineage._validate_file_freshness(deps, memo_key="stmt:a")
    _edit_keeping_size_and_time(next(iter(deps)))
    file_dep_snapshot.end_file_state_epoch()
    file_dep_snapshot.begin_file_state_epoch()
    assert not VirtualLineage._validate_file_freshness(deps, memo_key="stmt:b")


def test_a_write_in_the_run_is_seen_by_what_is_checked_after_it(tmp_path, checks):
    """An ordinary write -- it moves the size or the time. (One that keeps both
    is the documented same-stat limitation, bounded by the cell run.)"""
    deps = _inputs(tmp_path)
    assert VirtualLineage._validate_file_freshness(deps, memo_key="stmt:a")
    with open(next(iter(deps)), "a", encoding="utf-8") as fh:
        fh.write("appended\n")
    forget_file_state_this_run()  # a file-writing statement ran
    assert not VirtualLineage._validate_file_freshness(deps, memo_key="stmt:b")


def test_without_a_write_the_answer_holds_for_the_run(tmp_path, checks):
    """No file-writing statement in between: a later entry in the same run
    shares the answer -- the trust whole entries already had in their run."""
    deps = _inputs(tmp_path)
    assert VirtualLineage._validate_file_freshness(deps, memo_key="stmt:a")
    with open(next(iter(deps)), "a", encoding="utf-8") as fh:
        fh.write("appended\n")
    assert VirtualLineage._validate_file_freshness(deps, memo_key="stmt:b")


def test_mtimes_are_read_once_per_run(tmp_path, monkeypatch):
    deps = _inputs(tmp_path)
    looked: list[str] = []
    real = os.path.getmtime
    monkeypatch.setattr(os.path, "getmtime", lambda p: looked.append(p) or real(p))
    first = VirtualLineage._stat_file_deps(deps)
    second = VirtualLineage._stat_file_deps(deps)
    assert first == second and len(first) == N
    assert len(looked) <= N, "the mtimes were read again within one run"


def test_outside_a_run_nothing_is_kept(tmp_path, checks):
    file_dep_snapshot.end_file_state_epoch()
    deps = _inputs(tmp_path)
    assert VirtualLineage._validate_file_freshness(deps)
    assert VirtualLineage._validate_file_freshness(deps)
    assert len(checks) == 2 * N
    file_dep_snapshot.begin_file_state_epoch()
