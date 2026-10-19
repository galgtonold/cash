"""A statement's end stats each settled file it read once for its first two
looks, and the last look before its entry is written stats it again.

After a statement ran, each file it read was stat'ed four times: for its
deferred digest, for the check that no input moved, for its file lineage,
and for that check again before the entry is written. ~15 us a stat on
Windows: ~0.3 s a run for a cell reading 5,000 files. The first check takes
the stat the deferred digest was taken under, and the lineage the stats
that check took (`FileAccessTracker.end_stats`, ``checked_stats``); the
check before the store stats every file again.

What must hold: the lineage component is the one fresh stats give, for a
mix of relative, absolute and linked reads; a file a thread of the cell
rewrites while these are taken is caught by the last check, and read again
on the next run.
"""

from __future__ import annotations

import os
import sys
import threading
import warnings

import pytest

from cash.notebook.statement import file_deps, lineage
from cash.tracking import file_dep_snapshot, read_classification
from cash.tracking.file_tracker import FileAccessTracker
from tests._cell_driver import run_cash_cell


def _settled(path, text: str):
    path.write_text(text, encoding="utf-8")
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns - 3600 * 10**9, st.st_mtime_ns - 3600 * 10**9))
    return path


def _slash(path) -> str:
    return str(path).replace("\\", "/")


@pytest.fixture
def stores_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    file_dep_snapshot._HASH_MEMO.clear()


@pytest.fixture
def files(tmp_path, monkeypatch):
    """A relative read, an absolute one and, where links can be made, one
    through a linked folder; the cell reading them."""
    work = tmp_path / "work"
    (work / "data").mkdir(parents=True)
    _settled(work / "rel.txt", "relative ")
    absolute = _settled(tmp_path / "abs.txt", "absolute ")
    _settled(work / "data" / "linked.txt", "linked")
    reads = [repr("rel.txt"), repr(_slash(absolute))]
    try:
        os.symlink(work / "data", work / "link", target_is_directory=True)
        reads.append(repr("link/linked.txt"))
    except (OSError, NotImplementedError):
        pass  # Windows without the privilege: the other two still mix
    monkeypatch.chdir(work)
    cell = "text = " + " + ".join(f"open({r}, encoding='utf-8').read()" for r in reads)
    return work, cell


@pytest.fixture
def components(monkeypatch):
    """Each lineage component the statement took, with what it was given."""
    seen: list[tuple] = []
    real = file_deps.compute_file_hash_component

    def recording(accessed, remote=None, probes=None, stats=None):
        out = real(accessed, remote, probes, stats=stats)
        seen.append((set(accessed), remote, probes, dict(stats or {}), out))
        return out

    monkeypatch.setattr(lineage, "compute_file_hash_component", recording)
    return seen


def test_the_lineage_from_shared_stats_is_the_one_fresh_stats_give(
    cash_magics, stores_everything, files, components, monkeypatch
):
    _, cell = files
    stats_taken: list[str] = []
    real_stat = read_classification.regular_file_stat_result

    def counting(path):
        stats_taken.append(path)
        return real_stat(path)

    monkeypatch.setattr("cash.tracking.file_tracker.regular_file_stat_result", counting)
    run_cash_cell(cash_magics, cell)
    assert cash_magics.shell.user_ns["text"].startswith("relative absolute")
    (accessed, remote, probes, shared, component) = components[0]
    assert shared, "control: the lineage was given the check's stats"
    assert component == file_deps.compute_file_hash_component(accessed, remote, probes)
    assert component == file_deps.compute_file_hash_component(accessed, remote, probes, stats={})
    # The first check took the end stats; only the check before the store
    # stat'ed each file again.
    read = {p for p in accessed if os.path.isabs(p)}
    assert sorted(stats_taken) == sorted(read), stats_taken


def test_a_lookup_by_shared_stats_finds_the_entry(cash_magics, stores_everything, files):
    _, cell = files
    run_cash_cell(cash_magics, cell)
    lineage_first = cash_magics.tracking_state.variable_lineage["text"]
    cash_magics.shell.user_ns["text"] = None
    run_cash_cell(cash_magics, cell)
    assert cash_magics.shell.user_ns["text"].startswith("relative absolute"), "served"
    assert cash_magics.tracking_state.variable_lineage["text"] == lineage_first


THREAD = """
import threading
def _rewrite(go, done):
    go.wait(10)
    with open('rel.txt', 'w', encoding='utf-8') as fh:
        fh.write('NEW TEXT ')
    done.set()
threading.Thread(target=_rewrite, args=(go, done), daemon=True).start()
"""


@pytest.mark.parametrize("when", ["before-the-first-check", "between-the-checks"])
def test_a_file_a_thread_rewrites_as_the_statement_ends_is_read_again(cash_magics, stores_everything, files, when):
    """The cell leaves a thread running that rewrites a file it read, once
    cash has taken that file's end stat: in time for the first check to
    have used that stat, or after it, before the lineage. The check before
    the store stats it again and refuses."""
    _, cell = files
    go, done = threading.Event(), threading.Event()
    cash_magics.shell.user_ns.update(go=go, done=done)

    def let_it_write():
        if not go.is_set():
            go.set()
            assert done.wait(10), "the thread did not write"

    patches = pytest.MonkeyPatch()
    if when == "before-the-first-check":
        real = FileAccessTracker.inputs_changed_since_read

        def first_check(self, reuse_end_stats=False):
            if reuse_end_stats:
                let_it_write()
            return real(self, reuse_end_stats)

        patches.setattr(FileAccessTracker, "inputs_changed_since_read", first_check)
    else:
        real_component = file_deps.compute_file_hash_component

        def between(*args, **kwargs):
            let_it_write()
            return real_component(*args, **kwargs)

        patches.setattr(lineage, "compute_file_hash_component", between)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_cash_cell(cash_magics, THREAD + "\n" + cell)
    finally:
        patches.undo()
    assert go.is_set() and done.is_set(), "control: the thread wrote as the statement ended"
    assert cash_magics.shell.user_ns["text"].startswith("relative absolute")
    assert any(getattr(w.message, "code", None) == "STORE-INPUT-CHANGED" for w in caught), [
        str(w.message) for w in caught
    ]
    run_cash_cell(cash_magics, cell)
    assert cash_magics.shell.user_ns["text"].startswith("NEW TEXT absolute"), (
        "the value read from the old file was served"
    )


@pytest.mark.skipif(sys.platform == "win32", reason="Windows keeps no inode change time: a documented limitation")
def test_a_rewrite_that_puts_the_mtime_back_is_caught_too(cash_magics, stores_everything, files):
    work, cell = files
    real = FileAccessTracker.inputs_changed_since_read

    def first_check(self, reuse_end_stats=False):
        if reuse_end_stats:
            path = work / "rel.txt"
            before = os.stat(path)
            path.write_text("NEW TEXT ", encoding="utf-8")
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        return real(self, reuse_end_stats)

    with pytest.MonkeyPatch.context() as patches:
        patches.setattr(FileAccessTracker, "inputs_changed_since_read", first_check)
        run_cash_cell(cash_magics, cell)
    run_cash_cell(cash_magics, cell)
    assert cash_magics.shell.user_ns["text"].startswith("NEW TEXT"), "the value read from the old file was served"
