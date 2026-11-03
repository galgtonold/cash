"""A file read by a relative path is checked once per freshness check,
not once under each spelling.

A relative read is recorded twice: under the name the code used and under
that name made absolute from the working directory
(``FileAccessTracker.track_path``). The check stat'ed both, so p25's
``shingle_chunk`` hit took 2,000 stats for its 1,000 documents, 48,000 a
script run -- on Windows, where a stat opens the file, seconds of every
all-hit run. When the relative name still names the absolute one from
this working directory and both were recorded alike, the answer for one
is the answer for the other. What must hold: one stat per file; a
changed file still stale; and a relative name that now names another file
(the working directory moved) checked on its own.
"""

from __future__ import annotations

import os

from cash._paths import normalize_path
from cash.tracking import file_dep_snapshot
from cash.tracking.file_dep_snapshot import FreshnessMemo, snapshot_file_deps, snapshot_is_fresh

N = 20


def _write_docs(folder, text="document"):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(N):
        (folder / f"doc{i:02d}.txt").write_text(f"{text} {i}\n", encoding="utf-8")


def _snapshot_both_spellings(tmp_path, monkeypatch):
    project = tmp_path / "project"
    _write_docs(project / "docs")
    monkeypatch.chdir(project)
    paths = set()
    for i in range(N):
        relative = os.path.join("docs", f"doc{i:02d}.txt")
        paths.add(relative)
        paths.add(normalize_path(os.path.abspath(relative)))  # as the tracker records it: C:/... on Windows
    snap = snapshot_file_deps(paths)
    assert len(snap) == 2 * N, "control: each read recorded under both spellings"
    return project, snap


def _count_stats(monkeypatch) -> list:
    stats: list = []
    real = file_dep_snapshot.own_stat

    def counting(path):
        stats.append(path)
        return real(path)

    monkeypatch.setattr(file_dep_snapshot, "own_stat", counting)
    return stats


def test_each_file_is_stat_ed_once(tmp_path, monkeypatch):
    _project, snap = _snapshot_both_spellings(tmp_path, monkeypatch)
    stats = _count_stats(monkeypatch)

    assert snapshot_is_fresh(snap) == (True, None)
    assert len(stats) == N

    stats.clear()
    memo = FreshnessMemo()
    assert snapshot_is_fresh(snap, memo) == (True, None)
    # With a memo, a listing of the folder stands in for the stats where
    # it can (Windows: `stats_from_listings`).
    assert len(stats) == len([p for p in snap if os.path.isabs(p) and p not in memo.listed])


def test_a_changed_file_is_still_stale(tmp_path, monkeypatch):
    project, snap = _snapshot_both_spellings(tmp_path, monkeypatch)
    (project / "docs" / "doc07.txt").write_text("changed!!! 7\n", encoding="utf-8")

    fresh, stale = snapshot_is_fresh(snap)
    assert not fresh
    assert stale.path.endswith("doc07.txt")


def test_a_relative_name_that_names_another_file_is_checked_on_its_own(tmp_path, monkeypatch):
    _project, snap = _snapshot_both_spellings(tmp_path, monkeypatch)
    elsewhere = tmp_path / "elsewhere"
    _write_docs(elsewhere / "docs", text="other text")
    monkeypatch.chdir(elsewhere)

    fresh, stale = snapshot_is_fresh(snap)
    assert not fresh
    assert not os.path.isabs(stale.path)


def test_a_twin_marked_as_lying_beside_the_code_is_checked_once(tmp_path, monkeypatch):
    """The absolute spelling of a file under the code's folder also says
    where it lies beside the code (p25's documents): still one file."""
    _project, snap = _snapshot_both_spellings(tmp_path, monkeypatch)
    for path, entry in snap.items():
        if os.path.isabs(path):
            entry["code_rel"] = "docs/" + os.path.basename(path)
            entry["code_mod"] = "a_module_this_process_has_not_loaded"
    stats = _count_stats(monkeypatch)

    assert snapshot_is_fresh(snap) == (True, None)
    assert len(stats) == N
