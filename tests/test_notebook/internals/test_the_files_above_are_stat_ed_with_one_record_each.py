"""The stats the upstream check takes of the files above keep one record per file.

Before every cell below a loop over 24,000 documents, the check stats each
file the cells above read (`run_memo.stats_this_run`). Each path went
through a lookup helper that kept its own record of the path beside the
stat's: two tuples and two dictionary slots per file, kept for the cell
run, and a function call or two more per path. The bookkeeping cost as much
as the stats. One pass now, one record per file.
"""

from __future__ import annotations

import os

import pytest

from cash.notebook import run_memo
from cash.notebook.run_memo import stats_this_run
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


def _inputs(tmp_path) -> list[str]:
    paths = []
    for i in range(N):
        p = tmp_path / f"doc{i:03d}.txt"
        p.write_text(f"document {i}\n", encoding="utf-8")
        paths.append(str(p))
    return paths


def test_one_record_per_file(tmp_path):
    paths = _inputs(tmp_path)
    found = stats_this_run(paths)
    stats_this_run(reversed(paths))  # asked again, in any order, from any iterable

    assert {p: (resolved, st.st_size) for p, (resolved, st) in found.items()} == {
        p: (p, os.stat(p).st_size) for p in paths
    }
    run = run_memo.file_state_this_run()
    records = sum(len(v) for v in run.values() if isinstance(v, dict) and set(paths) <= set(v))
    assert records == N
