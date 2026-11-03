"""The stats the upstream check takes of the files above keep one record per file.

Before every cell below a loop over 24,000 documents, the check stats each
file the cells above read (`run_memo.stats_this_run`). Each path went
through a lookup helper that kept its own record of the path beside the
stat's: two tuples and two dictionary slots per file, kept for the cell
run, and a function call or two more per path. The bookkeeping cost as much
as the stats. One pass now, one record per file.

The record is the stat itself, which the garbage collector does not track:
a ``(path, stat)`` pair is tracked (a stat result is a collector type), and
one per file lived through the cell, was promoted to the oldest generation
and set off a full collection of the kernel's heap every few cells. The
pairs the answer hands out are made one at a time, as they are asked for.

Beside it the run keeps when each stat was taken, for the checks that take
only recent ones (``run["stat_date"]``): a slot per file holding one float
shared by every stat of its chunk, in a dict the collector does not track.
"""

from __future__ import annotations

import gc
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
    records = sum(len(v) for k, v in run.items() if k != "stat_date" and isinstance(v, dict) and set(paths) <= set(v))
    assert records == N


def test_the_records_are_not_tracked_by_the_garbage_collector(tmp_path):
    paths = _inputs(tmp_path)
    found = stats_this_run(paths)
    gc.collect()

    asked = set(paths)
    run = run_memo.file_state_this_run()
    records = [
        v for n, d in run.items() if n != "stat_date" and isinstance(d, dict) for k, v in d.items() if k in asked
    ]
    assert len(records) == N
    dates = run["stat_date"]
    assert not gc.is_tracked(dates)
    assert len({id(dates[p]) for p in paths}) == 1, "one date object per chunk"
    assert [r for r in records if gc.is_tracked(r)] == []
    pairs = [o for o in gc.get_objects() if type(o) is tuple and len(o) == 2 and type(o[0]) is str and o[0] in asked]
    assert pairs == [], "a pair per file is kept alive"
    assert dict(found.items()) == {p: (p, run["stat"][p]) for p in paths}
