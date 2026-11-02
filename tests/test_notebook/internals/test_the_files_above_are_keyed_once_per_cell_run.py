"""Each file the cells above read is keyed once per cell run, not once per check.

Several cached cells below a loop over 24,000 documents carry the same files,
and before every cell the check of each made a `stat_key` of every file's
stat again and compared them one by one: about 50 ms a cached cell. The run
keeps the keys with its stats, and a check whose files are all as recorded
compares the two dicts' items.
"""

from __future__ import annotations

import os
import types

import pytest

from cash.notebook import run_memo
from cash.notebook.tracking_state import TrackingState
from cash.notebook.upstream import cache_probe, virtual_lineage
from cash.notebook.upstream.cache_probe import CacheProbe
from cash.notebook.upstream.virtual_lineage import VirtualLineage
from cash.tracking import file_dep_snapshot
from cash.tracking.read_classification import stat_key

N = 30


@pytest.fixture(autouse=True)
def _one_cell_run(monkeypatch):
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    run_memo._FILE_STATE_THIS_RUN.clear()
    file_dep_snapshot.begin_file_state_epoch()
    yield
    file_dep_snapshot.end_file_state_epoch()
    run_memo._FILE_STATE_THIS_RUN.clear()


@pytest.fixture
def keys_made(monkeypatch):
    made: list[object] = []

    def counting(st):
        made.append(st)
        return stat_key(st)

    for module in (run_memo, virtual_lineage, cache_probe):
        monkeypatch.setattr(module, "stat_key", counting, raising=False)
    return made


def _deps(tmp_path) -> dict[str, tuple[int, int, int]]:
    deps = {}
    for i in range(N):
        p = tmp_path / f"doc{i:03d}.txt"
        p.write_text(f"document {i}\n", encoding="utf-8")
        deps[str(p)] = stat_key(os.stat(p))
    return deps


def _changed(deps) -> bool:
    check = types.SimpleNamespace(tracking_state=TrackingState())
    return VirtualLineage._check_cell_file_deps(check, deps, 0)


def test_three_cached_cells_with_the_same_files_key_each_file_once(tmp_path, keys_made):
    deps = _deps(tmp_path)

    assert [_changed(dict(deps)) for _ in range(3)] == [False, False, False]
    assert CacheProbe.stat_file_deps(dict(deps)) == deps
    assert CacheProbe.stat_file_deps(dict(deps)) == deps

    assert len(keys_made) == N


def test_a_file_changed_since_it_was_recorded_is_a_change(tmp_path):
    deps = _deps(tmp_path)
    path = next(iter(deps))
    size, mtime_ns, ctime_ns = deps[path]
    deps[path] = (size, mtime_ns - 1_000_000_000, ctime_ns)

    assert _changed(deps) is True


def test_a_file_gone_is_a_change(tmp_path):
    deps = _deps(tmp_path)
    gone = tmp_path / "gone.txt"
    deps[str(gone)] = (1, 1, 1)

    assert _changed(deps) is True
    assert str(gone) not in CacheProbe.stat_file_deps(deps)
