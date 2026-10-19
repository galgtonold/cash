"""The check that no input moved during a call reuses the snapshot's stat.

After a missed call's body ran, each file it read was stat'ed twice: once
for the entry's snapshot (`snapshot_file_deps`) and again, right after, by
the check that nothing moved since the read
(`FileAccessTracker.inputs_changed_since_read`). p25's workers read 6,000
files each and paid 6,000 extra stats. The check now compares with the stat
the snapshot took. A file rewritten during the body is still refused; one
rewritten after the snapshot is stored with the stat from before, so the
next call sees the change and recomputes.
"""

from __future__ import annotations

import os

import pytest

from cash.decorator import runtime

pytestmark = pytest.mark.core


def _files(tmp_path, n):
    folder = tmp_path / "data"
    folder.mkdir()
    paths = []
    for i in range(n):
        path = folder / f"f{i:03d}.txt"
        path.write_text(f"{i}\n", encoding="utf-8")
        paths.append(str(path))
    return folder, tuple(paths)


def test_each_file_a_call_read_is_stated_once_after_the_body(disk_cash, tmp_path, monkeypatch):
    folder, paths = _files(tmp_path, 40)
    body_done = []

    @disk_cash.cache(assume_safe=True)
    def total(paths):
        value = 0
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                value += int(fh.read())
        body_done.append(1)
        return value

    statted = []
    real = os.stat
    root = os.path.normcase(str(folder))

    def counting(path, *a, **k):
        if body_done and isinstance(path, str) and os.path.normcase(path).startswith(root):
            statted.append(path)
        return real(path, *a, **k)

    monkeypatch.setattr(os, "stat", counting)
    assert total(paths) == sum(range(40))
    monkeypatch.undo()
    per_file = {p: statted.count(p) for p in paths}
    # The snapshot's, and no second one for the check (it was two).
    assert set(per_file.values()) == {1}, per_file


def test_a_file_rewritten_after_the_snapshot_is_recomputed_next_call(disk_cash, tmp_path, monkeypatch):
    data = tmp_path / "data.txt"
    data.write_text("1 2 3\n", encoding="utf-8")
    runs = []
    real = runtime.snapshot_tracked_deps

    def then_rewrite(tracker, module=None):
        deps = real(tracker, module)
        if not runs[1:]:  # the first call only: a writer lands between the snapshot and the check
            data.write_text("1 2 3 4 5\n", encoding="utf-8")
        return deps

    monkeypatch.setattr(runtime, "snapshot_tracked_deps", then_rewrite)

    @disk_cash.cache(assume_safe=True)
    def total(path):
        runs.append(1)
        with open(path, encoding="utf-8") as fh:
            return sum(int(x) for x in fh.read().split())

    assert total(str(data)) == 6
    assert total(str(data)) == 15, "the entry stored before the rewrite was served for the new file"
    assert len(runs) == 2
    assert total(str(data)) == 15
    assert len(runs) == 2
