"""Unpacking an archive looks at each file it writes only a few times.

Cash notes each file a statement writes: as it is opened (its folder's real
path, and that it is the statement's own output), and after the statement
(its stat for the notebook's own writes, whether it is still there). A
statement unpacking 100,000 files paid ~12 looks per file where plain Python
pays one: the per-run memo of real paths filled up after ~20,000 files and
every later file had its whole path resolved, a component at a time, three
times; each file opened to write was stat-ed to rule out a folder, which
``open()`` cannot write; its stat was taken again after an ``lstat`` that
had already found it; and every written path was ``lstat``-ed to learn
whether *any* was still there. On Windows each look opens the file.
"""

from __future__ import annotations

import collections
import io
import os
import tarfile

import pytest

from cash.tracking import file_dep_snapshot
from tests._cell_driver import run_cash_cell

#: Past the files a writer's provenance is recorded for (``_MAX_PROVENANCE_FILES``).
FILES, FOLDERS = 1600, 4


@pytest.fixture
def archive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "data.tar"
    with tarfile.open(path, "w") as tar:
        for d in range(FOLDERS):
            for i in range(FILES // FOLDERS):
                data = f"review {d} {i}".encode()
                info = tarfile.TarInfo(f"set/part{d}/{i:04d}.txt")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return path


def test_each_file_unpacked_is_looked_at_a_few_times(cash_magics, archive, tmp_path, monkeypatch):
    # Fills the memo of real paths many times over, as 100,000 files do.
    monkeypatch.setattr(file_dep_snapshot, "_REALPATH_KEPT", 300, raising=False)
    run_cash_cell(cash_magics, "import tarfile")
    looks: collections.Counter = collections.Counter()
    root = str(tmp_path / "set")

    def counting(kind, real):
        def call(path, *args, **kwargs):
            if os.path.abspath(os.fspath(path)).startswith(root):
                looks[kind] += 1
            return real(path, *args, **kwargs)

        return call

    monkeypatch.setattr(os, "stat", counting("stat", os.stat))
    monkeypatch.setattr(os, "lstat", counting("lstat", os.lstat))
    run_cash_cell(cash_magics, "with tarfile.open('data.tar') as tar:\n    tar.extractall(filter='fully_trusted')")
    monkeypatch.undo()

    assert len(list((tmp_path / "set").rglob("*.txt"))) == FILES
    # tarfile's own: a stat of each file's folder (``os.path.exists``) and of
    # the file it made (``filter='fully_trusted'``); cash's: an ``lstat`` as the file
    # is opened (is it a link?), one after the statement for its stat.
    per_file = (looks["stat"] + looks["lstat"]) / FILES
    assert per_file <= 5, looks
