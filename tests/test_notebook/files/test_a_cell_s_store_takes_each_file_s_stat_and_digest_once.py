"""A statement's entry takes each file's stat and digest from the read and the
check, not again.

A cell reading 5,000 files paid five stats and two digests per file once its
statement ran: the relative spelling each read is also recorded under
(``FileAccessTracker.track_path``) was stat'ed and hashed by the snapshot as a
file of its own, and the snapshot stat'ed every file again right after the
check that no input moved had. The relative spelling now takes the entry of
the absolute path it names, and the snapshot takes the check's stat while the
digest from the read is still the file's.

That also closes a window: a file rewritten between the check and the
snapshot was recorded as the NEW file, stat and digest, next to a value
computed from the old one, and served on the next run. The entry now
describes the file as the statement read it.
"""

from __future__ import annotations

import os
import warnings

import pytest

from cash.notebook.statement import store
from cash.tracking import file_dep_snapshot, file_tracker
from tests._cell_driver import run_cash_cell


@pytest.fixture
def stores_everything(cash_instance):
    """Statements are stored however cheap."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0


@pytest.fixture
def docs(tmp_path, monkeypatch):
    folder = tmp_path / "docs"
    folder.mkdir()
    for i in range(20):
        (folder / f"d{i:02d}.txt").write_text("Terms" if i % 3 == 0 else "none", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return folder


CELL = "import glob\nn = sum('Terms' in open(p).read() for p in sorted(glob.glob('docs/*.txt')))"


def _hashes_per_file(monkeypatch, folder):
    hashed = []
    real = file_dep_snapshot.file_content_hash

    def counting(path, *a, **k):
        if os.path.basename(os.path.dirname(os.path.abspath(path))) == folder.name:
            hashed.append(os.path.basename(path))
        return real(path, *a, **k)

    for module in (file_dep_snapshot, file_tracker):
        monkeypatch.setattr(module, "file_content_hash", counting)
    return hashed


def test_each_file_a_cell_reads_by_relative_path_is_hashed_once(cash_magics, stores_everything, docs, monkeypatch):
    hashed = _hashes_per_file(monkeypatch, docs)
    run_cash_cell(cash_magics, CELL)
    assert cash_magics.shell.user_ns["n"] == 7
    per_file = {name: hashed.count(name) for name in sorted(os.listdir(docs))}
    # The read's digest; the snapshot hashed the relative spelling again (it was two).
    assert set(per_file.values()) == {1}, per_file


def test_a_file_edited_after_a_relative_read_runs_the_cell_again(cash_magics, stores_everything, docs):
    run_cash_cell(cash_magics, CELL)
    assert cash_magics.shell.user_ns["n"] == 7
    (docs / "d01.txt").write_text("Terms", encoding="utf-8")
    run_cash_cell(cash_magics, CELL)
    assert cash_magics.shell.user_ns["n"] == 8


def test_a_file_rewritten_between_the_check_and_the_snapshot_is_not_served(
    cash_magics, stores_everything, tmp_path, monkeypatch
):
    data = tmp_path / "data.txt"
    data.write_text("old", encoding="utf-8")
    real = store._snapshot_with_inherited
    armed = [True]

    def rewritten_first(*a, **k):
        if armed[0]:
            armed[0] = False
            temp = tmp_path / "data.txt.new"
            temp.write_text("NEW", encoding="utf-8")  # same size
            os.replace(temp, data)
        return real(*a, **k)

    monkeypatch.setattr(store, "_snapshot_with_inherited", rewritten_first)
    path = str(data).replace("\\", "/")
    cell = f"text = open({path!r}).read()"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_cash_cell(cash_magics, cell)
        assert cash_magics.shell.user_ns["text"] == "old"
        assert not armed[0], "the file was not rewritten before the snapshot"
        run_cash_cell(cash_magics, cell)
    assert cash_magics.shell.user_ns["text"] == "NEW", "the value read from the old file was served for the new one"
