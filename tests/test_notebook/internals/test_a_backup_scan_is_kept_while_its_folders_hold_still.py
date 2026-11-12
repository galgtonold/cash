"""``find_backup`` scans VS Code's backup folders, opening every backup in
them, two or three times a cell run; with no backup for the notebook the
answer was never kept, as no single file's stat stands for "none". It is
now kept against the stat of every path the scan looked at, so asking
again costs those stats, and anything that could change the answer -- a
backup appearing, a workspace folder appearing, a backup's header
rewritten -- moves one of them and brings a fresh scan.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import cash.notebook.vscode_backup as vb

#: Longer than any timestamp tick of a filesystem with sub-second stamps
#: (15.6 ms on Windows), so a change after it moves the folder's stamps.
_PAST_A_TICK_S = 0.03


def _backup(folder: Path, notebook: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    header = json.dumps({"mtime": 1, "size": 1})
    path.write_text(f'{notebook.as_uri()} {header}\n{{"cells": []}}', encoding="utf-8")
    return path


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A backup tree whose stamps count as settled at once, and a counter
    of the folder listings a scan makes in it."""
    root = tmp_path / "Backups"
    (root / "ws1").mkdir(parents=True)
    _backup(root / "ws2" / "file", tmp_path / "other.ipynb", "aaa")
    monkeypatch.setattr(vb, "backup_roots", lambda: [root])
    monkeypatch.setattr(vb, "SETTLED_SECONDS", 0.0)
    listings = []
    real_scandir = vb.os.scandir

    def scandir(path):
        listings.append(str(path))
        return real_scandir(path)

    monkeypatch.setattr(vb.os, "scandir", scandir)
    nb = tmp_path / "target.ipynb"
    nb.write_text("{}", encoding="utf-8")
    time.sleep(_PAST_A_TICK_S)
    return root, nb, listings


def test_no_backup_is_answered_again_without_a_scan(tree):
    root, nb, listings = tree
    assert vb.find_backup(str(nb)) is None
    assert listings, "the first call did not scan"
    listings.clear()
    assert vb.find_backup(str(nb)) is None
    assert listings == []


def test_a_backup_appearing_in_a_known_folder_is_found(tree):
    root, nb, _ = tree
    assert vb.find_backup(str(nb)) is None
    time.sleep(_PAST_A_TICK_S)
    want = _backup(root / "ws2" / "file", nb, "bbb")
    assert vb.find_backup(str(nb)) == want


def test_a_backup_in_a_new_workspace_folder_is_found(tree):
    root, nb, _ = tree
    assert vb.find_backup(str(nb)) is None
    time.sleep(_PAST_A_TICK_S)
    want = _backup(root / "ws3" / "file", nb, "ccc")
    assert vb.find_backup(str(nb)) == want


def test_a_file_folder_appearing_in_a_known_workspace_is_looked_in(tree):
    root, nb, _ = tree
    assert vb.find_backup(str(nb)) is None
    time.sleep(_PAST_A_TICK_S)
    want = _backup(root / "ws1" / "file", nb, "ddd")
    assert vb.find_backup(str(nb)) == want


def test_a_backup_rewritten_for_this_notebook_is_found(tree):
    root, nb, _ = tree
    assert vb.find_backup(str(nb)) is None
    time.sleep(_PAST_A_TICK_S)
    want = _backup(root / "ws2" / "file", nb, "aaa")  # same file, now this notebook's
    assert vb.find_backup(str(nb)) == want


def test_a_backup_going_away_is_noticed(tree):
    root, nb, _ = tree
    found = _backup(root / "ws2" / "file", nb, "eee")
    time.sleep(_PAST_A_TICK_S)
    assert vb.find_backup(str(nb)) == found
    found.unlink()
    assert vb.find_backup(str(nb)) is None


def test_an_answer_from_folders_changed_just_now_is_not_kept(tree, monkeypatch):
    """The control: a change inside a timestamp tick of the last can leave
    a folder's stamps as they were, so an answer from folders that changed
    within ``SETTLED_SECONDS`` is scanned for again."""
    root, nb, listings = tree
    monkeypatch.setattr(vb, "SETTLED_SECONDS", 60.0)
    assert vb.find_backup(str(nb)) is None
    listings.clear()
    assert vb.find_backup(str(nb)) is None
    assert listings, "an answer resting on stamps that may not move was kept"
