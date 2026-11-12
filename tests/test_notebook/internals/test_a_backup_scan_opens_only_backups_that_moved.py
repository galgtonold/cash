"""An editor that saves the notebook before each run changes a VS Code
backup folder each time (the notebook's own backup goes or is rewritten),
so ``find_backup`` scans again on most cell runs. That scan opened every
backup of every workspace to read its header. It now keeps each header it
read against the backup's stat, and opens only a backup that is new or has
moved since; a header read within a timestamp tick of its last write is
read again, since a write in that tick may not move the stat.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import cash.notebook.file_stamps as file_stamps
import cash.notebook.vscode_backup as vb

#: Longer than any timestamp tick of a filesystem with sub-second stamps.
_PAST_A_TICK_S = 0.03


def _backup(folder: Path, notebook: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    header = json.dumps({"mtime": 1, "size": 1})
    path.write_text(f'{notebook.as_uri()} {header}\n{{"cells": []}}', encoding="utf-8")
    return path


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """Ten workspaces of backups of other notebooks, and the list of the
    backups whose header a scan opened."""
    root = tmp_path / "Backups"
    for w in range(10):
        for j in range(3):
            _backup(root / f"ws{w}" / "file", tmp_path / f"other{w}_{j}.ipynb", f"b{j}")
    monkeypatch.setattr(vb, "backup_roots", lambda: [root])
    monkeypatch.setattr(file_stamps, "COARSE_TICK_NS", 0)
    opened: list[str] = []
    real_read_uri = vb._read_uri

    def read_uri(path):
        opened.append(Path(path).name)
        return real_read_uri(path)

    monkeypatch.setattr(vb, "_read_uri", read_uri)
    nb = tmp_path / "target.ipynb"
    nb.write_text("{}", encoding="utf-8")
    time.sleep(_PAST_A_TICK_S)
    return root, nb, opened


def test_a_scan_after_a_folder_changed_opens_only_the_new_backup(tree):
    root, nb, opened = tree
    assert vb.find_backup(str(nb)) is None
    assert len(opened) == 30
    opened.clear()
    time.sleep(_PAST_A_TICK_S)
    _backup(root / "ws0" / "file", root.parent / "unrelated.ipynb", "new")
    assert vb.find_backup(str(nb)) is None
    assert opened == ["new"]


def test_a_backup_rewritten_in_place_is_opened_again(tree):
    root, nb, opened = tree
    assert vb.find_backup(str(nb)) is None
    opened.clear()
    time.sleep(_PAST_A_TICK_S)
    want = _backup(root / "ws3" / "file", nb, "b1")  # now this notebook's
    assert vb.find_backup(str(nb)) == want
    assert opened == ["b1"]


def test_a_header_read_within_its_tick_is_read_again(tree, monkeypatch):
    """The control: a backup rewritten within the tick of its last write
    may keep its stat, so a header read then is not kept."""
    root, nb, opened = tree
    monkeypatch.setattr(file_stamps, "STAMP_TICK_NS", 60 * 10**9)
    monkeypatch.setattr(file_stamps, "COARSE_TICK_NS", 60 * 10**9)
    assert vb.find_backup(str(nb)) is None
    opened.clear()
    assert vb.find_backup(str(nb)) is None
    assert len(opened) == 30


def test_headers_stay_kept_over_many_scans(tree):
    root, nb, opened = tree
    assert vb.find_backup(str(nb)) is None
    opened.clear()
    for i in range(3):
        _backup(root / "ws0" / "file", root.parent / "unrelated.ipynb", f"new{i}")
        time.sleep(_PAST_A_TICK_S)
        assert vb.find_backup(str(nb)) is None
    assert opened == ["new0", "new1", "new2"]
