"""VS Code keeps unsaved notebook state on disk; cash can read it.

cash reads the cells it did not execute from the saved `.ipynb`, so an unsaved
edit is invisible and a downstream cell can restore a stale value. VS Code has
no in-page route to its live document -- the widget webview holds a
`standaloneModel` stub with an empty cell list, and output JS is sandboxed away
from the workbench. But VS Code persists dirty editors to a backup so it can
restore after a crash, and that file is plain JSON the kernel can read.

Format, measured on VS Code 1.123:

    line 1   file:///c%3A/path/nb.ipynb {"mtime":<epoch_ms>,"size":<bytes>,...}
    rest     the full notebook JSON

The header describes the SAVED file, which is what makes the backup usable
rather than a guess: cash can confirm a backup belongs to the version currently
on disk before trusting its body.
"""
from __future__ import annotations

import json
from pathlib import Path

from cash.notebook.vscode_backup import find_backup, parse_backup


def _write_backup(dirpath: Path, nb_path: Path, *, mtime_ms: int, size: int,
                  cells: list[str], name: str = "abc123") -> Path:
    """Write a file in VS Code's backup format."""
    uri = nb_path.as_uri()
    header = json.dumps({
        "mtime": mtime_ms, "ctime": mtime_ms, "size": size,
        "etag": "x", "orphaned": False,
        "typeId": "notebook/jupyter-notebook/jupyter-notebook",
    })
    body = json.dumps({
        "cells": [
            {"cell_type": "code", "id": f"c{i}", "metadata": {},
             "outputs": [], "execution_count": None, "source": src}
            for i, src in enumerate(cells)
        ],
        "nbformat": 4, "nbformat_minor": 5, "metadata": {},
    })
    target = dirpath / name
    target.write_text(f"{uri} {header}\n{body}", encoding="utf-8")
    return target


def test_parse_returns_uri_header_and_notebook(tmp_path):
    nb = tmp_path / "nb.ipynb"
    nb.write_text("{}", encoding="utf-8")
    bak = _write_backup(tmp_path, nb, mtime_ms=1700000000000, size=42,
                        cells=["THRESHOLD = 0.9"])

    parsed = parse_backup(bak)
    assert parsed is not None
    uri, meta, notebook = parsed
    assert uri == nb.as_uri()
    assert meta["mtime"] == 1700000000000
    assert meta["size"] == 42
    assert notebook["cells"][0]["source"] == "THRESHOLD = 0.9"


def test_parse_returns_none_for_garbage(tmp_path):
    """Degrade, never raise -- this is another product's private format and it
    can change without notice."""
    bad = tmp_path / "bad"
    bad.write_text("not a backup at all", encoding="utf-8")
    assert parse_backup(bad) is None


def test_parse_returns_none_when_the_body_is_not_json(tmp_path):
    nb = tmp_path / "nb.ipynb"
    nb.write_text("{}", encoding="utf-8")
    bad = tmp_path / "halfwritten"
    bad.write_text(f'{nb.as_uri()} {{"mtime":1,"size":2}}\n{{"cells": [', encoding="utf-8")
    assert parse_backup(bad) is None


def test_parse_returns_none_for_a_missing_file(tmp_path):
    assert parse_backup(tmp_path / "nope") is None


def test_find_backup_matches_by_uri(tmp_path, monkeypatch):
    import cash.notebook.vscode_backup as vb

    nb = tmp_path / "target.ipynb"
    nb.write_text("{}", encoding="utf-8")
    other = tmp_path / "other.ipynb"
    other.write_text("{}", encoding="utf-8")

    root = tmp_path / "Backups" / "ws" / "file"
    root.mkdir(parents=True)
    _write_backup(root, other, mtime_ms=1, size=1, cells=["x = 1"], name="aaa")
    want = _write_backup(root, nb, mtime_ms=2, size=2, cells=["y = 2"], name="bbb")

    monkeypatch.setattr(vb, "backup_roots", lambda: [tmp_path / "Backups"])
    assert find_backup(str(nb)) == want


def test_find_backup_returns_none_when_nothing_matches(tmp_path, monkeypatch):
    """The control: a notebook with no backup must not pick up someone else's."""
    import cash.notebook.vscode_backup as vb

    nb = tmp_path / "target.ipynb"
    nb.write_text("{}", encoding="utf-8")
    other = tmp_path / "other.ipynb"
    other.write_text("{}", encoding="utf-8")

    root = tmp_path / "Backups" / "ws" / "file"
    root.mkdir(parents=True)
    _write_backup(root, other, mtime_ms=1, size=1, cells=["x = 1"], name="aaa")

    monkeypatch.setattr(vb, "backup_roots", lambda: [tmp_path / "Backups"])
    assert find_backup(str(nb)) is None


def test_find_backup_survives_an_unreadable_root(tmp_path, monkeypatch):
    import cash.notebook.vscode_backup as vb

    nb = tmp_path / "target.ipynb"
    nb.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(vb, "backup_roots", lambda: [tmp_path / "does-not-exist"])
    assert find_backup(str(nb)) is None
