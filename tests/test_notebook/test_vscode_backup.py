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


def test_parse_returns_none_for_single_line_with_no_newline(tmp_path):
    """Isolate the newline guard: a file with no newline at all."""
    bad = tmp_path / "no_newline"
    bad.write_text("single line content", encoding="utf-8")
    assert parse_backup(bad) is None


def test_parse_returns_none_when_header_has_no_space(tmp_path):
    """Isolate the space guard: a header line with URI but no JSON."""
    bad = tmp_path / "no_space"
    bad.write_text("file:///c%3A/path/nb.ipynb\n{}", encoding="utf-8")
    assert parse_backup(bad) is None


def test_parse_returns_none_when_header_json_is_wrong_type(tmp_path):
    """Valid JSON but not a dict (list instead)."""
    bad = tmp_path / "wrong_type"
    bad.write_text("file:///c%3A/path/nb.ipynb [1,2,3]\n{}", encoding="utf-8")
    assert parse_backup(bad) is None


def test_parse_returns_none_when_notebook_json_is_wrong_type(tmp_path):
    """Valid JSON header but notebook body is not a dict (int instead)."""
    bad = tmp_path / "wrong_nb_type"
    bad.write_text('file:///c%3A/path/nb.ipynb {"mtime":1}\n42', encoding="utf-8")
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


def test_find_backup_survives_generator_raising_permission_error(tmp_path, monkeypatch):
    """A glob generator that raises PermissionError mid-iteration must not propagate.

    This tests that the iteration happens inside the try/except, not outside it.
    """
    import cash.notebook.vscode_backup as vb

    nb = tmp_path / "target.ipynb"
    nb.write_text("{}", encoding="utf-8")

    # Mock backup_roots to return a root that exists
    root = tmp_path / "Backups"
    root.mkdir()
    monkeypatch.setattr(vb, "backup_roots", lambda: [root])

    # Mock glob to return a generator that raises PermissionError mid-iteration
    def failing_glob(pattern):
        """Yield one item then raise PermissionError to simulate access denied."""
        raise PermissionError("Permission denied")
        yield  # pragma: no cover

    original_glob = Path.glob
    def patched_glob(self, pattern):
        if self == root:
            return failing_glob(pattern)
        return original_glob(self, pattern)

    monkeypatch.setattr(Path, "glob", patched_glob)
    assert find_backup(str(nb)) is None


import os
import time

from cash.notebook.vscode_backup import live_cells


def _setup(tmp_path, monkeypatch, *, backup_cells, file_bytes=b"{}", skew_ms=0):
    """A notebook on disk plus a backup whose header describes it."""
    import cash.notebook.vscode_backup as vb

    nb = tmp_path / "nb.ipynb"
    nb.write_bytes(file_bytes)
    st = os.stat(nb)
    root = tmp_path / "Backups" / "ws" / "file"
    root.mkdir(parents=True)
    _write_backup(root, nb, mtime_ms=int(st.st_mtime * 1000) + skew_ms,
                  size=st.st_size, cells=backup_cells)
    monkeypatch.setattr(vb, "backup_roots", lambda: [tmp_path / "Backups"])
    return nb


def test_a_matching_backup_supplies_live_cells(tmp_path, monkeypatch):
    nb = _setup(tmp_path, monkeypatch, backup_cells=["THRESHOLD = 0.9"])
    cells = live_cells(str(nb))
    assert cells is not None
    assert cells[0]["source"] == "THRESHOLD = 0.9"


def test_a_backup_for_a_different_saved_version_is_refused(tmp_path, monkeypatch):
    """THE safety property. The header describes the SAVED file; if it does not
    match what is on disk now, the backup belongs to some other version and
    using it would substitute one staleness for another."""
    nb = _setup(tmp_path, monkeypatch, backup_cells=["THRESHOLD = 0.9"])
    time.sleep(0.01)
    nb.write_bytes(b'{"cells": []}')        # the user saved; backup header now stale
    assert live_cells(str(nb)) is None


def test_a_size_mismatch_alone_is_enough_to_refuse(tmp_path, monkeypatch):
    """mtime granularity varies by filesystem, so size is the second check
    rather than a redundant one."""
    import cash.notebook.vscode_backup as vb

    nb = tmp_path / "nb.ipynb"
    nb.write_bytes(b"{}")
    st = os.stat(nb)
    root = tmp_path / "Backups" / "ws" / "file"
    root.mkdir(parents=True)
    _write_backup(root, nb, mtime_ms=int(st.st_mtime * 1000),
                  size=st.st_size + 999, cells=["x = 1"])
    monkeypatch.setattr(vb, "backup_roots", lambda: [tmp_path / "Backups"])
    assert live_cells(str(nb)) is None


def test_no_backup_means_none_not_an_error(tmp_path, monkeypatch):
    import cash.notebook.vscode_backup as vb

    nb = tmp_path / "nb.ipynb"
    nb.write_bytes(b"{}")
    monkeypatch.setattr(vb, "backup_roots", lambda: [tmp_path / "empty"])
    assert live_cells(str(nb)) is None


def test_a_missing_notebook_is_none_not_an_error(tmp_path, monkeypatch):
    import cash.notebook.vscode_backup as vb
    monkeypatch.setattr(vb, "backup_roots", lambda: [tmp_path / "Backups"])
    assert live_cells(str(tmp_path / "gone.ipynb")) is None


def test_a_same_size_resave_with_different_content_is_refused(tmp_path, monkeypatch):
    """A same-length edit -- THRESHOLD = 0.5 -> THRESHOLD = 0.9 -- leaves size
    unable to discriminate at all; mtime is the only thing standing between
    the user and a wrongly-trusted backup on this, the common case."""
    import cash.notebook.vscode_backup as vb

    nb = tmp_path / "nb.ipynb"
    nb.write_bytes(b"THRESHOLD = 0.5")
    st = os.stat(nb)
    root = tmp_path / "Backups" / "ws" / "file"
    root.mkdir(parents=True)
    _write_backup(root, nb, mtime_ms=int(st.st_mtime * 1000), size=st.st_size,
                  cells=["THRESHOLD = 0.5"])
    monkeypatch.setattr(vb, "backup_roots", lambda: [tmp_path / "Backups"])

    nb.write_bytes(b"THRESHOLD = 0.9")  # same size as above, different content
    resave_time = st.st_mtime + 10  # well outside the 2s tolerance
    os.utime(nb, (resave_time, resave_time))
    assert live_cells(str(nb)) is None


def test_a_skew_inside_the_tolerance_is_still_accepted(tmp_path, monkeypatch):
    """Pins the tolerance itself rather than leaving it incidental: filesystem
    mtime granularity varies (FAT is 2s), so a small skew between the header
    and the on-disk mtime must not cost the feature."""
    nb = _setup(tmp_path, monkeypatch, backup_cells=["THRESHOLD = 0.9"], skew_ms=1500)
    cells = live_cells(str(nb))
    assert cells is not None
    assert cells[0]["source"] == "THRESHOLD = 0.9"


def test_a_nul_byte_in_the_path_returns_none_not_a_raise(tmp_path, monkeypatch):
    """os.stat raises ValueError, not OSError, for an embedded NUL. Every
    failure mode here must degrade to None: the caller falls through to the
    saved file on None, but an uncaught exception breaks cell execution."""
    import cash.notebook.vscode_backup as vb
    monkeypatch.setattr(vb, "backup_roots", lambda: [tmp_path / "Backups"])
    assert live_cells("bad\x00path.ipynb") is None


def test_a_boolean_size_header_does_not_pass_as_an_integer(tmp_path, monkeypatch):
    """isinstance(True, int) is True in Python, so a header of {"size": true}
    would otherwise pass the type guard -- and for a 1-byte file, compare equal
    to it too. This is a foreign, undocumented format; tighten the guard."""
    import cash.notebook.vscode_backup as vb

    nb = tmp_path / "nb.ipynb"
    nb.write_bytes(b"x")  # 1 byte, so a bare True == 1 would otherwise match
    st = os.stat(nb)
    root = tmp_path / "Backups" / "ws" / "file"
    root.mkdir(parents=True)
    _write_backup(root, nb, mtime_ms=int(st.st_mtime * 1000), size=True,
                  cells=["x = 1"])
    monkeypatch.setattr(vb, "backup_roots", lambda: [tmp_path / "Backups"])
    assert live_cells(str(nb)) is None


def test_a_freshly_written_backup_is_allowed_to_settle(tmp_path, monkeypatch):
    """VS Code writes the backup on a trailing-edge ~1s debounce, so one touched
    moments ago may still be mid-write. Reading it then can yield a partial
    file -- which parses as None and silently costs the feature."""
    import cash.notebook.vscode_backup as vb

    nb = _setup(tmp_path, monkeypatch, backup_cells=["x = 1"])
    slept = []
    monkeypatch.setattr(vb._time, "sleep", lambda s: slept.append(s))

    backup = vb.find_backup(str(nb))
    os.utime(backup, None)                      # make it look freshly written
    vb._wait_for_backup_settle(backup)

    assert slept, "a freshly written backup was read without waiting for it to settle"


def test_a_settled_backup_is_not_waited_on(tmp_path, monkeypatch):
    """The control: the common case must cost nothing."""
    import cash.notebook.vscode_backup as vb

    nb = _setup(tmp_path, monkeypatch, backup_cells=["x = 1"])
    slept = []
    monkeypatch.setattr(vb._time, "sleep", lambda s: slept.append(s))

    backup = vb.find_backup(str(nb))
    old = time.time() - 60
    os.utime(backup, (old, old))
    vb._wait_for_backup_settle(backup)

    assert slept == [], f"waited on a long-settled backup: {slept}"
