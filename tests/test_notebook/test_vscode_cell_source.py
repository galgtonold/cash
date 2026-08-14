"""VS Code backup wiring in server_discovery: cash reads live cells from VS
Code's hot-exit backup through ``get_notebook_cells()`` /
``get_notebook_cells_with_ids()``, not just through the underlying
``vscode_backup`` module in isolation.

``test_vscode_backup.py`` covers ``vscode_backup.py``'s own contract
(``find_backup``, ``parse_backup``, ``live_cells``) but never imports
``server_discovery``, so a passing run of that file alone is not evidence
``_try_vscode_backup_cells`` is ever reached from the real entry points --
the exact failure mode CAS-274 hit, where a feature shipped dead while every
unit test passed. This file plants a real backup on disk and asserts through
the real ``server_discovery`` entry points instead, mirroring
``test_colab_cell_source.py``'s shape for the analogous Colab wiring.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from cash.notebook import server_discovery as sd
from cash.notebook import vscode_backup as vb


def _write_notebook(path: Path, sources: list[str]) -> None:
    cells = [{"cell_type": "code", "source": s} for s in sources]
    path.write_text(
        json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}),
        encoding="utf-8",
    )


def _write_backup(dirpath: Path, nb_path: Path, *, mtime_ms: int, size: int,
                  cells: list[str], name: str = "abc123") -> Path:
    """Write a file in VS Code's backup format (same shape as
    test_vscode_backup.py's helper of the same name)."""
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


def _plant_backup(tmp_path, monkeypatch, nb: Path, *, cells: list[str],
                  settled: bool = True, name: str = "abc123") -> Path:
    """Point vscode_backup at a fake Backups root holding one matching backup
    for *nb*, settled (aged past the debounce window) by default."""
    st = os.stat(nb)
    root = tmp_path / "Backups" / "ws" / "file"
    root.mkdir(parents=True, exist_ok=True)
    backup = _write_backup(root, nb, mtime_ms=int(st.st_mtime * 1000), size=st.st_size,
                           cells=cells, name=name)
    if settled:
        old = time.time() - 60
        os.utime(backup, (old, old))
    monkeypatch.setattr(vb, "backup_roots", lambda: [tmp_path / "Backups"])
    sd.invalidate_notebook_cells_cache()
    return backup


def _signal_vscode(monkeypatch, nb: Path) -> None:
    """Simulate VS Code's injected ``__vsc_ipynb_file__`` variable -- the
    signal ``_try_vscode_backup_cells`` gates on before it will even look for
    a backup. Tests that plant a backup and expect it to win must call this;
    ``_try_vscode_path`` is otherwise unmocked and returns None under plain
    pytest (no real VS Code kernel), which would make the gate refuse."""
    monkeypatch.setattr(sd, "_try_vscode_path", lambda: str(nb))


def test_get_notebook_cells_returns_backup_content_not_saved_file(tmp_path, monkeypatch):
    """The feature's whole point: an unsaved edit must win over the stale
    saved file. This is the wiring the review found nothing committed tests."""
    nb = tmp_path / "nb.ipynb"
    _write_notebook(nb, ["OLD_SAVED = 1"])
    _signal_vscode(monkeypatch, nb)
    _plant_backup(tmp_path, monkeypatch, nb, cells=["NEW_UNSAVED = 2"])

    assert sd.get_notebook_cells(str(nb)) == ["NEW_UNSAVED = 2"]


def test_get_notebook_cells_with_ids_returns_backup_content(tmp_path, monkeypatch):
    nb = tmp_path / "nb.ipynb"
    _write_notebook(nb, ["OLD_SAVED = 1"])
    _signal_vscode(monkeypatch, nb)
    _plant_backup(tmp_path, monkeypatch, nb, cells=["NEW_UNSAVED = 2"])

    result = sd.get_notebook_cells_with_ids(str(nb))
    assert [code for _id, code in result] == ["NEW_UNSAVED = 2"]


def test_no_backup_falls_through_to_the_saved_file(tmp_path, monkeypatch):
    """The negative control: with nothing planted, behaviour is exactly what
    it was before this feature existed."""
    nb = tmp_path / "nb.ipynb"
    _write_notebook(nb, ["OLD_SAVED = 1"])
    monkeypatch.setattr(vb, "backup_roots", lambda: [tmp_path / "empty"])
    sd.invalidate_notebook_cells_cache()

    assert sd.get_notebook_cells(str(nb)) == ["OLD_SAVED = 1"]


def test_no_backup_served_without_vscode_signal(tmp_path, monkeypatch):
    """The gate: no signal that this kernel is running inside VS Code means no
    backup is served, even though a perfectly valid, matching one exists. The
    concrete failure this closes: leave a notebook dirty in a VS Code window,
    then open and run that same file in JupyterLab -- without the gate, cash
    would use edits that are not on the screen being looked at. A hot-exit
    backup can also outlive its VS Code session by weeks, so "a matching
    backup exists" is never on its own evidence of anything current."""
    nb = tmp_path / "nb.ipynb"
    _write_notebook(nb, ["OLD_SAVED = 1"])
    monkeypatch.setattr(sd, "_try_vscode_path", lambda: None)  # explicitly not VS Code
    _plant_backup(tmp_path, monkeypatch, nb, cells=["NEW_UNSAVED = 2"])

    assert sd.get_notebook_cells(str(nb)) == ["OLD_SAVED = 1"], (
        "served a VS Code backup with no VS Code signal present"
    )


def test_backup_still_served_with_vscode_signal_present(tmp_path, monkeypatch):
    """The control paired with the test above: identical setup, but with the
    signal present, the backup still wins. The gate must not cost the feature
    anything in the case it exists to protect."""
    nb = tmp_path / "nb.ipynb"
    _write_notebook(nb, ["OLD_SAVED = 1"])
    _signal_vscode(monkeypatch, nb)
    _plant_backup(tmp_path, monkeypatch, nb, cells=["NEW_UNSAVED = 2"])

    assert sd.get_notebook_cells(str(nb)) == ["NEW_UNSAVED = 2"]


def test_a_malformed_cell_entry_returns_none_not_a_raise(tmp_path, monkeypatch):
    """A non-dict entry in an otherwise well-formed backup's cells list must
    degrade to None (fall through to the file), not raise out of
    get_notebook_cells(). live_cells() does no per-cell shape check, so
    guarding against this is _try_vscode_backup_cells's own responsibility --
    the review reproduced this raising AttributeError before the fix."""
    nb = tmp_path / "nb.ipynb"
    _write_notebook(nb, ["OLD_SAVED = 1"])
    _signal_vscode(monkeypatch, nb)
    st = os.stat(nb)
    root = tmp_path / "Backups" / "ws" / "file"
    root.mkdir(parents=True)
    uri = nb.as_uri()
    header = json.dumps({"mtime": int(st.st_mtime * 1000), "size": st.st_size})
    body = json.dumps({
        "cells": ["not a dict", {"cell_type": "code", "id": "c1", "source": "x = 1"}],
        "nbformat": 4, "nbformat_minor": 5, "metadata": {},
    })
    backup = root / "abc123"
    backup.write_text(f"{uri} {header}\n{body}", encoding="utf-8")
    old = time.time() - 60
    os.utime(backup, (old, old))
    monkeypatch.setattr(vb, "backup_roots", lambda: [tmp_path / "Backups"])
    sd.invalidate_notebook_cells_cache()

    # Must not raise; degrades to the saved file, same as any other
    # unusable-backup case.
    assert sd.get_notebook_cells(str(nb)) == ["OLD_SAVED = 1"]


def test_a_second_call_with_nothing_changed_does_not_rescan(tmp_path, monkeypatch):
    """The caching fix: a single cell run calls this ~3x (the magic once, the
    upstream checker twice) against unchanged state; only the first should
    pay for find_backup's directory scan and the settle wait."""
    nb = tmp_path / "nb.ipynb"
    _write_notebook(nb, ["OLD_SAVED = 1"])
    _signal_vscode(monkeypatch, nb)
    _plant_backup(tmp_path, monkeypatch, nb, cells=["NEW_UNSAVED = 2"])

    first = sd.get_notebook_cells(str(nb))  # cold: warms the cache

    calls = {"n": 0}
    real_find_backup = vb.find_backup

    def _counting(path):
        calls["n"] += 1
        return real_find_backup(path)

    monkeypatch.setattr(vb, "find_backup", _counting)

    second = sd.get_notebook_cells(str(nb))  # warm: nothing changed

    assert first == second == ["NEW_UNSAVED = 2"]
    assert calls["n"] == 0, f"second call re-scanned the backup directory: {calls['n']} find_backup call(s)"


def test_a_changed_backup_between_calls_is_picked_up(tmp_path, monkeypatch):
    """The cache's other half, and the more important one: it must not serve
    stale content forever. A same-path backup rewritten with new content
    between two calls (the notebook file on disk untouched throughout) must
    be reflected on the next read -- caching on the file's signature alone
    would have frozen the first edit and missed this."""
    nb = tmp_path / "nb.ipynb"
    _write_notebook(nb, ["OLD_SAVED = 1"])
    _signal_vscode(monkeypatch, nb)
    _plant_backup(tmp_path, monkeypatch, nb, cells=["FIRST_EDIT = 1"])

    assert sd.get_notebook_cells(str(nb)) == ["FIRST_EDIT = 1"]

    # The user keeps typing, still unsaved: the backup is rewritten in place
    # (same path, new content, later mtime); the notebook file is untouched.
    st = os.stat(nb)
    root = tmp_path / "Backups" / "ws" / "file"
    time.sleep(0.01)
    backup = _write_backup(root, nb, mtime_ms=int(st.st_mtime * 1000), size=st.st_size,
                           cells=["SECOND_EDIT = 2"], name="abc123")
    old = time.time() - 60
    os.utime(backup, (old, old))  # keep it settled so the wait doesn't interfere

    assert sd.get_notebook_cells(str(nb)) == ["SECOND_EDIT = 2"], "cache served a stale edit"
