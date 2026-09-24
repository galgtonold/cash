"""``cash inspect/clear nb.ipynb`` act on the cache the notebook's kernel uses.

They looked in ``<notebook dir>/.cash`` whatever the project configured, so
with ``[tool.cash] cache_dir`` set, ``inspect`` said there was no cache and
``clear`` removed nothing -- or another directory's ``.cash``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cash.__main__ import cmd_clear, cmd_inspect, notebook_cache_dir
from cash.backends.file_backend import FileBackend


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project whose pyproject.toml moves the cache, with a notebook one
    directory down and a stray ``.cash`` beside it that is not its cache."""
    monkeypatch.delenv("CASH_CACHE_DIR", raising=False)
    root = tmp_path / "proj"
    (root / "analysis").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[tool.cash]\ncache_dir = "caches/nb"\n', encoding="utf-8")
    nb = root / "analysis" / "report.ipynb"
    nb.write_text(json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}), encoding="utf-8")
    cache = root / "caches" / "nb"
    store = FileBackend(str(cache), flush_interval=0)
    store.set("app.f:1", 1, {"key": "app.f:1"})
    store.shutdown()
    stray = root / "analysis" / ".cash"
    stray.mkdir()
    (stray / "CACHE_VERSION").write_text("2", encoding="utf-8")
    monkeypatch.chdir(tmp_path)  # the CLI is run from elsewhere
    return SimpleNamespace(root=root, nb=nb, cache=cache, stray=stray)


def test_the_notebooks_cache_is_the_one_its_project_configures(project):
    assert notebook_cache_dir(str(project.nb)) == str(project.cache)


def test_without_a_setting_it_is_beside_the_notebook(tmp_path, monkeypatch):
    monkeypatch.delenv("CASH_CACHE_DIR", raising=False)
    nb = tmp_path / "loose" / "nb.ipynb"
    nb.parent.mkdir()
    nb.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert notebook_cache_dir(str(nb)) == str(nb.parent / ".cash")


def test_clear_removes_the_configured_cache_only(project, capsys):
    cmd_clear(SimpleNamespace(path=str(project.nb), all=False, force=False, tool=None))
    out = capsys.readouterr().out
    assert not project.cache.exists(), out
    assert project.stray.exists(), "a .cash that is not the notebook's cache was removed"


def test_inspect_reports_the_configured_cache(project, capsys):
    pytest.importorskip("nbformat")
    cmd_inspect(SimpleNamespace(path=str(project.nb), function=None, tool=None))
    out = capsys.readouterr().out
    assert f"Cache directory: {project.cache}" in out, out
    assert "Entries: 1" in out, out


@pytest.mark.parametrize(
    "flags",
    [{"function": "f"}, {"entry": "ab"}, {"expired": True}],
    ids=["function", "entry", "expired"],
)
def test_entry_flags_on_a_sqlite_cache_are_refused(tmp_path, monkeypatch, capsys, flags):
    """They scan for entry files, and a SQLite cache has none: the answer was
    "Cleared 0" or "No cached function matches" over a full cache."""
    from cash.backends.sqlite_backend import SQLiteBackend

    cache = tmp_path / "sq"
    db = SQLiteBackend(str(cache / "cache.db"))
    db.set("f:1", 1)
    db.shutdown()
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(path=str(cache), all=False, force=False, tool=None, entry=None, function=None, expired=False)
    for key, value in flags.items():
        setattr(args, key, value)
    with pytest.raises(SystemExit) as exit_info:
        cmd_clear(args)
    out = capsys.readouterr().out
    assert exit_info.value.code == 2, out
    assert "sqlite database" in out, out
    assert (cache / "cache.db").exists()
