"""``cash clear <dir>`` and ``cash clear --all`` clear a SQLite cache whole.

The sqlite backend keeps a cache in one ``cache.db`` and writes no
``CACHE_VERSION`` stamp and no ``.entry`` files, so the "does this look like
a cache" check refused it -- the very command ``cash clear --function``
points a SQLite user to. A database cash did not write is still refused.
"""

from __future__ import annotations

import sqlite3

import pytest

from cash.__main__ import cmd_clear
from tests._cli_args import cli_args


def _sqlite_cache(cache):
    from cash.backends.sqlite_backend import SQLiteBackend

    db = SQLiteBackend(str(cache / "cache.db"))
    db.set("f:1", 1)
    db.shutdown()
    assert not (cache / "CACHE_VERSION").exists(), "the fixture must be a cache without the stamp"
    return cache


def _clear(monkeypatch, tmp_path, **kwargs):
    monkeypatch.chdir(tmp_path)
    args = cli_args("clear", path=None, all=False, force=False, tool=None, entry=None, function=None, expired=False)
    for key, value in kwargs.items():
        setattr(args, key, value)
    cmd_clear(args)


def test_an_explicit_path_clears_a_sqlite_cache(tmp_path, monkeypatch, capsys):
    cache = _sqlite_cache(tmp_path / "sq")
    _clear(monkeypatch, tmp_path, path=str(cache))
    out = capsys.readouterr().out
    assert not cache.exists(), out
    assert "Cleared" in out, out


def test_all_clears_a_sqlite_cache(tmp_path, monkeypatch, capsys):
    cache = _sqlite_cache(tmp_path / "sq")
    monkeypatch.setenv("CASH_CACHE_DIR", str(cache))
    _clear(monkeypatch, tmp_path, all=True)
    out = capsys.readouterr().out
    assert not cache.exists(), out


def test_a_database_cash_did_not_write_is_still_refused(tmp_path, monkeypatch, capsys):
    """The control: a ``cache.db`` of the user's own is not a cash cache."""
    folder = tmp_path / "mine"
    folder.mkdir()
    conn = sqlite3.connect(folder / "cache.db")
    conn.execute("CREATE TABLE users (id INTEGER)")
    conn.commit()
    conn.close()
    with pytest.raises(SystemExit) as exit_info:
        _clear(monkeypatch, tmp_path, path=str(folder))
    out = capsys.readouterr().out
    assert exit_info.value.code == 1, out
    assert "does not look like a cash cache" in out, out
    assert (folder / "cache.db").exists()
