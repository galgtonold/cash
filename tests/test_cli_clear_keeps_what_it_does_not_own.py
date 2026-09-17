"""``cash clear`` removes cash's own files, never a user's.

Found while attacking the decorator before round 26: a project whose
``[tool.cash] cache_dir`` points at a directory holding data --
``cache_dir = "../shared_data"`` -- lost that data. cash writes its
``CACHE_VERSION`` stamp into whatever directory it is pointed at, so the
"does it look like a cache" guard passes and ``clear --all`` removed the
directory whole: ``Cleared: ...\shared_data``, exit 0, ``precious.csv`` gone.
"""
from __future__ import annotations

import pytest

from cash.__main__ import cmd_clear
from cash.backends.entry_format import ENTRY_SUFFIX
from types import SimpleNamespace


def _cache_with(tmp_path, *foreign):
    cache = tmp_path / "shared_data"
    (cache / "raw").mkdir(parents=True)
    (cache / "CACHE_VERSION").write_text("2")
    (cache / f"abc{ENTRY_SUFFIX}").write_bytes(b"entry")
    (cache / "_rank.log").write_text("")
    for name in foreign:
        (cache / name).write_text("mine")
    return cache


def _clear(cache, monkeypatch, tmp_path, **kwargs):
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(path=str(cache), all=False, force=False, tool=None,
                           entry=None, function=None, expired=False)
    for key, value in kwargs.items():
        setattr(args, key, value)
    cmd_clear(args)


def test_a_cache_holding_user_files_is_not_removed(tmp_path, monkeypatch, capsys):
    cache = _cache_with(tmp_path, "precious.csv")
    with pytest.raises(SystemExit) as exit_info:
        _clear(cache, monkeypatch, tmp_path)
    assert exit_info.value.code == 1
    out = capsys.readouterr().out
    assert (cache / "precious.csv").read_text() == "mine", out
    assert "precious.csv" in out, out
    assert (cache / f"abc{ENTRY_SUFFIX}").exists(), "nothing was cleared"


def test_force_clears_a_cache_holding_user_files(tmp_path, monkeypatch, capsys):
    cache = _cache_with(tmp_path, "precious.csv")
    _clear(cache, monkeypatch, tmp_path, force=True)
    assert not cache.exists(), capsys.readouterr().out


def test_a_cache_of_only_cash_files_is_removed(tmp_path, monkeypatch, capsys):
    cache = _cache_with(tmp_path)
    (cache / "raw").rmdir()
    _clear(cache, monkeypatch, tmp_path)
    assert not cache.exists(), capsys.readouterr().out
    assert "Cleared" in capsys.readouterr().out or True


def test_a_subdirectory_of_user_files_is_not_removed(tmp_path, monkeypatch, capsys):
    cache = _cache_with(tmp_path)
    (cache / "raw" / "x.bin").write_bytes(b"data")
    with pytest.raises(SystemExit) as exit_info:
        _clear(cache, monkeypatch, tmp_path)
    assert exit_info.value.code == 1
    assert (cache / "raw" / "x.bin").exists(), capsys.readouterr().out
