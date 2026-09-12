"""Round 20 (r20s4): configuring a packaged tool failed silently.

* ``Cash(config_path=...)`` naming a file the wheel did not ship: ignored, and
  the tool's cache lifetime with it.
* ``cache_dir = "~/crunch-cache"`` in that file became a directory literally
  named ``~`` beside the file, inside site-packages.
* ``cash info`` could not show what such a file resolves to.
* Lowering a tier's ``default_ttl`` freed no disk, ``cash inspect`` still showed
  the old expiry, and nothing could drop only the expired entries.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from types import SimpleNamespace

import pytest

from cash import config as cash_config
from cash.config import get_config

pytestmark = [pytest.mark.core]

needs_toml = pytest.mark.skipif(
    sys.version_info < (3, 11) and not __import__("importlib").util.find_spec("tomli"),
    reason="no TOML parser")


def test_a_config_file_that_does_not_exist_is_named(tmp_path, monkeypatch):
    monkeypatch.setattr(cash_config, "_CONFIG_NOTICES", set())
    missing = tmp_path / "pkg" / "cash.toml"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        get_config(config_path=str(missing), user_config_path=None, project_config_path=None)
    said = [str(w.message) for w in rec if "[CONFIG-FILE-MISSING]" in str(w.message)]
    assert said and str(missing) in said[0], [str(w.message) for w in rec]


@needs_toml
def test_a_home_relative_cache_dir_in_a_config_file_is_expanded(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    shipped = tmp_path / "site-packages" / "tool" / "cash.toml"
    shipped.parent.mkdir(parents=True)
    shipped.write_text('cache_dir = "~/crunch-cache"\n', encoding="utf-8")
    cfg = get_config(config_path=str(shipped), user_config_path=None, project_config_path=None)
    assert os.path.normcase(cfg.cache_dir) == os.path.normcase(str(home / "crunch-cache")), cfg.cache_dir


@needs_toml
def test_cash_info_shows_what_a_tools_config_file_resolves_to(tmp_path, capsys, monkeypatch):
    from cash.__main__ import cmd_info
    for key in [k for k in os.environ if k.startswith("CASH_")]:
        monkeypatch.delenv(key)
    shipped = tmp_path / "cash.toml"
    shipped.write_text("[tool.cash]\ncompress = true\n", encoding="utf-8")
    cmd_info(SimpleNamespace(config=str(shipped)))
    out = capsys.readouterr().out
    assert str(shipped) in out and "compress = True" in out.replace("true", "True"), out


def test_the_expiry_shown_and_cleared_follows_a_lowered_tier_default():
    from cash.__main__ import _effective_ttl
    assert _effective_ttl({"ttl": 86400}, tier_default=5) == 5
    assert _effective_ttl({"ttl": 86400, "ttl_declared": True}, tier_default=5) == 86400
    assert _effective_ttl({}, tier_default=5) == 5
    assert _effective_ttl({"ttl": 60}, tier_default=None) == 60


def test_clear_expired_frees_what_is_never_served_again(tmp_path, capsys, monkeypatch):
    from cash.__main__ import _clear_expired, _scan_entries
    from cash.backends.file_backend import FileBackend
    monkeypatch.setattr("cash.__main__._tier_default_ttl", lambda: None)
    backend = FileBackend(cache_dir=str(tmp_path / "cache"))
    backend.set("app.f:s::old", 1, {"key": "app.f:s::old", "ttl": 1, "created_at": time.time() - 60})
    backend.set("app.f:s::live", 2, {"key": "app.f:s::live", "ttl": 3600, "created_at": time.time()})
    backend.set("app.f:s::forever", 3, {"key": "app.f:s::forever"})
    backend._writes.wait_all()
    _clear_expired(str(tmp_path / "cache"))
    assert "Cleared 1 expired entry" in capsys.readouterr().out
    left = sorted(e.key for e in _scan_entries(tmp_path / "cache"))
    assert left == ["app.f:s::forever", "app.f:s::live"], left


def test_clear_expired_with_a_function_is_refused_not_widened(tmp_path):
    # --function used to win, deleting that function's live entries as well.
    from cash.__main__ import cmd_clear
    args = SimpleNamespace(path=str(tmp_path), all=False, tool=None, entry=None,
                           function="f", expired=True, force=False)
    with pytest.raises(SystemExit) as exit_:
        cmd_clear(args)
    assert exit_.value.code == 2
