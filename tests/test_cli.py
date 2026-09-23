"""Tests for cash CLI (__main__.py)."""

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cash.__main__ import (
    HOOK_FILENAME,
    HOOK_MARKER,
    _format_bytes,
    cmd_autoload,
    cmd_clear,
    cmd_info,
    cmd_inspect,
    cmd_version,
    main,
)
from cash.backends.entry_format import ENTRY_SUFFIX, pack_entry


def _autoload_on(*, mode="active", profile="default", force=False):
    return SimpleNamespace(state="on", mode=mode, profile=profile, force=force)


def _autoload_off(*, profile="default", force=False):
    return SimpleNamespace(state="off", mode="active", profile=profile, force=force)


class TestCLIFormatBytes:
    """Test _format_bytes helper."""

    def test_bytes(self):
        assert _format_bytes(500) == "500 B"

    def test_kilobytes(self):
        assert _format_bytes(2048) == "2.0 KiB"

    def test_megabytes(self):
        assert _format_bytes(5 * 1024 * 1024) == "5.0 MiB"

    def test_gigabytes(self):
        assert _format_bytes(2 * 1024 * 1024 * 1024) == "2.00 GiB"


class TestCLIVersion:
    """Test version command."""

    def test_version_outputs(self, capsys):
        from types import SimpleNamespace

        import cash

        cmd_version(SimpleNamespace())
        captured = capsys.readouterr()
        assert "cash" in captured.out
        assert cash.__version__ in captured.out


class TestCLIInfo:
    """Test info command."""

    def test_info_shows_config(self, capsys):
        from types import SimpleNamespace

        cmd_info(SimpleNamespace())
        captured = capsys.readouterr()
        assert "Backend" in captured.out
        assert "Cache dir" in captured.out

    def test_info_shows_what_this_projects_cache_holds(self, tmp_path, capsys, monkeypatch):
        """`cash info` named the cache dir but not how big it is --
        the number a user asks for when deciding whether to clear it."""
        from cash.config import get_config

        cache_dir = tmp_path / ".cash"
        cache_dir.mkdir()
        for i in range(3):
            (cache_dir / f"e{i}{ENTRY_SUFFIX}").write_bytes(b"x" * 2048)
        config = get_config()
        monkeypatch.setattr(config, "cache_dir", str(cache_dir))
        monkeypatch.setattr("cash.config.get_config", lambda **_: config)
        cmd_info(SimpleNamespace())
        out = capsys.readouterr().out
        line = next(l for l in out.splitlines() if l.strip().startswith("Holds:"))
        assert "3 entries" in line and "6.0 KiB" in line, out


class TestCLIInspect:
    """Test inspect command."""

    def test_inspect_cache_dir(self, tmp_path, capsys):
        """Inspect a cache directory."""
        # Create a fake cache dir
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()

        # Create some fake cache files
        meta = {"key": "test_key", "created_at": time.time(), "outputs": ["x", "y"]}
        (cache_dir / f"abc123{ENTRY_SUFFIX}").write_bytes(pack_entry(meta, b"fake data"))

        from types import SimpleNamespace

        cmd_inspect(SimpleNamespace(path=str(cache_dir)))
        captured = capsys.readouterr()
        assert "Total size:" in captured.out
        assert "Entries:" in captured.out

    def test_inspect_nonexistent(self, capsys, tmp_path, monkeypatch):
        """Inspect with no path and no cache should fail gracefully.

        The cache directory is pinned: without it the command resolves whatever
        cache the environment happens to offer (the repo's own, a sibling
        test's), so the test passed or failed by run order.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CASH_CACHE_DIR", str(tmp_path / "no-such-cache"))
        from types import SimpleNamespace

        with pytest.raises(SystemExit):
            cmd_inspect(SimpleNamespace(path=None))


class TestCLIClear:
    """Test clear command."""

    def test_clear_directory(self, tmp_path, capsys):
        """Clear should remove a cache directory named explicitly.

        It needs to LOOK like one (a CACHE_VERSION stamp, as every real cache
        has): an explicit path used to go straight to rmtree, and a user ran
        `cash clear .` in a project and lost its files.
        """
        cache_dir = tmp_path / "to_clear"
        cache_dir.mkdir()
        (cache_dir / "CACHE_VERSION").write_text("1")
        (cache_dir / "file.data").write_bytes(b"data")

        from types import SimpleNamespace

        cmd_clear(SimpleNamespace(path=str(cache_dir), all=False))
        assert not cache_dir.exists()
        captured = capsys.readouterr()
        assert "Cleared" in captured.out

    def test_clear_all(self, tmp_path, capsys, monkeypatch):
        """Clear --all should remove .cash in current directory."""
        monkeypatch.chdir(tmp_path)
        cache_dir = tmp_path / ".cash"
        cache_dir.mkdir()
        (cache_dir / "CACHE_VERSION").write_text("1")
        (cache_dir / "file.data").write_bytes(b"data")
        monkeypatch.setenv("CASH_CACHE_DIR", str(cache_dir))

        from types import SimpleNamespace

        cmd_clear(SimpleNamespace(path=None, all=True))
        assert not cache_dir.exists()

    def test_clear_no_args(self, capsys):
        """Clear without args should fail."""
        from types import SimpleNamespace

        with pytest.raises(SystemExit):
            cmd_clear(SimpleNamespace(path=None, all=False))

    def test_clear_all_no_cache(self, tmp_path, capsys, monkeypatch):
        """Clear --all when the resolved cache directory does not exist.

        The message names the directory now, because `--all` no longer means
        "./.cash" -- it means whatever the config resolved to, which the user
        may not be standing in.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CASH_CACHE_DIR", str(tmp_path / ".cash"))
        from types import SimpleNamespace

        cmd_clear(SimpleNamespace(path=None, all=True))
        captured = capsys.readouterr()
        assert "Nothing cleared: no cache at" in captured.out
        assert str(tmp_path) in captured.out
        # Where a running program's cache is when it is not here.
        assert "beside the script" in captured.out

    def test_clear_nonexistent_path(self, capsys):
        """Clear nonexistent path should fail."""
        from types import SimpleNamespace

        with pytest.raises(SystemExit):
            cmd_clear(SimpleNamespace(path="/nonexistent/path", all=False))

    def test_clear_notebook_with_cache(self, tmp_path, capsys):
        """Clear using a notebook path clears its .cash directory."""
        # Create a fake notebook and cache
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text('{"cells":[]}')
        cache_dir = tmp_path / ".cash"
        cache_dir.mkdir()
        (cache_dir / "CACHE_VERSION").write_text("1")
        (cache_dir / "data.meta").write_bytes(b"data")

        from types import SimpleNamespace

        cmd_clear(SimpleNamespace(path=str(nb_path), all=False))
        assert not cache_dir.exists()
        captured = capsys.readouterr()
        assert "Cleared" in captured.out

    def test_clear_notebook_without_cache(self, tmp_path, capsys):
        """Clear using a notebook with no cache prints message."""
        nb_path = tmp_path / "test.ipynb"
        nb_path.write_text('{"cells":[]}')

        from types import SimpleNamespace

        cmd_clear(SimpleNamespace(path=str(nb_path), all=False))
        captured = capsys.readouterr()
        assert "No cache found" in captured.out


class TestCLIMain:
    """Test main entry point."""

    def test_no_args_shows_help(self, capsys):
        """Running with no args should show help."""
        with patch("sys.argv", ["cash"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_version_subcommand(self, capsys):
        """Running 'version' subcommand."""
        with patch("sys.argv", ["cash", "version"]):
            main()
        captured = capsys.readouterr()
        assert "cash" in captured.out

    def test_inspect_subcommand_with_dir(self, tmp_path, capsys):
        """Running 'inspect' with a directory."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        with patch("sys.argv", ["cash", "inspect", str(cache_dir)]):
            main()
        captured = capsys.readouterr()
        assert "Cache directory:" in captured.out


class TestCLIInspectNotebook:
    """Test notebook inspection."""

    def test_inspect_notebook(self, tmp_path, capsys):
        """Inspect a notebook file."""
        nb_path = tmp_path / "test.ipynb"
        nb_content = {
            "cells": [
                {"cell_type": "code", "source": "%cash_on", "metadata": {}, "outputs": []},
                {"cell_type": "markdown", "source": "# Title", "metadata": {}},
                {"cell_type": "code", "source": "x = 1", "metadata": {}, "outputs": []},
            ],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        import json

        nb_path.write_text(json.dumps(nb_content))

        from cash.__main__ import _inspect_notebook

        _inspect_notebook(str(nb_path))
        captured = capsys.readouterr()
        assert "Code cells: 2" in captured.out
        assert "Markdown cells: 1" in captured.out
        assert "Uses cash: Yes" in captured.out

    def test_inspect_notebook_without_cash(self, tmp_path, capsys):
        """Inspect notebook that doesn't use cash."""
        nb_path = tmp_path / "test.ipynb"
        nb_content = {
            "cells": [
                {"cell_type": "code", "source": "x = 1", "metadata": {}, "outputs": []},
            ],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        import json

        nb_path.write_text(json.dumps(nb_content))

        from cash.__main__ import _inspect_notebook

        _inspect_notebook(str(nb_path))
        captured = capsys.readouterr()
        assert "Uses cash: No" in captured.out
        assert "Cache: not found" in captured.out

    def test_inspect_notebook_with_cache(self, tmp_path, capsys):
        """Inspect notebook that has a .cash directory."""
        nb_path = tmp_path / "test.ipynb"
        nb_content = {
            "cells": [{"cell_type": "code", "source": "x = 1", "metadata": {}, "outputs": []}],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        import json

        nb_path.write_text(json.dumps(nb_content))

        cache_dir = tmp_path / ".cash"
        cache_dir.mkdir()
        (cache_dir / "data.meta").write_bytes(b"data")

        from cash.__main__ import _inspect_notebook

        _inspect_notebook(str(nb_path))
        captured = capsys.readouterr()
        assert "Cache directory:" in captured.out

    def test_inspect_invalid_notebook(self, tmp_path, capsys):
        """Inspect an invalid notebook file."""
        nb_path = tmp_path / "bad.ipynb"
        nb_path.write_text("not valid json for notebook")

        from cash.__main__ import _inspect_notebook

        _inspect_notebook(str(nb_path))
        captured = capsys.readouterr()
        assert "Error reading notebook" in captured.out

    def test_inspect_cache_dir_with_metadata(self, tmp_path, capsys):
        """Inspect cache dir that has readable metadata with timestamp."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        meta = {"key": "stmt:abc123def456", "created_at": 1700000000.0, "outputs": ["result", "df"]}
        (cache_dir / f"entry1{ENTRY_SUFFIX}").write_bytes(pack_entry(meta, b""))

        from cash.__main__ import _inspect_cache_dir

        _inspect_cache_dir(str(cache_dir))
        captured = capsys.readouterr()
        assert "Entries: 1" in captured.out
        # A statement key has no function to name, so it groups under one
        # heading rather than being reported as a function called "stmt".
        assert "(notebook statements)" in captured.out

    def test_inspect_cache_dir_empty(self, tmp_path, capsys):
        """Inspect empty cache directory."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        from cash.__main__ import _inspect_cache_dir

        _inspect_cache_dir(str(cache_dir))
        captured = capsys.readouterr()
        assert "Entries: 0" in captured.out
        assert "(no readable entries)" in captured.out

    def test_inspect_cache_dir_corrupt_meta(self, tmp_path, capsys):
        """Inspect cache dir with unreadable metadata files."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "bad.meta").write_bytes(b"not a pickle")

        from cash.__main__ import _inspect_cache_dir

        _inspect_cache_dir(str(cache_dir))
        captured = capsys.readouterr()
        # One corrupt file must not cost the report for everything else;
        # it is skipped, and the directory still totals correctly.
        assert "Entries: 0" in captured.out
        assert "Total size:" in captured.out


@pytest.fixture
def fake_ipython_dir(tmp_path, monkeypatch):
    """Redirect the install/uninstall hook commands at a tmp_path-based
    startup directory so we never touch the user's real IPython config."""
    startup = tmp_path / "ipython" / "profile_default" / "startup"

    def _fake(profile: str):
        return tmp_path / "ipython" / f"profile_{profile}" / "startup"

    monkeypatch.setattr("cash.__main__._ipython_startup_dir", _fake)
    return startup


class TestCLIAutoloadOn:
    """Test `cash autoload on`."""

    def test_active_creates_hook(self, fake_ipython_dir, capsys):
        cmd_autoload(_autoload_on(mode="active"))

        hook_path = fake_ipython_dir / HOOK_FILENAME
        assert hook_path.exists(), "hook file should be created"
        body = hook_path.read_text(encoding="utf-8")
        assert HOOK_MARKER in body
        assert "import cash" in body
        assert 'run_line_magic("cash_on"' in body  # active mode runs %cash_on
        captured = capsys.readouterr()
        assert "mode=active" in captured.out

    def test_available_omits_cash_on(self, fake_ipython_dir):
        cmd_autoload(_autoload_on(mode="available"))

        body = (fake_ipython_dir / HOOK_FILENAME).read_text(encoding="utf-8")
        assert "import cash" in body
        assert "run_line_magic" not in body  # available mode does NOT auto-enable

    def test_is_idempotent(self, fake_ipython_dir, capsys):
        """Re-running with the same mode should be a no-op, not an error."""
        cmd_autoload(_autoload_on(mode="active"))
        capsys.readouterr()  # clear

        cmd_autoload(_autoload_on(mode="active"))
        captured = capsys.readouterr()
        assert "already on" in captured.out

    def test_refuses_to_clobber_different_content(self, fake_ipython_dir):
        """Without --force, must not overwrite a different file."""
        fake_ipython_dir.mkdir(parents=True)
        (fake_ipython_dir / HOOK_FILENAME).write_text("# user's own startup script\n", encoding="utf-8")

        with pytest.raises(SystemExit):
            cmd_autoload(_autoload_on(mode="active"))
        # File contents must be preserved
        assert (fake_ipython_dir / HOOK_FILENAME).read_text(encoding="utf-8") == "# user's own startup script\n"

    def test_force_overwrites(self, fake_ipython_dir):
        """--force replaces an existing file."""
        fake_ipython_dir.mkdir(parents=True)
        (fake_ipython_dir / HOOK_FILENAME).write_text("# something else\n", encoding="utf-8")

        cmd_autoload(_autoload_on(mode="available", force=True))
        body = (fake_ipython_dir / HOOK_FILENAME).read_text(encoding="utf-8")
        assert HOOK_MARKER in body
        assert "import cash" in body

    def test_can_switch_modes_with_force(self, fake_ipython_dir):
        """Switching from active -> available requires --force (different content)."""
        cmd_autoload(_autoload_on(mode="active"))
        with pytest.raises(SystemExit):
            cmd_autoload(_autoload_on(mode="available"))
        cmd_autoload(_autoload_on(mode="available", force=True))
        body = (fake_ipython_dir / HOOK_FILENAME).read_text(encoding="utf-8")
        assert "run_line_magic" not in body

    def test_custom_profile(self, fake_ipython_dir, tmp_path):
        cmd_autoload(_autoload_on(mode="active", profile="research"))
        custom = tmp_path / "ipython" / "profile_research" / "startup" / HOOK_FILENAME
        assert custom.exists()
        # Default profile should be untouched
        assert not (fake_ipython_dir / HOOK_FILENAME).exists()


class TestCLIAutoloadOff:
    """Test `cash autoload off`."""

    def test_removes_hook(self, fake_ipython_dir, capsys):
        cmd_autoload(_autoload_on(mode="active"))
        hook_path = fake_ipython_dir / HOOK_FILENAME
        assert hook_path.exists()

        cmd_autoload(_autoload_off())
        assert not hook_path.exists()
        captured = capsys.readouterr()
        assert "Autoload off" in captured.out

    def test_when_nothing_installed(self, fake_ipython_dir, capsys):
        cmd_autoload(_autoload_off())
        captured = capsys.readouterr()
        assert "not installed" in captured.out

    def test_refuses_unmarked_file(self, fake_ipython_dir):
        """Don't delete a file that doesn't carry our marker — it's not ours."""
        fake_ipython_dir.mkdir(parents=True)
        unrelated = fake_ipython_dir / HOOK_FILENAME
        unrelated.write_text("# user's own startup script\n", encoding="utf-8")

        with pytest.raises(SystemExit):
            cmd_autoload(_autoload_off())
        assert unrelated.exists()  # left alone

    def test_force_removes_unmarked(self, fake_ipython_dir):
        fake_ipython_dir.mkdir(parents=True)
        (fake_ipython_dir / HOOK_FILENAME).write_text("# something\n", encoding="utf-8")

        cmd_autoload(_autoload_off(force=True))
        assert not (fake_ipython_dir / HOOK_FILENAME).exists()


class TestInspectNamesWhatItWasGiven:
    """``cash inspect <path>`` reports THAT path, or says it is not there.

    Found while stress-testing the decorator: a path that does not
    exist fell through to the configured cache, so `cash inspect ./nope` (or a
    mistyped notebook name) printed a full, plausible report about an unrelated
    cache and exited 0. `cash clear` already refuses the same input.
    """

    def test_a_missing_path_is_not_silently_replaced(self, tmp_path, monkeypatch, capsys):
        cache = tmp_path / ".cash"
        cache.mkdir()
        (cache / "CACHE_VERSION").write_text("2")
        (cache / f"abc{ENTRY_SUFFIX}").write_bytes(pack_entry({"key": "k"}, b"v"))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CASH_CACHE_DIR", str(cache))
        with pytest.raises(SystemExit) as exit_info:
            cmd_inspect(SimpleNamespace(path="./nope", function=None, tool=None))
        assert exit_info.value.code == 1
        out = capsys.readouterr().out
        assert "nope" in out, out
        assert "Entries:" not in out, "it reported some other cache"

    def test_an_existing_path_is_still_inspected(self, tmp_path, monkeypatch, capsys):
        cache = tmp_path / "other"
        cache.mkdir()
        (cache / f"abc{ENTRY_SUFFIX}").write_bytes(pack_entry({"key": "k"}, b"v"))
        monkeypatch.chdir(tmp_path)
        cmd_inspect(SimpleNamespace(path=str(cache), function=None, tool=None))
        assert "Entries:" in capsys.readouterr().out


class TestTheCliSeesASqliteCache:
    """A sqlite cache is a database file, not a directory of entries.

    Found while stress-testing the decorator: `cash info` said
    "nothing yet (no cache written here)" and `cash inspect` said "No cache
    found" while a working sqlite cache sat in the directory, because both
    count `*.entry` files.
    """

    @staticmethod
    def _db(tmp_path):
        import sqlite3

        cache = tmp_path / ".cash"
        cache.mkdir()
        conn = sqlite3.connect(cache / "cache.db")
        conn.execute(
            "CREATE TABLE cache_entries (key TEXT PRIMARY KEY, data BLOB, "
            "metadata BLOB, created_at REAL, ttl REAL, access_count INTEGER)"
        )
        for i in range(3):
            conn.execute(
                "INSERT INTO cache_entries VALUES (?, ?, ?, ?, ?, ?)", (f"k{i}", b"x" * 1024, b"{}", 0.0, None, 0)
            )
        conn.commit()
        conn.close()
        return cache

    def test_info_counts_the_databases_entries(self, tmp_path, monkeypatch, capsys):
        cache = self._db(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("CASH_CACHE_DIR", str(cache))
        cmd_info(SimpleNamespace())
        line = next(l for l in capsys.readouterr().out.splitlines() if l.strip().startswith("Holds:"))
        assert "3 entries" in line, line

    def test_inspect_reports_the_database(self, tmp_path, monkeypatch, capsys):
        cache = self._db(tmp_path)
        monkeypatch.chdir(tmp_path)
        cmd_inspect(SimpleNamespace(path=str(cache), function=None, tool=None))
        out = capsys.readouterr().out
        assert "Entries: 3" in out.replace("    ", " ").replace("  ", " "), out
