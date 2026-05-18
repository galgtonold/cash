"""Tests for cash CLI (__main__.py)."""
import pytest
import pickle
import time
from types import SimpleNamespace
from unittest.mock import patch
from cash.__main__ import (
    main,
    cmd_version,
    cmd_info,
    cmd_inspect,
    cmd_clear,
    cmd_autoload,
    _format_bytes,
    HOOK_FILENAME,
    HOOK_MARKER,
)


def _autoload_on(*, mode="active", profile="default", force=False):
    return SimpleNamespace(state="on", mode=mode, profile=profile, force=force)


def _autoload_off(*, profile="default", force=False):
    return SimpleNamespace(state="off", mode="active", profile=profile, force=force)


class TestCLIFormatBytes:
    """Test _format_bytes helper."""

    def test_bytes(self):
        assert _format_bytes(500) == "500 B"

    def test_kilobytes(self):
        assert _format_bytes(2048) == "2.0 KB"

    def test_megabytes(self):
        assert _format_bytes(5 * 1024 * 1024) == "5.0 MB"

    def test_gigabytes(self):
        assert _format_bytes(2 * 1024 * 1024 * 1024) == "2.00 GB"


class TestCLIVersion:
    """Test version command."""

    def test_version_outputs(self, capsys):
        from types import SimpleNamespace
        cmd_version(SimpleNamespace())
        captured = capsys.readouterr()
        assert "cash" in captured.out
        assert "0.2.0" in captured.out


class TestCLIInfo:
    """Test info command."""

    def test_info_shows_config(self, capsys):
        from types import SimpleNamespace
        cmd_info(SimpleNamespace())
        captured = capsys.readouterr()
        assert "Backend" in captured.out
        assert "Cache dir" in captured.out


class TestCLIInspect:
    """Test inspect command."""

    def test_inspect_cache_dir(self, tmp_path, capsys):
        """Inspect a cache directory."""
        # Create a fake cache dir
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()

        # Create some fake cache files
        meta = {'key': 'test_key', 'created_at': time.time(), 'outputs': ['x', 'y']}
        with open(cache_dir / "abc123.meta", 'wb') as f:
            pickle.dump(meta, f)
        (cache_dir / "abc123.data").write_bytes(b"fake data")

        from types import SimpleNamespace
        cmd_inspect(SimpleNamespace(path=str(cache_dir)))
        captured = capsys.readouterr()
        assert "Total files:" in captured.out
        assert "Cache entries:" in captured.out

    def test_inspect_nonexistent(self, capsys, tmp_path, monkeypatch):
        """Inspect nonexistent path should fail gracefully."""
        monkeypatch.chdir(tmp_path)
        from types import SimpleNamespace
        with pytest.raises(SystemExit):
            cmd_inspect(SimpleNamespace(path=None))


class TestCLIClear:
    """Test clear command."""

    def test_clear_directory(self, tmp_path, capsys):
        """Clear should remove a directory."""
        cache_dir = tmp_path / "to_clear"
        cache_dir.mkdir()
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
        (cache_dir / "file.data").write_bytes(b"data")

        from types import SimpleNamespace
        cmd_clear(SimpleNamespace(path=None, all=True))
        assert not cache_dir.exists()

    def test_clear_no_args(self, capsys):
        """Clear without args should fail."""
        from types import SimpleNamespace
        with pytest.raises(SystemExit):
            cmd_clear(SimpleNamespace(path=None, all=False))

    def test_clear_all_no_cache(self, tmp_path, capsys, monkeypatch):
        """Clear --all when no .cash dir exists."""
        monkeypatch.chdir(tmp_path)
        from types import SimpleNamespace
        cmd_clear(SimpleNamespace(path=None, all=True))
        captured = capsys.readouterr()
        assert "No .cash directory" in captured.out

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
        with patch('sys.argv', ['cash']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_version_subcommand(self, capsys):
        """Running 'version' subcommand."""
        with patch('sys.argv', ['cash', 'version']):
            main()
        captured = capsys.readouterr()
        assert "cash" in captured.out

    def test_inspect_subcommand_with_dir(self, tmp_path, capsys):
        """Running 'inspect' with a directory."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        with patch('sys.argv', ['cash', 'inspect', str(cache_dir)]):
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
            "nbformat_minor": 5
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
            "nbformat_minor": 5
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
            "nbformat_minor": 5
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

        meta = {
            'key': 'stmt:abc123def456',
            'created_at': 1700000000.0,
            'outputs': ['result', 'df']
        }
        with open(cache_dir / "entry1.meta", 'wb') as f:
            pickle.dump(meta, f)

        from cash.__main__ import _inspect_cache_dir
        _inspect_cache_dir(str(cache_dir))
        captured = capsys.readouterr()
        assert "Total files: 1" in captured.out
        assert "Cache entries: ~1" in captured.out
        assert "Recent entries:" in captured.out
        assert "result, df" in captured.out

    def test_inspect_cache_dir_empty(self, tmp_path, capsys):
        """Inspect empty cache directory."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        from cash.__main__ import _inspect_cache_dir
        _inspect_cache_dir(str(cache_dir))
        captured = capsys.readouterr()
        assert "Total files: 0" in captured.out

    def test_inspect_cache_dir_corrupt_meta(self, tmp_path, capsys):
        """Inspect cache dir with unreadable metadata files."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "bad.meta").write_bytes(b"not a pickle")

        from cash.__main__ import _inspect_cache_dir
        _inspect_cache_dir(str(cache_dir))
        captured = capsys.readouterr()
        assert "could not read metadata" in captured.out


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
