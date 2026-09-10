"""The per-user cache is for an installed TOOL run from nowhere in particular.

#104 sent every installed console script run from anywhere to a per-user,
per-tool cache. Round 17 found what else that rule catches, and all five
testers hit the first one:

* **cash's own CLI.** ``cash.exe`` is a console script too, so it resolved a
  per-user ``…/cash/cash`` that nothing writes to. ``cash inspect`` found no
  cache and ``cash clear --all`` exited 0 having cleared nothing, while the
  real cache kept serving. ``python -m cash`` disagreed with ``cash``.
* **pytest.** ``pytest.exe`` sent every project's test suite into one shared
  per-user ``…/cash/pytest``; ``python -m pytest`` in the same project used a
  different cache.

The rule now: installed code run inside a project anchors to that project;
the per-user location is for a launcher run from somewhere no project claims;
and ``cash`` never takes it.

In-process, with the launcher faked through ``sys.argv[0]`` and ``__main__``,
so each rule is exercised on every platform in milliseconds. The real-venv
tests in ``test_installed_console_script_cache_dir.py`` cover the installed
shapes end to end.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from cash import config

pytestmark = pytest.mark.core

_SCRIPTS = "Scripts" if os.name == "nt" else "bin"


@pytest.fixture
def launched_as(monkeypatch):
    """Pretend this process is an installed console script called *name*."""
    def _launch(name: str):
        exe = Path(sys.prefix) / _SCRIPTS / (name + (".exe" if os.name == "nt" else ""))
        monkeypatch.setattr(sys, "argv", [str(exe)])
        # A console script's __main__ is the generated launcher: no user file.
        monkeypatch.setitem(sys.modules, "__main__", types.ModuleType("__main__"))
        monkeypatch.setattr(config, "_interactive_shell_is_running", lambda: False)
        monkeypatch.delenv("CASH_CACHE_DIR", raising=False)
    return _launch


def _no_project_above(path: Path) -> bool:
    return not any(
        (d / m).exists() for d in [path, *path.parents] for m in config._PROJECT_MARKERS
    )


def _config_cache_dir() -> str:
    # user config skipped: a real XDG file on the machine running the suite
    # must not decide the answer.
    return str(config.get_config(user_config_path=None).cache_dir)


def test_cash_itself_never_takes_the_per_user_cache(launched_as, tmp_path, monkeypatch):
    """THE BUG: `cash info/inspect/clear` resolved `…/cash/cash`."""
    if not _no_project_above(tmp_path):
        pytest.skip("a project marker above the temp dir decides this case")
    monkeypatch.chdir(tmp_path)
    launched_as("cash")

    assert config._installed_entry_point_cache_dir() is None
    got = _config_cache_dir()
    assert "cash" + os.sep + "cash" not in got, got
    assert got == os.path.normpath(str(tmp_path / ".cash"))


def test_a_launcher_inside_a_project_anchors_to_the_project(launched_as, tmp_path, monkeypatch):
    """THE OTHER BUG: `pytest` sent every project's tests to one shared cache.

    Run from a SUBDIRECTORY, because that is where `pytest tests/` often is,
    and the old cwd-relative answer would have put a cache in tests/.
    """
    project = tmp_path / "proj"
    (project / "tests").mkdir(parents=True)
    (project / "pyproject.toml").write_text('[project]\nname = "p"\nversion = "0"\n',
                                            encoding="utf-8")
    monkeypatch.chdir(project / "tests")
    launched_as("pytest")

    assert config._installed_entry_point_cache_dir() is None
    assert config.project_anchor() == project
    assert _config_cache_dir() == os.path.normpath(str(project / ".cash"))


def test_cash_inside_a_project_sees_the_projects_cache(launched_as, tmp_path, monkeypatch):
    """`cash inspect` from a project subdirectory finds the project's cache."""
    project = tmp_path / "proj"
    (project / "src").mkdir(parents=True)
    (project / "pyproject.toml").write_text('[project]\nname = "p"\nversion = "0"\n',
                                            encoding="utf-8")
    monkeypatch.chdir(project / "src")
    launched_as("cash")

    assert _config_cache_dir() == os.path.normpath(str(project / ".cash"))


def test_a_tool_run_from_nowhere_still_gets_its_per_user_cache(launched_as, tmp_path, monkeypatch):
    """The control: #104's case is untouched."""
    if not _no_project_above(tmp_path):
        pytest.skip("a project marker above the temp dir decides this case")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "_per_user_cache_root", lambda: tmp_path / "peruser")
    launched_as("reportgen")

    assert _config_cache_dir() == str(tmp_path / "peruser" / "reportgen")


def test_a_repl_keeps_the_cwd(tmp_path, monkeypatch):
    """The other control: no running program at all is not installed code."""
    project = tmp_path / "proj"
    (project / "sub").mkdir(parents=True)
    (project / "pyproject.toml").write_text('[project]\nname = "p"\nversion = "0"\n',
                                            encoding="utf-8")
    monkeypatch.chdir(project / "sub")
    monkeypatch.setattr(sys, "argv", [""])
    monkeypatch.setitem(sys.modules, "__main__", types.ModuleType("__main__"))
    monkeypatch.setattr(config, "_interactive_shell_is_running", lambda: False)

    assert config.project_anchor() == project / "sub"
