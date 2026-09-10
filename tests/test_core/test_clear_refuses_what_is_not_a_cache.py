"""`cash clear` removes caches, and nothing else.

CAS-107, reported BLOCKING by round-17 tester r17s2. An explicit path went
straight to ``shutil.rmtree``:

    cd myproject && cash clear .
    # deleted the project's files, then raised PermissionError trying to
    # remove the directory it was standing in  (exit 1)
    cash clear C:/some/project
    # "Cleared: …"  (exit 0) — the whole directory gone

Only ``--all`` had a "does this look like a cache?" check, and its refusal
message then recommended naming the directory explicitly — the unguarded form.

Now every removal is checked: no ``CACHE_VERSION`` and no ``.entry`` files means
it is refused unless ``--force`` says otherwise; and the current directory or
anything containing it is refused always.

Through the real CLI in a subprocess, because what is under test is what an
operator types.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.core


def _cash(*argv, cwd):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    return subprocess.run([sys.executable, "-m", "cash", *argv], cwd=str(cwd),
                          capture_output=True, text=True, env=env)


def _a_project(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "notes.txt").write_text("do not lose me", encoding="utf-8")
    (root / "app.py").write_text("print(1)\n", encoding="utf-8")
    return root


def _a_cache(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "CACHE_VERSION").write_text("1", encoding="utf-8")
    (root / "abc.entry").write_bytes(b"x")
    return root


def test_clear_dot_in_a_project_deletes_nothing(tmp_path):
    """THE BUG: `cash clear .` wiped the project it was run in."""
    project = _a_project(tmp_path / "project")

    result = _cash("clear", ".", cwd=project)

    assert result.returncode == 1, result.stdout + result.stderr
    assert (project / "notes.txt").exists(), "cash clear . deleted a project file"
    assert (project / "app.py").exists()
    assert "Traceback" not in result.stderr


def test_an_explicit_path_that_is_not_a_cache_is_refused(tmp_path):
    """The same bug from anywhere else: an absolute path to a project."""
    project = _a_project(tmp_path / "project")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    result = _cash("clear", str(project), cwd=elsewhere)

    assert result.returncode == 1, result.stdout
    assert "does not look like a cash cache" in result.stdout
    assert (project / "notes.txt").exists()


def test_the_refusal_no_longer_recommends_the_unguarded_form(tmp_path):
    """`--all`'s message used to say "or name the directory explicitly"."""
    project = _a_project(tmp_path / "project")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    result = _cash("clear", str(project), cwd=elsewhere)

    assert "name the directory explicitly" not in result.stdout
    assert "--force" in result.stdout


def test_a_real_cache_named_explicitly_is_cleared(tmp_path):
    """The control: the command still does its job."""
    cache = _a_cache(tmp_path / "somecache")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    result = _cash("clear", str(cache), cwd=elsewhere)

    assert result.returncode == 0, result.stdout + result.stderr
    assert not cache.exists()


def test_force_clears_an_unmarked_directory(tmp_path):
    """The deliberate override, for a cache that lost its stamp."""
    unmarked = tmp_path / "unmarked"
    unmarked.mkdir()
    (unmarked / "blob.bin").write_bytes(b"x")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    result = _cash("clear", str(unmarked), "--force", cwd=elsewhere)

    assert result.returncode == 0, result.stdout + result.stderr
    assert not unmarked.exists()


def test_the_current_directory_is_refused_even_with_force(tmp_path):
    """Standing inside a real cache: still no, and --force does not change it."""
    cache = _a_cache(tmp_path / "somecache")

    result = _cash("clear", ".", "--force", cwd=cache)

    assert result.returncode == 1, result.stdout
    assert "current directory" in result.stdout
    assert (cache / "CACHE_VERSION").exists()


def test_a_directory_containing_the_cwd_is_refused(tmp_path):
    """An ancestor of where you stand is never what you meant."""
    cache = _a_cache(tmp_path / "somecache")
    inside = cache / "sub"
    inside.mkdir()

    result = _cash("clear", str(cache), "--force", cwd=inside)

    assert result.returncode == 1, result.stdout
    assert (cache / "CACHE_VERSION").exists()
