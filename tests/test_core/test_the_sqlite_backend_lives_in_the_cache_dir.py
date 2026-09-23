"""The sqlite backend's database is a file INSIDE the cache directory.

Found while attacking the decorator before round 26: the factory passed
``db_path=config.cache_dir``, so the cache DIRECTORY was the database FILE.
A fresh project got a SQLite database literally named ``.cash``; a project that
had used the default backend first (so ``.cash/`` exists) died with
``sqlite3.OperationalError: unable to open database file``; and the CLI, which
looks for entries in a directory, reported "nothing here" either way.
``SQLiteBackend``'s own default says ``.cash/cache.db``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

PROGRAM = textwrap.dedent("""
    import time
    import cash

    @cash.cache
    def slow(n):
        time.sleep(0.3)
        return n * 2

    print("RESULT", slow(3))
""")


def _project(tmp_path, pyproject):
    (tmp_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    (tmp_path / "run.py").write_text(PROGRAM, encoding="utf-8")
    return tmp_path


def _run(project, *args):
    done = subprocess.run(
        [sys.executable, *(args or ("run.py",))],
        cwd=str(project),
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return done


@pytest.mark.parametrize(
    "pyproject",
    [
        '[tool.cash]\nbackend = "sqlite"\n',
        '[tool.cash]\n[[tool.cash.tiers]]\ntype = "sqlite"\n',
    ],
)
@pytest.mark.timeout(300)
def test_the_database_goes_inside_the_cache_directory(tmp_path, pyproject):
    project = _project(tmp_path, pyproject)
    done = _run(project)
    assert "RESULT 6" in done.stdout, done.stderr[-1500:]
    assert (project / ".cash").is_dir(), "the cache directory was replaced by a database file"
    assert (project / ".cash" / "cache.db").is_file(), sorted(p.name for p in project.iterdir())


@pytest.mark.timeout(300)
def test_it_does_not_crash_when_the_cache_directory_already_exists(tmp_path):
    project = _project(tmp_path, "[tool.cash]\n")
    assert "RESULT 6" in _run(project).stdout  # default backend makes .cash/
    (project / "pyproject.toml").write_text('[tool.cash]\nbackend = "sqlite"\n', encoding="utf-8")
    done = _run(project)
    assert "RESULT 6" in done.stdout, done.stderr[-1500:]
    assert "Traceback" not in done.stderr, done.stderr[-1500:]


@pytest.mark.timeout(300)
def test_a_database_at_the_old_location_keeps_working(tmp_path):
    """Anyone who ran the sqlite backend before has a database FILE named
    `.cash`; the fix must not make `makedirs` crash on it, or lose it."""
    project = _project(tmp_path, '[tool.cash]\nbackend = "sqlite"\n')
    import sqlite3

    sqlite3.connect(str(project / ".cash")).close()  # the old layout
    done = _run(project)
    assert "RESULT 6" in done.stdout, done.stderr[-1500:]
    assert "Traceback" not in done.stderr, done.stderr[-1500:]
    assert (project / ".cash").is_file(), "the old database was replaced"
