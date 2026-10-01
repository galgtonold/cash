"""The sqlite backend's database is a file INSIDE the cache directory.

Found while stress-testing the decorator: the factory passed
``db_path=config.cache_dir``, so the cache DIRECTORY was the database FILE.
A fresh project got a SQLite database literally named ``.cash``; a project that
had used the default backend first (so ``.cash/`` exists) died with
``sqlite3.OperationalError: unable to open database file``; and the CLI, which
looks for entries in a directory, reported "nothing here" either way.
``SQLiteBackend``'s own default says ``.cash/cache.db``.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

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
    # No CASH_CACHE_DIR: the project's pyproject.toml decides.
    return run_python(*(args or ("run.py",)), cwd=project, cache_dir=None, timeout=180, check=False)


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
