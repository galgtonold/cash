"""``file_depends_on=`` with a directory or a glob pattern.

A directory was recorded by its own timestamp only, which an edit to a file
inside does not move; a glob was recorded as a missing file of that literal
name. Either way an edit was served stale, with no warning. A directory now
covers every file under it, and a glob its matches and where they are listed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.core


def _read(path):
    """Read without a tracked reader, as a subprocess or C library would."""
    fd = os.open(path, os.O_RDONLY)
    try:
        return os.read(fd, 100).decode()
    finally:
        os.close(fd)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    later = time.time() + 5  # a new timestamp, whatever the clock's resolution
    os.utime(path, (later, later))


def _files_seen_by_a_child(directory):
    """List *directory* in another process, which cash cannot see."""
    code = "import os, sys; print(sum(len(f) for _, _, f in os.walk(sys.argv[1])))"
    return int(subprocess.check_output([sys.executable, "-c", code, str(directory)]))


@pytest.fixture
def data(tmp_path):
    d = tmp_path / "data"
    _write(d / "a.txt", "v1")
    _write(d / "sub" / "b.txt", "w1")
    return d


def test_an_edit_inside_a_declared_directory_recomputes(disk_cash, data):
    @disk_cash.cache(file_depends_on=str(data))
    def by_dir():
        return _read(data / "sub" / "b.txt")

    assert by_dir() == "w1"
    assert by_dir() == "w1" and by_dir.cache_info()["hits"] == 1
    _write(data / "sub" / "b.txt", "w2")
    assert by_dir() == "w2"


def test_a_new_file_in_a_declared_directory_recomputes(disk_cash, data):
    @disk_cash.cache(file_depends_on=str(data))
    def count():
        return _files_seen_by_a_child(data)  # @cash:assume-safe

    assert count() == 2
    (data / "sub" / "c.txt").write_text("x", encoding="utf-8")
    assert count() == 3


def test_an_edit_to_a_glob_match_recomputes(disk_cash, data):
    @disk_cash.cache(file_depends_on=str(data / "*.txt"))
    def by_glob():
        return _read(data / "a.txt")

    assert by_glob() == "v1"
    assert by_glob() == "v1" and by_glob.cache_info()["hits"] == 1
    _write(data / "a.txt", "v2")
    assert by_glob() == "v2"


def test_a_new_glob_match_recomputes(disk_cash, data):
    @disk_cash.cache(file_depends_on=str(data / "*.txt"))
    def count():
        return _files_seen_by_a_child(data)  # @cash:assume-safe

    assert count() == 2
    (data / "new.txt").write_text("x", encoding="utf-8")
    assert count() == 3
