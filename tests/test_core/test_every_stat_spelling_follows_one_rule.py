"""Every spelling of a metadata question about a path records the same answer.

A path that is not there is recorded as absent whoever asked; a file that is
there becomes a dependency only when the user's own code asked. A library
stats and probes paths for itself all the time, and none of those may
become a dependency of the cached call around it, whichever call it used.
"""

from __future__ import annotations

import pytest

from cash.tracking.file_tracker import FileAccessTracker

_SPELLINGS_SOURCE = """
import os, pathlib
SPELLINGS = {
    "os.stat": lambda p: os.stat(p).st_size,
    "os.lstat": lambda p: os.lstat(p).st_size,
    "os.path.getsize": lambda p: os.path.getsize(p),
    "os.path.getmtime": lambda p: os.path.getmtime(p),
    "Path.stat": lambda p: pathlib.Path(p).stat().st_size,
}
"""


def _spellings(filename):
    namespace = {}
    exec(compile(_SPELLINGS_SOURCE, filename, "exec"), namespace)
    return namespace["SPELLINGS"]


# The same calls made by the user's code (this file) and by code that is not
# the user's: a pseudo-filename is no user path.
ASKERS = {"user": _spellings(__file__), "library": _spellings("<a library>")}
SPELLINGS = list(ASKERS["user"])


def _ask(asker, spelling, path):
    with FileAccessTracker() as tracker:
        try:
            ASKERS[asker][spelling](str(path))
        except FileNotFoundError:
            pass
    return tracker


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("asker", ["user", "library"])
def test_a_missing_path_is_absent_whoever_asked(tmp_path, spelling, asker):
    missing = tmp_path / "missing.txt"
    tracker = _ask(asker, spelling, missing)
    assert any(p.endswith("missing.txt") for p in tracker.get_absent_files())


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_a_file_the_user_asked_about_is_a_dependency(tmp_path, spelling):
    there = tmp_path / "there.txt"
    there.write_text("abc", encoding="utf-8")
    tracker = _ask("user", spelling, there)
    assert any(p.endswith("there.txt") for p in tracker.get_accessed_files())


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_a_file_a_library_asked_about_is_not(tmp_path, spelling):
    there = tmp_path / "there.txt"
    there.write_text("abc", encoding="utf-8")
    tracker = _ask("library", spelling, there)
    assert not any(p.endswith("there.txt") for p in tracker.get_accessed_files())
    assert not any(p.endswith("there.txt") for p in tracker.get_present_files())


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_a_file_a_notebook_statement_asked_about_is_a_dependency(tmp_path, spelling):
    from cash.notebook.compiled_source import register_cell_source

    there = tmp_path / "there.txt"
    there.write_text("abc", encoding="utf-8")
    ask = _spellings(register_cell_source(_SPELLINGS_SOURCE))[spelling]
    with FileAccessTracker() as tracker:
        ask(str(there))
    assert any(p.endswith("there.txt") for p in tracker.get_accessed_files())
