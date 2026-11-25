"""Importing a module inside a cached call records nothing the import system
looked for.

The first import of a module writes its compiled file: ``__pycache__`` and
the ``.pyc`` in it are not there before and are there after. Python's import
system stats both. Were those recorded as "not there" for the call that
imported the module, the entry would be stale in the next process, and a
statement that imports one of the user's modules would never be served after
a restart.
"""

import importlib
import os
import sys

from cash.tracking.file_tracker import FileAccessTracker


def _forget(name):
    sys.modules.pop(name, None)
    importlib.invalidate_caches()


def test_a_first_import_records_no_missing_compiled_file(tmp_path, monkeypatch):
    (tmp_path / "first_import_helper.py").write_text("K = 3\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    _forget("first_import_helper")
    tracker = FileAccessTracker()
    try:
        with tracker:
            module = importlib.import_module("first_import_helper")
    finally:
        _forget("first_import_helper")
    assert module.K == 3
    assert (tmp_path / "__pycache__").is_dir(), "the import was to write its compiled file"
    assert tracker.get_absent_files() == set()


def test_a_missing_file_the_program_asks_about_is_still_recorded(tmp_path):
    missing = tmp_path / "not_there.txt"
    tracker = FileAccessTracker()
    with tracker:
        try:
            os.stat(missing)
        except FileNotFoundError:
            pass
    assert {os.path.basename(p) for p in tracker.get_absent_files()} == {"not_there.txt"}
