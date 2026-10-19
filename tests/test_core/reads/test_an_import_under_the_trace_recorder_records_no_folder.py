"""The trace recorder judges a library's code by its file before it reads the
frame. Its first verdict on a file may list the file's import folder: that
is cash's own read, and a call being tracked must not record it."""

from __future__ import annotations

import sys

import pytest

from cash.tracking.file_tracker import FileAccessTracker

pytestmark = pytest.mark.skipif(sys.version_info < (3, 12), reason="sys.monitoring needs Python 3.12")


def test_importing_a_user_module_while_the_recorder_runs_records_no_folder(tmp_path, monkeypatch):
    from cash.decorator import code_trace

    if code_trace._recorder() is None:
        pytest.skip("another tool holds the monitoring tool ids")
    (tmp_path / "module_the_recorder_sees_first.py").write_text("VALUE = 7\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    tracker = FileAccessTracker()
    try:
        with tracker:
            import module_the_recorder_sees_first  # noqa: F401 - the import is the read under test

            seen = {p.replace("\\", "/") for p in tracker.get_accessed_files()}
    finally:
        sys.modules.pop("module_the_recorder_sees_first", None)
    root = str(tmp_path).replace("\\", "/")
    assert not [p for p in seen if p.startswith(root) or root.startswith(p)], seen
