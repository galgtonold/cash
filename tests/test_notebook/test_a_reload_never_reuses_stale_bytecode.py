"""A reload of an edited helper runs the edited code, whatever the .pyc says.

Python decides whether a module's ``.pyc`` is current from the source's mtime
in WHOLE SECONDS and its size. An edit that keeps the size (``sum`` -> ``max``)
within the second of the first import therefore looks current, and
``importlib.reload`` runs the old bytecode. ``FunctionTracker.reload_module``
deleted the ``.pyc`` first to prevent that -- and ignored a failed delete. On
Windows a file just written is routinely held open for a moment by an
antivirus or indexer scan, so under load the delete failed, the reload ran
the old code, and the badge said MODULE RELOADED over a stale answer: the
intermittent ``test_a_helper_edit_reaches_a_cell_below::test_a_from_import``
failure.
"""

import importlib
import os
import sys
import uuid

import pytest

from cash.notebook.function_tracker import FunctionTracker


@pytest.fixture
def helper(tmp_path, monkeypatch):
    name = "stale_pyc_" + uuid.uuid4().hex[:8]
    path = tmp_path / f"{name}.py"
    path.write_text("def f(rows):\n    return sum(rows)\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name, path
    sys.modules.pop(name, None)


def _edit_within_the_same_second(path):
    first = os.stat(path).st_mtime
    path.write_text("def f(rows):\n    return max(rows)\n", encoding="utf-8")  # same size
    second = int(first) + 0.5
    os.utime(path, (second, second))
    assert int(os.stat(path).st_mtime) == int(first)


def test_a_reload_runs_the_edit_when_the_pyc_cannot_be_deleted(helper, monkeypatch):
    name, path = helper
    module = importlib.import_module(name)
    assert module.f([1, 2, 3, 4]) == 10
    tracker = FunctionTracker()
    tracker.track_module(name)
    _edit_within_the_same_second(path)

    real_remove = os.remove

    def locked(p, *a, **k):  # what an antivirus scan's open handle does
        if str(p).endswith(".pyc"):
            raise PermissionError(32, "The process cannot access the file", str(p))
        return real_remove(p, *a, **k)

    monkeypatch.setattr(os, "remove", locked)

    assert tracker.reload_module(name)
    assert sys.modules[name].f([1, 2, 3, 4]) == 4, "the reload ran the pre-edit bytecode"


def test_a_reload_runs_the_edit_when_the_pyc_is_deleted(helper):
    name, path = helper
    importlib.import_module(name)
    tracker = FunctionTracker()
    tracker.track_module(name)
    _edit_within_the_same_second(path)
    assert tracker.reload_module(name)
    assert sys.modules[name].f([1, 2, 3, 4]) == 4
