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

from cash.tracking.function_tracker import FunctionTracker


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


# ---------------------------------------------------------------------------
# The import itself ran stale bytecode. A fresh process -- Restart & Run All --
# imports the helper through its `.pyc`, whose header still matches the edited
# file, and tracking took that module as the file's baseline: no edit was ever
# seen, the old code kept running, and every key built from the module
# described the new text. A cell below printed, and persisted, the pre-edit
# value (`test_a_helper_imported_in_the_cash_on_cell[True]`, when the edit
# landed in the second of the first import).
# ---------------------------------------------------------------------------


def _import_through_a_stale_pyc(name, path):
    """Import *name* the way a restarted kernel does, from a ``.pyc`` compiled
    from the pre-edit file whose header the same-size, same-mtime edit keeps."""
    import py_compile

    first = os.stat(path)
    py_compile.compile(
        str(path),
        cfile=importlib.util.cache_from_source(str(path)),
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
        doraise=True,
    )
    path.write_text("def f(rows):\n    return max(rows)\n", encoding="utf-8")  # same size
    os.utime(path, ns=(first.st_atime_ns, first.st_mtime_ns))
    sys.modules.pop(name, None)
    importlib.invalidate_caches()
    module = importlib.import_module(name)
    assert module.f([1, 2, 3, 4]) == 10, "Python compiled the edit: the stale .pyc was never used"
    return module


def test_tracking_a_module_imported_from_stale_bytecode_runs_the_file(helper):
    name, path = helper
    module = _import_through_a_stale_pyc(name, path)

    FunctionTracker().track_module(name)

    assert module.f([1, 2, 3, 4]) == 4, "the module kept running the pre-edit bytecode"


def test_a_name_imported_from_stale_bytecode_follows_the_reload(helper):
    name, path = helper
    module = _import_through_a_stale_pyc(name, path)
    user_ns = {"hm": module, "f": module.f}

    FunctionTracker().track_module(name, user_ns)

    assert user_ns["hm"].f([1, 2, 3, 4]) == 4
    assert user_ns["f"]([1, 2, 3, 4]) == 4, "`from helper import f` kept the pre-edit function"


def test_a_sub_module_imported_from_stale_bytecode_runs_its_file(helper, tmp_path):
    """The stale module is one the tracked module imports: it is reloaded, and
    so is its importer, whose `from sub import f` held the old function."""
    name, path = helper
    parent = "parent_" + name
    (tmp_path / f"{parent}.py").write_text(
        f"from {name} import f\n\ndef g(rows):\n    return f(rows)\n", encoding="utf-8"
    )
    _import_through_a_stale_pyc(name, path)
    try:
        importlib.import_module(parent)
        assert sys.modules[parent].g([1, 2, 3, 4]) == 10

        FunctionTracker().track_module(parent)

        assert sys.modules[parent].g([1, 2, 3, 4]) == 4, "the importer kept the sub-module's pre-edit function"
    finally:
        sys.modules.pop(parent, None)


def test_a_module_imported_from_current_bytecode_is_not_reloaded(helper):
    """Control: tracking reloads only what the check shows is stale."""
    name, path = helper
    import py_compile

    py_compile.compile(
        str(path),
        cfile=importlib.util.cache_from_source(str(path)),
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
        doraise=True,
    )
    module = importlib.import_module(name)
    loaded = module.f

    FunctionTracker().track_module(name, {"f": loaded})

    assert sys.modules[name].f is loaded, "a module imported from current bytecode was reloaded"
