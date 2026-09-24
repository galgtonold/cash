"""A ``.pyc`` whose header matches the source is not proof it was compiled from it.

Python records the source's mtime in WHOLE seconds and its size in a ``.pyc``
header, and loads the bytecode when both match. Save a helper, import it, then
save a same-size edit (``sum`` -> ``max``) inside the same second: the header
still matches, so every later import -- a restarted process included -- runs
the first save's code. Cash took a matching header as proof that the loaded
code was the file on disk, and keyed the old code's results by the new text.
"""

import importlib
import os
import py_compile
import sys
import time
import uuid

import pytest

from cash import source_norm
from cash.source_norm import loaded_code_matches_disk

OLD = "def f(rows):\n    return sum(rows)\n"
NEW = "def f(rows):\n    return max(rows)\n"


@pytest.fixture
def helper(tmp_path, monkeypatch):
    name = "stale_hdr_" + uuid.uuid4().hex[:8]
    path = tmp_path / f"{name}.py"
    path.write_text(OLD, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name, path
    sys.modules.pop(name, None)


def _compile(path):
    py_compile.compile(
        str(path),
        cfile=importlib.util.cache_from_source(str(path)),
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
        doraise=True,
    )


def _import(name):
    sys.modules.pop(name, None)
    importlib.invalidate_caches()
    return importlib.import_module(name)


def loaded_module_matches_disk(module):
    # Imported here so the file collects against a tree without it.
    from cash.source_norm import loaded_module_matches_disk as check

    return check(module)


def _stale(name, path):
    """Import *name* from bytecode of OLD while the file says NEW, at an equal
    size and an equal mtime."""
    first = os.stat(path)
    _compile(path)
    path.write_text(NEW, encoding="utf-8")
    os.utime(path, ns=(first.st_atime_ns, first.st_mtime_ns))
    module = _import(name)
    assert module.f([1, 2, 3, 4]) == 10, "Python compiled the edit: the stale .pyc was never used"
    return module


def test_a_module_loaded_from_a_stale_pyc_does_not_match_its_file(helper):
    assert loaded_module_matches_disk(_stale(*helper)) is False


def test_a_module_loaded_from_a_current_pyc_matches_its_file(helper):
    """Control: the same timing, no edit."""
    name, path = helper
    _compile(path)
    assert loaded_module_matches_disk(_import(name)) is True


def test_a_module_compiled_from_its_source_matches_its_file(helper):
    """Control: no ``.pyc`` at all."""
    name, path = helper
    pyc = importlib.util.cache_from_source(str(path))
    module = _import(name)
    if os.path.exists(pyc):
        os.remove(pyc)
    assert loaded_module_matches_disk(module) is True


def test_a_function_loaded_from_a_stale_pyc_is_not_proven_by_its_header(helper, monkeypatch):
    """The decorator's check: both files predate the process, and the header
    matches -- which a same-second, same-size edit keeps."""
    monkeypatch.setattr(source_norm, "_PROCESS_START", time.time() + 60)
    module = _stale(*helper)
    assert loaded_code_matches_disk(module.f) is False


def test_a_pyc_written_after_the_source_settled_is_still_proof(helper, monkeypatch):
    """Control: the fast path stands when the ``.pyc`` postdates the save by
    more than a tick -- nothing is compiled to decide."""
    name, path = helper
    old = time.time() - 100
    os.utime(path, (old, old))
    _compile(path)
    module = _import(name)
    monkeypatch.setattr(source_norm, "_PROCESS_START", time.time() + 60)

    def no_compile(_path):
        raise AssertionError("the header was proof, and the file was compiled anyway")

    monkeypatch.setattr(source_norm, "_compiled_module", no_compile)
    assert loaded_code_matches_disk(module.f) is True
    assert loaded_module_matches_disk(module) is True
