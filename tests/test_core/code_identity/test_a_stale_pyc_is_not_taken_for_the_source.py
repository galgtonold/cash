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

from cash import loaded_code, process_start
from cash.loaded_code import loaded_code_matches_disk

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
    from cash.loaded_code import loaded_module_matches_disk as check

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
    monkeypatch.setattr(process_start, "_PROCESS_START", time.time() + 60)
    module = _stale(*helper)
    assert loaded_code_matches_disk(module.f) is False


def _after_tick(folder, ns):
    """Wait until a fresh write's timestamp is past *ns*: a coarse clock puts
    writes made close together in one tick, which no comparison can order."""
    probe = folder / "tick.probe"
    for _ in range(500):
        probe.write_bytes(b"x")
        stamp = os.stat(probe).st_mtime_ns
        os.remove(probe)
        if stamp > ns:
            return
        time.sleep(0.002)  # the clock's tick, polled


def _settled_then_restored(name, path):
    """Compile OLD long after its save, then save NEW at the same size with
    the old mtime put back (``cp -p``, ``tar``, ``os.utime``): the header
    matches and the ``.pyc`` postdates the recorded mtime by far."""
    old = time.time() - 100
    os.utime(path, (old, old))
    _after_tick(path.parent, os.stat(path).st_ctime_ns)
    _compile(path)
    first = os.stat(path)
    path.write_text(NEW, encoding="utf-8")
    os.utime(path, ns=(first.st_atime_ns, first.st_mtime_ns))
    module = _import(name)
    assert module.f([1, 2, 3, 4]) == 10, "Python compiled the edit: the stale .pyc was never used"
    return module


def test_a_pyc_compiled_after_the_source_settled_is_no_proof_once_the_mtime_is_put_back(helper, monkeypatch):
    monkeypatch.setattr(process_start, "_PROCESS_START", time.time() + 60)
    module = _settled_then_restored(*helper)
    assert loaded_code_matches_disk(module.f) is False
    assert loaded_module_matches_disk(module) is False


@pytest.mark.skipif(
    os.name == "nt", reason="no change time on Windows: the file is compiled instead (tested below with it off)"
)
def test_the_proof_by_header_compiles_nothing_for_a_file_saved_before_its_pyc(helper, monkeypatch):
    """The cold path's cost: a helper checked out, then imported (its ``.pyc``
    written a tick or more after the save), is proven by one stat and one
    16-byte read, not by compiling its module."""
    name, path = helper
    old = time.time() - 100
    os.utime(path, (old, old))
    _after_tick(path.parent, os.stat(path).st_ctime_ns)
    _compile(path)
    module = _import(name)
    monkeypatch.setattr(process_start, "_PROCESS_START", time.time() + 60)

    def no_compile(_path):
        raise AssertionError("the header was proof, and the file was compiled anyway")

    monkeypatch.setattr(loaded_code, "_compiled_module", no_compile)
    monkeypatch.setattr(loaded_code, "_read_text", no_compile)
    started = time.perf_counter()
    for _ in range(200):
        assert loaded_code_matches_disk(module.f) is True
    assert time.perf_counter() - started < 1.0, "200 header checks should take milliseconds"
    assert loaded_module_matches_disk(module) is True


def test_without_a_change_time_a_put_back_mtime_is_caught_by_the_code(helper, monkeypatch):
    """Windows: ``st_ctime`` is the creation time, so no stat tells a
    put-back save from none. The header proves nothing there; the loaded code
    is compared with the file compiled."""
    monkeypatch.setattr(loaded_code, "_HAS_CHANGE_TIME", False)
    monkeypatch.setattr(process_start, "_PROCESS_START", time.time() + 60)
    module = _settled_then_restored(*helper)
    assert loaded_code_matches_disk(module.f) is False
    assert loaded_module_matches_disk(module) is False


def test_without_a_change_time_an_unedited_file_still_matches(helper, monkeypatch):
    """Control: the compiled comparison passes a file nobody touched."""
    monkeypatch.setattr(loaded_code, "_HAS_CHANGE_TIME", False)
    name, path = helper
    old = time.time() - 100
    os.utime(path, (old, old))
    _compile(path)
    module = _import(name)
    monkeypatch.setattr(process_start, "_PROCESS_START", time.time() + 60)
    assert loaded_code_matches_disk(module.f) is True
    assert loaded_module_matches_disk(module) is True


def _fresh_process_memos(monkeypatch):
    """What a new process starts with: nothing compiled, no record read."""
    from cash._memo import LruMemo

    monkeypatch.setattr(loaded_code, "_MODULE_CODE_CACHE", LruMemo(8))
    monkeypatch.setattr(loaded_code, "_RECORDS", LruMemo(8))
    monkeypatch.setattr(loaded_code, "_CODE_INDEX", LruMemo(8))


def _counting_compiles(monkeypatch):
    compiled = []
    real = loaded_code._compiled_text

    def counting(path, text):
        compiled.append(path)
        return real(path, text)

    monkeypatch.setattr(loaded_code, "_compiled_text", counting)
    return compiled


def test_without_a_change_time_a_later_process_compares_by_the_texts_digest(helper, tmp_path, monkeypatch):
    """Windows' cost: compiling every helper file in every process. What the
    text compiles to is kept on disk under the text's digest, so a later
    process reads and digests the file and compiles nothing -- and a
    same-size save with the mtime put back, a different text, finds no
    record and is compiled and caught."""
    monkeypatch.setenv("CASH_ANALYSIS_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(loaded_code, "_HAS_CHANGE_TIME", False)
    name, path = helper
    old = time.time() - 100
    os.utime(path, (old, old))
    _compile(path)
    module = _import(name)
    monkeypatch.setattr(process_start, "_PROCESS_START", time.time() + 60)
    compiled = _counting_compiles(monkeypatch)
    _fresh_process_memos(monkeypatch)
    assert loaded_code_matches_disk(module.f) is True
    assert loaded_module_matches_disk(module) is True
    assert compiled, "the first process compiles the text to record it"

    compiled.clear()
    _fresh_process_memos(monkeypatch)
    assert loaded_code_matches_disk(module.f) is True
    assert loaded_module_matches_disk(module) is True
    assert compiled == [], "the recorded text was compiled again"

    first = os.stat(path)
    path.write_text(NEW, encoding="utf-8")
    os.utime(path, ns=(first.st_atime_ns, first.st_mtime_ns))
    _fresh_process_memos(monkeypatch)
    assert loaded_code_matches_disk(module.f) is False
    assert loaded_module_matches_disk(module) is False
    assert compiled, "the edited text found the old text's record"
