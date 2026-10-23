"""The RAM tier's copy of a result does not open the user's file to write.

The RAM tier keeps a copy of a result, made by a pickle round trip, so a
caller changing what it got cannot change the entry. A class of the user's
that pickles as ``(open, (path, "w"))`` -- a sink, a log -- was rebuilt by
that copy on every store: the file was emptied, once more than plain Python
did, and on every call, since the copy held an open file no disk tier could
write and nothing was stored. A rebuild that would open a file to write is
not run: the result is not stored, and the file keeps what it held.
"""

from __future__ import annotations

import pytest

import cash
from cash.backends.memory_backend import InMemoryBackend


class Truncating:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (open, (self.path, "w"))


class Reading:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (_reopen, (self.path,))


def _reopen(path):
    return Reading(path)


@pytest.fixture
def sink(tmp_path):
    path = tmp_path / "sink.txt"
    path.write_text("keep me", encoding="utf-8")
    return path


def test_a_cached_call_leaves_the_file_as_it_was(tmp_path, sink):
    c = cash.Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
    held = Truncating(str(sink))

    @c.cache
    def with_sink(n):
        return {"sink": held, "n": n}

    got = with_sink(1)
    assert got["sink"] is held
    assert sink.read_text(encoding="utf-8") == "keep me"


@pytest.mark.parametrize("mode", ["w", "a", "x", "r+", "wb"])
def test_no_writing_mode_is_rebuilt(sink, mode):
    class Opener:
        def __reduce__(self):
            return (open, (str(sink), mode))

    backend = InMemoryBackend()
    with pytest.raises(cash.exceptions.CacheBackendError):
        backend.set("k", {"v": Opener()}, metadata={"copy_required": True})
    assert sink.read_text(encoding="utf-8") == "keep me"


def test_a_rebuild_that_writes_nothing_is_still_copied(sink):
    backend = InMemoryBackend()
    value = {"v": Reading(str(sink))}
    backend.set("k", value, metadata={"copy_required": True})
    _meta, got = backend.get("k")
    assert type(got["v"]) is Reading and got["v"] is not value["v"]


class ReopensInSetstate:
    """A sink that reopens its file when unpickled."""

    def __init__(self, path):
        self.path = path

    def __getstate__(self):
        return {"path": self.path}

    def __setstate__(self, state):
        self.path = state["path"]
        self.handle = open(self.path, "w", encoding="utf-8")


class OpensInInit:
    def __init__(self, path):
        self.path = path
        open(path, "w", encoding="utf-8").close()

    def __reduce__(self):
        return (OpensInInit, (self.path,))


class SwallowsTheError:
    def __init__(self, path):
        self.path = path

    def __getstate__(self):
        return {"path": self.path}

    def __setstate__(self, state):
        self.path = state["path"]
        try:
            open(self.path, "w", encoding="utf-8").close()
        except Exception:
            pass


class ReadsInSetstate:
    def __init__(self, path):
        self.path = path

    def __getstate__(self):
        return {"path": self.path}

    def __setstate__(self, state):
        self.path = state["path"]
        with open(self.path, encoding="utf-8") as f:
            self.text = f.read()


def _path_open(path):
    import pathlib

    class PathOpener:
        def __reduce__(self):
            return (pathlib.Path.open, (pathlib.Path(path), "w"))

    return PathOpener()


def _os_open(path):
    import os

    class OsOpener:
        def __reduce__(self):
            return (os.open, (path, os.O_WRONLY | os.O_TRUNC))

    return OsOpener()


@pytest.mark.parametrize(
    "make",
    [
        lambda path: _new(ReopensInSetstate, path),
        lambda path: _new(OpensInInit, path),
        lambda path: _new(SwallowsTheError, path),
        _path_open,
        _os_open,
    ],
    ids=["setstate", "constructor_named_by_reduce", "error_caught_in_setstate", "pathlib_open", "os_open"],
)
def test_a_write_however_the_rebuild_asks_for_it_is_not_made(sink, make):
    """Not a list of openers: the mode Python reports for the open itself,
    from ``__setstate__``, a constructor a pickle names, or any opener."""
    value = {"v": make(str(sink))}
    sink.write_text("keep me", encoding="utf-8")
    backend = InMemoryBackend()
    with pytest.raises(cash.exceptions.CacheBackendError):
        backend.set("k", value, metadata={"copy_required": True})
    assert sink.read_text(encoding="utf-8") == "keep me"


def _new(cls, path):
    """An object of *cls* made without running its ``__init__``."""
    obj = cls.__new__(cls)
    obj.path = path
    return obj


def test_a_rebuild_that_reads_its_file_is_copied(sink):
    backend = InMemoryBackend()
    value = {"v": _new(ReadsInSetstate, str(sink))}
    backend.set("k", value, metadata={"copy_required": True})
    _meta, got = backend.get("k")
    assert got["v"].text == "keep me"


def test_a_result_whose_rebuild_writes_is_not_stored_on_disk_either(tmp_path, sink):
    """Refused in RAM, it was still written to disk, and every later hit
    unpickled it and emptied the file: ``'keep me'``, then ``''``."""

    @cash.Cash(cache_dir=str(tmp_path / "cache"), register_magic=False).cache
    def with_sink(n):
        return {"sink": _new(ReopensInSetstate, str(sink)), "n": n}

    for _ in range(3):
        sink.write_text("keep me", encoding="utf-8")
        with_sink(1)
        assert sink.read_text(encoding="utf-8") == "keep me"

    @cash.Cash(cache_dir=str(tmp_path / "cache"), register_magic=False).cache
    def with_sink(n):
        return {"sink": _new(ReopensInSetstate, str(sink)), "n": n}

    with_sink(1)
    assert sink.read_text(encoding="utf-8") == "keep me"


def test_a_disk_entry_whose_rebuild_writes_is_a_miss(tmp_path, sink):
    """Written before (by an older cash, or to disk alone as a big value):
    the hit is not served, and the file is not touched."""
    from cash.backends.file_backend import FileBackend

    backend = FileBackend(str(tmp_path / "cache"))
    backend.set("k", {"v": _new(ReopensInSetstate, str(sink))}, {})
    sink.write_text("keep me", encoding="utf-8")
    assert FileBackend(str(tmp_path / "cache")).get("k") == (None, None)
    assert sink.read_text(encoding="utf-8") == "keep me"


class _Changes:
    def __init__(self, path, change):
        self.path, self.change = path, change

    def __getstate__(self):
        return {"path": self.path, "change": self.change}

    def __setstate__(self, state):
        self.__dict__.update(state)
        _CHANGES[self.change](self.path)


def _sqlite(path):
    import sqlite3

    sqlite3.connect(path + ".db").close()


_CHANGES = {
    "truncate": lambda path: __import__("os").truncate(path, 0),
    "remove": lambda path: __import__("os").remove(path),
    "rename": lambda path: __import__("os").rename(path, path + ".old"),
    "mkdir": lambda path: __import__("os").mkdir(path + ".d"),
    "sqlite": _sqlite,
}


@pytest.mark.parametrize("change", list(_CHANGES))
def test_a_change_to_a_file_other_than_an_open_is_not_made(sink, change):
    """Writes not made through ``open``: the file was emptied, removed or
    renamed, a folder or database made, by the RAM tier's copy."""
    import os

    backend = InMemoryBackend()
    with pytest.raises(cash.exceptions.CacheBackendError):
        backend.set("k", {"v": _Changes(str(sink), change)}, metadata={"copy_required": True})
    assert sink.read_text(encoding="utf-8") == "keep me"
    assert sorted(os.listdir(sink.parent)) == ["sink.txt"]


class _ImportsLazily:
    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        import lazily_imported  # noqa: F401 - imported on first rebuild

        self.imported = True


def test_a_module_imported_in_a_rebuild_is_imported(tmp_path, monkeypatch):
    """Python's own ``.pyc`` write for it counted as a writing open: the
    import was aborted, and every store refused."""
    import sys

    (tmp_path / "lazily_imported.py").write_text("X = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "lazily_imported", raising=False)
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    backend = InMemoryBackend()
    for i in range(2):
        backend.set(f"k{i}", {"v": _ImportsLazily()}, metadata={"copy_required": True})
    assert "lazily_imported" in sys.modules
    assert sys.dont_write_bytecode is False


class _UsesScratch:
    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "model.bin"), "wb") as f:
                f.write(b"x")
            os.mkdir(os.path.join(folder, "parts"))
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as f:
            f.write("x")
        fd, path = tempfile.mkstemp()
        os.close(fd)
        os.remove(path)
        self.loaded = True


def test_a_rebuild_using_scratch_files_of_its_own_is_copied():
    """A library that loads through a temporary file or folder changes
    nothing that was there before."""
    backend = InMemoryBackend()
    backend.set("k", {"v": _UsesScratch()}, metadata={"copy_required": True})
    _meta, got = backend.get("k")
    assert got["v"].loaded
