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
