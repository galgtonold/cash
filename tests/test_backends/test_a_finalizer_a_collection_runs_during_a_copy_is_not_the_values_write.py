"""A finalizer a garbage collection runs during a copy is not the value's write.

The RAM tier's copy of a result refuses a rebuild that would change a file
(`no_file_writes`). A collection can start at any allocation the copy makes,
and the finalizers it runs belong to garbage earlier work left: a
``jupyter_client.KernelManager`` from a finished kernel removes its
``tmpXXXX.json`` connection file in ``__del__``. That removal was refused as
the copy's own, the finalizer failed, and a result as plain as ``4`` was not
stored. Plain Python would have run the finalizer wherever the collection
fell, so it runs, and the copy is made.
"""

from __future__ import annotations

import gc

import pytest

from cash import _gc_context
from cash.backends.memory_backend import InMemoryBackend

pytestmark = pytest.mark.skipif(
    not _gc_context.INSTALLED, reason="collections get a context of their own on CPython only"
)


class _RemovesItsFile:
    """Garbage in a cycle, which only a collection frees."""

    def __init__(self, path, seen):
        self.path = path
        self.seen = seen
        self.cycle = self

    def __del__(self):
        try:
            self.path.unlink()
        except BaseException as exc:
            self.seen.append(exc)
        else:
            self.seen.append("removed")


class _CollectsWhileCopied:
    """A value whose copy allocates enough for a collection to start, made
    certain here by starting one."""

    def __reduce__(self):
        gc.collect()
        return (_CollectsWhileCopied, ())


def test_a_collection_during_a_copy_runs_the_finalizers_and_the_copy_is_made(tmp_path):
    left_over = tmp_path / "tmpconnection.json"
    left_over.write_text("{}", encoding="utf-8")
    seen: list = []
    _RemovesItsFile(left_over, seen)  # garbage now, freed by the next collection

    backend = InMemoryBackend()
    backend.set("k", {"v": _CollectsWhileCopied()}, metadata={"copy_required": True})

    assert seen == ["removed"]
    assert not left_over.exists()
    found = backend.get("k")
    assert found is not None
    assert isinstance(found[1]["v"], _CollectsWhileCopied)


def test_the_values_own_removal_is_still_refused(tmp_path):
    kept = tmp_path / "kept.json"
    kept.write_text("{}", encoding="utf-8")

    backend = InMemoryBackend()
    with pytest.raises(Exception, match="os.remove"):
        backend.set("k", {"v": _RemovesWhenCopied(str(kept))}, metadata={"copy_required": True})
    assert kept.exists()


class _RemovesWhenCopied:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (_remove_and_make, (self.path,))


def _remove_and_make(path):
    import os

    os.remove(path)
    return object()
