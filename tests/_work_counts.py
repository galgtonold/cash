"""Counters for the work-count tests (``test_*/internals/test_work_counts.py``).

A work count is a number that grows with what cash does, not with how fast
the machine is: bytes hashed, bytes pickled, deep-copy steps, calls of one
method. A test asserts it under a bound, so growth fails CI without the
noise of a timer. Each counter wraps the one public seam its work passes
through, with ``pytest.MonkeyPatch``, and leaves everything else alone.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import pickle
from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest


@dataclass
class Count:
    calls: int = 0
    bytes: int = 0
    seen: list = field(default_factory=list)

    def add(self, data=None) -> None:
        self.calls += 1
        if data is not None:
            self.bytes += memoryview(data).nbytes


class _CountingHash:
    """A hashlib object that counts what it is fed; digests are unchanged."""

    def __init__(self, real, count: Count):
        self._real = real
        self._count = count

    def update(self, data) -> None:
        self._count.add(data)
        self._real.update(data)

    def copy(self):
        return _CountingHash(self._real.copy(), self._count)

    def __getattr__(self, name):
        return getattr(self._real, name)


@contextlib.contextmanager
def hashed_bytes() -> Iterator[Count]:
    """Bytes fed to ``hashlib`` (sha256, blake2b, md5, sha1) or to the leaf
    hasher of big data (`cash.bulk_digest.leaf_hasher`: SHA-256 or BLAKE3)
    inside the block: every key and every content digest cash takes goes
    through one of them."""
    from cash import bulk_digest, row_blocks

    count = Count()

    def counting(real_ctor):
        def make(data=b"", *args, **kwargs):
            h = _CountingHash(real_ctor(*args, **kwargs), count)
            if data:
                h.update(data)
            return h

        return make

    with pytest.MonkeyPatch.context() as mp:
        for name in ("sha256", "blake2b", "md5", "sha1"):
            mp.setattr(hashlib, name, counting(getattr(hashlib, name)))
        leaf = counting(bulk_digest.leaf_hasher)
        mp.setattr(bulk_digest, "leaf_hasher", leaf)
        mp.setattr(row_blocks, "leaf_hasher", leaf)
        yield count


def _counting_pickler(base: type, count: Count) -> type:
    """*base*, a pickler class, counting what it writes into *count*."""

    class CountingPickler(base):
        def __init__(self, file, *args, buffer_callback=None, **kwargs):
            def on_buffer(buf):
                count.bytes += memoryview(buf.raw()).nbytes
                return buffer_callback(buf) if buffer_callback else True

            super().__init__(file, *args, buffer_callback=on_buffer if buffer_callback else None, **kwargs)
            self._out = file

        def dump(self, obj):
            start = self._out.tell() if hasattr(self._out, "tell") else 0
            super().dump(obj)
            count.calls += 1
            count.bytes += (self._out.tell() - start) if hasattr(self._out, "tell") else 0

    return CountingPickler


@contextlib.contextmanager
def pickled_bytes() -> Iterator[Count]:
    """Bytes pickled inside the block, by ``pickle.dumps`` or a
    ``pickle.Pickler`` writing to a seekable buffer (cash's entry writer);
    arrays handed out of band count their buffers' size. Also cash's own
    pickler subclass for a value walked before it is pickled
    (`kept_state._Overriding`, always so on Windows 3.10), which subclasses
    the ``pickle.Pickler`` of import time and so passes the patch by."""
    from cash import kept_state

    count = Count()
    real_dumps = pickle.dumps

    def dumps(obj, *args, **kwargs):
        out = real_dumps(obj, *args, **kwargs)
        count.add(out)
        return out

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pickle, "dumps", dumps)
        mp.setattr(pickle, "Pickler", _counting_pickler(pickle.Pickler, count))
        mp.setattr(kept_state, "_Overriding", _counting_pickler(kept_state._Overriding, count))
        yield count


@contextlib.contextmanager
def deepcopy_steps() -> Iterator[Count]:
    """``copy.deepcopy`` calls inside the block, its own recursion included:
    one per container or object it walks into."""
    count = Count()
    real = copy.deepcopy

    def deepcopy(obj, *args, **kwargs):
        count.add()
        return real(obj, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(copy, "deepcopy", deepcopy)
        yield count


@contextlib.contextmanager
def ram_copied_bytes() -> Iterator[Count]:
    """Blocks of pandas tables the RAM tier copies inside the block, and
    their bytes: each goes through `frame_sharing._copy_values`."""
    from cash.backends import frame_sharing

    count = Count()
    real = frame_sharing._copy_values

    def copy_values(values, *args, **kwargs):
        count.calls += 1
        count.bytes += int(getattr(values, "nbytes", 0))
        return real(values, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(frame_sharing, "_copy_values", copy_values)
        yield count


@contextlib.contextmanager
def method_calls(owner, name: str) -> Iterator[Count]:
    """Calls of ``owner.name`` (a function or method) inside the block."""
    count = Count()
    real = getattr(owner, name)

    def spy(*args, **kwargs):
        count.add()
        return real(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(owner, name, spy)
        yield count
