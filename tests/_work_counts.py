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
    """Bytes fed to ``hashlib`` (sha256, blake2b, md5, sha1) inside the block:
    every key and every content digest cash takes goes through one of them."""
    count = Count()
    with pytest.MonkeyPatch.context() as mp:
        for name in ("sha256", "blake2b", "md5", "sha1"):
            real_ctor = getattr(hashlib, name)

            def make(data=b"", *args, _real=real_ctor, **kwargs):
                h = _CountingHash(_real(*args, **kwargs), count)
                if data:
                    h.update(data)
                return h

            mp.setattr(hashlib, name, make)
        yield count


@contextlib.contextmanager
def pickled_bytes() -> Iterator[Count]:
    """Bytes pickled inside the block, by ``pickle.dumps`` or a
    ``pickle.Pickler`` writing to a seekable buffer (cash's entry writer);
    arrays handed out of band count their buffers' size."""
    count = Count()
    real_dumps = pickle.dumps
    real_pickler = pickle.Pickler

    def dumps(obj, *args, **kwargs):
        out = real_dumps(obj, *args, **kwargs)
        count.add(out)
        return out

    class CountingPickler(real_pickler):
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

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pickle, "dumps", dumps)
        mp.setattr(pickle, "Pickler", CountingPickler)
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
