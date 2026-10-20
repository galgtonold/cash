"""Environment variables read while a cached body runs, wherever the read is.

The key folds the environment reads the analyzer finds written out in the
user's code (`EnvironmentFold`, `statement_environment_component`). A read
inside an installed package (a team package's ``os.environ["TENANT"]``)
was not seen, and the first value's result was served for every later one.
While a body runs on a miss, every read of ``os.environ`` (``[...]``,
``.get``, ``os.getenv``) made in an installed package's code is noted here.
The entry keeps a digest of each value (`environment_digests`), and a lookup
serves it only while they all match (`environment_unchanged`).

``os.environ.__getitem__`` is replaced while observation is on
(`io_watch.add_patcher`), and costs one ContextVar read per lookup when no
body is being recorded.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import sys
import threading
from collections.abc import Iterable, Iterator
from contextvars import ContextVar
from typing import Any

from . import io_watch

__all__ = ["environment_digests", "environment_unchanged", "note_reads", "recording"]

#: The names the body being recorded read, or None outside one.
_READS: ContextVar[set[str] | None] = ContextVar("cash_env_reads", default=None)
_ENVIRON = type(os.environ)
_ORIGINAL = _ENVIRON.__getitem__
_ORIGINAL_ITER = _ENVIRON.__iter__
#: Per thread: the reader frame that is copying the whole environment
#: (``{**os.environ}``, as loky hands it to a worker) and how many of its
#: lookups are left. A copy reads no one value for itself.
_copying = threading.local()
#: Frames of the lookup itself, between the reader and ``__getitem__``.
_LOOKUP_MODULES = frozenset({"os", "_collections_abc", "collections.abc"})
#: The lookups among those that read one name the caller gave.
_ONE_NAME = frozenset({"get", "getenv", "__contains__"})


#: code file -> whether a read made there counts (see `_counts`).
_COUNTED_FILES: dict[str, bool] = {}


def _counts(filename: str) -> bool:
    """A read counts when it is made in an installed package's code: the
    user's own reads are keyed from their text already, the standard
    library's (``tempfile`` reading ``TMPDIR``) are its own setup, and
    cash's are not the body's."""
    counted = _COUNTED_FILES.get(filename)
    if counted is None:
        from ..install_paths import is_cash_path, normcase_path, site_roots

        try:
            path = normcase_path(os.path.abspath(filename))
        except (TypeError, ValueError, OSError):
            path = ""
        installed = path.startswith(site_roots()) or any(seg in path for seg in ("/site-packages/", "/dist-packages/"))
        counted = bool(path) and installed and not is_cash_path(filename)
        if len(_COUNTED_FILES) > 4096:
            _COUNTED_FILES.clear()
        _COUNTED_FILES[filename] = counted
    return counted


def _reader(depth: int) -> tuple[Any, bool]:
    """The frame reading the environment, past the lookup's own; and whether
    the lookup reads one name the caller gave (not ``items()`` or ``copy()``)."""
    frame = sys._getframe(depth)
    one = True
    while frame is not None and frame.f_globals.get("__name__") in _LOOKUP_MODULES:
        one = one and frame.f_code.co_name in _ONE_NAME
        frame = frame.f_back
    return frame, one


def _reader_counts() -> bool:
    frame, one = _reader(3)
    if frame is None or not one:
        return False
    copy = getattr(_copying, "state", None)
    if copy is not None and copy[0] is frame:
        if copy[1] > 1:
            _copying.state = (frame, copy[1] - 1)
        else:
            _copying.state = None
        return False
    return _counts(frame.f_code.co_filename)


def _getitem(self, key):
    reads = _READS.get()
    if reads is not None and self is os.environ and isinstance(key, str) and _reader_counts():
        reads.add(key)
    return _ORIGINAL(self, key)


def _iter(self):
    if _READS.get() is not None and self is os.environ:
        frame, _ = _reader(2)
        if frame is not None:
            # Its lookups of every name follow: they are a copy.
            _copying.state = (frame, len(self._data))
    return _ORIGINAL_ITER(self)


def _install() -> None:
    _ENVIRON.__getitem__ = _getitem
    _ENVIRON.__iter__ = _iter


def _remove() -> None:
    _ENVIRON.__getitem__ = _ORIGINAL
    _ENVIRON.__iter__ = _ORIGINAL_ITER


io_watch.add_patcher(_install, _remove)


@contextlib.contextmanager
def recording() -> Iterator[set[str]]:
    """Note the environment variables read inside the block; what a block
    nested in it reads counts for the outer one too."""
    outer = _READS.get()
    reads: set[str] = set()
    token = _READS.set(reads)
    try:
        yield reads
    finally:
        _READS.reset(token)
        if outer is not None:
            outer |= reads


def note_reads(names: Iterable[str]) -> None:
    """Count *names* as read by the body being recorded: what a cached call
    served inside it read when it ran."""
    reads = _READS.get()
    if reads is not None:
        reads.update(names)


def _value_digest(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return "-"
    # A digest, not the value: these are often credentials.
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def environment_digests(names: set[str]) -> dict[str, str]:
    """name -> a digest of what each of *names* holds now ("-" unset)."""
    return {name: _value_digest(name) for name in sorted(names)}


def environment_unchanged(recorded: dict[str, str]) -> bool:
    """Do the variables of *recorded* (`environment_digests`) hold the same now?"""
    return all(_value_digest(name) == digest for name, digest in recorded.items())
