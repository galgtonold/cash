"""What source text a helper walk read, beyond the callables it visited.

A stored purity report (`cash.analysis.report_store`) is used again in a
later process only while what the walk worked it out from is the same. The
text of each callable the walk visited is checked on its own; what else the
walk read of a file is noted here while it reads it, so the stored report
checks that too:

* the text of a callable it did not visit (a special method of an object a
  body names, read to decide whether to walk it, or a function a rule reads
  to judge a call): `note_object`;
* the whole text of a file (``inspect.getsource`` of a module, a file read
  as bytes): `note_file`;
* the scan of which globals a module's functions change
  (`mutable_globals.module_modified_globals`), which reads every function
  in the module: `note_scan`.

Every place in the walk that reads source calls one of these before any
memo of its own answers, so a walk that finds the answer in a memo another
walk filled notes the read all the same. Outside `recording` they do
nothing.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator
from typing import Any

__all__ = ["Reads", "note_file", "note_object", "note_scan", "recording", "replay"]


class Reads:
    """The reads noted inside one `recording`."""

    __slots__ = ("files", "objects", "scans")

    def __init__(self) -> None:
        #: id -> the object whose own text was read (held: an id is unique
        #: only among live objects).
        self.objects: dict[int, Any] = {}
        self.files: dict[str, None] = {}
        #: Module names whose global-change scan was read.
        self.scans: dict[str, None] = {}

    def merge(self, other: Reads) -> None:
        self.objects.update(other.objects)
        self.files.update(other.files)
        self.scans.update(other.scans)

    def __bool__(self) -> bool:
        return bool(self.objects or self.files or self.scans)


_ACTIVE: contextvars.ContextVar[Reads | None] = contextvars.ContextVar("cash_source_reads", default=None)


@contextlib.contextmanager
def recording() -> Iterator[Reads]:
    """Note the reads made inside; a recording around this one notes them too."""
    outer = _ACTIVE.get()
    reads = Reads()
    token = _ACTIVE.set(reads)
    try:
        yield reads
    finally:
        _ACTIVE.reset(token)
        if outer is not None:
            outer.merge(reads)


def note_object(obj: Any) -> None:
    """The own source text of *obj* (a function, class, method or code
    object) is being read."""
    reads = _ACTIVE.get()
    if reads is not None:
        reads.objects[id(obj)] = obj


def note_file(path: Any) -> None:
    """The whole text of the file at *path* is being read."""
    reads = _ACTIVE.get()
    if reads is not None and isinstance(path, str):
        reads.files[path] = None


def note_scan(module_name: Any) -> None:
    """The global-change scan of the module named *module_name* is being read."""
    reads = _ACTIVE.get()
    if reads is not None and isinstance(module_name, str):
        reads.scans[module_name] = None


def replay(reads: Reads) -> None:
    """Note again *reads*, noted where an answer a memo keeps was worked out."""
    active = _ACTIVE.get()
    if active is not None:
        active.merge(reads)
