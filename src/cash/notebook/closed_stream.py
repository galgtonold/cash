"""A closed file a statement leaves bound, stored as what it is.

``with open(p) as fh: text = fh.read()`` leaves ``fh`` bound to a closed
file. It holds no data, but it is one of the statement's outputs, and a
file object does not pickle: the entry could not reach disk (the cell ran
again after every restart) and the failed write was reported in the cell's
output on every run. It is stored as its name and mode, and restored as a
closed file object of the same type, name, mode and encoding, which is all
a closed file has to give.
"""

from __future__ import annotations

import io
import os
from typing import Any

__all__ = ["ClosedStream", "stored_form"]

_FILE_TYPES = (io.TextIOWrapper, io.BufferedReader, io.BufferedWriter, io.BufferedRandom, io.FileIO)


class ClosedStream:
    """What a closed file is stored as. Copied or unpickled, it is the
    closed file again (:func:`_closed_file`).

    Nothing in it changes, and its copy is not it but the closed file, which
    does not pickle: the RAM tier keeps it as it is when storing
    (``_cash_stored_as_is``), so a later disk write of that entry gets the
    stored form, and a hit still copies it into the closed file."""

    _cash_stored_as_is = True

    def __init__(self, name: str | bytes, mode: str, encoding: str | None, errors: str | None) -> None:
        self.args = (name, mode, encoding, errors)

    def __reduce__(self) -> tuple[Any, tuple]:
        return _closed_file, self.args


def stored_form(value: Any) -> Any:
    """*value*, or a :class:`ClosedStream` for a closed file opened by name."""
    if type(value) not in _FILE_TYPES or not getattr(value, "closed", False):
        return value
    name = getattr(value, "name", None)
    mode = getattr(value, "mode", None)
    if not isinstance(name, (str, bytes)) or not isinstance(mode, str):
        return value
    text = isinstance(value, io.TextIOWrapper)
    return ClosedStream(
        name, mode, getattr(value, "encoding", None) if text else None, getattr(value, "errors", None) if text else None
    )


def _closed_file(name: str | bytes, mode: str, encoding: str | None, errors: str | None) -> Any:
    """A closed file object as ``open(name, mode)`` makes, without opening
    *name*: the descriptor it is built over is the null device's."""
    def null_device(_path: Any, _flags: int) -> int:
        return os.open(os.devnull, os.O_RDONLY)

    stream = open(name, mode, encoding=encoding, errors=errors, opener=null_device)  # noqa: SIM115 - closed at once
    stream.close()
    return stream
