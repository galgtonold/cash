"""The content of records holding dates, as bytes to hash.

A frame's object column of sessions, each a list of ``(action, datetime)``
records (student_8's 87,000 sessions, two million records), was keyed by
pickling it: a ``datetime`` pickles through its ``__reduce_ex__``, a
microsecond and more each, so the content hash of the frame took 3.5 s
where building it took 0.2 s. `dated_records_bytes` writes such a
column's content as tagged bytes instead, read at C speed by the
``dated_records`` kernel (`cash._native`), with this module's Python
version as the reference and the fallback: both give the same bytes.

The form, every number little-endian:

- a list: ``[`` and its length (8 bytes), then its items; a tuple: ``(``;
- a ``str``: ``s``, the length of its UTF-8 (surrogates passed through,
  8 bytes), then that UTF-8;
- an ``int`` that fits 64 bits: ``i`` and its 8 bytes, signed;
- a ``float``: ``f`` and the 8 bytes of the double;
- ``True``, ``False``, ``None``: ``T``, ``F``, ``N``;
- ``bytes``: ``b``, the length (8 bytes), then the bytes;
- a naive ``datetime``: ``D`` and the 10 bytes pickle keeps of it (year
  in two bytes, month with the fold in its top bit, day, hour, minute,
  second, microsecond in three bytes, each big-endian);
- a ``date``: ``d``, year in two bytes, month, day.

Only exact types are written: a subclass (pandas' ``Timestamp``, a clock
test double's date), an aware datetime, an int past 64 bits, any other
type, a list or tuple *levels* deep, or a column none of whose first
*probe* items holds a date, gives None, and the caller pickles the items
as before. So a column without dates keys as it did, and one with them
keys on what they are: two equal columns give the same bytes, two that
differ in any value, type or nesting give different ones.
"""

from __future__ import annotations

import datetime
import struct
from typing import Any

from cash import _native
from cash._user_attrs import attribute_or

__all__ = ["dated_records_bytes"]

#: The items looked at for a date before the rest are read.
PROBE = 64
#: How deep lists and tuples may nest below the column.
LEVELS = 8

_KERNEL = attribute_or(_native.kernels, "dated_records", None)

_INT64 = struct.Struct("<q")
_FLOAT = struct.Struct("<d")
_LENGTH = struct.Struct("<Q")


def dated_records_bytes(items: list[Any], probe: int = PROBE, levels: int = LEVELS) -> bytes | None:
    """The content of *items*, an exact list, in the module's form, or
    None when it is not written so (see the module docstring)."""
    if type(items) is not list:
        return None
    if _KERNEL is not None:
        found = _KERNEL(items, probe, levels)
        if found is not NotImplemented:
            return found
    return python_dated_records_bytes(items, probe, levels)


class _NotThisForm(Exception):
    pass


def python_dated_records_bytes(items: list[Any], probe: int = PROBE, levels: int = LEVELS) -> bytes | None:
    """`dated_records_bytes` in Python: the reference the kernel matches."""
    out: list[bytes] = [b"[", _LENGTH.pack(len(items))]
    dated = [False]
    try:
        for at, item in enumerate(items):
            if at == probe and not dated[0]:
                return None
            _write(item, 1, levels, out, dated)
    except _NotThisForm:
        return None
    return b"".join(out) if dated[0] else None


def _datetime(obj: datetime.datetime, dated: list[bool]) -> bytes:
    if obj.tzinfo is not None:
        raise _NotThisForm
    dated[0] = True
    # Protocol 4 keeps the fold: the 10 bytes the module names.
    return b"D" + obj.__reduce_ex__(4)[1][0]


def _date(obj: datetime.date, dated: list[bool]) -> bytes:
    dated[0] = True
    return b"d" + obj.__reduce_ex__(4)[1][0]


def _str(obj: str, dated: list[bool]) -> bytes:
    data = obj.encode("utf-8", "surrogatepass")
    return b"s" + _LENGTH.pack(len(data)) + data


def _int(obj: int, dated: list[bool]) -> bytes:
    if not -(1 << 63) <= obj < 1 << 63:
        raise _NotThisForm
    return b"i" + _INT64.pack(obj)


def _bytes(obj: bytes, dated: list[bool]) -> bytes:
    return b"b" + _LENGTH.pack(len(obj)) + obj


_LEAVES: dict[type, Any] = {
    str: _str,
    int: _int,
    float: lambda obj, dated: b"f" + _FLOAT.pack(obj),
    bool: lambda obj, dated: b"T" if obj else b"F",
    type(None): lambda obj, dated: b"N",
    bytes: _bytes,
    datetime.datetime: _datetime,
    datetime.date: _date,
}


def _write(obj: Any, depth: int, levels: int, out: list[bytes], dated: list[bool]) -> None:
    kind = type(obj)
    if kind is not list and kind is not tuple:
        leaf = _LEAVES.get(kind)
        if leaf is None:
            raise _NotThisForm
        out.append(leaf(obj, dated))
        return
    if depth >= levels:
        raise _NotThisForm
    out.append((b"[" if kind is list else b"(") + _LENGTH.pack(len(obj)))
    leaves = _LEAVES
    append = out.append
    for item in obj:
        leaf = leaves.get(type(item))
        if leaf is not None:
            append(leaf(item, dated))
        elif type(item) is tuple and depth + 1 < levels:
            # A record: its leaves written here, not by a call each.
            append(b"(" + _LENGTH.pack(len(item)))
            for part in item:
                leaf = leaves.get(type(part))
                if leaf is not None:
                    append(leaf(part, dated))
                else:
                    _write(part, depth + 2, levels, out, dated)
        else:
            _write(item, depth + 1, levels, out, dated)
