"""Object arrays of dates and times, stored as columns of numbers.

A pandas column of Python ``datetime`` objects (object dtype: what
``.apply`` returns, a log read without ``parse_dates``) pickled each
object on its own: about 1.8 s a million rows, and a 15 s stall of the
notebook's thread for one user's logs on Windows. `reduce_datetime_objects`
stores such an array as the fields pickle itself keeps of each object
(a ``datetime``'s ten bytes, a ``Timestamp``'s integer), one column of
numbers per kind of object, which protocol 5 hands out of band, and
`datetime_objects` rebuilds each object from them.

Only what comes back exactly is taken this way: the array's items are
``datetime.datetime``, ``datetime.date``, ``pandas.Timestamp`` (never a
subclass), ``None`` and ``pandas.NaT``; the aware ones of a kind share
one ``tzinfo`` object, a ``datetime.timezone`` or a ``zoneinfo.ZoneInfo``,
pickled once as pickle pickles it. Anything else -- another type in the
array, two time zones, a ``pytz`` zone, a short array -- is pickled as
before. What is not kept is which items were one object: a datetime held
twice comes back as two equal ones, as it would from two pickles.
"""

from __future__ import annotations

import datetime
import operator
import sys
from itertools import repeat
from typing import Any

__all__ = ["DATETIME_OBJECTS_MIN_ITEMS", "datetime_objects", "reduce_datetime_objects"]

#: Arrays shorter than this are pickled as before: the columns cost more
#: than they save.
DATETIME_OBJECTS_MIN_ITEMS = 1024

# What each kind of item is stored as, by a fixed number: the stored
# bytes, and the digest taken over them, depend on the items alone.
_NONE, _NAT, _DATETIME, _DATE, _TIMESTAMP = range(5)

#: Items looked at before the whole array is: an array of strings or
#: numbers costs as many type checks, never a pass over its items.
_PROBES = 16

_STATE = operator.itemgetter(0)
_ARGS = operator.itemgetter(1)
_TZINFO = operator.attrgetter("tzinfo")
_REDUCE_V4 = operator.methodcaller("__reduce_ex__", 4)


def _kinds() -> dict[type, int]:
    """Each item type the columns take, and the number it is stored as."""
    kinds = {type(None): _NONE, datetime.datetime: _DATETIME, datetime.date: _DATE}
    pd = sys.modules.get("pandas")
    timestamp, nat = getattr(pd, "Timestamp", None), getattr(pd, "NaT", None)
    if isinstance(timestamp, type) and nat is not None:
        kinds[timestamp] = _TIMESTAMP
        kinds[type(nat)] = _NAT
    return kinds


def _plain_zone(tz: Any) -> bool:
    """Is *tz* a ``tzinfo`` whose own pickle the columns may keep as one
    object for every item: the standard library's, not a subclass?"""
    if type(tz) is datetime.timezone:
        return True
    zoneinfo = sys.modules.get("zoneinfo")
    return zoneinfo is not None and type(tz) is zoneinfo.ZoneInfo


def reduce_datetime_objects(array: Any) -> tuple | None:
    """*array*'s reduce as columns (`datetime_objects`), or None to pickle
    it as before: not an object array of the items the module names."""
    np = sys.modules.get("numpy")
    try:
        if (
            array.dtype.kind != "O"
            or np is None
            or type(array) is not np.ndarray
            or array.dtype != np.dtype(object)
            or array.dtype.metadata is not None
            or array.size < DATETIME_OBJECTS_MIN_ITEMS
        ):
            return None
        kinds = _kinds()
        size = array.size
        step = max(size // _PROBES, 1)
        probe = array.flat
        if any(type(probe[i]) not in kinds for i in range(0, size, step)):
            return None
        return _columns(np, array, kinds)
    except Exception:  # noqa: BLE001 - anything unforeseen: pickled as before
        return None


def _columns(np: Any, array: Any, kinds: dict[type, int]) -> tuple | None:
    fortran = array.flags.f_contiguous and not array.flags.c_contiguous
    order = "F" if fortran else "C"
    flat = array.ravel(order=order)
    items = flat.tolist()
    types = list(map(type, items))
    present = set(types)
    if not present <= kinds.keys() or present <= {type(None)}:
        return None
    codes = None
    if len(present) > 1:
        codes = np.fromiter(map(kinds.__getitem__, types), np.uint8, len(items))
    parts = []
    for kind in sorted(kinds[t] for t in present):
        if codes is None:
            group = items
        else:
            group = flat[codes == kind].tolist()
        part = _encode(np, kind, group)
        if part is None:
            return None
        parts.append((kind, part))
    return datetime_objects, (array.shape, fortran, codes, parts)


def _encode(np: Any, kind: int, group: list) -> tuple | None:
    """*group*, items all of *kind*, as the numbers that rebuild them, or
    None when they would not come back exactly."""
    if kind == _NONE:
        return ()
    if kind == _NAT:
        nat = sys.modules["pandas"].NaT
        return () if group.count(nat) == len(group) else None  # count: identity first
    if kind == _DATE:
        # A date's pickle is (date, (4 bytes,)): year (2), month, day.
        states = b"".join(map(_STATE, map(_ARGS, map(_REDUCE_V4, group))))
        if len(states) != 4 * len(group):
            return None
        return (np.frombuffer(states, np.uint8).reshape(-1, 4),)
    if kind == _DATETIME:
        tz = group[0].tzinfo
        if len(set(map(id, map(_TZINFO, group)))) != 1 or not (tz is None or _plain_zone(tz)):
            return None
        # A datetime's pickle at protocol 4 is (datetime, (10 bytes[, tz])):
        # year (2), month (fold in its top bit), day, hour, minute, second,
        # microsecond (3): every field it restores but the zone.
        states = b"".join(map(_STATE, map(_ARGS, map(_REDUCE_V4, group))))
        if len(states) != 10 * len(group):
            return None
        return np.frombuffer(states, np.uint8).reshape(-1, 10), tz
    return _encode_timestamps(np, group)


def _encode_timestamps(np: Any, group: list) -> tuple | None:
    """pandas Timestamps as pandas pickles each: ``(rebuild, (value, None,
    tz, resolution))``, the same *rebuild*, zone and resolution for all."""
    first = group[0]
    unit = getattr(first, "unit", None)
    reduced = list(map(_REDUCE_V4, group))
    rebuild, args = reduced[0][0], reduced[0][1]
    if unit is None or len(args) != 4 or args[1] is not None:
        return None
    _value, _freq, tz, resolution = args
    if not (tz is None or _plain_zone(tz)):
        return None
    if any(r[0] is not rebuild or len(r) != 2 for r in reduced):
        return None
    every = list(map(_ARGS, reduced))
    if (
        len(set(map(len, every))) != 1
        or len(set(map(id, map(operator.itemgetter(2), every)))) != 1
        or set(map(operator.itemgetter(3), every)) != {resolution}
        or set(map(operator.itemgetter(1), every)) != {None}
    ):
        return None
    if tz is not None and sys.modules["pandas"].DatetimeTZDtype(unit, tz).tz is not tz:
        return None  # pandas would give the column another zone object
    values = np.fromiter(map(_STATE, every), np.int64, len(every))
    return values, rebuild, tz, resolution, unit


def datetime_objects(shape: tuple, fortran: bool, codes: Any, parts: list) -> Any:
    """The object array `reduce_datetime_objects` stored: of *shape*,
    Fortran-ordered when *fortran*, its items rebuilt from *parts*
    (``(kind, numbers)`` per kind), placed where *codes* says (one kind:
    None)."""
    import numpy as np

    size = 1
    for n in shape:
        size *= n
    flat = np.empty(size, dtype=object)
    for kind, part in parts:
        where = slice(None) if codes is None else np.flatnonzero(codes == kind)
        if kind == _NONE:
            continue  # np.empty's object array holds None
        if kind == _NAT:
            flat[where] = _pandas().NaT
            continue
        flat[where] = _decode(np, kind, part)
    out = np.empty(shape, dtype=object, order="F" if fortran else "C")
    out[...] = flat.reshape(shape, order="F" if fortran else "C")
    return out


def _pandas() -> Any:
    import pandas

    return pandas


def _decode(np: Any, kind: int, part: tuple) -> Any:
    """The items of one kind, as an object array: a date or datetime
    built from its pickled fields as pickle builds it, ``date(state)`` or
    ``datetime(state, tz)``."""
    if kind == _DATE:
        (fields,) = part
        return _built(np, datetime.date, fields)
    if kind == _DATETIME:
        fields, tz = part
        return _built(np, datetime.datetime, fields, tz)
    return _decode_timestamps(np, *part)


def _built(np: Any, cls: type, fields: Any, *tz: Any) -> Any:
    """``cls(state, *tz)`` for each row of *fields*, a pickle state each."""
    fields = np.ascontiguousarray(fields, dtype=np.uint8)
    states = fields.view(f"V{fields.shape[1]}").ravel().tolist()
    made = map(cls, states, *(repeat(zone) for zone in tz))
    return np.fromiter(made, dtype=object, count=len(states))


def _object_array(np: Any, items: list) -> Any:
    """*items* as a 1-d object array, each item as it is."""
    out = np.empty(len(items), dtype=object)
    out[:] = items
    return out


#: Items `_decode_timestamps` checks against pandas' own rebuild.
_CHECKED = 8


def _decode_timestamps(np: Any, values: Any, rebuild: Any, tz: Any, resolution: Any, unit: str) -> Any:
    """Timestamps as pandas' own pickle rebuilds each, ``rebuild(value,
    None, tz, resolution)``: made a column at a time by pandas, checked
    against *rebuild* item by item at a few places, and rebuilt item by
    item when any differs (another pandas)."""
    pd = _pandas()
    try:
        index = pd.DatetimeIndex(values.view(f"M8[{unit}]"))
        if tz is not None:
            index = index.tz_localize("UTC").tz_convert(tz)
        items = np.asarray(index.astype(object), dtype=object)
        n = len(values)
        step = max(n // _CHECKED, 1)
        if len(items) == n and all(
            _same_timestamp(items[i], rebuild(int(values[i]), None, tz, resolution))
            for i in [*range(0, n, step), n - 1]
        ):
            return items
    except Exception:  # noqa: BLE001 - rebuilt item by item below
        pass
    return _object_array(np, list(map(rebuild, values.tolist(), repeat(None), repeat(tz), repeat(resolution))))


def _same_timestamp(a: Any, b: Any) -> bool:
    return (
        type(a) is type(b)
        and a == b
        and a.tzinfo is b.tzinfo
        and a.fold == b.fold
        and a.unit == b.unit
        and a.asm8 == b.asm8
    )
