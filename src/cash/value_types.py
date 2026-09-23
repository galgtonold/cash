"""The sets of types cash tests values against, kept in one place.

Most are tested with ``type(v) in X``, the exact type, not ``isinstance``:
``isinstance`` is true for SUBCLASSES, so a user class deriving from str, int,
float or bytes -- and every IntEnum member, whose type is the user's own enum
class -- would be taken for a primitive, although it carries code and can
carry state. Each name below says when ``isinstance`` is meant instead.

They are TUPLES, not frozensets: ``x in frozenset`` hashes ``x``, and a class
whose metaclass defines ``__eq__`` without ``__hash__`` is itself unhashable,
so a frozenset raises TypeError on an instance of one. Tuple ``in`` compares
with ``is``/``==`` and never hashes. Measured over 200k elements it is also
faster than the isinstance form: 4.0ms vs 5.1ms (ints), 3.2ms vs 3.4ms
(strs), 3.6ms vs 4.3ms (mixed). Where ``in`` scans in order, the order is how
often each type shows up in an argument.
"""

from __future__ import annotations

import datetime
import decimal
import enum
import fractions
import functools
import pathlib
import sys
import uuid

#: Exact types that carry no user code and hold nothing else.
CODELESS_PRIMS = (str, int, float, bool, type(None), bytes, complex, bytearray)

#: `CODELESS_PRIMS` without ``bytearray``, the one that can be written into.
#: Their ``repr`` is their full value, the same in every process.
IMMUTABLE_PRIMS = (str, int, float, bool, type(None), bytes, complex)

#: Exact types of the builtin containers.
BUILTIN_CONTAINERS = (dict, list, tuple, set, frozenset)

#: What plain data nests in: exact lists and tuples (`cash._plain_data`).
PLAIN_SEQS = (list, tuple)

#: The immutable value types a parser puts in a row -- a ``date`` column, a
#: ``Decimal`` amount. Leaves of plain data beside the primitives: without
#: them, rows holding a date left the fast path and cost 16x their body per
#: call to key (round 20).
PARSED_VALUE_TYPES = (datetime.date, datetime.datetime, datetime.time, datetime.timedelta, decimal.Decimal)

#: Leaves of plain data for keying. ``bytearray`` pickles by value, so it is
#: one; it is mutable, so it is not in `IMMUTABLE_LEAF_TYPES`.
LEAF_TYPES = CODELESS_PRIMS + PARSED_VALUE_TYPES
IMMUTABLE_LEAF_TYPES = IMMUTABLE_PRIMS + PARSED_VALUE_TYPES

#: Standard-library value types that cannot change once built. Tested with
#: ``isinstance``: an ``Enum`` member's type is the user's own enum class.
IMMUTABLE_VALUE_TYPES = (
    datetime.date,
    datetime.time,
    datetime.timedelta,
    datetime.tzinfo,
    decimal.Decimal,
    fractions.Fraction,
    uuid.UUID,
    pathlib.PurePath,
    enum.Enum,
)


@functools.lru_cache(maxsize=4)
def _writable_types(numpy: object, pandas: object) -> tuple[type, ...]:
    types: list[type] = [dict, list, set, bytearray]
    for module, names in ((numpy, ("ndarray",)), (pandas, ("DataFrame", "Series"))):
        for name in names:
            attr = getattr(module, name, None)
            if isinstance(attr, type):
                types.append(attr)
    return tuple(types)


def writable_types() -> tuple[type, ...]:
    """Types a caller can write INTO (``isinstance``): the builtin mutable
    containers, and numpy's and pandas' arrays and frames once imported.

    Built once per set of those modules loaded, not on every call: the
    modules are looked up, never imported here."""
    return _writable_types(sys.modules.get("numpy"), sys.modules.get("pandas"))
