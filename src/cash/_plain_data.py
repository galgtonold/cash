"""Looking at big plain data at C speed.

"Plain" means exact lists and tuples, nested, over exact primitives -- the rows
a parser returns, a matrix of floats as lists. Such a value holds no set,
no dict, no code and no object state, so the Python-level walks cash otherwise
makes over it -- to key it, to copy it into the RAM tier, to size it -- can be
answered one LEVEL at a time with ``chain.from_iterable`` and ``map`` instead
of one element at a time. Round 19 measured what the walks cost: a warm run of
a two-million-row parser and two consumers took 9.4 s where no cache took 0.7 s.

Every function here gives up (returns None or False) on anything else -- a
dict, a set, an object, a subclass, a cycle, too many levels -- and the caller
falls back to its general walk.
"""
from __future__ import annotations

import copyreg
import datetime
import decimal
import io
import operator
import pickle
import random
import sys
from itertools import chain
from typing import Any

#: Leaves: exact primitives, and the immutable value types a parser puts in a
#: row -- a ``date`` column, a ``Decimal`` amount. Without those, rows holding a
#: date left the fast path and cost 16x their body per call to key (round 20).
#: ``bytearray`` is a leaf for keying (it pickles by value) but is mutable, so
#: a value holding one is not `profile`'s immutable.
_VALUE_TYPES = (datetime.date, datetime.datetime, datetime.time, datetime.timedelta,
                decimal.Decimal)
LEAF_TYPES = (str, int, float, bool, type(None), bytes, complex, bytearray, *_VALUE_TYPES)
IMMUTABLE_LEAF_TYPES = (str, int, float, bool, type(None), bytes, complex, *_VALUE_TYPES)
SEQS = (list, tuple)
MAX_LEVELS = 16


def _fake_clock() -> tuple[tuple, dict]:
    """``(leaf types, pickler dispatch entries)`` for a loaded clock test double.

    Under freezegun, ``date(2025, 10, 1)`` written in a module is a
    ``freezegun.api.FakeDate``: equal to the real date, but pickled under its
    own class -- so every key holding one moved, and a frozen run never shared
    an entry with a real one (round 20). Reduced as the real class reduces,
    it is keyed as the value it is.
    """
    mod = sys.modules.get("freezegun.api")
    if mod is None:
        return (), {}
    known = _FAKE_CLOCK.get(id(mod))
    if known is not None and known[0] is mod:
        return known[1]
    leaves: list[type] = []
    table: dict[type, Any] = {}
    for name, base in (("FakeDate", datetime.date), ("FakeDatetime", datetime.datetime)):
        cls = getattr(mod, name, None)
        if isinstance(cls, type) and issubclass(cls, base):
            leaves.append(cls)
            table[cls] = lambda obj, base=base: (base, base.__reduce_ex__(obj, 2)[1])
    _FAKE_CLOCK.clear()
    _FAKE_CLOCK[id(mod)] = (mod, (tuple(leaves), table))
    return tuple(leaves), table


#: id(freezegun.api) -> (the module, what `_fake_clock` found in it)
_FAKE_CLOCK: dict[int, tuple[Any, tuple[tuple, dict]]] = {}


def _dump(value: Any, fast: bool) -> bytes:
    buf = io.BytesIO()
    pickler = pickle.Pickler(buf, protocol=pickle.DEFAULT_PROTOCOL)
    pickler.fast = fast
    table = _fake_clock()[1]
    if table:
        pickler.dispatch_table = {**copyreg.dispatch_table, **table}
    pickler.dump(value)
    return buf.getvalue()


def key_dumps(value: Any) -> bytes:
    """``pickle.dumps(value)`` for a cache key: a clock test double's date
    pickles as the date (`_fake_clock`)."""
    if not _fake_clock()[1]:
        return pickle.dumps(value)
    return _dump(value, fast=False)


def _levels(value: Any):
    """Yield ``(flat, types)`` for each level below *value*; stop at leaves.

    ``flat`` is every item one level down, ``types`` their exact types. Raises
    ``_NotPlain`` as soon as a level holds anything but leaves and sequences.
    """
    fakes = _fake_clock()[0]
    leaves = LEAF_TYPES + fakes if fakes else LEAF_TYPES
    level = [value]
    for _ in range(MAX_LEVELS):
        flat = list(chain.from_iterable(level))
        types = set(map(type, flat))
        yield flat, types
        if all(t in leaves for t in types):
            return
        if not all(t in leaves or t in SEQS for t in types):
            raise _NotPlain
        level = flat if all(t in SEQS for t in types) else [
            x for x in flat if type(x) in SEQS]
    raise _NotPlain                     # deeper than MAX_LEVELS, or a cycle


class _NotPlain(Exception):
    pass


def is_plain(value: Any) -> bool:
    """Is *value* a list or tuple of plain data?"""
    if type(value) not in SEQS:
        return False
    try:
        for _level in _levels(value):
            pass
    except (_NotPlain, TypeError):      # TypeError: an unhashable type among them
        return False
    return True


def dict_rows(value: Any) -> tuple[tuple, list] | None:
    """``(sorted keys, rows as tuples)`` for a list of dicts, or None.

    ``csv.DictReader`` rows and JSON records: dicts that share one set of
    string (or int) keys, with plain values. Keyed as dicts they took the
    general path -- every dict walked and rebuilt in Python to put its keys in
    order -- about 10x the plain-rows cost (round 20). Their content is the
    keys once and a tuple of values per row, which ``map(itemgetter(...))``
    builds at C speed, in key order, so two lists equal but for their dicts'
    insertion order have the same form.
    """
    if type(value) is not list or not value or set(map(type, value)) != {dict}:
        return None
    try:
        orders = set(map(tuple, value))
        if len({frozenset(o) for o in orders}) != 1:
            return None
        keys = tuple(sorted(next(iter(orders))))
    except TypeError:                   # keys that do not sort together
        return None
    if not keys or not all(type(k) in (str, int) for k in keys):
        return None
    getter = operator.itemgetter(*keys)
    rows = list(map(getter, value)) if len(keys) > 1 else [(v,) for v in map(getter, value)]
    if not is_plain(rows):
        return None
    return keys, rows


def dict_rows_profile(value: Any) -> int | None:
    """The size of a list of dicts `dict_rows` accepts whose values are all
    immutable, or None -- the case `list(map(dict, value))` copies completely."""
    found = dict_rows(value)
    if found is None:
        return None
    total = sys.getsizeof(value) + _level_size(value)
    try:
        for depth, (flat, types) in enumerate(_levels(found[1])):
            if not all(t in IMMUTABLE_LEAF_TYPES or t is tuple for t in types):
                return None
            if depth:                   # the values; level 0 is the temporary tuples
                total += _level_size(flat)
    except (_NotPlain, TypeError):
        return None
    return total


def identity_snapshot(value: Any) -> list[tuple | None] | None:
    """What a call could change in *value*, as object identities, or None.

    One tuple of items per level, for every level whose parents include a
    list -- a tuple cannot change, so the items below tuples only are not
    held. Comparing identities after a call (`identity_changed`) sees a sort,
    an append, a ``del``, a ``rows[i] = ...`` or a ``row[3] = ...`` at C speed:
    the re-hash it replaces was skipped for a big argument, so ``rows.sort()``
    on a million parsed rows was stored and the warm run skipped it (round 20).
    The tuples hold their items, so an id cannot be reused while they exist.
    None for anything that is not plain data, and for a ``bytearray`` leaf,
    which changes in place without changing its identity.
    """
    if type(value) not in SEQS:
        return None
    snapshot: list[tuple | None] = []
    parents_can_change = type(value) is list
    try:
        for flat, types in _levels(value):
            if bytearray in types:
                return None
            snapshot.append(tuple(flat) if parents_can_change else None)
            parents_can_change = list in types
    except (_NotPlain, TypeError):
        return None
    return snapshot


def identity_changed(value: Any, snapshot: list[tuple | None]) -> bool:
    """Did *value* change since `identity_snapshot` took *snapshot*?"""
    try:
        levels = list(_levels(value)) if type(value) in SEQS else None
    except (_NotPlain, TypeError):
        return True
    if levels is None or len(levels) != len(snapshot):
        return True
    for (flat, _types), before in zip(levels, snapshot):
        if before is not None and (len(flat) != len(before)
                                   or not all(map(operator.is_, flat, before))):
            return True
    return False


def pickle_unshared(value: Any) -> bytes:
    """``pickle.dumps(value)`` without the memo: 5x faster on big plain data.

    The memo -- a dict entry per tuple and string written, so a second
    reference can be written as a back-reference -- was 80 % of pickling two
    million rows (0.7 s of 0.9 s). Without it the bytes are the content alone:
    a string or a list reached twice pickles like two equal ones. Plain data
    has no cycle (`_levels` gives up on one), which is the one thing the memo
    was needed for.
    """
    return _dump(value, fast=True)


#: A level with more items than this is sized from `SIZE_SAMPLE` of them.
SIZE_EXACT_UP_TO = 1 << 16
SIZE_SAMPLE = 1 << 14


def _level_size(flat: list) -> int:
    """``sys.getsizeof`` summed over *flat*, estimated from a sample when big.

    A call per item was 0.33 s of storing two million parsed rows in the RAM
    tier. The sample is random, not every k-th item: rows flatten to a
    repeating pattern -- int, int, str -- that a fixed stride can land on one
    column of. Seeded by the length, so one value always gets one size.
    """
    n = len(flat)
    if n <= SIZE_EXACT_UP_TO:
        return sum(map(sys.getsizeof, flat))
    picks = random.Random(n).sample(range(n), SIZE_SAMPLE)
    return sum(map(sys.getsizeof, map(flat.__getitem__, picks))) * n // SIZE_SAMPLE


def profile(value: Any) -> tuple[int, bool, list[set]] | None:
    """``(size, immutable, level types)`` for plain data from one walk, else None.

    *size* is ``sys.getsizeof`` summed over *value* and everything in it, an
    estimate for a memory cap: a leaf shared by many references (a small int,
    an interned string) is counted once per reference, where a recursive walk
    with a seen-set counts it once, and a level of more than
    `SIZE_EXACT_UP_TO` items is sized from a sample (`_level_size`).

    *immutable* is whether every container below *value* is a tuple and every
    leaf immutable. Then a new top-level list -- or the tuple itself -- is a
    complete deep copy: nothing the caller holds can reach anything the copy
    holds that could change.
    """
    if type(value) not in SEQS:
        return None
    total = sys.getsizeof(value)
    immutable = True
    levels: list[set] = []
    try:
        for flat, types in _levels(value):
            total += _level_size(flat)
            levels.append(types)
            if immutable and not all(t in IMMUTABLE_LEAF_TYPES or t is tuple for t in types):
                immutable = False
    except (_NotPlain, TypeError):
        return None
    return total, immutable, levels


def size_of(value: Any) -> int | None:
    """The size half of `profile`."""
    found = profile(value)
    return None if found is None else found[0]


def immutable_below(value: Any) -> bool:
    """The immutable half of `profile`, stopping at the first level that is not."""
    if type(value) not in SEQS:
        return False
    try:
        for _flat, types in _levels(value):
            if not all(t in IMMUTABLE_LEAF_TYPES or t is tuple for t in types):
                return False
    except (_NotPlain, TypeError):
        return False
    return True


def level_types(value: Any) -> list[set] | None:
    """The exact types found at each level below *value*, or None if not plain."""
    if type(value) not in SEQS:
        return None
    try:
        return [types for _flat, types in _levels(value)]
    except (_NotPlain, TypeError):
        return None


def copy_plain(value: Any, immutable: bool | None = None,
               levels: list[set] | None = None) -> tuple[bool, Any]:
    """``(True, copy)`` for plain data, ``(False, None)`` for anything else.

    Tuples of immutables all the way down need a new top list at most. Rows
    that are LISTS of immutables -- ``csv.reader`` output with a field or two
    converted -- need a new list per row, which ``map(list, rows)`` builds at C
    speed: a pickle round trip was 4.8 s of a never-stored 2.5M-row parse
    (round 20). Deeper nests, and ``bytearray`` leaves, still take the pickle
    round trip, which is exact for these types and runs in C. *immutable* and
    *levels* are for a caller that has already profiled *value*.
    """
    if immutable is None:
        immutable = immutable_below(value)
        if not immutable and not is_plain(value):
            return False, None
    if immutable:
        return True, (list(value) if type(value) is list else value)
    if levels is None:
        levels = level_types(value)
    if (levels is not None and len(levels) == 2
            and all(t in IMMUTABLE_LEAF_TYPES for t in levels[1])
            and all(t in SEQS or t in IMMUTABLE_LEAF_TYPES for t in levels[0])):
        rows = (list(map(list, value)) if levels[0] == {list}
                else [list(x) if type(x) is list else x for x in value])
        return True, (tuple(rows) if type(value) is tuple else rows)
    return True, pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
