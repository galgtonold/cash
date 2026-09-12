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

import io
import operator
import pickle
import random
import sys
from itertools import chain
from typing import Any

#: Leaves: exact primitives. ``bytearray`` is one for keying (it pickles by
#: value) but is mutable, so a value holding one is not `profile`'s immutable.
LEAF_TYPES = (str, int, float, bool, type(None), bytes, complex, bytearray)
IMMUTABLE_LEAF_TYPES = (str, int, float, bool, type(None), bytes, complex)
SEQS = (list, tuple)
MAX_LEVELS = 16


def _levels(value: Any):
    """Yield ``(flat, types)`` for each level below *value*; stop at leaves.

    ``flat`` is every item one level down, ``types`` their exact types. Raises
    ``_NotPlain`` as soon as a level holds anything but leaves and sequences.
    """
    level = [value]
    for _ in range(MAX_LEVELS):
        flat = list(chain.from_iterable(level))
        types = set(map(type, flat))
        yield flat, types
        if all(t in LEAF_TYPES for t in types):
            return
        if not all(t in LEAF_TYPES or t in SEQS for t in types):
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
    buf = io.BytesIO()
    pickler = pickle.Pickler(buf, protocol=pickle.DEFAULT_PROTOCOL)
    pickler.fast = True
    pickler.dump(value)
    return buf.getvalue()


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


def profile(value: Any) -> tuple[int, bool] | None:
    """``(size, immutable)`` for plain data from one walk, else None.

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
    try:
        for flat, types in _levels(value):
            total += _level_size(flat)
            if immutable and not all(t in IMMUTABLE_LEAF_TYPES or t is tuple for t in types):
                immutable = False
    except (_NotPlain, TypeError):
        return None
    return total, immutable


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


def copy_plain(value: Any, immutable: bool | None = None) -> tuple[bool, Any]:
    """``(True, copy)`` for plain data, ``(False, None)`` for anything else.

    Tuples of immutables all the way down need a new top list at most; other
    plain data (lists inside) is copied by a pickle round trip, which is exact
    for these types and runs in C. *immutable* is for a caller that has
    already profiled *value* and knows it is plain: it saves looking again.
    """
    if immutable is None:
        immutable = immutable_below(value)
        if not immutable and not is_plain(value):
            return False, None
    if immutable:
        return True, (list(value) if type(value) is list else value)
    return True, pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
