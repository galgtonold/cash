"""Looking at big plain data at C speed.

"Plain" means exact lists and tuples, nested, over exact primitives -- the rows
a parser returns, a matrix of floats as lists. Such a value holds no set, no
dict, no code and no object state, so the Python-level walks cash otherwise
makes over it -- to key it, to copy it into the RAM tier, to size it -- can be
answered one LEVEL at a time with ``chain.from_iterable`` and ``map`` instead
of one element at a time. Round 19 measured what the walks cost: a warm run of
a two-million-row parser and two consumers took 9.4 s where no cache took 0.7 s.

Every function here gives up (returns None or False) on anything else -- a
dict, a set, an object, a subclass, a cycle, too many levels -- and the caller
falls back to its general walk.
"""
from __future__ import annotations

import pickle
import sys
from itertools import chain
from typing import Any

#: Leaves: exact primitives. ``bytearray`` is one for keying (it pickles by
#: value) but is mutable, so `immutable_below` does not accept it.
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


def container_ids(value: Any) -> list[int] | None:
    """The ids of every list and tuple in *value* if it is plain, else None."""
    if type(value) not in SEQS:
        return None
    ids = [id(value)]
    try:
        for flat, types in _levels(value):
            if all(t in SEQS for t in types):
                ids.extend(map(id, flat))
            elif not all(t in LEAF_TYPES for t in types):
                ids.extend(id(x) for x in flat if type(x) in SEQS)
    except (_NotPlain, TypeError):      # TypeError: an unhashable type among them
        return None
    return ids


def immutable_below(value: Any) -> bool:
    """Is every container below *value* a tuple, and every leaf immutable?

    Then a new top-level list -- or the tuple itself -- is a complete deep
    copy: nothing the caller holds can reach anything the copy holds that
    could change.
    """
    if type(value) not in SEQS:
        return False
    try:
        for _flat, types in _levels(value):
            if not all(t in IMMUTABLE_LEAF_TYPES or t is tuple for t in types):
                return False
    except (_NotPlain, TypeError):
        return False
    return True


def copy_plain(value: Any) -> tuple[bool, Any]:
    """``(True, copy)`` for plain data, ``(False, None)`` for anything else.

    Tuples of immutables all the way down need a new top list at most; other
    plain data (lists inside) is copied by a pickle round trip, which is exact
    for these types and runs in C.
    """
    if immutable_below(value):
        return True, (list(value) if type(value) is list else value)
    if container_ids(value) is None:
        return False, None
    return True, pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def size_of(value: Any) -> int | None:
    """``sys.getsizeof`` summed over *value* and everything in it, or None.

    A leaf shared by many references (a small int, an interned string) is
    counted once per reference, where a recursive walk with a seen-set counts
    it once: an estimate for a memory cap, erring high.
    """
    if type(value) not in SEQS:
        return None
    total = sys.getsizeof(value)
    try:
        for flat, _types in _levels(value):
            total += sum(map(sys.getsizeof, flat))
    except (_NotPlain, TypeError):
        return None
    return total
