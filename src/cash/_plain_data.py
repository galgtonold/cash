"""Looking at big plain data at C speed.

"Plain" means exact lists and tuples, nested, over exact primitives -- the rows
a parser returns, a matrix of floats as lists. Such a value holds no set,
no dict, no code and no object state, so the Python-level walks cash otherwise
makes over it -- to key it, to copy it into the RAM tier, to size it -- can be
answered one LEVEL at a time with ``chain.from_iterable`` and ``map`` instead
of one element at a time. What the walks cost, measured: a warm run of
a two-million-row parser and two consumers took 9.4 s where no cache took 0.7 s.

Every function here gives up (returns None or False) on anything else -- a
dict, a set, an object, a subclass, a cycle, too many levels -- and the caller
falls back to its general walk.
"""

from __future__ import annotations

import contextlib
import datetime
import functools
import gc
import operator
import pickle
import random
import sys
from collections import deque
from itertools import chain, compress, repeat
from typing import Any

from . import kept_state
from .value_types import IMMUTABLE_LEAF_TYPES, LEAF_TYPES, PLAIN_SEQS

MAX_LEVELS = 16


def fake_clock() -> tuple[tuple, dict]:
    """``(leaf types, pickler dispatch entries)`` for a loaded clock test double.

    Under freezegun, ``date(2025, 10, 1)`` written in a module is a
    ``freezegun.api.FakeDate``: equal to the real date, but pickled under its
    own class -- so every key holding one moved, and a frozen run never shared
    an entry with a real one. Reduced as the real class reduces,
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


#: id(freezegun.api) -> (the module, what `fake_clock` found in it)
_FAKE_CLOCK: dict[int, tuple[Any, tuple[tuple, dict]]] = {}


def _dump(value: Any, fast: bool, keyed: bool = False) -> bytes:
    return kept_state.dumps(value, fast=fast, extra=fake_clock()[1], keyed=keyed)


def key_dumps(value: Any) -> bytes:
    """``pickle.dumps(value)`` for a cache key: a clock test double's date
    pickles as the date (`fake_clock`), and a C base's reduce keeps the
    subclass's attributes (`kept_state`), and so does one a class of the
    user's leaves out of its own pickled state (`kept_state.chooses_its_state`)."""
    return _dump(value, fast=False, keyed=True)


def content_dumps(value: Any) -> bytes:
    """The bytes of *value*'s content: pickled without the memo
    (`pickle_unshared`), or with it when *value* holds a cycle.

    Through the memo, a second reference to one object is written as a
    back-reference, so a dict whose two values are one string pickled
    differently from an equal dict whose values are two: equal values keyed
    apart for how a parser happened to build them. A cycle -- possible only
    inside an object pickle stores as it asks to -- needs the memo.
    """
    try:
        return _dump(value, fast=True, keyed=True)
    except (ValueError, RecursionError):  # fast mode refuses a cycle
        return key_dumps(value)


def _levels(value: Any, extra: tuple = ()):
    """Yield ``(flat, types)`` for each level below *value*; stop at leaves.

    ``flat`` is every item one level down, ``types`` their exact types. Raises
    ``_NotPlain`` as soon as a level holds anything but leaves and sequences.
    *extra* are more leaf types (`numpy_scalar_types`).
    """
    fakes = fake_clock()[0] + extra
    leaves = LEAF_TYPES + fakes if fakes else LEAF_TYPES
    level = [value]
    for _ in range(MAX_LEVELS):
        flat = list(chain.from_iterable(level))
        types = set(map(type, flat))
        # Before the level is yielded: a caller sizes it, and a frame asked its
        # size walks every string it holds (4.3 s of one 12 s statement).
        if not all(t in leaves or t in PLAIN_SEQS for t in types):
            raise _NotPlain
        yield flat, types
        if all(t in leaves for t in types):
            return
        level = flat if all(t in PLAIN_SEQS for t in types) else [x for x in flat if type(x) in PLAIN_SEQS]
    raise _NotPlain  # deeper than MAX_LEVELS, or a cycle


class _NotPlain(Exception):
    pass


def is_plain(value: Any) -> bool:
    """Is *value* a list or tuple of plain data?"""
    if type(value) not in PLAIN_SEQS:
        return False
    try:
        for _level in _levels(value):
            pass
    except (_NotPlain, TypeError):  # TypeError: an unhashable type among them
        return False
    return True


def aliases(value: Any) -> tuple[tuple, bool] | None:
    """``(repeats, numpy)``: where *value* holds one list (or bytearray) more
    than once, and whether numpy scalars are among its leaves; or None if it
    is not plain data. A numpy number is a leaf here (`numpy_scalar_types`),
    keyed by `level_key_bytes`.

    ``[[0] * 3] * 3`` is one row three times; written into, it changes in
    three places, where three equal rows change in one. Pickled without the
    memo, the two are the same bytes, so a key must carry this too:
    ``((level, position), (level, position first met))`` per repeat, empty
    when every list appears once. See `sharing` for how they are found.
    """
    found = sharing(value)
    return None if found is None else (found[0], found[2])


def sharing(value: Any, *, tree: bool = False, held_twice_at: int | None = None) -> tuple[tuple, dict, bool] | None:
    """``(repeats, shared, numpy)`` for plain data -- or, with *tree*, for
    exact dicts, lists and tuples nested over leaves (`tree_levels`) -- or None.

    *repeats* is `aliases`' list of containers met twice. *shared* is
    ``{id: (level, position)}`` for every writable container (a list, a dict,
    a bytearray) that something besides its parent references: the only ones
    another argument can share. *numpy* says whether numpy scalars are among
    the leaves (never with *tree*).

    Found at C speed for the common case. A container held once has one
    reference from its parent and one from the level's flat list; only an
    item with more (`_unshared_refs`) can repeat, and only those are compared
    by identity. A reference held elsewhere, such as a variable naming one
    row, only makes that row a candidate. *held_twice_at* names a level
    whose items the caller's own temporaries hold once more (`dict_rows`).
    """
    kinds = TREE_NODES if tree else PLAIN_SEQS
    if type(value) not in kinds:
        return None
    repeats: list = []
    first: dict[int, tuple] = {}
    held: list = []
    numbers = () if tree else numpy_scalar_types()
    has_numbers = False
    try:
        levels = tree_levels(value) if tree else _levels(value, numbers)
        for depth, (flat, types) in enumerate(levels):
            if numbers and not has_numbers:
                has_numbers = not types.isdisjoint(numbers)
            if types.isdisjoint(_WRITABLE):
                continue
            if types <= _WRITABLE:
                items, extra = flat, 0
            else:
                items, extra = [x for x in flat if type(x) in _WRITABLE], 1
            base = _unshared_refs() + extra + (depth == held_twice_at)
            refs = list(map(sys.getrefcount, items))
            if not refs or max(refs) <= base:
                continue
            for pos in compress(range(len(refs)), map(base.__lt__, refs)):
                item = items[pos]
                seen = first.get(id(item))
                if seen is None:
                    first[id(item)] = (depth, pos)
                    held.append(item)
                else:
                    repeats.append(((depth, pos), seen))
    except (_NotPlain, TypeError):
        return None
    return tuple(repeats), first, has_numbers


def is_tree(value: Any, leaves: tuple | None = None) -> bool:
    """Is *value* JSON-like data over *leaves* (`tree_levels`)? Such a value
    holds no object, so a check for one kind of object inside it is answered
    "none" without a walk that visits the items one at a time."""
    if type(value) not in TREE_NODES:
        return False
    try:
        for _level in tree_levels(value, leaves):
            pass
    except (_NotPlain, TypeError):  # TypeError: an unhashable type among them
        return False
    return True


#: What a tree nests in (`tree_levels`): exact dicts, lists and tuples.
TREE_NODES = (dict, list, tuple)
_NODES = frozenset(TREE_NODES)
_COPY_LEAVES = frozenset(IMMUTABLE_LEAF_TYPES)


def tree_levels(value: Any, leaves: tuple | None = None):
    """`_levels` for JSON-like data: exact dicts, lists and tuples, nested,
    over the leaves of plain data; a dict's keys must be leaves too.

    ``flat`` holds a level's items: the items of its lists and tuples, then
    the values of its dicts. Pickled, such a value is its content and
    nothing else -- a dict keeps its order, a list and a tuple differ -- so
    records parsed from JSON are keyed by one pickle at C speed. Walked one
    container at a time, 20k records cost 16x ``json.dumps`` per hit.

    *leaves*, when given, replaces the leaf types.
    """
    if leaves is None:
        fakes = fake_clock()[0]
        leaves = LEAF_TYPES + fakes if fakes else LEAF_TYPES
    level = [value]
    for _ in range(MAX_LEVELS):
        kinds = set(map(type, level))
        if dict in kinds:
            if len(kinds) == 1:
                dicts, seqs = level, []
            else:
                dicts = [c for c in level if type(c) is dict]
                seqs = [c for c in level if type(c) is not dict]
            if not all(t in leaves for t in set(map(type, chain.from_iterable(dicts)))):
                raise _NotPlain
            flat = list(chain.from_iterable(seqs))
            flat.extend(chain.from_iterable(map(dict.values, dicts)))
        else:
            flat = list(chain.from_iterable(level))
        types = set(map(type, flat))
        if not all(t in leaves or t in TREE_NODES for t in types):
            raise _NotPlain
        yield flat, types
        if all(t in leaves for t in types):
            return
        level = flat if all(t in TREE_NODES for t in types) else [x for x in flat if type(x) in TREE_NODES]
    raise _NotPlain  # deeper than MAX_LEVELS, or a cycle


def numpy_scalar_types() -> tuple:
    """The exact numpy number types (``np.float64``, ``np.int32``, ...),
    or ``()`` while numpy is not loaded. Not ``longdouble``: its bytes carry
    padding that is not the value.

    ``list(arr)`` and iterating an array give numpy scalars, not Python
    numbers. Not leaves, such a list left the fast path: every scalar was
    walked for sets and pickled one by one, 7 us each, and a hit on 200k of
    them cost 1.4 s where the body took 7 ms.
    """
    return _numpy_scalars()[1]


def numpy_scalar_set() -> frozenset:
    """`numpy_scalar_types` as a set, for a test per value: asked of every
    value a key walks, so answered from one comparison when numpy is as
    it was."""
    known = _NUMPY_SCALARS
    if sys.modules.get("numpy") is known[0]:
        return known[2]
    return _numpy_scalars()[2]


def _numpy_scalars() -> tuple[Any, tuple, frozenset]:
    global _NUMPY_SCALARS
    numpy = sys.modules.get("numpy")
    if numpy is not _NUMPY_SCALARS[0]:
        found = () if numpy is None else tuple(dict.fromkeys(numpy.dtype(c).type for c in "?bhilqpBHILQPefdFD"))
        _NUMPY_SCALARS = (numpy, found, frozenset(found))
    return _NUMPY_SCALARS


#: ``(the numpy module or None, its number types, the same as a set)``
_NUMPY_SCALARS: tuple[Any, tuple, frozenset] = (None, (), frozenset())


def level_key_bytes(value: Any) -> bytes:
    """The key bytes of plain data holding numpy scalars, level by level.

    For each level: the item types (once, when they are all one); the length of each
    list or tuple; each numpy type's values as one array (exact, and C
    speed where pickling a scalar is not); and the other leaves pickled.
    With the top type, that is all the value is, so nothing else keys alike.
    """
    import numpy

    numbers = numpy_scalar_types()
    parts = [type(value).__name__.encode()]
    for flat, types in _levels(value, numbers):
        if len(types) == 1:
            parts.append(pickle.dumps(next(iter(types)), protocol=4))
        else:  # mixed, or none: the level below ``[np.float64(1), []]`` is empty
            parts.append(pickle.dumps(list(map(type, flat)), protocol=4))
        seqs = flat if types <= _SEQS else [x for x in flat if type(x) in PLAIN_SEQS]
        if seqs:
            parts.append(pickle.dumps(list(map(len, seqs)), protocol=4))
        for kind in sorted(types.intersection(numbers), key=lambda t: t.__name__):
            same = flat if len(types) == 1 else [x for x in flat if type(x) is kind]
            parts.append(kind.__name__.encode() + numpy.array(same, dtype=kind).tobytes())
        rest = types.difference(numbers).difference(_SEQS)
        if rest:
            parts.append(pickle_unshared(flat if len(types) == len(rest) else [x for x in flat if type(x) in rest]))
    return b"".join(len(part).to_bytes(8, "little") + part for part in parts)


_SEQS = frozenset(PLAIN_SEQS)


_WRITABLE = frozenset({list, bytearray, dict})


def _shared_probe() -> int:
    """`sys.getrefcount` of a list its parent and one flat list hold, read
    the way `aliases` reads it."""
    parent = [[]]
    flat = list(chain.from_iterable([parent]))
    return max(map(sys.getrefcount, flat))


def _unshared_refs() -> int:
    """What `aliases` reads for an item held once: measured, because what
    ``sys.getrefcount`` counts besides the holders varies across Python
    versions."""
    global _UNSHARED_REFS
    if _UNSHARED_REFS is None:
        _UNSHARED_REFS = _shared_probe()
    return _UNSHARED_REFS


_UNSHARED_REFS: int | None = None


def dict_rows(value: Any) -> tuple[tuple, list] | None:
    """``(keys, rows as tuples)`` for a list of dicts, or None.

    ``csv.DictReader`` rows and JSON records: dicts that share one sequence
    of string (or int) keys, in one order, with plain values. Keyed as dicts
    they took the general path -- every dict walked and rebuilt in Python --
    about 10x the plain-rows cost. Their content is the keys once and a
    tuple of values per row, which ``map(itemgetter(...))`` builds at C
    speed. The keys stay in the rows' own order, which code reads (a
    header, a frame's columns); rows whose orders differ take the general
    path, which keeps each dict's order.
    """
    found = dict_rows_unchecked(value)
    if found is None or not is_plain(found[1]):
        return None
    return found


def dict_rows_unchecked(value: Any) -> tuple[tuple, list] | None:
    """`dict_rows` before its rows are checked to be plain data."""
    if type(value) is not list or not value or set(map(type, value)) != {dict}:
        return None
    try:
        orders = set(map(tuple, value))
    except TypeError:
        return None
    if len(orders) != 1:
        return None
    if max(map(sys.getrefcount, value)) > _unshared_refs() - 1 and len(set(map(id, value))) != len(value):
        return None  # one dict more than once: the general path keys that
    keys = next(iter(orders))
    if not keys or not all(type(k) in (str, int) for k in keys):
        return None
    getter = operator.itemgetter(*keys)
    rows = list(map(getter, value)) if len(keys) > 1 else [(v,) for v in map(getter, value)]
    return keys, rows


def shared_rows(value: list) -> dict[int, tuple]:
    """``{id: (-1, position)}`` for the dicts of a `dict_rows` value that
    something besides the list references: the ones another argument can
    share."""
    refs = list(map(sys.getrefcount, value))
    base = _unshared_refs() - 1
    return {id(value[pos]): (-1, pos) for pos in compress(range(len(refs)), map(base.__lt__, refs))}


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
            if depth:  # the values; level 0 is the temporary tuples
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
    on a million parsed rows was stored and the warm run skipped it.
    The tuples hold their items, so an id cannot be reused while they exist.
    None for anything that is not plain data, and for a ``bytearray`` leaf,
    which changes in place without changing its identity.
    """
    if type(value) not in PLAIN_SEQS:
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
        levels = list(_levels(value)) if type(value) in PLAIN_SEQS else None
    except (_NotPlain, TypeError):
        return True
    if levels is None or len(levels) != len(snapshot):
        return True
    for (flat, _types), before in zip(levels, snapshot):
        if before is not None and (len(flat) != len(before) or not all(map(operator.is_, flat, before))):
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
    return sum(map(sys.getsizeof, map(flat.__getitem__, _picks(n)))) * n // SIZE_SAMPLE


@functools.lru_cache(maxsize=16)
def _picks(n: int) -> tuple[int, ...]:
    """The positions `_level_size` samples a level of *n* items at.

    Kept: drawing them was 2.5 ms of the 3.5 ms a 200,000-item level took,
    and storing one value sizes its levels more than once. With replacement:
    `choices` draws a float per pick, where `sample` draws bits until one
    falls in range, 2-3x slower (10 ms at 200,000).
    """
    return tuple(random.Random(n).choices(range(n), k=SIZE_SAMPLE))


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
    if type(value) not in PLAIN_SEQS:
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
    if type(value) not in PLAIN_SEQS:
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
    if type(value) not in PLAIN_SEQS:
        return None
    try:
        return [types for _flat, types in _levels(value)]
    except (_NotPlain, TypeError):
        return None


def copy_plain(value: Any, immutable: bool | None = None, levels: list[set] | None = None) -> tuple[bool, Any]:
    """``(True, copy)`` for plain data, ``(False, None)`` for anything else.

    Tuples of immutables all the way down need a new top list at most. Rows
    that are LISTS of immutables -- ``csv.reader`` output with a field or two
    converted -- need a new list per row, which ``map(list, rows)`` builds at C
    speed: a pickle round trip was 4.8 s of a never-stored 2.5M-row parse.
    Deeper nests are copied a level at a time (`spine_copy`); ``bytearray``
    leaves and a list held twice take the pickle round trip, which is exact
    for these types and runs in C. *immutable* and
    *levels* are for a caller that has already profiled *value*.
    """
    if immutable is None:
        immutable = immutable_below(value)
    if immutable:
        return True, (list(value) if type(value) is list else value)
    if levels is None:
        levels = level_types(value)
        if levels is None:  # not plain
            return False, None
    if (
        levels is not None
        and len(levels) == 2
        and all(t in IMMUTABLE_LEAF_TYPES for t in levels[1])
        and all(t in PLAIN_SEQS or t in IMMUTABLE_LEAF_TYPES for t in levels[0])
        and not _row_held_twice(value)
    ):
        rows = list(map(list, value)) if levels[0] == {list} else [list(x) if type(x) is list else x for x in value]
        return True, (tuple(rows) if type(value) is tuple else rows)
    spine = spine_copy(value)
    if spine is not None:
        return True, spine
    return True, pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def _row_held_twice(value: list | tuple) -> bool:
    """Is one list a row of *value* twice (``[row, row]``)? A list per row
    would split it in two; `spine_copy` sees it and leaves it to pickle,
    which keeps it one list."""
    if all(type(x) is list for x in value):
        rows, held_once = value, _unshared_refs() - 1
    else:
        rows, held_once = [x for x in value if type(x) is list], _unshared_refs()
    # A row held twice has more references than one held once: only then
    # are the ids compared, as `dict_rows_unchecked` does.
    return bool(rows) and max(map(sys.getrefcount, rows)) > held_once and len(set(map(id, rows))) != len(rows)


@contextlib.contextmanager
def _gc_paused():
    """Hold the cyclic collector off while a big copy allocates containers.

    Every few thousand new lists it would walk every object the process
    holds, the two million tuples of the value being copied among them:
    half of `spine_copy`'s time on a parsed log. Left as it was found, so a
    program that had switched it off keeps it off.
    """
    was_on = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_on:
            gc.enable()


def _of_kind(items: list, kind: type) -> list[bool]:
    """Which of *items* are exactly of *kind*, at C speed."""
    return list(map(operator.is_, map(type, items), repeat(kind)))


def _put_at(target: list, mask: list[bool], values) -> None:
    """``target[i] = next(values)`` for each *i* where *mask* is true, at C
    speed: a level mixing kinds -- 300,000 record dicts beside the RNG
    state's tuple in a notebook entry -- was a Python step per item."""
    deque(map(target.__setitem__, compress(range(len(target)), mask), values), maxlen=0)


def spine_copy(value: Any, memo: dict[int, Any] | None = None, walk: list | None = None) -> Any:
    """A copy of nested plain data, or of JSON-like data (exact dicts, lists
    and tuples over immutable leaves), that rebuilds only what can be written
    into.

    The lists and dicts are copied, and so is every tuple with one somewhere
    below it. A tuple of immutables all the way down is shared: nothing can
    reach into it. Built a level at a time, so the work is one step per
    CONTAINER, never one per leaf -- a parsed log of 87,000 sessions holding
    two million ``(name, datetime)`` pairs is 175,000 containers. Both
    ``deepcopy`` and a pickle round trip pay per leaf, and a ``datetime`` is
    slow to pickle: 14 to 25 s for that log, on every store and every RAM hit.
    A dict is rebuilt from its keys and its copied values at C speed: a
    300,000-key index went through ``deepcopy`` on every hit, 3x what
    building it cost.

    Returns None for anything it does not answer exactly, and the caller
    copies another way: a leaf that can be written into (``bytearray``), a
    dict key that is not an immutable leaf, a list or dict reachable twice
    (the copy would split it in two, and ``deepcopy`` and pickle keep it as
    one), or anything else.

    *memo*, when given, is the table ``copy.deepcopy`` keeps, ``id(original)
    -> copy``: this adds every container it made, so a value held by two
    names in one stored entry still comes back as one object; and gives up
    when a list or dict in *value* is already there -- copied for another
    name, which must keep sharing it.

    *walk* is `tree_walk` of *value*, when the caller has it.
    """
    if type(value) not in TREE_NODES:
        return None
    with _gc_paused():
        return _spine_copy(value, memo, walk)


#: `copy_plan` follows a dict of at most this many keys, this near the top,
#: into its values: a notebook entry's payload, its ``variables``.
_PLAN_KEYS = 32
_PLAN_DEPTH = 3
#: How `copy_by_plan` copies a part: shared as it is (an immutable leaf, a
#: tuple of immutables), a new list of the same items, a new dict per row,
#: or `spine_copy`.
_SHARE, _NEW_LIST, _ROWS, _SPINE = "share", "list", "rows", "spine"


def copy_plan(value: Any, _depth: int = 0) -> Any:
    """How to copy *value* -- made by `spine_copy`, so nothing in it is
    reached twice, and private, so never written -- a part at a time, each
    the fastest way it allows: `copy_by_plan`.

    `spine_copy` asks of every hit whether each part is still JSON-like and
    reached once, a pass over every leaf: for a notebook entry holding
    300,000 record dicts that walk was 3x copying the rows. The answer for a
    stored value cannot change, so it is found here, once.
    """
    kind = type(value)
    if kind is dict and len(value) <= _PLAN_KEYS and _depth < _PLAN_DEPTH:
        return {key: copy_plan(item, _depth + 1) for key, item in value.items()}
    if kind in _COPY_LEAVES:
        return _SHARE
    if kind is tuple or kind is list:
        if immutable_below(value):
            return _SHARE if kind is tuple else _NEW_LIST
        if (
            kind is list
            and value
            and all(map(operator.is_, map(type, value), repeat(dict)))
            and set(map(type, chain.from_iterable(map(dict.values, value)))) <= _COPY_LEAVES
        ):
            return _ROWS
    return _SPINE


def copy_by_plan(value: Any, plan: Any) -> Any:
    """A copy of *value* made as *plan* (`copy_plan`) says, or None if a
    part could not be copied that way."""
    if type(plan) is dict:
        copied = {}
        for key, part in plan.items():
            item = copy_by_plan(value[key], part)
            if item is None and value[key] is not None:
                return None
            copied[key] = item
        return copied
    if plan is _SHARE:
        return value
    if plan is _NEW_LIST:
        return list(value)
    if plan is _ROWS:
        return list(map(dict, value))
    return spine_copy(value)


def _tree_walk(value: Any, leaves: frozenset):
    """Yield ``(containers, flat, types, key types)`` for each level of
    JSON-like *value* (exact dicts, lists and tuples over *leaves*): the
    containers at the level, what they hold in their order (a dict's
    values), the exact types of that, and of the level's dict keys. Stops
    after a level of leaves; raises `_NotPlain` on anything else, a dict key
    that is not a leaf, or more than `MAX_LEVELS` levels (or a cycle).

    `tree_levels` without the order: its flat puts the lists' items before
    the dicts' values, where a copy needs each container's items together.
    """
    level = [value]
    for _ in range(MAX_LEVELS):
        kinds = set(map(type, level))
        key_types: set = set()
        if dict in kinds:
            if len(kinds) == 1:
                dicts, held = level, map(dict.values, level)
            else:
                is_dict = _of_kind(level, dict)
                dicts = list(compress(level, is_dict))
                held = list(level)
                _put_at(held, is_dict, map(dict.values, dicts))
            key_types = set(map(type, chain.from_iterable(dicts)))
            if not key_types <= leaves:
                raise _NotPlain
            flat = list(chain.from_iterable(held))
        else:
            flat = list(chain.from_iterable(level))
        types = set(map(type, flat))
        if not types <= leaves | _NODES:
            raise _NotPlain
        yield level, flat, types, key_types
        if types.isdisjoint(_NODES):
            return
        level = flat if types <= _NODES else list(compress(flat, map(_NODES.__contains__, map(type, flat))))
    raise _NotPlain


def tree_walk(value: Any) -> list | None:
    """The levels of JSON-like *value* over any leaf (`_tree_walk`), or None
    for anything else: walked once for both `tree_size` and `spine_copy`
    when a value is sized and then copied, each walk a pass over every leaf."""
    if type(value) not in TREE_NODES:
        return None
    try:
        with _gc_paused():  # a dict's values view per dict, see `_gc_paused`
            return list(_tree_walk(value, frozenset(LEAF_TYPES + fake_clock()[0])))
    except (_NotPlain, TypeError):
        return None


def tree_size(value: Any, walk: list | None = None) -> int | None:
    """`profile`'s size for JSON-like data (`_tree_walk`): every container,
    key and leaf ``sys.getsizeof``-summed a level at a time; None for
    anything else. *walk* is `tree_walk` of *value*, when the caller has it."""
    levels = tree_walk(value) if walk is None else walk
    if levels is None:
        return None
    total = sys.getsizeof(value)
    try:
        for level, flat, _types, _key_types in levels:
            dicts = list(compress(level, _of_kind(level, dict))) if dict in set(map(type, level)) else ()
            if dicts:
                total += _level_size(list(chain.from_iterable(dicts)))
            total += _level_size(flat)
    except (_NotPlain, TypeError):
        return None
    return total


def _copy_levels(walk: list):
    """*walk*'s levels while they hold only what a copy may share (`_COPY_LEAVES`),
    as `_tree_walk` over those leaves would yield them."""
    for level, flat, types, key_types in walk:
        if not (types <= _COPY_LEAVES | _NODES and key_types <= _COPY_LEAVES):
            raise _NotPlain
        yield level, flat, types, key_types


def _spine_copy(value: list | tuple | dict, memo: dict[int, Any] | None, walk: list | None = None) -> Any:
    # `nodes[k]`: the containers at level k; `flats[k]`: what they hold.
    nodes: list[list] = []
    flats: list[list] = []
    frozen: list[bool] = []
    try:
        levels = _tree_walk(value, _COPY_LEAVES) if walk is None else _copy_levels(walk)
        for level, flat, types, _key_types in levels:
            nodes.append(level)
            flats.append(flat)
            # Tuples over tuples and immutables, and so the levels below
            # (checked on the way back): shared as they are.
            frozen.append(types <= _COPY_LEAVES | {tuple} and set(map(type, level)) == {tuple})
    except (_NotPlain, TypeError):
        return None
    # A list or dict held twice -- here, or by a name copied before (*memo*).
    # Only one something else references can be (`sharing`): a container its
    # parent holds once is referenced by the parent and the level's list.
    held_elsewhere = [value]
    for k in range(1, len(nodes)):
        base = _unshared_refs() + (nodes[k] is not flats[k - 1])
        refs = list(map(sys.getrefcount, nodes[k]))
        if refs and max(refs) > base:
            held_elsewhere.extend(compress(nodes[k], map(base.__lt__, refs)))
    shared = [id(c) for c in held_elsewhere if type(c) is not tuple]
    if len(set(shared)) != len(shared) or (memo and not memo.keys().isdisjoint(shared)):
        return None
    for k in range(len(nodes) - 2, -1, -1):
        frozen[k] = frozen[k] and frozen[k + 1]
    new_flat = flats[-1]
    for k in range(len(nodes) - 1, -1, -1):
        if frozen[k]:
            rebuilt = nodes[k]
        elif k + 1 == len(nodes) or frozen[k + 1]:
            # What the level holds is shared: each container is copied on its
            # own, at C speed when the level is all lists or all dicts.
            kinds = set(map(type, nodes[k]))
            if len(kinds) == 1 and tuple not in kinds:
                rebuilt = list(map(next(iter(kinds)), nodes[k]))
                if memo is not None:
                    memo.update(zip(map(id, nodes[k]), rebuilt))
            else:
                # Tuples shared, each other kind copied by its own type.
                rebuilt = list(nodes[k])
                for kind in kinds - {tuple}:
                    mask = _of_kind(nodes[k], kind)
                    made = list(map(kind, compress(nodes[k], mask)))
                    _put_at(rebuilt, mask, made)
                    if memo is not None:
                        memo.update(zip(map(id, compress(nodes[k], mask)), made))
        else:
            rebuilt = []
            at = 0
            for node in nodes[k]:
                end = at + len(node)
                kind = type(node)
                if kind is list:
                    copied = new_flat[at:end]
                elif kind is dict:
                    copied = dict(zip(node, new_flat[at:end]))
                else:
                    part = new_flat[at:end]
                    copied = node if all(map(operator.is_, part, node)) else tuple(part)
                rebuilt.append(copied)
                if memo is not None and copied is not node:
                    memo[id(node)] = copied
                at = end
        if k == 0:
            return rebuilt[0]
        if frozen[k]:
            # Level k - 1 copies its containers on their own and never reads
            # what they hold: rebuilding its flat would be a Python step per
            # item -- 200,000 for a list of ints next to the RNG state's
            # tuple, the most expensive part of storing it.
            continue
        flat = flats[k - 1]
        if len(flat) == len(rebuilt):
            new_flat = rebuilt
        else:
            new_flat = list(flat)
            _put_at(new_flat, list(map(_NODES.__contains__, map(type, flat))), rebuilt)
    return None  # unreachable: k reaches 0 above
