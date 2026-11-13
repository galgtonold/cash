"""Pickling that keeps the attributes a C base class's reduce drops.

``pickle`` stores an object as its ``__reduce_ex__`` describes it. A class
written in C that defines its own reduce -- ``numpy.ndarray``,
``datetime.date`` -- describes its own data and nothing else, so a Python
subclass's instance attributes are lost: numpy's own ``InfoArray`` example
pickles without its ``info``. Two such arrays with equal data and a different
``info`` keyed alike, and a stored one came back without it.

`dumps` pickles such an instance with its ``__dict__`` added to the state it
restores (`restore_with_dict`), and everything else exactly as ``pickle``
does. It is used wherever cash pickles a value it keys or stores.
"""

from __future__ import annotations

import array
import contextlib
import contextvars
import copy
import copyreg
import datetime
import functools
import hashlib
import io
import itertools
import logging
import operator
import pickle
import sys
import threading
import types
import uuid
import weakref
from collections.abc import Callable, Iterator
from typing import Any

from . import _native, _weak
from ._own_work import own_work
from ._user_attrs import instance_of
from .live_state import is_live_stream, live_reduce
from .value_types import WEAK_PROXY_TYPES, is_dead_proxy

__all__ = [
    "Ahead",
    "DefinedAgain",
    "FlatValue",
    "by_name",
    "chooses_its_state",
    "defined_again",
    "held_now",
    "dumps",
    "flatten",
    "left_out_attrs",
    "load_flat",
    "named_now",
    "one_keying",
    "reducers",
    "restore_with_dict",
    "restores_alike_apart",
    "same_definition",
    "set_walk_order",
    "sorted_set_ids",
    "stopped_at_defined_again",
    "store_memo",
    "too_deep",
]


def dumps(value: Any, *args: Any, **kwargs: Any) -> bytes:
    """`_dumps_value` inside `one_keying`; see there. A value holding a
    function whose cell ran again since is pickled again with it by its
    name, when that name holds the same definition now (`defined_again`):
    not with a *sink*, whose bytes are written already, nor when buffers
    handed out could not be taken back (*buffer_callback* without
    *on_retry*)."""
    try:
        return _dumps_once(value, *args, **kwargs)
    except pickle.PicklingError as exc:
        again = (
            stopped_at_defined_again(exc)
            and kwargs.get("sink") is None
            and (kwargs.get("buffer_callback") is None or kwargs.get("on_retry") is not None)
        )
        if not again:
            raise
    if kwargs.get("on_retry") is not None:
        kwargs["on_retry"]()
    kwargs["reducer_override"] = defined_again(kwargs.get("reducer_override"))
    return _dumps_once(value, *args, **kwargs)


def _dumps_once(*args: Any, **kwargs: Any) -> bytes:
    if _keying() is not None:
        return _dumps_value(*args, **kwargs)
    with one_keying():
        return _dumps_value(*args, **kwargs)


def stopped_at_defined_again(exc: pickle.PicklingError) -> bool:
    """Did pickling stop at a function or class its module names another
    of now (`defined_again`)? CPython's wording for it (3.10 to 3.14); were
    it to change, such a value is not stored, as before."""
    return "not the same object as" in str(exc)


def _dumps_value(
    value: Any,
    protocol: int = pickle.DEFAULT_PROTOCOL,
    *,
    fast: bool = False,
    buffer_callback: Any = None,
    extra: dict | None = None,
    persistent_id: Any = None,
    keyed: bool = False,
    on_retry: Any = None,
    aliases: bool | dict = False,
    sink: Any = None,
    reducer_override: Any = None,
    sorted_sets: bool = False,
    after: Any = None,
) -> bytes:
    """``pickle.dumps(value, protocol)``, keeping the instance attributes a C
    base's reduce drops. *fast* is the pickler's memo-less mode; *extra*
    are reducers by type that come first (a clock test double's);
    *persistent_id* is the pickler's hook of that name. *keyed*
    bytes are for hashing, never loaded: an instance of the user's own class
    also keeps the attributes its own reduce leaves out (`_reduce_for_key`).
    *aliases* (key bytes only) marks a changeable object met again instead
    of pickle's memo back-referencing whatever object it meets again
    (`key_aliases`), so equal values built apart key alike.

    A value too deep for the pickler's recursion is pickled flat
    (`FlatValue`), which ``pickle.loads`` reads back as the value; when it
    is, *on_retry* is called first, as *buffer_callback* is called again
    for the buffers of the second attempt. *keyed* bytes are not: a value
    too deep to key raises RecursionError, as before.

    With *sink*, an object with a ``write`` method, the bytes are written
    to it a frame at a time as the pickler makes them, the same bytes in
    the same order, and the result is empty: a digest is taken as they
    come, and whoever reads them can stop the pickle by raising from
    ``write`` (`value_hash.budgeted_hash`). Only for *keyed* bytes, which
    are never pickled flat. *reducer_override* is the pickler's hook of
    that name; a value pickled flat is pickled without it.

    *sorted_sets* (key bytes only) writes each set and frozenset of two or
    more items with its items sorted (`sorted_set_ids`), as *aliases* bytes
    always do. *after* is called with the pickler once it has pickled
    *value*; not for a value walked first or pickled flat."""
    if sink is not None and not keyed:
        raise ValueError("dumps: a sink takes keyed bytes only")
    if keyed and sorted_sets and not aliases:
        persistent_id = _sets_first(persistent_id, sorted_set_ids(protocol, extra))
    if FLAT_FIRST and not _surely_shallow(value):
        return _dumps_walked_first(
            value, protocol, fast, buffer_callback, extra, persistent_id, keyed, aliases, sink, reducer_override
        )
    try:
        return _dumps(
            value,
            protocol,
            fast,
            buffer_callback,
            extra,
            persistent_id,
            keyed,
            aliases,
            sink,
            None,
            reducer_override,
            after,
        )
    except (RecursionError, MemoryError) as exc:
        if keyed or not too_deep(exc):
            # A key is never taken over a value pickle cannot follow: the
            # caller says the value is too deep to key (`canonical_form`).
            raise
    # Deeper than the pickler may recurse (a linked list of a few thousand
    # nodes): pickled flat instead, as `FlatValue`. Buffers handed out by the
    # attempt that failed are taken back first.
    if on_retry is not None:
        on_retry()
    flat = flatten(value, _KeepDict(protocol, extra, keyed), _held_leaf(_persistent_leaf(persistent_id), aliases))
    return _dumps(flat, protocol, fast, buffer_callback, extra, persistent_id, keyed, aliases)


def dumps_also_keyed(*args: Any, **kwargs: Any) -> tuple[bytes, bool]:
    """`_dumps_also_keyed` inside `one_keying`; see there."""
    with one_keying():
        return _dumps_also_keyed(*args, **kwargs)


def _dumps_also_keyed(
    value: Any,
    protocol: int = pickle.DEFAULT_PROTOCOL,
    *,
    buffer_callback: Any = None,
    extra: dict | None = None,
    on_retry: Any = None,
    sink: Any = None,
    reducer_override: Any = None,
    after: Any = None,
    watch: list | None = None,
) -> tuple[bytes, bool]:
    """``dumps(value, protocol, ...)``, the bytes a store keeps, and whether
    they are also the *keyed* bytes of the same call: True when no object
    in *value* is one a key pickles otherwise (`_KeepDict` with *keyed*) --
    a record array with padding, a user's object whose reduce leaves an
    attribute out, an object holding views of itself.

    Then one pickle serves both the store and a digest of the value's
    content (`stored_pickle.object_digest`), where two were made before.
    False also for a value pickled flat (too deep for the pickler), which
    keyed bytes refuse.

    With *sink*, the bytes are written to it as they are made (as `dumps`
    writes them) and the stream returned is empty; a value too deep for
    the pickler then raises RecursionError, as keyed bytes do: its digest
    is taken over its keyed bytes, which refuse it too.
    *reducer_override* is the pickler's hook of that name. *after* is
    called with the pickler once it has pickled *value*, its memo still
    whole; not for a value pickled flat. *watch*, a list, is appended to
    as each such object is met (a *sink* may stop the pickle there)."""
    differs: list[bool] = [] if watch is None else watch
    table = _KeepDict(protocol, extra, watch=differs)
    if FLAT_FIRST and not _surely_shallow(value):
        walk = _Walk(getattr(sink, "walked", None), reducer_override, differs)
        if not _at_most_deep(value, _SAFE_DEPTH, table, _flat_leaf(None), walk):
            if sink is not None:
                raise RecursionError("pickled flat first on this Python: digested over its keyed bytes")
            stream = dumps(value, protocol, buffer_callback=buffer_callback, extra=extra, on_retry=on_retry)
            return stream, False
        if walk.reduced:
            reducer_override = walk.override()
    try:
        stream = _dumps(
            value,
            protocol,
            False,
            buffer_callback,
            extra,
            None,
            False,
            sink=sink,
            table=table,
            override=reducer_override,
            after=after,
        )
        return stream, not differs
    except (RecursionError, MemoryError) as exc:
        if sink is not None or not too_deep(exc):
            raise
    if on_retry is not None:
        on_retry()
    flat = flatten(value, _KeepDict(protocol, extra), None)
    return _dumps(flat, protocol, False, buffer_callback, extra, None, False), False


def _dumps_walked_first(
    value: Any,
    protocol: int,
    fast: bool,
    buffer_callback: Any,
    extra: dict | None,
    persistent_id: Any,
    keyed: bool,
    aliases: bool | dict = False,
    sink: Any = None,
    reducer_override: Any = None,
) -> bytes:
    """`dumps` where the pickler would run out of C stack before its
    recursion limit (`FLAT_FIRST`): the value is walked on the heap first,
    and handed to the pickler only when it is shallow enough
    (`_SAFE_DEPTH`). Deeper, it is pickled flat; for a key, as deep as
    pickle follows elsewhere, and past that it raises RecursionError as the
    pickler raises on other Pythons."""
    # The reducers both the walk and the pickler take: one without the memo
    # writes no state setter, as `_dumps` picks for its own (`_reduce_with_dict`).
    unshared = fast and persistent_id is None and not aliases
    reducers = _KeepDict(protocol, extra, keyed, unshared=unshared)
    leaf = _held_leaf(_flat_leaf(persistent_id), aliases)
    walk = _Walk(getattr(sink, "walked", None), reducer_override)
    # A container too may be taken whole: one the copy has made already
    # (*persistent_id*), or one the rest of a key holds (*aliases*).
    held = leaf if persistent_id is not None or type(aliases) is dict else None
    if _at_most_deep(value, _SAFE_DEPTH, reducers, leaf, walk, held):
        override = walk.override() if walk.reduced else reducer_override
        return _dumps(
            value, protocol, fast, buffer_callback, extra, persistent_id, keyed, aliases, sink, reducers, override
        )
    deepest = _pickler_room() if keyed else None
    flat = flatten(value, reducers, leaf, deepest)
    return _dumps(flat, protocol, fast, buffer_callback, extra, persistent_id, keyed, aliases, sink)


def _pickler_room() -> int:
    """How many objects deep the pickler would follow a value from here on
    other Pythons: it counts each against the recursion limit, on top of
    the frames already running."""
    depth = 0
    frame = sys._getframe(1)
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return max(sys.getrecursionlimit() - depth, 0)


def _at_most_deep(
    value: Any, limit: int, reducers: Any, leaf: Any, walk: _Walk | None = None, held: Any = None
) -> bool:
    """Is *value* at most *limit* objects deep, as the pickler follows it?

    `flatten` made a step for every object to learn the same, and on
    `FLAT_FIRST` Pythons every value not `_surely_shallow` was flattened
    before it was pickled: seconds for each pickle of a list of 200,000
    rows, each key of it included, which ran a test keying such a list past
    its timeout on Windows 3.10. Here the value is taken a level at a time,
    the builtin containers of a level opened at C speed, and nothing is
    built. The levels count the longest chain of objects each holding the
    next, which the pickler, following each object once, never goes deeper
    than; so this may say deep where `flatten` would not, never the other
    way. A cycle, or anything else met again level after level, is left to
    `flatten`, which walks each object once (and refuses an object rebuilt
    from arguments that hold it, which the pickler would recurse through
    until the stack ran out).

    With *walk*, each object's reduce is kept for the pickle that follows
    (`_Walk.override`), so it runs once, and *walk*'s check runs after each.

    *held*, when given, is asked of each builtin container before it is
    opened, as the pickler asks its persistent id first: one it takes is
    pickled whole, and nothing below it is followed. Unasked, the walk of a
    stored entry naming a list the RAM copy had made already opened every
    record and datetime of it, and asked of each whether it was one: 24,000
    calls for a log of 3,000 sessions the pickle itself never looked into.
    """
    level = _not_atoms([value])
    seen: set[int] = set()
    alive: list[list] = []  # the levels, so no id in *seen* is reused by a reduce's parts
    rewalked = 0
    for _ in range(limit):
        if not level:
            return True
        before = len(seen)
        seen.update(map(id, level))
        added = len(seen) - before
        if added < len(level):
            # Each object once in a level: a class every instance's reduce
            # names is one object, not one per instance.
            level = list(dict(zip(map(id, level), level)).values())
            rewalked += len(level) - added  # met at a level above
            if rewalked > _REWALKED_FREE:
                # Much of it met again: a cycle, or objects shared all the
                # way down. `flatten` walks each object once.
                return False
        alive.append(level)
        level = _next_level(level, reducers, leaf, walk, held)
    return not level


#: How many objects `_at_most_deep` walks again before it leaves the value
#: to `flatten`: a few shared at every level (a class, a dtype) are not a
#: cycle.
_REWALKED_FREE = 1024


def _next_level(level: list, reducers: Any, leaf: Any, walk: _Walk | None = None, held: Any = None) -> list:
    """What the objects of *level* hold that is not an atom; not what a
    builtin container *held* takes holds (`_at_most_deep`)."""
    kinds = list(map(type, level))
    opened = list(map(_OPENED_BY_ITEMS.__contains__, kinds))
    found: list[Any] = []
    containers = list(itertools.compress(level, opened))
    if containers and held is not None:
        containers = list(itertools.filterfalse(held, containers))
    if containers:
        found += _not_atoms(list(itertools.chain.from_iterable(containers)))  # a dict's keys too
        dicts = list(itertools.compress(containers, map(_DICT.__contains__, map(type, containers))))
        found += _not_atoms(list(itertools.chain.from_iterable(map(dict.values, dicts))))
    if len(containers) < len(level):
        for obj in itertools.compress(level, map(operator.not_, opened)):
            parts = _reduced_parts(obj, reducers, leaf, walk)
            if parts:
                found += _not_atoms(parts)
    return found


#: Opened by `_next_level` without a reduce: what iterating them gives is
#: what the pickler saves in them (a dict's values follow its keys).
_OPENED_BY_ITEMS: frozenset[type] = frozenset({list, tuple, set, frozenset, dict})
_DICT: frozenset[type] = frozenset({dict})


def _not_atoms(items: list) -> list:
    """The items of *items* that are not `_FLAT_ATOMS`, picked out without
    a Python step per item."""
    return list(itertools.compress(items, map(operator.not_, map(_FLAT_ATOMS.__contains__, map(type, items)))))


class _Walk:
    """What a walk before a pickle (`_at_most_deep`) learned of each object
    it reduced, for the pickle to take instead of reducing it again
    (`override`): the user's ``__reduce_ex__`` runs once per pickle, as on
    other Pythons. *check* runs after each reduce with the bytes of data
    reduced so far (a digest's deadline, `stored_pickle._Sink.walked`):
    the time a pickle takes is mostly its reduces, which here run first.
    *watch* (`dumps_also_keyed`'s) is told of an array a key packs
    (`flat_array`), which the walk does not reduce, so *check* can stop the
    pickle there before it reduces what follows."""

    __slots__ = ("check", "data", "first", "reduced", "watch")

    def __init__(self, check: Callable[[int], Any] | None = None, first: Any = None, watch: list | None = None) -> None:
        self.check = check
        self.data = 0
        #: The pickle's own ``reducer_override``, asked before the table as
        #: the pickler asks it.
        self.first = first
        self.watch = watch
        #: id -> (object, reduce with its item iterators as lists)
        self.reduced: dict[int, tuple[Any, tuple]] = {}

    def reduce(self, obj: Any, reducers: Any) -> Any:
        """*obj* reduced as the pickle will reduce it, kept: its iterators
        made lists."""
        reduced = NotImplemented if self.first is None else self.first(obj)
        if reduced is NotImplemented:
            reduced = reducers[type(obj)](obj)
        if not isinstance(reduced, str):
            reduced = tuple(list(part) if i in (3, 4) and part is not None else part for i, part in enumerate(reduced))
        self.reduced[id(obj)] = (obj, reduced)
        if self.check is not None:
            self.data += _data_bytes(reduced)
            self.check(self.data)
        return reduced

    def flat_array(self, obj: Any) -> None:
        """Met *obj*, an array pickled whole: *watch* is told when a key
        packs its records (`packed_records`), as the store's reduce of it
        will tell it, and *check* runs."""
        try:
            dtype = obj.dtype
            if (dtype.fields is not None or dtype.subdtype is not None) and _packed_dtype(
                dtype.type, dtype
            ) is not None:
                self.watch.append(True)
        except Exception:  # noqa: BLE001 - left to the store's reduce, as before
            return
        if self.check is not None:
            self.check(self.data)

    def override(self) -> Callable[[Any], Any]:
        """A ``reducer_override`` giving each object its kept reduce, with
        fresh iterators (the pickler may save an object twice when it keeps
        no memo); the pickle's own (`first`), or the table, for the rest."""
        reduced = self.reduced
        then = self.first

        def override(obj: Any) -> Any:
            found = reduced.get(id(obj))
            if found is None or found[0] is not obj:
                return NotImplemented if then is None else then(obj)
            parts = found[1]
            if isinstance(parts, str):
                return parts
            return tuple(iter(part) if i in (3, 4) and part is not None else part for i, part in enumerate(parts))

        return override


_DATA = (bytes, bytearray, str)


def _data_bytes(reduced: Any) -> int:
    """About the bytes of data a reduce hands the pickler at its first
    level: its strings and bytes, in its arguments, its state and its
    items. An estimate of how far a pickle is along, never more."""
    if isinstance(reduced, str):
        return 0
    found = 0
    for i, part in enumerate(reduced[1:5], 1):
        if part is None:
            continue
        if isinstance(part, _DATA):
            found += len(part)
            continue
        items = part.values() if isinstance(part, dict) else part if isinstance(part, (tuple, list)) else ()
        if i == 4:  # dict items: (key, value) pairs
            items = [v for pair in items for v in pair]
        found += sum(len(item) for item in items if isinstance(item, _DATA))
    return found


def _reduced_parts(obj: Any, reducers: Any, leaf: Any, walk: _Walk | None = None) -> list:
    """What pickling *obj*, not a builtin container, saves inside it, as
    `flatten` follows it: nothing for an object pickled whole. *walk* keeps
    the reduce (`_Walk`)."""
    obj_type = type(obj)
    if walk is not None and walk.watch is not None and _flat_array(obj):
        walk.flat_array(obj)  # a library's type, a leaf: told here
        return []
    if leaf(obj) or obj_type in _WHOLE or issubclass(obj_type, type) or _flat_array(obj):
        return []
    reduced = reducers[obj_type](obj) if walk is None else walk.reduce(obj, reducers)
    if isinstance(reduced, str):
        return []  # a global, pickled by name
    _rebuild, args, *rest = reduced
    rest += [None] * (4 - len(rest))
    state, listitems, dictitems = rest[:3]
    parts = [args]
    if state is not None:
        parts.append(state)
    if listitems is not None:
        parts += listitems
    if dictitems is not None:
        parts += itertools.chain.from_iterable(dictitems)
    return parts


#: How many objects deep the pickler is let follow a value on `FLAT_FIRST`
#: Pythons. Each costs it two or three levels of C recursion.
_SAFE_DEPTH = 100


#: Python 3.10 on Windows runs out of C stack before the recursion limit
#: stops the pickler or ``deepcopy``: its threads have a 2 MB stack, and
#: pickling a chain a few hundred objects long took the process down rather
#: than raise. There the user's objects and the containers holding them are
#: always pickled flat, and only a library's objects, whose pickling is a
#: level or two deep, by the pickler itself. Later Pythons, and 3.10 on the
#: 8 MB and larger stacks of Linux and macOS, raise RecursionError first.
FLAT_FIRST = sys.version_info < (3, 11) and sys.platform == "win32"


def _persistent_leaf(persistent_id: Any) -> Any:
    """What `flatten` pickles whole: what *persistent_id* takes."""
    return None if persistent_id is None else (lambda obj: persistent_id(obj) is not None)


def _held_leaf(leaf: Any, aliases: bool | dict) -> Any:
    """*leaf*, and a container the rest of a key holds (`key_aliases`'s
    *outside*): kept as itself in the flat steps, the key pickle marks it
    as held there, where a step of its own would key it like a copy."""
    if type(aliases) is not dict:
        return leaf
    if leaf is None:
        return lambda obj: id(obj) in aliases
    return lambda obj: id(obj) in aliases or leaf(obj)


def _flat_leaf(persistent_id: Any) -> Any:
    """What `flatten` pickles whole on `FLAT_FIRST` Pythons: what
    *persistent_id* takes, and a library's objects (`_library_type`)."""

    def leaf(obj: Any) -> bool:
        if persistent_id is not None and persistent_id(obj) is not None:
            return True
        return _library_type(type(obj))

    return leaf


def _library_type(t: type) -> bool:
    """Is *t* a library's type (pandas, numpy, the standard library's
    ``datetime``), not one of the user's nor a builtin container? Remembered
    per type."""
    try:
        return _LIBRARY_TYPES[t]
    except KeyError:
        pass
    except TypeError:  # a class whose metaclass makes it unhashable
        return False
    module = t.__module__
    if module in _WALKED_MODULES:
        found = False
    else:
        from .install_paths import is_user_code_module

        found = not is_user_code_module(sys.modules.get(module))
    _LIBRARY_TYPES[t] = found
    return found


#: Their types hold other objects as the user's do: walked by `flatten`.
_WALKED_MODULES = frozenset({"builtins", "collections", "copyreg", "__main__"})

_LIBRARY_TYPES: weakref.WeakKeyDictionary = _weak.WeakKeyDictionary()


def _dumps(
    value: Any,
    protocol: int,
    fast: bool,
    buffer_callback: Any,
    extra: dict | None,
    persistent_id: Any,
    keyed: bool,
    aliases: bool | dict = False,
    sink: Any = None,
    table: _KeepDict | None = None,
    override: Any = None,
    after: Any = None,
) -> bytes:
    buf = io.BytesIO() if sink is None else sink
    if override is None:
        pickler = pickle.Pickler(buf, protocol=protocol, buffer_callback=buffer_callback)
    else:
        pickler = _Overriding(buf, protocol=protocol, buffer_callback=buffer_callback)
        pickler.override = override
    pickler.fast = fast
    if persistent_id is not None:
        pickler.persistent_id = persistent_id
    elif aliases:
        # Key bytes: sharing marked for the objects whose sharing means
        # something, never through pickle's memo (`key_aliases`).
        pickler.fast = True
        pickler.persistent_id = key_aliases(
            aliases if type(aliases) is dict else None, sorted_set_ids(protocol, extra) if keyed else None
        )
    if table is None:
        # Without the memo (and without a hook that marks an object met
        # again), an object's state cannot be set by a state setter: pickle
        # writes the object again for the setter to take, and with no memo
        # to write it as a back-reference, reduces it again, without end.
        unshared = pickler.fast and persistent_id is None and not aliases
        table = _KeepDict(protocol, extra, keyed, unshared=unshared)
    pickler.dispatch_table = table
    with own_work:
        pickler.dump(value)
    if after is not None:
        after(pickler)
    return b"" if sink is not None else buf.getvalue()


def store_memo(value: Any, protocol: int, *, extra: dict | None, sink: Any) -> dict[int, tuple[int, Any]]:
    """The memo of *value*'s store pickle (as `dumps_also_keyed` makes it),
    ``id(obj) -> (index, obj)``: every object the pickle holds once and
    refers back to, which is every object whose identity it keeps. The
    bytes go to *sink* and are dropped; buffers are handed out of band
    and not copied. Raises what pickling raises."""
    found: dict[int, tuple[int, Any]] = {}
    _dumps(
        value,
        protocol,
        False,
        _apart,
        extra,
        None,
        False,
        sink=sink,
        table=_KeepDict(protocol, extra),
        after=lambda pickler: found.update(pickler.memo.copy()),
    )
    return found


def _apart(_buffer: Any) -> bool:
    return False


def restores_alike_apart(obj: Any, table: dict) -> bool:
    """Does *obj*, an object the store pickles of two values both hold,
    restore alike whether the two are pickled together or apart: a value
    no program tells from an equal one (a string, a number, a tuple -- its
    items are asked on their own --, a numpy dtype or scalar), or an
    object pickled by its name, which every pickle restores to the one
    object that name holds (a class, a function, a ``__reduce__`` that
    gives a name)? *table* is a store's reducers (`_KeepDict`)."""
    t = type(obj)
    if t in _ATOMS or t is types.FunctionType or _identityless(t):
        return True  # a function: the C pickler writes it by name, before the table
    try:
        reducer = table[t]
    except KeyError:
        return issubclass(t, type)  # a class: by name
    except Exception:  # noqa: BLE001 - not known: kept apart
        return False
    try:
        return isinstance(reducer(obj), str)
    except Exception:  # noqa: BLE001 - not known: kept apart
        return False


class _Overriding(pickle.Pickler):
    """A pickler whose ``reducer_override`` is *override* (the C pickler
    takes the hook from its class only)."""

    override: Any = None

    def reducer_override(self, obj: Any) -> Any:
        return self.override(obj)


def key_aliases(outside: dict[int, Any] | None = None, sets: Any = None) -> Callable[[Any], Any]:
    """A key pickle's ``persistent_id`` (`dumps` with *aliases*): an object
    that can change met a second time is written as a mark naming its first
    appearance; anything else is written in full every time.

    Pickle's memo writes ANY object met twice as a back-reference, by
    identity: a string, a tuple, a numpy dtype as much as a list. Which
    immutable objects happen to be one object is how they were built, not
    what they hold: a freshly fitted model's parameter names are the same
    string objects as the keys of its own ``__dict__``, and its arrays share
    numpy's one ``float64`` dtype, while the same model loaded from a pickle
    holds copies. Equal models gave different key bytes, so a cached call
    taking a model a cached call returned missed once in every new process
    that loaded the model from the cache. Two names for one list, dict or
    object still key apart from two equal ones, as through the memo, and a
    cycle is closed by the mark.

    Every object it marks is kept for the dump: a temporary a reduce made
    and dropped must not hand its id to the next.

    *outside*, ``id(obj) -> where``, are the containers the rest of the key
    holds, each kept by the caller: one of them met here is written as a
    mark of where, its content keyed there. An object the key walk left to
    pickle whole holding a list the walk keyed too, or a plain argument
    keyed by its own digest, keyed like an equal copy before
    (`canonical_form.canonical_bytes`).

    *sets*, when given, is asked of each set and frozenset written in full
    (`sorted_set_ids`): a set met again is marked as any changeable object
    is, and only its first appearance is written sorted.
    """
    seen: dict[int, tuple[int, Any]] = {}
    atoms = _ATOMS
    by_name = _BY_NAME
    kernel = getattr(_native.kernels, "key_alias_id", None)
    if kernel is not None:
        # The same answers at C speed: asked of every object a key writes.
        return functools.partial(kernel, seen, atoms, by_name, {}, _identityless, outside, sets)

    def persistent_id(obj: Any, type: Any = type, id: Any = id) -> Any:
        t = type(obj)
        if t is frozenset and sets is not None:
            return sets(obj)
        if t in atoms or t in by_name or issubclass(t, type) or _identityless(t):
            return None
        found = seen.get(id(obj))
        if found is not None:
            return ("__cash_alias__", found[0])
        if outside is not None:
            where = outside.get(id(obj))
            if where is not None:
                return ("__cash_held__", where)
        seen[id(obj)] = (len(seen), obj)
        return sets(obj) if t is set and sets is not None else None

    return persistent_id


#: What a key pickle writes for a set or frozenset of two or more items, in
#: place of the set (`sorted_set_ids`): this mark, the set's type and its
#: items in sorted order.
SET_MARK = "__cash_set__"


def sorted_set_ids(protocol: int, extra: dict | None) -> Callable[[Any], Any]:
    """A key pickle's ``persistent_id`` that writes each set and frozenset
    of two or more items as ``(SET_MARK, its type's name, its items
    sorted)`` -- and the order it iterates in where that is not the plain
    one (`set_walk_order`) -- and lets the pickle write anything else as it
    would: None.

    Pickle writes a set's items in the order the set iterates in, its hash
    table's slot order. That order is not the set's content: it depends on
    the order the items went in and, for strings, on ``PYTHONHASHSEED``,
    and a set rebuilt from a pickle can iterate in another order than the
    one pickled. A frozenset a ``functools.cache`` memo answered with,
    restored from a store, digested apart from the one stored, and read as
    changed. Sorted (`_sorted_items`), equal sets write equal bytes in any
    process. A set of one item, or none, has one order, and is written as
    it always was.

    The same set met again gets the same tuple, which the pickle's memo
    writes as a back-reference, as it did the set: one set held twice
    still keys apart from two equal sets. Every set answered for is kept
    for the dump, so no other object takes its id. *protocol* and *extra*
    are the pickle's, for an item whose order is told by its own pickle.

    With the C extension, asked at C speed (``cwalk.c``'s
    ``sorted_set_id``, which sorts a set of strings, ints or bytes alone
    itself): the pickle asks it of every object it writes.
    """
    made: dict[int, tuple[Any, tuple]] = {}

    def forms(obj: Any) -> tuple | None:
        if len(obj) < 2:
            return None
        key = id(obj)
        found = made.get(key)
        if found is not None:
            return found[1]
        walked = list(obj)
        items = tuple(_sorted_items(walked, protocol, extra))
        order = set_walk_order(walked, items)
        written = (SET_MARK, _SET_NAMES[type(obj)], items) + (() if order is None else (order,))
        made[key] = (obj, written)
        return written

    kernel = getattr(_native.kernels, "sorted_set_id", None)
    if kernel is not None:
        return functools.partial(kernel, forms, made, _SET_SPEC)

    def persistent_id(obj: Any, type: Any = type) -> Any:
        t = type(obj)
        return forms(obj) if t is set or t is frozenset else None

    return persistent_id


def set_walk_order(walked: list, ordered: tuple | list) -> bytes | None:
    """The order a set iterates its items that hash alike in every process
    in (`_hashed_alike`): each one's place in *ordered* (the set's items,
    the same objects, sorted), in the order the set iterates them, as
    little-endian uint32 bytes. None when fewer than two of its items hash
    so, or when those iterate in sorted order. *walked* is the set's items
    as it iterates them.

    Equal sets can iterate in other orders: two items that collide in the
    table sit in the order they went in (``{1, 9}`` and ``{9, 1}``), and
    deletions or a copied larger table move items too. ``list(s)``,
    ``s.pop()`` and ``next(iter(s))`` read that order, so a key that left it
    out served ``first = s.pop()`` the item another set's order put first.

    Only numbers' order: their hashes, and so the order they iterate in
    after the same history, are the same in every process, and the key with
    their order in it is too. Strings iterate in an order ``PYTHONHASHSEED``
    picks per process, so a set of strings keys order-free, as before, and
    in a set of both only the numbers' order among themselves is kept. A
    set iterating its numbers in sorted order keys as before. With the C
    extension, at C speed (``cwalk.c``)."""
    kernel = getattr(_native.kernels, "set_walk_order", None)
    if kernel is not None:
        return kernel(walked if type(walked) is list else list(walked), ordered, _NONE_ALIKE)
    alike = [item for item in walked if _hashed_alike(item)]
    if len(alike) < 2:
        return None
    place = {id(item): at for at, item in enumerate(ordered)}
    order = array.array("I", [place[id(item)] for item in alike])
    if all(map(operator.lt, order, itertools.islice(order, 1, None))):
        return None
    if sys.byteorder != "little":
        order.byteswap()
    return order.tobytes()


#: Whether ``None`` hashes the same in every process: from Python 3.12 on
#: (before, its hash is its address).
_NONE_ALIKE = sys.version_info >= (3, 12)


def _hashed_alike(item: Any) -> bool:
    """Does *item* hash the same in every process? An int or bool, a float
    or complex number other than NaN (whose hash is its address), ``None``
    (3.12+), or a tuple or frozenset of those."""
    kind = type(item)
    if kind is int or kind is bool:
        return True
    if item is None:
        return _NONE_ALIKE
    if kind is float or kind is complex:
        return item == item
    if kind is tuple or kind is frozenset:
        return all(map(_hashed_alike, item))
    return False


def _sets_first(persistent_id: Any, sets: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """*sets* (`sorted_set_ids`), then *persistent_id* when given."""
    if persistent_id is None:
        return sets

    def both(obj: Any) -> Any:
        found = sets(obj)
        return found if found is not None else persistent_id(obj)

    return both


#: Items a set of only these sorts by their own values: no two equal.
_SELF_ORDERED = frozenset({str, int, bytes})

_SET_NAMES = {set: "set", frozenset: "frozenset"}
#: What ``cwalk.c``'s ``sorted_set_id`` writes a form of: these very objects,
#: so a pickle's memo writes each once, as for the Python hook's forms.
_SET_SPEC = (SET_MARK, _SET_NAMES[set], _SET_NAMES[frozenset])


#: A complex number's place among others with no NaN in them.
_COMPLEX_PARTS = operator.attrgetter("real", "imag")


def _sorted_items(items: Any, protocol: int, extra: dict | None) -> list:
    """The items of a set in an order its content alone decides, whatever
    order it iterates in (`sorted_set_ids`). Strings, ints or bytes alone
    sort as they are, and so do floats or complex numbers alone, none of
    them NaN; anything else by `_order_key`. Which way a set sorts depends
    on its items alone, so equal sets sort alike."""
    kinds = set(map(type, items))
    if len(kinds) == 1:
        kind = next(iter(kinds))
        if kind in _SELF_ORDERED:
            return sorted(items)
        if kind is float and all(x == x for x in items):
            return sorted(items)
        if kind is complex and all(z == z for z in items):
            return sorted(items, key=_COMPLEX_PARTS)
    return sorted(items, key=functools.partial(_order_key, protocol=protocol, extra=extra))


def _order_key(item: Any, protocol: int, extra: dict | None) -> tuple:
    """Where *item* sorts among a set's items (`_sorted_items`): its type's
    name, then its value -- by itself for a string, an int or bytes, by its
    ``repr`` for another number or None, item by item for a tuple or
    frozenset -- or, for anything else, the SHA-256 of its key pickle,
    written with its own sets sorted and blind to which of its parts are one
    object (`key_aliases`). Two items share a key only when their content
    pickles alike; such items keep the order the set gives them."""
    t = type(item)
    if t is str or t is int or t is bytes:
        return (t.__name__, item)
    if t is float or t is complex or t is bool or item is None:
        return (t.__name__, repr(item))
    if t is tuple:
        return ("tuple", tuple(_order_key(part, protocol, extra) for part in item))
    if t is frozenset:
        return ("frozenset", tuple(sorted(_order_key(part, protocol, extra) for part in item)))
    pickled = dumps(item, protocol, extra=extra, keyed=True, aliases=True)
    return ("", hashlib.sha256(pickled).digest())


#: Types whose instances cannot change and are pickled by value (tuples and
#: frozensets: their items are marked, if anything, on their own).
_ATOMS = frozenset({str, bytes, int, float, complex, bool, type(None), tuple, frozenset, range, slice, type(Ellipsis)})

#: Types pickled by name: their identity is their name.
_BY_NAME = frozenset({types.FunctionType, types.BuiltinFunctionType, types.MethodDescriptorType, types.ModuleType})


def _identityless(t: type) -> bool:
    """Is *t* a numpy dtype or scalar type: values that cannot change, so
    which object holds one says nothing? Remembered per type."""
    try:
        return _IDENTITYLESS[t]
    except KeyError:
        pass
    except TypeError:
        return False
    numpy = sys.modules.get("numpy")
    found = numpy is not None and (
        issubclass(t, numpy.dtype) or (issubclass(t, numpy.generic) and t not in (numpy.object_, numpy.void))
    )
    _IDENTITYLESS[t] = found
    return found


_IDENTITYLESS: weakref.WeakKeyDictionary = _weak.WeakKeyDictionary()


def _surely_shallow(value: Any) -> bool:
    """Is *value* one the pickler walks a level or two of at most: an atom,
    a plain array, or a short container of atoms? Such a value needs no
    large stack (`_big_stack`), and most of what is pickled is one."""
    kind = type(value)
    if kind in _FLAT_ATOMS or _flat_array(value):
        return True
    if kind is dict:
        return len(value) <= _SHALLOW_ITEMS and all(
            type(k) in _FLAT_ATOMS and type(v) in _FLAT_ATOMS for k, v in value.items()
        )
    if kind in (list, tuple, set, frozenset):
        return len(value) <= _SHALLOW_ITEMS and all(type(item) in _FLAT_ATOMS for item in value)
    return False


#: The most items `_surely_shallow` looks through before it says no.
_SHALLOW_ITEMS = 64


def too_deep(exc: BaseException) -> bool:
    """Did *exc* stop a walk that went deeper than the stack allows?

    A RecursionError; or, on Windows before Python 3.12, the MemoryError
    the interpreter raises when it finds its C stack nearly used up.
    """
    return isinstance(exc, RecursionError) or (isinstance(exc, MemoryError) and str(exc) == "Stack overflow")


class _Slot(int):
    """In a `FlatValue`'s steps: the object made at that step number."""

    __slots__ = ()

    def __reduce__(self) -> tuple:
        return _Slot, (int(self),)


#: Pickled as they are in a `FlatValue`: none holds another object.
_FLAT_ATOMS: frozenset[type] = frozenset(
    {type(None), bool, int, float, complex, str, bytes, type(Ellipsis), type(NotImplemented), range}
)


class FlatValue:
    """A value as a flat list of steps that make it, pickled where the value
    itself is too deep to pickle: ``load_flat(steps, root)`` when loaded.

    Each object the value holds is one step (made, then filled), and refers
    to the others by step number (`_Slot`), so the stream is as deep as one
    object however long the chain. The steps are those pickle's own stream
    takes, in its order: an object is made from its reduce, then its
    state is set once everything in it is built, so ``__setstate__`` sees
    what it would see in a plain pickle.
    """

    __slots__ = ("depth", "root", "steps")

    def __init__(self, steps: list, root: Any, depth: int = 0) -> None:
        self.steps = steps
        self.root = root
        #: How many objects deep the value is: the longest chain of
        #: objects, each holding the next, that the walk followed.
        self.depth = depth

    def __reduce__(self) -> tuple:
        return load_flat, (self.steps, self.root)


def flatten(value: Any, reducers: Any, leaf: Any = None, deepest: int | None = None) -> FlatValue:
    """*value* as a `FlatValue`, walked on a stack of constant size.

    *reducers* is ``type -> reducer`` (`_KeepDict`); *leaf*, when given,
    says which objects to pickle as they are (the pickler's persistent ids).
    Raises ``pickle.PicklingError`` for an object rebuilt from arguments
    that hold the object itself, as pickle cannot store one either, and
    RecursionError, as the pickler does, for a value more than *deepest*
    objects deep.
    """
    steps: list = []
    slots: dict[int, _Slot] = {}
    alive: list = []  # every object walked, so no id in *slots* is reused
    making: set[int] = set()

    def made(obj: Any) -> _Slot:
        """The step number of *obj*, made by the step appended next."""
        slot = _Slot(len(slots_in_order))
        slots_in_order.append(slot)
        slots[id(obj)] = slot
        alive.append(obj)
        return slot

    slots_in_order: list[_Slot] = []

    def whole(obj: Any) -> bool:
        """Pickled as it is, without a step: it holds no object."""
        obj_type = type(obj)
        return obj_type in _FLAT_ATOMS or obj_type in _WHOLE or issubclass(obj_type, type)

    def start(obj: Any) -> tuple[bool, Any]:
        obj_type = type(obj)
        if obj_type in _FLAT_ATOMS:
            return True, obj
        key = id(obj)
        if key in slots:
            return True, slots[key]
        if leaf is not None and leaf(obj):
            return True, obj  # asked first, as the pickler asks its persistent_id first
        if obj_type is list:
            return False, _list(obj)
        if obj_type is dict:
            return False, _dict(obj)
        if obj_type is tuple:
            if all(map(whole, obj)):
                return True, obj  # pickled in the step that holds it
            return False, _tuple(obj)
        if obj_type in _WHOLE or issubclass(obj_type, type) or _flat_array(obj):
            return True, obj
        reduced = reducers[obj_type](obj)
        if isinstance(reduced, str):
            return True, obj  # a global, pickled by name
        if key in making:
            raise pickle.PicklingError(f"a {obj_type.__qualname__} is rebuilt from arguments that hold it")
        return False, _reduced(obj, reduced)

    def _list(obj: list) -> Any:
        slot = made(obj)
        steps.append(("list", slot))
        items = []
        for item in obj:
            items.append((yield item))
        steps.append(("extend", slot, items))
        return slot

    def _dict(obj: dict) -> Any:
        slot = made(obj)
        steps.append(("dict", slot))
        pairs = []
        for k, v in obj.items():
            pairs.append((yield k))
            pairs.append((yield v))
        steps.append(("setitems", slot, pairs))
        return slot

    def _tuple(obj: tuple) -> Any:
        items = []
        for item in obj:
            items.append((yield item))
        if id(obj) in slots:  # made meanwhile, through a cycle
            return slots[id(obj)]
        slot = made(obj)
        steps.append(("tuple", slot, items))
        return slot

    def _reduced(obj: Any, reduced: tuple) -> Any:
        rebuild, args, *rest = reduced
        rest += [None] * (4 - len(rest))
        state, listitems, dictitems, setter = rest[:4]
        making.add(id(obj))
        args = yield args
        making.discard(id(obj))
        slot = made(obj)
        steps.append(("new", slot, rebuild, args))
        # pickle's order: the items, then the state (`load_flat`).
        if listitems is not None:
            items = []
            for item in listitems:
                items.append((yield item))
            steps.append(("extend", slot, items))
        if dictitems is not None:
            pairs = []
            for k, v in dictitems:
                pairs.append((yield k))
                pairs.append((yield v))
            steps.append(("setitems", slot, pairs))
        if state is not None:
            state = yield state
            steps.append(("build", slot, state, setter))
        return slot

    depth = 0
    done, root = start(value)
    if not done:
        stack = [root]
        sent: Any = None
        while stack:
            if len(stack) > depth:
                depth = len(stack)
                if deepest is not None and depth > deepest:
                    raise RecursionError("maximum recursion depth exceeded while pickling an object")
            try:
                part = stack[-1].send(sent)
            except StopIteration as finished:
                stack.pop()
                sent = finished.value
                continue
            done, got = start(part)
            if done:
                sent = got
            else:
                stack.append(got)
                sent = None
        root = sent
    return FlatValue(steps, root, depth)


def load_flat(steps: list, root: Any) -> Any:
    """The value a `FlatValue` holds, made step by step. Named in stored
    entries: keep its name."""
    made: list = []
    # Slots whose state is set. A value stored before the steps took pickle's
    # order (state, then items) was rebuilt wrong -- a dict whose
    # ``__dict__`` is itself lost its attributes -- so such a stream is
    # refused, and the entry is a miss.
    built: set = set()

    def get(part: Any) -> Any:
        return made[part] if type(part) is _Slot else part

    for step in steps:
        kind = step[0]
        if built and kind in ("extend", "setitems") and step[1] in built:
            raise pickle.UnpicklingError("a flat value stored in an older step order")
        if kind == "list":
            made.append([])
        elif kind == "dict":
            made.append({})
        elif kind == "tuple":
            made.append(tuple(get(item) for item in step[2]))
        elif kind == "new":
            made.append(step[2](*get(step[3])))
        elif kind == "build":
            _build(made[step[1]], get(step[2]), step[3])
            built.add(step[1])
        elif kind == "extend":
            target, items = made[step[1]], [get(item) for item in step[2]]
            if isinstance(target, list):
                target.extend(items)
            else:
                extend = getattr(target, "extend", None)
                if extend is not None:
                    extend(items)
                else:
                    for item in items:
                        target.append(item)
        elif kind == "setitems":
            target, pairs = made[step[1]], step[2]
            for i in range(0, len(pairs), 2):
                target[get(pairs[i])] = get(pairs[i + 1])
        else:
            raise pickle.UnpicklingError(f"unknown step {kind!r} in a flat value")
    return get(root)


def _build(obj: Any, state: Any, setter: Any) -> None:
    """Set *obj*'s *state*, as pickle's ``BUILD`` does."""
    if setter is not None:
        setter(obj, state)
        return
    setstate = getattr(obj, "__setstate__", None)
    if setstate is not None:
        setstate(state)
        return
    slots = None
    if isinstance(state, tuple) and len(state) == 2:
        state, slots = state
    if state:
        inst_dict = obj.__dict__
        for k, v in state.items():
            inst_dict[sys.intern(k) if type(k) is str else k] = v
    if slots:
        for k, v in slots.items():
            setattr(obj, k, v)


#: Pickled as they are in a `FlatValue`: by name, or holding no object.
_WHOLE: frozenset[type] = frozenset(
    {
        types.FunctionType,
        types.BuiltinFunctionType,
        types.ModuleType,
        types.CodeType,
        bytearray,
        pickle.PickleBuffer,
    }
)


def _flat_array(obj: Any) -> bool:
    """A numpy array of plain values: pickled as it is in a `FlatValue`,
    its data out of band as ever."""
    if "numpy" not in sys.modules:
        return False
    ndarray = sys.modules["numpy"].ndarray
    # ``isinstance`` asks an object that is not an array for its
    # ``__class__``, which runs a ``__getattribute__`` of its own:
    # `instance_of` answers such an object by its real type.
    return instance_of(obj, ndarray) and not obj.dtype.hasobject


def reducers(protocol: int = pickle.DEFAULT_PROTOCOL, extra: dict | None = None) -> dict:
    """Each type's reducer, as `dumps` reduces it: ``reducers(p)[type(obj)](obj)``
    is what pickling *obj* at protocol *p* saves (a class has none: it is
    pickled by name). *extra*: reducers taken first, as `dumps` takes them."""
    return _KeepDict(protocol, extra)


#: Lock types a value may hold for its own use (`live_state.live_reduce`).
_LOCK_TYPES = frozenset({type(threading.Lock()), type(threading.RLock())})


def _refused(obj: Any) -> Any:
    """A key's reduce of a writer cash puts in front of a stream: refused,
    as the stream it stands for is."""
    raise TypeError(f"cannot key {type(obj).__name__!r} object")


def _stand_in(obj: Any, reducer: Any, watch: list | None) -> Any:
    """*reducer*'s reduce of a writer cash puts in front of a stream, which
    stores as the stream (``notebook/_tee.py``); *watch* is told, as a key
    refuses it (`_refused`)."""
    if watch is not None:
        watch.append(True)
    return reducer(obj)


def _or_live(obj: Any, reducer: Any, watch: list | None = None) -> Any:
    """*reducer*'s reduce of *obj*; for one pickle refuses, a standard stream
    or a free lock, the stand-in `live_state.live_reduce` gives. What pickled
    before pickles as it did. The stand-in is in the stored bytes only, which
    *watch* (`_KeepDict`) is told of: keyed bytes still refuse the object.

    A standard stream's own reduce may succeed and its state fail later
    (ipykernel's ``OutStream`` holds the session's HMAC): one is asked whether
    it pickles whole, which only the few streams bound in ``sys`` are."""
    if is_live_stream(obj) and not _pickles(obj):
        if watch is not None:
            watch.append(True)
        return live_reduce(obj)
    try:
        return reducer(obj)
    except (TypeError, pickle.PicklingError):
        live = live_reduce(obj)
        if live is None:
            raise
        if watch is not None:
            watch.append(True)
        return live


def _pickles(obj: Any) -> bool:
    """Whether ``pickle`` takes *obj* whole."""
    try:
        pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:  # noqa: BLE001 - whatever its own reduce raises
        return False
    return True


#: The standard library's types (not a subclass of one) whose
#: ``__reduce_ex__`` is always ``(class, args)``, no state.
_NO_STATE_TYPES = frozenset({datetime.datetime, datetime.date, datetime.time, datetime.timedelta})


class Ahead:
    """An *extra* reducer (`dumps`) tried before the type's own: *reduce*
    gives an object's reduce value, or None for the reducer the table
    would have used without it (`_KeepDict`), for the objects it leaves
    as they were."""

    __slots__ = ("reduce",)

    def __init__(self, reduce: Callable[[Any], Any]) -> None:
        self.reduce = reduce


def _ahead_of(first: Callable[[Any], Any], reducer: Any) -> Callable[[Any], Any]:
    """*first*'s reduce of an object, or *reducer*'s when *first* gives None
    (`Ahead`)."""

    def reduce(obj: Any) -> Any:
        reduced = first(obj)
        return reducer(obj) if reduced is None else reduced

    return reduce


class _KeepDict(dict):
    """A pickler's ``dispatch_table``: each type's reducer, decided at its
    first instance. The ordinary ``__reduce_ex__`` for most, so the stream
    is the one ``pickle.dumps`` writes; `_reduce_with_dict` for a type whose
    reduce drops its instances' ``__dict__`` (`_drops_dict`)."""

    def __init__(
        self,
        protocol: int,
        extra: dict | None,
        keyed: bool = False,
        watch: list | None = None,
        unshared: bool = False,
    ) -> None:
        extra = dict(extra or ())
        #: The *extra* reducers tried before the type's own (`Ahead`).
        self._ahead = {t: extra.pop(t).reduce for t in [t for t, r in extra.items() if type(r) is Ahead]}
        super().__init__(extra)
        self._protocol = protocol
        self._keyed = keyed
        #: A pickle without the memo, never loaded (`_reduce_with_dict`).
        self._unshared = unshared
        #: Store bytes only (`dumps_also_keyed`): gets an item when an
        #: object is reduced otherwise than a key reduces it.
        self._watch = watch
        #: This pickle's one object for each attribute name (`_names_apart`).
        self._names: dict[str, str] = {}
        #: This pickle's one object for each numpy dtype (`_dtypes_apart`).
        self._dtypes: dict[Any, Any] = {}

    def __missing__(self, t: type) -> Any:
        if _numpy_scalar_type(t):
            # A scalar reduces to its dtype and bytes, no state: no names.
            reducer = _scalar_dtype_apart(self._reducer(t), self._dtypes)
        else:
            reducer = self._reducer(t)
            if t not in _NO_STATE_TYPES or type(reducer) is not operator.methodcaller:
                # A datetime reduces to its class and its bytes, never a
                # state: nothing for `_names_apart` to part, and no Python
                # call per object (two million rows hold as many datetimes).
                reducer = _names_apart(reducer, self._names)
            if _holds_dtype(t):
                reducer = _dtypes_apart(reducer, self._dtypes)
        if self._keyed:
            # A key refuses a stream and a lock as it did: what stands in for
            # one is stored, never keyed (`live_state`).
            if getattr(t, "_cash_wrapped_stream", None) is not None:
                reducer = _refused
        elif getattr(t, "_cash_wrapped_stream", None) is not None:
            reducer = functools.partial(_stand_in, reducer=reducer, watch=self._watch)
        elif issubclass(t, io.IOBase) or t in _LOCK_TYPES:
            reducer = functools.partial(_or_live, reducer=reducer, watch=self._watch)
        first = self._ahead.get(t)
        if first is not None:
            reducer = _ahead_of(first, reducer)
        self[t] = reducer
        return reducer

    def _reducer(self, t: type) -> Any:
        reducer = self._chosen_reducer(t)
        if runs_users_state(t):
            # Counted each time it runs: what it runs is the user's code,
            # which may change what cash read before it (`user_state_runs`).
            return functools.partial(_counted, reducer=reducer)
        return reducer

    def _chosen_reducer(self, t: type) -> Any:
        if issubclass(t, type):
            raise KeyError(t)  # a class: pickled by name, as pickle does
        if t in WEAK_PROXY_TYPES and self._watch is None:
            return functools.partial(_reduce_weak_proxy, protocol=self._protocol)
        registered = copyreg.dispatch_table.get(t)
        if self._watch is not None:
            reducer = _watched(self._store_reducer(t, registered), t, registered, self._watch)
            if t.__name__ in _GLOBAL_RNG_TYPE_NAMES:
                reducer = functools.partial(_watched_rng, reducer=reducer, watch=self._watch)
            return reducer
        if self._keyed and t.__name__ == "ndarray" and t is getattr(sys.modules.get("numpy"), "ndarray", None):
            # A record array's padding bytes are whatever memory held:
            # keyed without them (`packed_records`).
            return functools.partial(_reduce_array_for_key, protocol=self._protocol)
        if self._keyed and (chooses_its_state(t) or (registered is not None and _is_users_class(t))):
            # A user's class whose state its own reduce -- or one registered
            # with ``copyreg.pickle`` -- decides: key it with what it leaves out.
            reducer = functools.partial(_reduce_for_key, protocol=self._protocol, registered=registered)
        elif registered is not None:
            reducer = registered
        elif _plain_container_subclass(t):
            reducer = _reduce_container_subclass
        else:
            if _drops_dict(t):
                reducer = _reducer_with_dict(self._protocol, self._unshared)
            else:
                reducer = operator.methodcaller("__reduce_ex__", self._protocol)
            if cached_properties(t) and (self._keyed or _leaves_views_out(t)):
                which = None if self._keyed else stored_views
                reducer = functools.partial(_reduce_without_views, reducer=reducer, which=which)
        if self._keyed and may_stand_by_reference(t):
            # A logger reduces to ``getLogger(name)``: the key would hold
            # its name and none of its level (`_reduce_by_reference_for_key`).
            reducer = functools.partial(_reduce_by_reference_for_key, reducer=reducer)
        if self._keyed and t.__name__ in _GLOBAL_RNG_TYPE_NAMES:
            # One instance of the class may be a module's global RNG.
            reducer = functools.partial(_reduce_rng_for_key, reducer=reducer)
        return reducer

    def _store_reducer(self, t: type, registered: Any) -> Any:
        """The reducer a store's pickle (not *keyed*) takes for *t*."""
        if registered is not None:
            return registered
        if _plain_container_subclass(t):
            return _reduce_container_subclass
        if _drops_dict(t):
            reducer = _reducer_with_dict(self._protocol, self._unshared)
        else:
            reducer = operator.methodcaller("__reduce_ex__", self._protocol)
        if cached_properties(t) and _leaves_views_out(t):
            reducer = functools.partial(_reduce_without_views, reducer=reducer, which=stored_views)
        return reducer


def dead_weak_proxy() -> None:
    """What a ``weakref.proxy`` whose object is gone stands for in a key
    pickle: one reference to this function, the same for every such
    proxy (`value_types.is_dead_proxy`)."""


def _reduce_weak_proxy(proxy: Any, protocol: int) -> Any:
    """A live proxy pickles as the object it refers to, as pickle itself
    does it; a dead one as `dead_weak_proxy`, where pickle raised
    ReferenceError and the key could not be built."""
    if is_dead_proxy(proxy):
        return (dead_weak_proxy, ())
    return proxy.__reduce_ex__(protocol)


def _names_apart(reducer: Any, names: dict[str, str]) -> Callable[[Any], Any]:
    """*reducer*, giving each attribute name of an object's state (a dict,
    or ``__slots__``' ``(dict, slots)``) this pickle's own object for it,
    never one the value also holds elsewhere.

    The pickle's memo writes an object met twice as a back-reference, by
    identity. A fitted ``ColumnTransformer``'s ``remainder='drop'`` is the
    very string object of its ``OneHotEncoder``'s attribute name ``drop``
    (both interned): the second came out as a back-reference to the first.
    Loading interns the attribute names again (to Python's own ``'drop'``),
    so the restored ``remainder`` is the pickle's object and no name: its
    pickle wrote the string twice, and the restored pipeline digested apart
    from the one it was stored from (every fit reading it missed after a
    restart). The names are written as objects of their own, the same
    whichever objects the value held, as loading leaves them."""

    apart = _names_apart_of(names)

    def reduce(obj: Any) -> Any:
        return apart(reducer(obj))

    kernel = getattr(_native.kernels, "reduce_dict_apart", None)
    protocol = getattr(reducer, "with_dict_protocol", None)
    if kernel is not None and protocol is not None:
        # The same answers at C speed: a key pickle asks it of every pandas
        # Timestamp it writes.
        return functools.partial(kernel, reduce, apart, protocol)
    kernel = getattr(_native.kernels, "reduce_apart", None)
    protocol = _reduce_ex_protocol(reducer)
    if kernel is not None and protocol is not None:
        # The same answers at C speed for an object reduced by its own
        # ``__reduce_ex__``: a store pickle asks it of every object of a
        # class of the user's (a trie of 250,000 slotted nodes: 1.2 s of a
        # 1.7 s store, where pickle alone took 0.45 s).
        return functools.partial(kernel, apart, names, protocol)
    return reduce


def _reduce_ex_protocol(reducer: Any) -> int | None:
    """The protocol of *reducer* when it is exactly
    ``operator.methodcaller("__reduce_ex__", protocol)``; else None."""
    if type(reducer) is not operator.methodcaller:
        return None
    try:
        _kind, (name, *args) = reducer.__reduce__()[:2]
    except (TypeError, ValueError):
        return None
    if name != "__reduce_ex__" or len(args) != 1 or type(args[0]) is not int:
        return None
    return args[0]


def _names_apart_of(names: dict[str, str]) -> Callable[[Any], Any]:
    """`_names_apart`'s change to a reduce value already made."""

    def apart(reduced: Any) -> Any:
        if type(reduced) is not tuple or len(reduced) < 3:
            return reduced
        state = reduced[2]
        if type(state) is dict:
            try:
                parted = {names[key]: value for key, value in state.items()}
            except (KeyError, TypeError):
                parted = _state_apart(state, names)
        elif type(state) is tuple and len(state) == 2:
            # ``(dict, slots)``: a slotted object's state, one per object of
            # a trie of them, so without a generator per object.
            attrs, slots = state
            if (attrs is not None and type(attrs) is not dict) or (slots is not None and type(slots) is not dict):
                return reduced
            parted = (
                None if attrs is None else _state_apart(attrs, names),
                None if slots is None else _state_apart(slots, names),
            )
        else:
            return reduced
        return (reduced[0], reduced[1], parted, *reduced[3:])

    return apart


def _holds_dtype(t: type) -> bool:
    """Does a *t* instance reduce to its numpy dtype (an array, a numpy
    scalar)?"""
    numpy = sys.modules.get("numpy")
    if numpy is None:
        return False
    try:
        return issubclass(t, (numpy.ndarray, numpy.generic))
    except TypeError:
        return False


def _numpy_scalar_type(t: type) -> bool:
    """Is *t* one of numpy's own scalar types (``numpy.float64``), not a
    subclass of one?"""
    numpy = sys.modules.get("numpy")
    return numpy is not None and t.__module__ == "numpy" and isinstance(t, type) and issubclass(t, numpy.generic)


def _scalar_dtype_apart(reducer: Any, dtypes: dict[Any, Any]) -> Callable[[Any], Any]:
    """`_dtypes_apart` for a numpy scalar's reduce, ``(scalar, (dtype,
    bytes))``: the last dtype met and the pickle's own for it are kept, so
    a run of scalars of one dtype (a model's thousands of ``float64``) asks
    the table once."""
    last: list[Any] = [None, None]

    def reduce(obj: Any) -> Any:
        reduced = reducer(obj)
        if type(reduced) is not tuple or len(reduced) != 2 or type(reduced[1]) is not tuple or not reduced[1]:
            return reduced
        args = reduced[1]
        dt = args[0]
        if dt is not last[0]:
            numpy = sys.modules["numpy"]
            if not isinstance(dt, numpy.dtype):
                return reduced
            last[0], last[1] = dt, _dtype_apart(dt, dtypes)
        return (reduced[0], (last[1], *args[1:]))

    return reduce


def _dtypes_apart(reducer: Any, dtypes: dict[Any, Any]) -> Callable[[Any], Any]:
    """*reducer*, giving the dtype an array or numpy scalar reduces to this
    pickle's own object for it, one per equal dtype, never one the value
    holds.

    As for attribute names (`_names_apart`): the pickle's memo writes a
    dtype met twice as a back-reference, by identity. ``np.int64(3)`` holds
    numpy's one ``int64`` dtype, which loading does not give back: it makes
    a dtype of its own (``np.dtype('i8', False, True)`` copies). A fitted
    forest's or SVC's pickle then wrote its scalars' dtype once and
    referred back to it, and the restored model's wrote it again: the
    restored model digested apart from the one it was stored from, and
    every statement reading it after a restart missed. Dtypes that pickle
    apart (other metadata, alignment, record type) stay apart."""

    def reduce(obj: Any) -> Any:
        reduced = reducer(obj)
        if type(reduced) is not tuple or len(reduced) < 2:
            return reduced
        args = _parts_apart(reduced[1], dtypes)
        state = _parts_apart(reduced[2], dtypes) if len(reduced) > 2 else None
        if args is None and state is None:
            return reduced
        return (
            reduced[0],
            reduced[1] if args is None else args,
            *((reduced[2] if state is None else state,) if len(reduced) > 2 else ()),
            *reduced[3:],
        )

    return reduce


def _parts_apart(parts: Any, dtypes: dict[Any, Any]) -> tuple | None:
    """*parts*, a tuple, with each numpy dtype in it the pickle's own;
    None when it holds none."""
    if type(parts) is not tuple:
        return None
    numpy = sys.modules.get("numpy")
    dtype = getattr(numpy, "dtype", None)
    if dtype is None:
        return None
    apart = None
    for i, part in enumerate(parts):
        if isinstance(part, dtype):
            if apart is None:
                apart = list(parts)
            apart[i] = _dtype_apart(part, dtypes)
    return None if apart is None else tuple(apart)


def _dtype_apart(dt: Any, dtypes: dict[Any, Any]) -> Any:
    """The pickle's own dtype that pickles as *dt* does. Keyed by that
    pickle, not by equality: an aligned struct and the same layout packed,
    or a record dtype and the plain structured one, are equal dtypes that
    load apart. A dtype met before is found by its identity (*dtypes*
    holds it, so the id stays its own) without pickling it again."""
    seen = dtypes.get(id(dt))
    if seen is not None and seen[0] is dt:
        return seen[1]
    try:
        key = pickle.dumps(dt, protocol=pickle.HIGHEST_PROTOCOL)
        own = dtypes.get(key)
        if own is None:
            own = dtypes[key] = copy.copy(dt)
    except Exception:  # noqa: BLE001 - a dtype that will not hash or copy: as it is
        return dt
    dtypes[id(dt)] = (dt, own)
    return own


def _state_apart(state: dict, names: dict[str, str]) -> dict:
    """*state* with each string key the pickle's own object for it; as it
    is when a key is not a string."""
    try:
        return {names[key]: value for key, value in state.items()}
    except (KeyError, TypeError):
        pass
    for key in state:
        if type(key) is not str:
            return state
        if key not in names:
            # A new object, equal (one character is Python's one object).
            names[key] = key[:1] + key[1:] if len(key) > 1 else key
    return {names[key]: value for key, value in state.items()}


#: The class names of the module-global RNGs (`global_rng_module`): the
#: cheap test, made once per type, before the identity one.
_GLOBAL_RNG_TYPE_NAMES = frozenset({"Random", "RandomState"})


def global_rng_module(obj: object) -> str | None:
    """The module whose global RNG *obj* is (``"random"``,
    ``"numpy.random"``), or None.

    The module functions (``random.random``, ``np.random.normal``) draw
    from these singletons, and cash's global RNG channel owns their state:
    it is seeded, captured and replayed as the module's, not as an
    object's. Unseeded, that state is fresh entropy in every process, so a
    value that merely holds the singleton (a scipy distribution's
    ``_random_state``) could not be keyed by it: it missed in every new
    process.
    """
    if obj is getattr(sys.modules.get("random"), "_inst", None):
        return "random"
    if obj is getattr(sys.modules.get("numpy.random.mtrand"), "_rand", None):
        return "numpy.random"
    return None


def _reduce_rng_for_key(obj: Any, reducer: Any) -> Any:
    """A module's global RNG by the module's name; any other generator of
    the class as *reducer* reduces it, by its state."""
    module = global_rng_module(obj)
    if module is not None:
        return (_global_rng, (module,))
    return reducer(obj)


def _global_rng(module: str) -> Any:
    """The global RNG of *module* (`global_rng_module`): what a key's
    stand-in for it loads back as."""
    if module == "random":
        return sys.modules["random"]._inst
    return sys.modules["numpy.random.mtrand"]._rand


def _watched_rng(obj: Any, reducer: Any, watch: list) -> Any:
    """*reducer*'s reduce of *obj*, noting in *watch* a module's global RNG,
    which a key names and a store keeps the state of."""
    if global_rng_module(obj) is not None:
        watch.append(True)
    return reducer(obj)


def _watched(reducer: Any, t: type, registered: Any, watch: list) -> Any:
    """*reducer*, a store's for *t*, noting in *watch* each object a key
    would reduce otherwise (`_KeepDict.__missing__` with *keyed*): what
    `dumps_also_keyed` answers from. Each check is the one the keyed
    reducer makes to decide whether it changes anything."""
    if t.__name__ == "ndarray" and t is getattr(sys.modules.get("numpy"), "ndarray", None):

        def array(obj: Any) -> Any:
            try:
                if packed_records(obj) is not obj:
                    watch.append(True)
            except Exception:  # noqa: BLE001 - keyed as it is then, as `_reduce_array_for_key` does
                pass
            return reducer(obj)

        return array
    if chooses_its_state(t) or (registered is not None and _is_users_class(t)):
        plain = registered is not None or not _drops_dict(t)

        def state(obj: Any) -> Any:
            reduced = reducer(obj)
            if not plain:
                watch.append(True)  # a key reduces it without the dict this adds
            elif left_out_attrs(obj, None if isinstance(reduced, str) or len(reduced) < 3 else reduced[2]):
                watch.append(True)
            return reduced

        return state
    if may_stand_by_reference(t):
        reducer = _watched_by_reference(_watched_views(reducer, t, registered, watch), watch)
    else:
        reducer = _watched_views(reducer, t, registered, watch)
    return reducer


def _watched_by_reference(reducer: Any, watch: list) -> Callable[[Any], Any]:
    """*reducer*, noting in *watch* an object it reduces by reference while
    it holds attributes a key reduces it with (`_reduce_by_reference_for_key`)."""

    def by_reference(obj: Any) -> Any:
        reduced = reducer(obj)
        if stands_by_reference(reduced):
            _note_met(obj, reduced)
            if _FOLDING.get() and _own_attrs(obj):
                watch.append(True)
        return reduced

    return by_reference


def _watched_views(reducer: Any, t: type, registered: Any, watch: list) -> Any:
    """`_watched` for a class that is not the user's choosing its state."""
    if registered is None and cached_properties(t):

        def views(obj: Any) -> Any:
            reduced = reducer(obj)
            if not isinstance(reduced, str) and derived_views(obj) - stored_views(obj):
                watch.append(True)  # a view the stored bytes keep and a key leaves out
            return reduced

        return views
    return reducer


def cached_properties(t: type) -> tuple[str, ...]:
    """The attribute names *t*'s ``functools.cached_property`` members fill
    in an instance's ``__dict__``. Remembered per type."""
    try:
        return _CACHED_PROPERTIES[t]
    except KeyError:
        pass
    except TypeError:  # a class whose metaclass makes it unhashable
        return ()
    names: list[str] = []
    for klass in t.__mro__:
        for name, member in vars(klass).items():
            if isinstance(member, functools.cached_property) and name not in names:
                names.append(member.attrname or name)
    found = _CACHED_PROPERTIES[t] = tuple(names)
    return found


_CACHED_PROPERTIES: weakref.WeakKeyDictionary = _weak.WeakKeyDictionary()

#: What a view may hold beside its links to the object it views: settings,
#: not data (``DegreeView``'s ``_weight``).
_VIEW_SETTINGS = (type(None), bool, int, float, str, bytes)


def derived_views(obj: Any) -> frozenset[str]:
    """The ``cached_property`` values in *obj*'s ``__dict__`` that are views
    of *obj*: every attribute of the value is *obj* itself (directly, by a
    weak reference, or as a bound method's ``__self__``), one of the other
    values *obj* holds, or a scalar setting -- and at least one is a link.

    networkx fills ``G.nodes``, ``G.edges``, ``G.adj`` and ``G.degree`` on
    first use, so ``print(G)`` changed G's pickle and read as an in-place
    change of G, and a call keyed on G's content missed after it. A view
    holds nothing the object does not, so a key leaves it out. A cached
    value that is a snapshot (a number, a list) is kept: it may be stale.
    """
    names = cached_properties(type(obj))
    state = getattr(obj, "__dict__", None)
    if not names or not isinstance(state, dict):
        return frozenset()
    present = [name for name in names if name in state]
    if not present:
        return frozenset()
    held = {id(value) for name, value in state.items() if name not in present}
    held.add(id(obj))
    return frozenset(name for name in present if _views(state[name], held))


def _views(value: Any, held: set[int]) -> bool:
    if id(value) in held or type(value) in _VIEW_SETTINGS:
        return False
    attrs = _instance_attrs(value)
    if not attrs:
        return False
    linked = False
    for part in attrs.values():
        if isinstance(part, weakref.ref):
            part = part()
        target = getattr(part, "__self__", part) if isinstance(part, types.BuiltinMethodType) else part
        if id(target) in held:
            linked = True
        elif type(part) not in _VIEW_SETTINGS:
            return False
    return linked


def _reduce_without_views(obj: Any, reducer: Any, which: Any = None) -> Any:
    """*reducer*'s reduce of *obj*, with the views `derived_views` finds
    left out of its state (those *which* gives, when given:
    `stored_views` for a store's bytes)."""
    reduced = reducer(obj)
    if isinstance(reduced, str):
        return reduced
    views = (which or derived_views)(obj)
    if not views:
        return reduced
    rebuild, args, *rest = reduced
    if not rest:
        return reduced
    state = rest[0]
    if isinstance(state, dict):
        state = {k: v for k, v in state.items() if k not in views}
    elif isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], dict):
        state = ({k: v for k, v in state[0].items() if k not in views}, state[1])
    else:
        return reduced
    return (rebuild, args, state, *rest[1:])


def stored_views(obj: Any) -> frozenset[str]:
    """The views of *obj* (`derived_views`) its stored pickle leaves out:
    networkx's, which hold nothing but links to the graph and are made
    again on first use. Building a graph fills ``G.adj``, so its stored
    bytes were not its keyed bytes, and a call storing a graph pickled it
    twice, once to digest and once to keep (`dumps_also_keyed`).

    Only these: a view of a user's class may hold a counter beside its
    link, which a restore must not reset."""
    views = derived_views(obj)
    if not views:
        return views
    state = obj.__dict__
    return frozenset(name for name in views if type(state[name]).__module__ in _VIEW_MODULES_LEFT_OUT)


#: The modules whose views a store leaves out (`stored_views`).
_VIEW_MODULES_LEFT_OUT = frozenset({"networkx.classes.coreviews", "networkx.classes.reportviews"})


def _leaves_views_out(t: type) -> bool:
    """Does a store's pickle of a *t* look for views to leave out
    (`stored_views`): a networkx graph class, or a subclass of one, that
    does not choose its own state (a key keeps such a class's views)?
    Any other class is pickled by its own reduce, as before, with no
    Python step per instance."""
    return not chooses_its_state(t) and any(k.__module__.startswith("networkx.") for k in t.__mro__)


def packed_records(array: Any) -> Any:
    """*array* without the padding of its record dtype, or *array* itself
    when its dtype has none.

    A record dtype's gaps (a ``uint8`` field after ``float64`` ones, an
    aligned struct) hold whatever the memory held: a fitted
    scikit-learn tree's node array keyed differently in every process,
    and a cached call taking a forest missed once for each. The copy holds
    the same fields, packed; nested records are packed too.
    """
    dtype = array.dtype
    if dtype.fields is None and dtype.subdtype is None:
        return array
    # The scalar type too: ``np.dtype((np.record, D))`` equals ``D`` and
    # hashes alike, but numpy packs it keeping its record type.
    packed = _packed_dtype(dtype.type, dtype)
    if packed is None:
        return array
    return array.astype(packed)


@functools.lru_cache(maxsize=256)
def _packed_dtype(scalar_type: type, dtype: Any) -> Any:
    """*dtype*, of *scalar_type*, packed (`packed_records`), or None when it
    has no padding.
    Once per dtype: a forest's trees share one node dtype, and repacking it
    for each of their 14,000 arrays took about 0.15 s a read of a voting
    classifier's four models."""
    from numpy.lib import recfunctions

    packed = recfunctions.repack_fields(dtype, recurse=True)
    return None if packed.itemsize == dtype.itemsize else packed


def _reduce_array_for_key(array: Any, protocol: int) -> Any:
    try:
        packed = packed_records(array)
    except Exception:  # noqa: BLE001 - a dtype recfunctions does not take: as it is
        packed = array
    if packed is array:
        return array.__reduce_ex__(protocol)
    # The dtype itself, not its ``descr``: an aligned struct and the same
    # layout packed have one ``descr``, and arrays of the same values in
    # each keyed alike.
    return _packed_array, (array.dtype, packed)


def _packed_array(dtype: Any, packed: Any) -> Any:
    """What `_reduce_array_for_key` names (key bytes are never loaded)."""
    import numpy as np

    return packed.astype(np.dtype(dtype))


#: The builtin containers `_plain_container_subclass` takes a subclass of.
_CONTAINER_BASES = (dict, list, tuple, set, frozenset)

#: What decides how a class is pickled, when a class of its own defines it.
_PICKLE_HOOKS = ("__reduce__", "__reduce_ex__", "__getstate__", "__setstate__", "__getnewargs__", "__getnewargs_ex__")

#: By class, held weakly: a class made per call is not kept alive by it.
_PLAIN_SUBCLASS: weakref.WeakKeyDictionary[type, bool] = _weak.WeakKeyDictionary()


def _plain_container_subclass(t: type) -> bool:
    """Whether *t* is a class written in Python on a dict, list, tuple, set
    or frozenset that holds nothing but the builtin's items, its
    ``__dict__`` and its slots, and leaves pickling to the builtin (or to
    ``collections.Counter``, whose reduce is ``dict(self)``), but has
    methods of its own that pickling reads or fills it with: then
    `_reduce_container_subclass` copies it without running a method of
    the class. Pickled by its own route, a subclass whose ``__iter__``,
    ``__len__`` or ``keys`` prints and counts printed and counted as cash
    stored it, where plain Python never asked. A namedtuple keeps its
    route, as does a class on a C subclass (``OrderedDict``,
    ``defaultdict``), which keeps more than the items."""
    try:
        return _PLAIN_SUBCLASS[t]
    except KeyError:
        pass
    except TypeError:  # a class whose metaclass makes it unhashable
        return False
    found = _PLAIN_SUBCLASS[t] = _decide_plain_container_subclass(t)
    return found


def _decide_plain_container_subclass(t: type) -> bool:
    mro = t.__mro__
    base = next((klass for klass in mro if klass in _CONTAINER_BASES), None)
    if base is None or t is base or "_fields" in vars(t):
        return False
    for klass in mro[: mro.index(base)]:
        if not klass.__flags__ & _HEAP_TYPE or "__new__" in vars(klass) and klass.__module__ != "collections":
            return False
        if klass.__module__ != "collections" and any(hook in vars(klass) for hook in _PICKLE_HOOKS):
            return False
        if klass.__module__ == "collections" and klass.__name__ != "Counter":
            return False
    # Pickle's own way runs none of the class's methods, where it has none
    # of the ones it reads and fills a container with: kept as it is.
    return any(
        klass.__module__ == "collections" or any(name in vars(klass) for name in _CONTAINER_METHODS)
        for klass in mro[: mro.index(base)]
    )


#: What storing, copying and sizing a container read or fill it with.
_CONTAINER_METHODS = (
    "__iter__",
    "__len__",
    "__getitem__",
    "__setitem__",
    "__contains__",
    "__reversed__",
    "__getattribute__",
    "__getattr__",
    "__setattr__",
    "__sizeof__",
    "items",
    "keys",
    "values",
    "append",
    "extend",
    "add",
    "update",
    "copy",
    "__copy__",
    "__deepcopy__",
)


#: ``Py_TPFLAGS_HEAPTYPE``: a class a ``class`` statement made.
_HEAP_TYPE = 1 << 9


def _reduce_container_subclass(obj: Any) -> Any:
    """*obj*, a `_plain_container_subclass`, as its builtin's items, its
    ``__dict__`` and its slots, each read by the builtin's or ``object``'s
    own methods: rebuilt by `rebuild_container_subclass`, which runs none
    of the class's methods either."""
    kind = type(obj)
    base = next(klass for klass in kind.__mro__ if klass in _CONTAINER_BASES)
    items = list(dict.items(obj)) if base is dict else list(base.__iter__(obj))
    try:
        attrs = object.__getattribute__(obj, "__dict__")
    except AttributeError:
        attrs = None
    from cash.closure_sharing import slot_readers  # imports this module

    slots = {}
    # By the class dicts' member descriptors, named as Python stores them:
    # ``__slots__ = 'tag'`` is one slot, and ``'__x'`` is ``_Priv__x``.
    for descriptor in slot_readers(kind):
        try:
            slots[descriptor.__name__] = descriptor.__get__(obj, kind)
        except AttributeError:  # an unset slot
            pass
    return (rebuild_container_subclass, (kind, base, items, dict(attrs) if attrs else None, slots or None))


def rebuild_container_subclass(kind: type, base: type, items: list, attrs: dict | None, slots: dict | None) -> Any:
    """A *kind* holding *items*, *attrs* and *slots*, made by the builtin
    *base*'s own methods (`_reduce_container_subclass`)."""
    if base in (tuple, frozenset):
        obj = base.__new__(kind, items)
    else:
        obj = base.__new__(kind)
        if base is dict:
            dict.update(obj, items)
        elif base is list:
            list.extend(obj, items)
        else:
            set.update(obj, items)
    if attrs:
        object.__getattribute__(obj, "__dict__").update(attrs)
    for name, value in (slots or {}).items():
        object.__setattr__(obj, name, value)
    return obj


def _drops_dict(t: type) -> bool:
    """Does pickling a *t* instance lose its ``__dict__``?

    When *t*'s reduce is a C type's and that C type has no instance dict of
    its own to know about: the dict is the Python subclass's. Remembered
    per type.
    """
    try:
        return _DROPS_DICT[t]
    except KeyError:
        pass
    except TypeError:  # a class whose metaclass makes it unhashable
        return _decide_drops_dict(t)
    found = _DROPS_DICT[t] = _decide_drops_dict(t)
    return found


def _decide_drops_dict(t: type) -> bool:
    if not t.__dictoffset__:
        return False
    for klass in t.__mro__:
        own = klass.__dict__
        method = own.get("__reduce_ex__") or own.get("__reduce__")
        if method is None:
            continue
        if klass is object or isinstance(method, _PYTHON_METHODS):
            return False  # object's reduce keeps the dict; a Python one is the class's own say
        return not klass.__dictoffset__
    return False


_PYTHON_METHODS = (types.FunctionType, classmethod, staticmethod)

_DROPS_DICT: weakref.WeakKeyDictionary = _weak.WeakKeyDictionary()


def _reducer_with_dict(protocol: int, unshared: bool = False) -> Any:
    """`_reduce_with_dict` for *protocol*, with the common case -- an
    instance with nothing in its ``__dict__`` -- first and in one call: a
    column of 1.9 million ``pandas.Timestamp`` went through it one at a
    time, 1.8 s on top of pickle's own 3.4 s. *unshared*: for a pickle
    without the memo (`_reduce_with_dict`)."""

    def reduce(obj: Any) -> Any:
        try:
            empty = not obj.__dict__
        except AttributeError:
            empty = True
        if empty:
            return obj.__reduce_ex__(protocol)
        return _reduce_with_dict(obj, protocol, unshared)

    reduce.with_dict_protocol = protocol  # type: ignore[attr-defined]
    return reduce


def _reduce_with_dict(obj: Any, protocol: int, unshared: bool = False) -> Any:
    """*obj*'s own reduce, with its ``__dict__`` restored after its state.

    *unshared*: for a pickle without the memo, which is only ever hashed
    (`pickle_unshared`, `content_dumps`). There the state is written as it
    is, with no state setter: a setter takes the object written again, and
    with no memo to write it as a back-reference pickle reduces it again,
    without end -- a crash, not a RecursionError, under CPython's C pickler
    (a ``datetime`` in a frame's list, its ``tzinfo`` a class of the
    user's)."""
    reduced = obj.__reduce_ex__(protocol)
    attrs = getattr(obj, "__dict__", None)
    if not attrs or isinstance(reduced, str):
        return reduced
    rebuild, args, *rest = reduced
    rest += [None] * (4 - len(rest))
    state, listitems, dictitems, setter = rest[:4]
    if unshared:
        return rebuild, args, (state, setter, dict(attrs)), listitems, dictitems
    return rebuild, args, (state, setter, dict(attrs)), listitems, dictitems, restore_with_dict


#: How many times a pickle of cash's has run a reducer of the user's.
_USER_STATE_RUNS = [0]
#: The `_USER_STATE_RUNS` of the last run of each class's reducer.
_USER_STATE_TYPES: weakref.WeakKeyDictionary = _weak.WeakKeyDictionary()
#: The `_USER_STATE_RUNS` of the last run of one for a class not in it.
_USER_STATE_UNNAMED = [0]


def user_state_runs() -> int:
    """How many times cash's pickles have run a ``__reduce__``,
    ``__reduce_ex__``, ``__getstate__`` or ``__getnewargs__`` of the user's,
    or a reducer of theirs registered with ``copyreg``: code that may have
    changed what cash read of other objects before it pickled."""
    return _USER_STATE_RUNS[0]


def _counted(obj: Any, reducer: Any) -> Any:
    _refuse_writer(type(obj))
    _note_user_state_run(type(obj))
    return reducer(obj)


class ReducerNotRun(TypeError):
    """A reducer of the user's cash does not run: it writes what the
    notebook holds (`set_reducer_veto`). A TypeError, as pickle raises for
    an object it cannot pickle, so every caller that falls back for one
    falls back for this."""


#: Set by the notebook (`reducer_writes.watch`): whether a class's reducer
#: of the user's must not run in a pickle of cash's.
_REDUCER_VETO: list[Any] = [None, None]


def set_reducer_veto(veto: Any, reason: Any = None) -> None:
    """Ask *veto* (a class -> bool) before a pickle of cash's runs a
    reducer of the user's; one it answers True for is not run. *reason*
    (a class -> str), when given, says why in the error."""
    _REDUCER_VETO[0] = veto
    _REDUCER_VETO[1] = reason


#: How deep this thread is in a call of a ``@cash.cache`` function
#: (`reducers_run`): its pickles run every reducer, vetoed or not.
_IN_A_CACHED_CALL = threading.local()


@contextlib.contextmanager
def reducers_run() -> Iterator[None]:
    """Run every reducer of the user's in the pickles of cash's made here,
    as outside a notebook: the veto (`set_reducer_veto`) is the notebook's
    rule for its statements, and a ``@cash.cache`` function always stores
    its result."""
    depth = getattr(_IN_A_CACHED_CALL, "depth", 0)
    _IN_A_CACHED_CALL.depth = depth + 1
    try:
        yield
    finally:
        _IN_A_CACHED_CALL.depth = depth


def in_reducers_run() -> bool:
    """Whether this thread is in `reducers_run`: a write it hands to a
    background thread runs there the same way."""
    return bool(getattr(_IN_A_CACHED_CALL, "depth", 0))


def _refuse_writer(t: type) -> None:
    veto, reason = _REDUCER_VETO
    if veto is not None and not getattr(_IN_A_CACHED_CALL, "depth", 0) and veto(t):
        why = reason(t) if reason is not None else "it changes other values"
        raise ReducerNotRun(f"cash does not run the reducer of {t.__qualname__}: {why}")


def next_mark() -> int:
    """Move `user_state_runs` on by one, naming no class, and return it."""
    _USER_STATE_RUNS[0] += 1
    return _USER_STATE_RUNS[0]


def _note_user_state_run(t: type) -> None:
    _USER_STATE_RUNS[0] += 1
    try:
        _USER_STATE_TYPES[t] = _USER_STATE_RUNS[0]
    except TypeError:  # a class that takes no weak reference, or no hash
        _USER_STATE_UNNAMED[0] = _USER_STATE_RUNS[0]


def user_state_types_since(runs: int) -> tuple[list[type], bool]:
    """The classes whose reducers of the user's cash's pickles ran since
    `user_state_runs` was *runs*, and whether one ran that could not be
    named (a class that takes no weak reference)."""
    return [t for t, last in list(_USER_STATE_TYPES.items()) if last > runs], _USER_STATE_UNNAMED[0] > runs


def reduce_counted(obj: Any, protocol: int) -> Any:
    """``obj.__reduce_ex__(protocol)``, counted in `user_state_runs` when it
    runs code of the user's (`runs_users_state`): for reduces cash calls
    itself, outside its picklers."""
    if runs_users_state(type(obj)):
        _refuse_writer(type(obj))
        _note_user_state_run(type(obj))
    return obj.__reduce_ex__(protocol)


def runs_users_state(t: type) -> bool:
    """Does pickling an instance of *t* run code of the user's: their own
    ``__getstate__``, ``__reduce__``, ``__reduce_ex__``, ``__getnewargs__``
    or ``__getnewargs_ex__``, or a reducer of theirs registered with
    ``copyreg``? Remembered per type, but for a ``copyreg`` registration,
    which may come after the type was first asked about."""
    registered = copyreg.dispatch_table.get(t)
    if registered is not None:
        return users_function(registered)
    try:
        return _RUNS_USERS_STATE[t]
    except KeyError:
        pass
    except TypeError:  # a class whose metaclass makes it unhashable
        return _decide_runs_users_state(t)
    found = _RUNS_USERS_STATE[t] = _decide_runs_users_state(t)
    return found


def _decide_runs_users_state(t: type) -> bool:
    return any(
        users_function(method)
        for klass in t.__mro__
        if klass is not object
        for name in (*_STATE_METHODS, *_NEWARGS_METHODS)
        if isinstance(method := klass.__dict__.get(name), _PYTHON_METHODS)
    )


def users_function(function: Any) -> bool:
    """Is *function* code of the user's: of a module of theirs, or, when
    its module is not loaded, of a file of theirs?"""
    from .install_paths import is_user_code_file, is_user_code_module

    function = getattr(function, "__func__", function)
    module = sys.modules.get(getattr(function, "__module__", None) or "")
    if module is not None:
        return is_user_code_module(module)
    code = getattr(function, "__code__", None)
    return code is not None and is_user_code_file(code.co_filename)


_NEWARGS_METHODS = ("__getnewargs__", "__getnewargs_ex__")

_RUNS_USERS_STATE: weakref.WeakKeyDictionary = _weak.WeakKeyDictionary()


def chooses_its_state(t: type) -> bool:
    """Is *t* a class of the user's whose own Python ``__getstate__``,
    ``__reduce__`` or ``__reduce_ex__`` decides what pickle stores?

    Such a class may leave a setting out of its saved state (a precision, a
    device, a threshold) that its methods still read, so a key of what it
    pickles alone served one setting the other's result. A library's class
    is left to say what its state is. Remembered per type.
    """
    try:
        return _CHOOSES_STATE[t]
    except KeyError:
        pass
    except TypeError:  # a class whose metaclass makes it unhashable
        return _decide_chooses_state(t)
    found = _CHOOSES_STATE[t] = _decide_chooses_state(t)
    return found


def _decide_chooses_state(t: type) -> bool:
    for klass in t.__mro__:
        if klass is object:
            return False
        own = klass.__dict__
        if any(isinstance(own.get(name), _PYTHON_METHODS) for name in _STATE_METHODS):
            from .install_paths import is_user_code_module

            return is_user_code_module(sys.modules.get(klass.__module__))
    return False


_STATE_METHODS = ("__getstate__", "__reduce__", "__reduce_ex__")

_CHOOSES_STATE: weakref.WeakKeyDictionary = _weak.WeakKeyDictionary()


def _is_users_class(t: type) -> bool:
    from .install_paths import is_user_code_module

    return is_user_code_module(sys.modules.get(t.__module__))


def _reduce_for_key(obj: Any, protocol: int, registered: Any = None) -> Any:
    """*obj*'s own reduce (or the one *registered* for its type with
    ``copyreg.pickle``), with the instance attributes its state leaves out
    beside it (`chooses_its_state`). Only hashed, never loaded."""
    reduced = obj.__reduce_ex__(protocol) if registered is None else registered(obj)
    if isinstance(reduced, str):
        # By its name, as a module-level singleton is: nothing of its state.
        left_out = left_out_attrs(obj, None)
        if not left_out:
            return reduced
        return _by_name_reduce(obj, reduced, ("__cash_left_out__", None, left_out))
    rebuild, args, *rest = reduced
    rest += [None] * (4 - len(rest))
    left_out = left_out_attrs(obj, rest[0])
    if not left_out:
        return reduced
    return rebuild, args, ("__cash_left_out__", rest[0], left_out), *rest[1:4]


def may_stand_by_reference(t: type) -> bool:
    """Can an instance of *t* pickle by reference -- by its name, or by a
    call that looks it up (`stands_by_reference`) -- while it holds
    attributes of its own? Its class has a reduce of its own and its
    instances an ``__dict__`` or ``__slots__``. Remembered per type.

    Most classes cannot: the ones that pickle their state, the ones whose
    instances hold no attributes (a numpy ufunc, a bound method), and the
    compiled ones whose reduce rebuilds from numbers (`_REDUCES_BY_VALUE`:
    a pandas Timestamp) are reduced as before, with no Python step per
    instance. A function and a class are pickled by name before a reducer
    is asked (their code and their attributes are read where they are
    used); a module never pickles."""
    try:
        return _MAY_STAND_BY_REFERENCE[t]
    except KeyError:
        pass
    except TypeError:  # a class whose metaclass makes it unhashable
        return _decide_may_stand_by_reference(t)
    found = _MAY_STAND_BY_REFERENCE[t] = _decide_may_stand_by_reference(t)
    return found


def _decide_may_stand_by_reference(t: type) -> bool:
    if not t.__dictoffset__ and not any(klass.__dict__.get("__slots__") for klass in t.__mro__):
        return False
    for klass in t.__mro__:
        own = klass.__dict__
        if "__reduce_ex__" in own or "__reduce__" in own:
            return klass is not object and (klass.__module__, klass.__qualname__) not in _REDUCES_BY_VALUE
    return False


#: Compiled classes whose reduce rebuilds the object from numbers and
#: objects it holds, never from a name: a pandas Timestamp pickles as
#: ``_unpickle_timestamp(<nanoseconds>, None, <tzinfo or None>, <unit>)``, a
#: Timedelta as ``_timedelta_unpickle(<nanoseconds>, <unit>)``. A subclass
#: with a reduce of its own is decided by that.
_REDUCES_BY_VALUE = frozenset(
    {
        ("pandas._libs.tslibs.timestamps", "_Timestamp"),
        ("pandas", "Timedelta"),
        # pandas 2 names its Timedelta for the module that defines it.
        ("pandas._libs.tslibs.timedeltas", "Timedelta"),
    }
)


_MAY_STAND_BY_REFERENCE: weakref.WeakKeyDictionary = _weak.WeakKeyDictionary()

#: What a reduce rebuilds an object from its class and its state with: by
#: value, never a lookup.
_REBUILDS_BY_VALUE = (copyreg.__newobj__, copyreg.__newobj_ex__, copyreg._reconstructor)  # type: ignore[attr-defined]


def stands_by_reference(reduced: Any) -> bool:
    """Does *reduced*, an object's reduce value, stand for it by reference:
    its name (a string), or a call that is not its class and is given no
    state (``getLogger(name)``), so loading looks the object up?"""
    if isinstance(reduced, str):
        return True
    if type(reduced) is not tuple or len(reduced) < 2:
        return False
    rebuild = reduced[0]
    if len(reduced) > 2 and reduced[2] is not None:
        return False
    newobj, newobj_ex, reconstructor = _REBUILDS_BY_VALUE
    return (
        not isinstance(rebuild, type)
        and rebuild is not newobj
        and rebuild is not newobj_ex
        and rebuild is not reconstructor
    )


def _reduce_by_reference_for_key(obj: Any, reducer: Any) -> Any:
    """*reducer*'s reduce of *obj*, with the attributes *obj* holds as its
    state when the reduce stands for it by reference (`stands_by_reference`).
    Only hashed, never loaded.

    A logger pickles as ``getLogger(name)``: its level, its handlers and its
    parent are what the name finds then. Keyed by its name alone, a value
    holding one digested the same after ``log.setLevel(10)``: the call was
    taken to change nothing, and a reader of the logger was served the
    level it read before. Inside `folding_by_reference` its attributes are
    keyed as a pickle by value would hold them, as `left_out_attrs` keys
    what a class of the user's leaves out: one that cannot be pickled (a
    handler holding a lock and a stream) by its type. A cache it fills as
    it is read (a logger's levels found enabled) is part of what it holds.

    Elsewhere -- a variable's content digest, which every "did it change?"
    check of the notebook compares -- it is keyed by its name, as pickle
    stores it, and noted for `met_by_reference`: the name finds the object
    from anywhere, so it changes through calls that name no variable
    holding it (``logging.getLogger('app').setLevel(10)``), and a statement
    reading it is checked by its state instead (`reference_states`). Keyed
    there with its state, a call that only logged changed the model holding
    the logger, and everything read from the model ran again on every run."""
    reduced = reducer(obj)
    if not stands_by_reference(reduced):
        return reduced
    _note_met(obj, reduced)
    if not _FOLDING.get():
        return reduced
    attrs = left_out_attrs(obj, None, own_only=True)
    if type(obj) in _STDLIB_LOGGERS:
        attrs = logger_state(obj, attrs)
    elif "_cache" in attrs and _pandas_index(type(obj)):
        attrs = {name: value for name, value in attrs.items() if name != "_cache"}
    if not attrs:
        return reduced
    if isinstance(reduced, str):
        return _by_name_reduce(obj, reduced, attrs)
    return (reduced[0], reduced[1], attrs, *reduced[3:])


def _pandas_index(t: type) -> bool:
    """Is *t* a pandas Index (or a subclass)? Its reduce rebuilds it from
    its labels and name through a function, which reads as standing by
    reference, so its attributes are keyed beside it: but for ``_cache``,
    pandas' memo of what it computed from the labels (its lookup engine,
    filled by the first ``df["x"]``), not state. Keyed, a lookup changed
    the key of an object holding the frame on pandas 2, and a call reading
    a column was taken to change its argument and never stored."""
    pd = sys.modules.get("pandas")
    index = getattr(pd, "Index", None) if pd is not None else None
    return isinstance(index, type) and issubclass(t, index)


def _own_attrs(obj: Any) -> dict:
    """*obj*'s attributes (`_instance_attrs`) that are its own state: all
    but an object its class holds under the same name -- the very object,
    by identity, not one equal to it.

    Every logger holds its class's one manager, which holds every logger in
    the process: keyed with it, a value holding a logger changed whenever
    any library made a logger, and keying it nested a pickle per logger
    until the stack ran out. What a class holds is the class's state, read
    where the class is.

    Nor the ``_cache`` dict pandas' ``cache_readonly`` properties fill as
    they are read (`_fills_cache_readonly`): a pandas 2 Index notes its
    lookup engine there on the first column read, and keyed with it, a
    value holding a frame changed whenever a call read a column of it."""
    attrs = _instance_attrs(obj)
    if not attrs:
        return attrs
    t = type(obj)
    own: dict = {}
    for name, value in attrs.items():
        if any(klass.__dict__.get(name, _NOT_HELD) is value for klass in t.__mro__):
            continue
        if name == "_cache" and type(value) is dict and _fills_cache_readonly(t):
            continue
        if type(value) is types.FunctionType and value.__dict__.get(_INSTRUMENTATION) is True:
            # Cash's own, set on the object while a statement runs and
            # taken off after (`mark_instrumentation`): not its state.
            continue
        own[name] = value
    return own


_NOT_HELD = object()


def _fills_cache_readonly(t: type) -> bool:
    """Does *t* have a pandas ``cache_readonly`` property? Each one keeps
    what it computed from the object, which pandas does not change in
    place (an Index), in the object's ``_cache`` dict: derived from the
    rest, never state of its own. Known by the descriptor, not by the
    attribute's name. Remembered per type."""
    try:
        return _FILLS_CACHE_READONLY[t]
    except KeyError:
        pass
    except TypeError:  # a class whose metaclass makes it unhashable
        return _decide_fills_cache_readonly(t)
    found = _FILLS_CACHE_READONLY[t] = _decide_fills_cache_readonly(t)
    return found


def _decide_fills_cache_readonly(t: type) -> bool:
    for klass in t.__mro__:
        for value in klass.__dict__.values():
            kind = type(value)
            if kind.__name__ == "CachedProperty" and kind.__module__ == "pandas._libs.properties":
                return True
    return False


_FILLS_CACHE_READONLY: weakref.WeakKeyDictionary = _weak.WeakKeyDictionary()

#: The attribute `mark_instrumentation` sets on a function of cash's own.
_INSTRUMENTATION = "__cash_instrumentation__"


def mark_instrumentation(fn: types.FunctionType) -> types.FunctionType:
    """Mark *fn*, a function cash sets on a user's object while a statement
    runs (a memo's ``cache_clear`` that records the call), as cash's own:
    left out of the object's state (`_own_attrs`), which a run compares
    before and after with it put back."""
    fn.__dict__[_INSTRUMENTATION] = True
    return fn


#: Whether a key's pickle reduces an object pickled by reference with its
#: state (`folding_by_reference`).
_FOLDING: contextvars.ContextVar[bool] = contextvars.ContextVar("cash_folding_by_reference", default=False)


@contextlib.contextmanager
def folding_by_reference() -> Iterator[None]:
    """Within it, a key's pickle of an object pickled by reference (a
    logger) holds the attributes it holds (`_reduce_by_reference_for_key`):
    for an argument of a cached function, and for the state a lookup of a
    statement compares (`state_digest`)."""
    token = _FOLDING.set(True)
    try:
        yield
    finally:
        _FOLDING.reset(token)


def state_digest(obj: Any) -> str:
    """The digest of *obj*'s keyed pickle with the state of each object
    pickled by reference in it (`folding_by_reference`). Raises what
    pickling raises.

    Taken in a keying of its own, whatever keying is open around it: one
    open there may be telling *obj* what it leaves out, and answers it
    without its attributes, so the digest a statement took as it ran
    differed from the one taken after it for an object nothing changed."""
    outer = _keying()
    _KEYING.now = None
    try:
        with folding_by_reference(), one_keying():
            return hashlib.sha256(dumps(obj, keyed=True)).hexdigest()
    finally:
        _KEYING.now = outer


class _Met(threading.local):
    """`met_by_reference`'s list, on this thread."""

    found: list | None = None


_MET = _Met()


def met_by_reference() -> list | None:
    """The list `_note_met` adds to on this thread, or None when no one
    collects (`value_hash.compute_hash` sets it)."""
    return _MET.found


def collect_met(found: list | None) -> list | None:
    """Make *found* the list `_note_met` adds to on this thread; the one
    before is returned, to set back."""
    before = _MET.found
    _MET.found = found
    return before


def _note_met(obj: Any, reduced: Any) -> None:
    """Note *obj*, reduced by reference to *reduced*, for the collector
    (`met_by_reference`): its recipe (`reference_recipe`) and the object,
    when the reduce looks it up (`looks_up`)."""
    found = _MET.found
    if found is not None and looks_up(reduced):
        found.append((reduced, obj))
        note_kind_by_reference(type(obj), reduced)


#: The classes an instance of which was seen pickled by reference to a
#: lookup by a name (`note_kind_by_reference`).
_KINDS_BY_REFERENCE: set[type] = set()


def note_kind_by_reference(kind: type, reduced: Any) -> None:
    """An instance of *kind* was seen pickled by reference to *reduced*.
    When that finds it by a name (a logger's ``getLogger('app')``, not a
    timestamp's ``_unpickle_timestamp(<numbers>)``, which makes it again
    from what it holds), `seen_by_reference` takes one of them for such
    without reducing it."""
    if kind in _KINDS_BY_REFERENCE:
        return
    if type(reduced) is str or (
        type(reduced) is tuple
        and len(reduced) >= 2
        and type(reduced[1]) is tuple
        and any(type(arg) is str for arg in reduced[1])
    ):
        _KINDS_BY_REFERENCE.add(kind)


def seen_by_reference(kind: type) -> bool:
    """Was an instance of *kind* seen pickled by reference to a lookup by a
    name? A set lookup: no code of the object's runs."""
    return kind in _KINDS_BY_REFERENCE


def looks_up(reduced: Any) -> bool:
    """May *reduced*, a reduce that stands by reference
    (`stands_by_reference`), find its object by a name: a string, or a call
    given a name among names and numbers (``getLogger('app')``), or given
    nothing (``getLogger()``)? A call given bytes or a container is given
    content (a tensor's storage pickles as ``_load_from_bytes(data)``), and
    one given numbers alone the value (a timestamp's
    ``_unpickle_timestamp(<nanoseconds>, ...)``): each builds an object
    from it. Whether the call finds this very object is asked of its recipe
    (`finds_itself`)."""
    if isinstance(reduced, str):
        return True
    args = reduced[1]
    if type(args) is not tuple:
        return False
    named = not args
    for arg in args:
        kind = type(arg)
        if kind is str:
            named = True
        elif kind is tuple:
            # Nested: the general reading.
            return _all_atoms(args) and _names(args)
        elif arg is not None and kind not in _NUMBERS:
            return False
    return named


#: The other atoms a lookup's call may be given (`looks_up`).
_NUMBERS = frozenset({int, float, bool})


def _all_atoms(value: Any) -> bool:
    if type(value) is tuple:
        return all(_all_atoms(item) for item in value)
    return value is None or type(value) in (str, int, float, bool)


def _names(value: Any) -> bool:
    if type(value) is tuple:
        return any(_names(item) for item in value)
    return type(value) is str


def finds_itself(obj: Any, recipe: bytes) -> bool:
    """Does *recipe* (`reference_recipe`) find *obj* itself, as a lookup
    does, rather than build another object? Only then is the state a
    pickle of it leaves out the state of the object every holder of it
    shares, and the object found later by the recipe the one it was."""
    try:
        return find_by_recipe(recipe) is obj
    except Exception:  # noqa: BLE001 - whatever loading the recipe raises: not found
        return False


def reference_recipe(obj: Any, reduced: Any) -> bytes | None:
    """The bytes that find *obj* again from its reduce value *reduced*, as
    loading a pickle of it does (`find_by_recipe`); None when they cannot
    be pickled."""
    recipe = (by_name, (pickle.whichmodule(obj, reduced), reduced)) if isinstance(reduced, str) else reduced[:2]
    try:
        return pickle.dumps(recipe, protocol=4)
    except Exception:  # noqa: BLE001 - whatever stops this pickle: no recipe
        return None


def find_by_recipe(recipe: bytes) -> Any:
    """The object *recipe* (`reference_recipe`) finds now: what a pickle of
    it would load. Raises what loading raises."""
    rebuild, args = pickle.loads(recipe)
    return rebuild(*args)


#: The standard library's own logger classes, by identity: a subclass
#: defined elsewhere is keyed by all its attributes.
_STDLIB_LOGGERS = (logging.Logger, logging.RootLogger)

#: What a standard library logger's output follows: its own attributes
#: below, and its manager's ``disable`` (`logger_state`). ``_cache``, the
#: levels found enabled, is left out: ``isEnabledFor`` fills it from these
#: and every ``setLevel`` or ``logging.disable`` clears it, so a
#: ``log.info(...)`` changes no state. A Python adding an attribute fails
#: ``test_the_logger_s_attributes_are_all_described``.
LOGGER_STATE = ("name", "level", "parent", "propagate", "handlers", "disabled", "filters")
LOGGER_DERIVED = ("_cache",)


def logger_state(logger: logging.Logger, attrs: dict) -> dict:
    """The state of *logger*, a standard library one, from *attrs*, its
    attributes as a key holds them: those in `LOGGER_STATE`, and its
    manager's ``disable``. A standard library handler is keyed by the
    level, formatter, filters and stream or file it holds
    (`_handler_state`)."""
    state = {name: attrs[name] for name in LOGGER_STATE if name in attrs}
    if "handlers" in state:
        state["handlers"] = [_handler_state(handler) for handler in logger.handlers]
    state["manager.disable"] = logger.manager.disable
    return state


def _stdlib_logging(obj: Any) -> bool:
    """Is *obj* of a class the standard library's ``logging`` package
    defines, itself, not a subclass of one?"""
    t = type(obj)
    module = sys.modules.get(t.__module__) if t.__module__ in ("logging", "logging.handlers") else None
    return module is not None and vars(module).get(t.__qualname__) is t


def _handler_state(handler: Any) -> Any:
    """What a handler's output follows: its attributes, its formatter's
    (`_formatter_state`), and the stream it writes to (`_stream_key`), not
    its lock or what the stream holds. A standard library handler's
    attributes as they are. One of a subclass of ``logging.Handler``
    defined elsewhere: its filters as they are and the rest as `_held_key`
    keys them, so one set to write elsewhere is keyed otherwise and one
    gathering what it handles (pytest's ``LogCaptureHandler``) is not
    rekeyed by each record. Anything else in a logger's handlers by its
    type."""
    if logging.Handler not in type(handler).__mro__:
        return type(handler)
    own = not _stdlib_logging(handler)
    state: dict[str, Any] = {"__class__": type(handler)}
    for name, value in _instance_attrs(handler).items():
        if name in ("lock", "_builtin_open"):
            continue
        if name == "formatter":
            value = _formatter_state(value)
        elif name == "stream":
            value = _stream_key(value)
        elif own and name != "filters":
            value = _held_key(value)
        state[name] = value
    return state


def _held_key(value: Any) -> Any:
    """What a handler or formatter of a class defined elsewhere holds:
    a plain value as it is; code (a function, a bound method) by its type's
    name, as what a handler does with a record does not change a result;
    any other object as that very object (`_stream_key`)."""
    if type(value) in _HANDLER_VALUES:
        return value
    if callable(value):
        kind = type(value)
        return (kind.__module__, kind.__qualname__)
    return _stream_key(value)


#: What a handler of a class defined elsewhere holds that is keyed by value.
_HANDLER_VALUES = (str, bytes, int, float, complex, bool, type(None))


#: Streams whose ``name`` is read in C: the file a handler writes to.
_NAMED_STREAMS = (io.TextIOWrapper, io.BufferedWriter, io.FileIO)


def _stream_key(stream: Any) -> Any:
    """A stream a handler writes to, or another object it holds: a file by
    its type and name; any other (an ``io.StringIO``, a list) as the very
    object, by a token of its own in this session (`_STREAM_TOKENS`), so a
    handler set to write elsewhere is keyed otherwise. A token is new in each session: an entry keyed on one is
    not served after a restart."""
    if stream is None:
        return None
    if type(stream) in _NAMED_STREAMS:
        name = stream.name
        if type(name) is str:
            return (type(stream), name)
    key = id(stream)
    known = _STREAM_TOKENS.get(key)
    if known is not None and known[0]() is stream:
        return (type(stream), known[1])
    token = uuid.uuid4().hex
    try:
        held = _weak.ref(stream, lambda _gone, key=key: _STREAM_TOKENS.pop(key, None))
    except TypeError:  # one that takes no weak reference: held, until the session ends
        held = functools.partial(operator.itemgetter(0), (stream,))
    _STREAM_TOKENS[key] = (held, token)
    return (type(stream), token)


#: ``id(stream) -> (a call giving the stream, token)`` (`_stream_key`): held
#: weakly where it can be, and emptied with the session (`forget_streams`).
_STREAM_TOKENS: dict[int, tuple[Any, str]] = {}


def buffering_handlers(logger: Any) -> list[str]:
    """The handlers a standard library *logger*'s records reach that hold
    each record they handle (a ``MemoryHandler``'s buffer, a
    ``QueueHandler``'s queue), by their class's name: keyed with it, every
    record logged changes the logger's state."""
    found: list[str] = []
    seen: set[int] = set()
    while type(logger) in _STDLIB_LOGGERS and id(logger) not in seen:
        seen.add(id(logger))
        for handler in _instance_attrs(logger).get("handlers") or ():
            if logging.Handler in type(handler).__mro__ and {"buffer", "queue"} & _instance_attrs(handler).keys():
                found.append(type(handler).__name__)
        if not _instance_attrs(logger).get("propagate"):
            break
        logger = _instance_attrs(logger).get("parent")
    return found


def forget_streams() -> None:
    """Forget the tokens of the streams handlers wrote to, as a fresh kernel
    would."""
    _STREAM_TOKENS.clear()


def _formatter_state(formatter: Any) -> Any:
    """A standard library formatter by its format, date format and style.
    One of a class defined elsewhere (pytest's, colorlog's): what it holds
    as `_held_key` keys it, as a handler of one is (`_handler_state`)."""
    if formatter is None:
        return None
    own = not _stdlib_logging(formatter)
    if own and logging.Formatter not in type(formatter).__mro__:
        return formatter
    style = vars(formatter).get("_style")
    defaults = vars(style).get("_defaults") if _stdlib_logging(style) else style
    if not own:
        return (type(formatter), vars(formatter).get("_fmt"), vars(formatter).get("datefmt"), type(style), defaults)
    state: dict[str, Any] = {"__class__": type(formatter)}
    for name, value in _instance_attrs(formatter).items():
        if name == "_style" and _stdlib_logging(style):
            value = (type(style), defaults)
        else:
            value = _held_key(value)
        state[name] = value
    return state


def _by_name_reduce(obj: Any, name: str, state: Any) -> tuple:
    """A reduce value naming *obj* as pickle does for the *name* its reduce
    gave, with *state* beside it: keyed, never loaded."""
    module = pickle.whichmodule(obj, name)
    return by_name, (module, name), state


class _Keying:
    """What one keying has told (`one_keying`)."""

    __slots__ = ("left_out", "picklable")

    def __init__(self) -> None:
        #: ``id(obj)`` -> ``(obj, state, left_out_attrs(obj, state))``, or
        #: ``(obj, _TELLING, None)`` while its attributes are told.
        self.left_out: dict[int, tuple[Any, Any, dict | None]] = {}
        #: ``id(value)`` -> ``(value, whether it pickles)`` (`_picklable`).
        self.picklable: dict[int, tuple[Any, bool]] = {}


#: The keying this thread is in, if any. Per thread: another thread keying
#: at the same time has objects of its own being told.
_KEYING = threading.local()

#: In `_Keying.left_out`, the state of an object whose left-out attributes
#: are being told.
_TELLING = object()


def _keying() -> _Keying | None:
    return getattr(_KEYING, "now", None)


@contextlib.contextmanager
def one_keying() -> Iterator[None]:
    """Work out each object's `left_out_attrs` at most once inside the
    block, for one value being pickled or keyed.

    Telling whether a left-out attribute can be pickled pickles it, and an
    attribute that is itself an object of such a class has its own left-out
    attributes pickled to tell, and is pickled again as part of the state:
    a chain of them was pickled 2**depth times (16 deep, a second; 30 deep,
    hours). Remembered, each object is told once.

    What is remembered is about the object alone, never about where it was
    met: the answer for a value is the same however deep the stack is and
    whatever was keyed before it in the block."""
    if _keying() is not None:
        yield
        return
    _KEYING.now = _Keying()
    try:
        yield
    finally:
        _KEYING.now = None


def left_out_attrs(obj: Any, state: Any, *, own_only: bool = False) -> dict:
    """The attributes of *obj* -- in its ``__dict__`` or its ``__slots__``
    -- that *state*, what its reduce saves, does not hold: by name, or
    under their name with another value (a ``__getstate__`` that saves a
    portable default in place of the live setting).

    *own_only*: only those `_own_attrs` gives, for an object pickled by
    reference (`_reduce_by_reference_for_key`).

    One that cannot be pickled (a lock or a connection the class leaves out
    for that reason) is given as its type: its content is nothing pickle can
    read. Whether one pickles does not depend on what any object leaves
    out, as that is pickled or given as its type: an attribute that leads
    back to *obj* pickles when the rest of it does, as pickle stores a cycle.
    """
    keying = _keying()
    if keying is None:
        with one_keying():
            return left_out_attrs(obj, state, own_only=own_only)
    # Inline below: a chain of such objects nests a pickle per level, and
    # each frame here counts against the recursion limit.
    memo = keying.left_out
    known = memo.get(id(obj))
    if known is not None and known[0] is obj:
        if known[1] is _TELLING:
            # Met again inside a pickle telling whether one of its own
            # attributes pickles. What it leaves out cannot decide that (see
            # above), and telling it again would recurse without end, until
            # a stack -- however deep it already was -- ran out.
            return {}
        if known[1] is state:
            return known[2]  # type: ignore[return-value]
    attrs = _own_attrs(obj) if own_only else _instance_attrs(obj)
    if not attrs:
        return {}
    memo[id(obj)] = (obj, _TELLING, None)
    saved: dict = {}
    if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
        if isinstance(state[0], dict):
            saved.update(state[0])
        saved.update(state[1])  # (dict, slots)
    elif isinstance(state, dict):
        saved = state
    left_out = {}
    try:
        for name, value in attrs.items():
            if name in saved and _same(saved[name], value):
                continue
            if not _picklable(value):
                value = ("__cash_unpicklable__", f"{type(value).__module__}.{type(value).__qualname__}")
            left_out[name] = value
    except BaseException:
        memo.pop(id(obj), None)
        raise
    # Remembered for this state object only: a reduce that makes its state
    # anew is told again, cheaply, as whether its attributes pickle is
    # remembered (`_picklable`). An equal state is not taken for it:
    # ``1 == True``.
    memo[id(obj)] = (obj, state, left_out)
    return left_out


def _picklable(value: Any) -> bool:
    """Does *value* pickle as a key pickles it? Inside `one_keying` each
    object is asked once, and what was found to pickle is not pickled again
    to tell whether what holds it does: the bytes are thrown away, so it
    stands in by a persistent id. Each object is then pickled once to tell,
    however deep the attributes it leaves out nest."""
    keying = _keying()
    memo = keying.picklable if keying is not None else None
    if memo is not None:
        known = memo.get(id(value))
        if known is not None and known[0] is value:
            return known[1]

    def told(obj: Any) -> Any:
        if obj is value or memo is None:
            return None
        known = memo.get(id(obj))
        return ("__cash_picklable__", id(obj)) if known is not None and known[0] is obj and known[1] else None

    try:
        if FLAT_FIRST:
            dumps(value, keyed=True, persistent_id=told)
        else:  # `dumps` itself, two frames fewer (see `left_out_attrs`)
            _dumps(value, pickle.DEFAULT_PROTOCOL, False, None, None, told, True)
        ok = True
    except (RecursionError, MemoryError) as exc:
        if not too_deep(exc):
            ok = False
        else:
            # No answer about *value*: where the stack ran out depends on how
            # deep it was when asked. Taken for one, the value would key one
            # way from here and another from a deeper call; the keying as a
            # whole is too deep instead, as a pickle of it would be.
            raise
    except Exception:  # noqa: BLE001 - whatever pickle refuses
        ok = False
    if memo is not None:
        memo[id(value)] = (value, ok)
    return ok


def _instance_attrs(obj: Any) -> dict:
    """*obj*'s ``__dict__`` and the ``__slots__`` it has set."""
    attrs = dict(getattr(obj, "__dict__", None) or ())
    for klass in type(obj).__mro__:
        slots = klass.__dict__.get("__slots__")
        if not slots:
            continue
        for name in (slots,) if isinstance(slots, str) else slots:
            if name in ("__dict__", "__weakref__"):
                continue
            if name.startswith("__") and not name.endswith("__"):
                name = f"_{klass.__name__.lstrip('_')}{name}"  # name-mangled slot
            if name in attrs:
                continue
            try:
                attrs[name] = getattr(obj, name)
            except AttributeError:  # a slot never set
                continue
    return attrs


def _same(saved: Any, value: Any) -> bool:
    """Is *saved*, the state's value under an attribute's name, the
    attribute's own *value*? Of one type and equal; when equality gives no
    plain answer (an array), the attribute is keyed beside the state."""
    if saved is value:
        return True
    if type(saved) is not type(value):
        return False
    try:
        return bool(saved == value)
    except Exception:  # noqa: BLE001 - an ambiguous or failing comparison
        return False


def restore_with_dict(obj: Any, packed: tuple) -> None:
    """Set *obj*'s state as pickle would, then its ``__dict__``
    (`_reduce_with_dict`). Named in stored entries: keep its name."""
    state, setter, attrs = packed
    if setter is not None:
        setter(obj, state)
    elif state is not None:
        setstate = getattr(obj, "__setstate__", None)
        if setstate is not None:
            setstate(state)
        else:
            slots = None
            if isinstance(state, tuple) and len(state) == 2:
                state, slots = state
            if state:
                obj.__dict__.update(state)
            for name, value in (slots or {}).items():
                setattr(obj, name, value)
    obj.__dict__.update(attrs)


# -- functions defined again --------------------------------------------------


def named_now(obj: Any) -> Any:
    """The class or function *obj*'s module now names where *obj* was
    defined, when that is another object of the same kind; else None.

    What a pickle round trip from disk resolves the name to. A notebook cell
    re-run defines a new class under the old name, and pickle refuses an
    instance of the old one ("not the same object as __main__.Fit").
    """
    module = sys.modules.get(getattr(obj, "__module__", None) or "")
    qualname = getattr(obj, "__qualname__", None)
    if module is None or not isinstance(qualname, str) or "<locals>" in qualname:
        return None
    current: Any = module
    for part in qualname.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    if current is obj:
        return None
    if isinstance(obj, type) and isinstance(current, type):
        return current
    if type(obj) is types.FunctionType and type(current) is types.FunctionType:
        return current
    return None


def held_now(obj: Any) -> Any:
    """What a copy of a value holds in place of *obj*, a function whose
    module names another function now (`named_now`): that one when it is
    the same definition (`same_definition`), as a stored copy loads it by
    name (`defined_again`); else *obj* itself, as the value held it.

    A notebook re-running the cell defining ``column_ratio`` makes a new
    function; a pipeline restored from RAM held the first one, so pickling
    it (``joblib.dump``) failed as in no plain Python run."""
    current = named_now(obj)
    if current is None or type(obj) is not types.FunctionType:
        return obj
    return current if same_definition(obj, current) else obj


def defined_again(then: Any = None) -> Callable[[Any], Any]:
    """A pickler's ``reducer_override`` for a value that holds a function
    whose cell ran again: its module names another function now, and pickle
    refuses the one held ("not the same object as __main__.column_ratio").

    When the one named now is the same definition (`same_definition`), the
    function is pickled by its name, and loads as the one that name holds
    then, as any function pickled by name does. When it is not, pickling
    raises PicklingError saying so: the value is not stored. *then* is the
    override to ask about everything else.

    A notebook that repeats a cell defining ``column_ratio`` after building
    a pipeline on it held the first function: E6's fitted grid search was
    never stored, and ran again after every restart (~48 s).
    """

    def override(obj: Any) -> Any:
        if type(obj) is types.FunctionType:
            current = named_now(obj)
            if current is not None:
                if not same_definition(obj, current):
                    name = obj.__qualname__
                    raise DefinedAgain(
                        f"it holds {name} as defined before its cell ran again with different code, "
                        f"and a stored copy could only load the {name} defined now"
                    )
                return by_name, (obj.__module__, obj.__qualname__)
        return NotImplemented if then is None else then(obj)

    return override


class DefinedAgain(pickle.PicklingError):
    """A value holds a function whose cell ran again with other code since
    (`defined_again`): pickled by its name, it would load as the new one."""


def by_name(module: str, qualname: str) -> Any:
    """What *module* names *qualname* now, as pickle finds a function it
    pickled by name (`defined_again`). Named in stored entries: keep its
    name."""
    import importlib

    found: Any = importlib.import_module(module)
    for part in qualname.split("."):
        found = getattr(found, part)
    return found


def same_definition(
    old: types.FunctionType, new: types.FunctionType, _pending: set[tuple[int, int]] | None = None
) -> bool:
    """Do *old* and *new* do the same, so that a value holding *old* can
    hold *new* instead: the same code (lines aside), the same globals, no
    closure, and alike defaults, attributes, docstring and annotations.

    A default that is a function is alike when its module now names one of
    the same definition, which a copy loaded from disk holds in its place:
    ``def ensemble(c, tsps=(greedy, rep_2opt))`` run again after the cells
    defining ``greedy`` and ``rep_2opt`` holds the new ones, which do what
    the old ones did. *_pending*: the pairs being compared further up, taken
    as alike (functions naming each other as defaults)."""
    from cash.decorator.function_identity import code_fingerprint

    pending = set() if _pending is None else _pending
    if (id(old), id(new)) in pending:
        return True
    if old.__globals__ is not new.__globals__ or old.__closure__ is not None or new.__closure__ is not None:
        return False
    a, b = old.__code__, new.__code__
    if (a.co_argcount, a.co_posonlyargcount, a.co_kwonlyargcount, a.co_flags) != (
        b.co_argcount,
        b.co_posonlyargcount,
        b.co_kwonlyargcount,
        b.co_flags,
    ):
        return False
    pairs = [
        (old.__defaults__, new.__defaults__),
        (old.__kwdefaults__, new.__kwdefaults__),
        (old.__dict__, new.__dict__),
        (old.__doc__, new.__doc__),
        (getattr(old, "__type_params__", ()), getattr(new, "__type_params__", ())),
    ]
    annotate = getattr(old, "__annotate__", None), getattr(new, "__annotate__", None)
    if sys.version_info >= (3, 14):
        # Annotations are made when asked for: compare what makes them.
        if (annotate[0] is None) != (annotate[1] is None):
            return False
        if annotate[0] is not None and not same_definition(annotate[0], annotate[1], pending):
            return False
    else:
        pairs.append((old.__annotations__, new.__annotations__))
    if code_fingerprint(a) != code_fingerprint(b):
        return False
    pending.add((id(old), id(new)))
    try:
        return all(_alike(x, y, pending) for x, y in pairs)
    finally:
        pending.discard((id(old), id(new)))


_ALIKE_ATOMS = (str, bytes, int, float, complex, bool, type(None))


def _alike(a: Any, b: Any, pending: set[tuple[int, int]]) -> bool:
    """*a* is *b*, both are equal plain data, or *a* is a function whose
    module now names *b*, of the same definition (`same_definition`)."""
    if a is b:
        return True
    t = type(a)
    if t is not type(b):
        return False
    if t in _ALIKE_ATOMS:
        return a == b
    if t in (tuple, list):
        return len(a) == len(b) and all(_alike(x, y, pending) for x, y in zip(a, b))
    if t is dict:
        return a.keys() == b.keys() and all(_alike(v, b[k], pending) for k, v in a.items())
    if t is types.FunctionType:
        return named_now(a) is b and same_definition(a, b, pending)
    return False
