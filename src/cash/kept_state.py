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

import contextlib
import copy
import copyreg
import functools
import io
import itertools
import operator
import pickle
import sys
import threading
import types
import weakref
from collections.abc import Callable, Iterator
from typing import Any

__all__ = [
    "DefinedAgain",
    "FlatValue",
    "LoadsOtherwise",
    "by_name",
    "by_name_as_then",
    "chooses_its_state",
    "defined_again",
    "dumps",
    "flatten",
    "left_out_attrs",
    "load_flat",
    "loads_otherwise",
    "named_now",
    "one_keying",
    "reducers",
    "reduces_its_own",
    "restore_with_dict",
    "restores_alike_apart",
    "same_class",
    "same_definition",
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
    then = kwargs.get("reducer_override")

    def dump(override: Callable[[Any], Any]) -> bytes:
        if kwargs.get("on_retry") is not None:
            kwargs["on_retry"]()
        kwargs["reducer_override"] = override
        return _dumps_once(value, *args, **kwargs)

    return by_name_as_then(dump, then)


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
    aliases: bool = False,
    sink: Any = None,
    reducer_override: Any = None,
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
    that name; a value pickled flat is pickled without it."""
    if sink is not None and not keyed:
        raise ValueError("dumps: a sink takes keyed bytes only")
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
    flat = flatten(value, _KeepDict(protocol, extra, keyed), _persistent_leaf(persistent_id))
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
    whole; not for a value pickled flat."""
    differs: list[bool] = []
    table = _KeepDict(protocol, extra, watch=differs)
    if FLAT_FIRST and not _surely_shallow(value):
        walk = _Walk(getattr(sink, "walked", None), reducer_override)
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
    aliases: bool = False,
    sink: Any = None,
    reducer_override: Any = None,
) -> bytes:
    """`dumps` where the pickler would run out of C stack before its
    recursion limit (`FLAT_FIRST`): the value is walked on the heap first,
    and handed to the pickler only when it is shallow enough
    (`_SAFE_DEPTH`). Deeper, it is pickled flat; for a key, as deep as
    pickle follows elsewhere, and past that it raises RecursionError as the
    pickler raises on other Pythons."""
    reducers, leaf = _KeepDict(protocol, extra, keyed), _flat_leaf(persistent_id)
    walk = _Walk(getattr(sink, "walked", None), reducer_override)
    if _at_most_deep(value, _SAFE_DEPTH, reducers, leaf, walk):
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


def _at_most_deep(value: Any, limit: int, reducers: Any, leaf: Any, walk: _Walk | None = None) -> bool:
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
        level = _next_level(level, reducers, leaf, walk)
    return not level


#: How many objects `_at_most_deep` walks again before it leaves the value
#: to `flatten`: a few shared at every level (a class, a dtype) are not a
#: cycle.
_REWALKED_FREE = 1024


def _next_level(level: list, reducers: Any, leaf: Any, walk: _Walk | None = None) -> list:
    """What the objects of *level* hold that is not an atom."""
    kinds = list(map(type, level))
    opened = list(map(_OPENED_BY_ITEMS.__contains__, kinds))
    found: list[Any] = []
    containers = list(itertools.compress(level, opened))
    if containers:
        found += _not_atoms(list(itertools.chain.from_iterable(containers)))  # a dict's keys too
        dicts = list(itertools.compress(level, map(_DICT.__contains__, kinds)))
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
    the time a pickle takes is mostly its reduces, which here run first."""

    __slots__ = ("check", "data", "first", "reduced")

    def __init__(self, check: Callable[[int], Any] | None = None, first: Any = None) -> None:
        self.check = check
        self.data = 0
        #: The pickle's own ``reducer_override``, asked before the table as
        #: the pickler asks it.
        self.first = first
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

_LIBRARY_TYPES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _dumps(
    value: Any,
    protocol: int,
    fast: bool,
    buffer_callback: Any,
    extra: dict | None,
    persistent_id: Any,
    keyed: bool,
    aliases: bool = False,
    sink: Any = None,
    table: _KeepDict | None = None,
    override: Any = None,
    after: Any = None,
) -> bytes:
    buf = io.BytesIO() if sink is None else sink
    plain = getattr(_PLAIN, "ids", None)
    if plain:
        return _dumps_plain(
            value,
            protocol,
            fast,
            buffer_callback,
            extra,
            persistent_id,
            keyed,
            aliases,
            buf,
            sink,
            table,
            override,
            after,
            plain,
        )
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
        pickler.persistent_id = key_aliases()
    pickler.dispatch_table = _KeepDict(protocol, extra, keyed) if table is None else table
    pickler.dump(value)
    if after is not None:
        after(pickler)
    return b"" if sink is not None else buf.getvalue()


def _dumps_plain(
    value: Any,
    protocol: int,
    fast: bool,
    buffer_callback: Any,
    extra: dict | None,
    persistent_id: Any,
    keyed: bool,
    aliases: bool,
    buf: Any,
    sink: Any,
    table: _KeepDict | None,
    override: Any,
    after: Any,
    plain: dict[int, type],
) -> bytes:
    """`_dumps` with `_PlainNaming`, for `by_name_as_then`."""
    pickler = _PlainNaming(buf, protocol=protocol, buffer_callback=buffer_callback)
    pickler.override = override
    pickler.plain = plain
    pickler.fast = fast
    if persistent_id is not None:
        pickler.persistent_id = persistent_id
    elif aliases:
        pickler.fast = True
        pickler.persistent_id = key_aliases()
    pickler.dispatch_table = _TableGet(_KeepDict(protocol, extra, keyed) if table is None else table)
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


def key_aliases() -> Callable[[Any], Any]:
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
    """
    seen: dict[int, tuple[int, Any]] = {}
    atoms = _ATOMS
    by_name = _BY_NAME

    def persistent_id(obj: Any, type: Any = type, id: Any = id) -> Any:
        t = type(obj)
        if t in atoms or t in by_name or issubclass(t, type) or _identityless(t):
            return None
        found = seen.get(id(obj))
        if found is not None:
            return ("__cash_alias__", found[0])
        seen[id(obj)] = (len(seen), obj)
        return None

    return persistent_id


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


_IDENTITYLESS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


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
    return isinstance(obj, ndarray) and not obj.dtype.hasobject


def reducers(protocol: int = pickle.DEFAULT_PROTOCOL, extra: dict | None = None) -> dict:
    """Each type's reducer, as `dumps` reduces it: ``reducers(p)[type(obj)](obj)``
    is what pickling *obj* at protocol *p* saves (a class has none: it is
    pickled by name). *extra*: reducers taken first, as `dumps` takes them."""
    return _KeepDict(protocol, extra)


class _KeepDict(dict):
    """A pickler's ``dispatch_table``: each type's reducer, decided at its
    first instance. The ordinary ``__reduce_ex__`` for most, so the stream
    is the one ``pickle.dumps`` writes; `_reduce_with_dict` for a type whose
    reduce drops its instances' ``__dict__`` (`_drops_dict`)."""

    def __init__(self, protocol: int, extra: dict | None, keyed: bool = False, watch: list | None = None) -> None:
        super().__init__(extra or ())
        self._protocol = protocol
        self._keyed = keyed
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
            reducer = _names_apart(self._reducer(t), self._names)
            if _holds_dtype(t):
                reducer = _dtypes_apart(reducer, self._dtypes)
        self[t] = reducer
        return reducer

    def _reducer(self, t: type) -> Any:
        if issubclass(t, type):
            raise KeyError(t)  # a class: pickled by name, as pickle does
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
        else:
            if _drops_dict(t):
                reducer = _reducer_with_dict(self._protocol)
            else:
                reducer = operator.methodcaller("__reduce_ex__", self._protocol)
            if cached_properties(t) and (self._keyed or _leaves_views_out(t)):
                which = None if self._keyed else stored_views
                reducer = functools.partial(_reduce_without_views, reducer=reducer, which=which)
        if self._keyed and t.__name__ in _GLOBAL_RNG_TYPE_NAMES:
            # One instance of the class may be a module's global RNG.
            reducer = functools.partial(_reduce_rng_for_key, reducer=reducer)
        return reducer

    def _store_reducer(self, t: type, registered: Any) -> Any:
        """The reducer a store's pickle (not *keyed*) takes for *t*."""
        if registered is not None:
            return registered
        if _drops_dict(t):
            reducer = _reducer_with_dict(self._protocol)
        else:
            reducer = operator.methodcaller("__reduce_ex__", self._protocol)
        if cached_properties(t) and _leaves_views_out(t):
            reducer = functools.partial(_reduce_without_views, reducer=reducer, which=stored_views)
        return reducer


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

    def reduce(obj: Any) -> Any:
        reduced = reducer(obj)
        if type(reduced) is not tuple or len(reduced) < 3:
            return reduced
        state = reduced[2]
        if type(state) is dict:
            try:
                apart = {names[key]: value for key, value in state.items()}
            except (KeyError, TypeError):
                apart = _state_apart(state, names)
        elif type(state) is tuple and len(state) == 2 and all(part is None or type(part) is dict for part in state):
            apart = tuple(None if part is None else _state_apart(part, names) for part in state)
        else:
            return reduced
        return (reduced[0], reduced[1], apart, *reduced[3:])

    return reduce


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
            elif not isinstance(reduced, str) and left_out_attrs(obj, reduced[2] if len(reduced) > 2 else None):
                watch.append(True)
            return reduced

        return state
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


_CACHED_PROPERTIES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

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
    from numpy.lib import recfunctions

    packed = recfunctions.repack_fields(dtype, recurse=True)
    if packed.itemsize == dtype.itemsize:
        return array
    return array.astype(packed)


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

_DROPS_DICT: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _reducer_with_dict(protocol: int) -> Any:
    """`_reduce_with_dict` for *protocol*, with the common case -- an
    instance with nothing in its ``__dict__`` -- first and in one call: a
    column of 1.9 million ``pandas.Timestamp`` went through it one at a
    time, 1.8 s on top of pickle's own 3.4 s."""

    def reduce(obj: Any) -> Any:
        try:
            empty = not obj.__dict__
        except AttributeError:
            empty = True
        if empty:
            return obj.__reduce_ex__(protocol)
        return _reduce_with_dict(obj, protocol)

    return reduce


def _reduce_with_dict(obj: Any, protocol: int) -> Any:
    """*obj*'s own reduce, with its ``__dict__`` restored after its state."""
    reduced = obj.__reduce_ex__(protocol)
    attrs = getattr(obj, "__dict__", None)
    if not attrs or isinstance(reduced, str):
        return reduced
    rebuild, args, *rest = reduced
    rest += [None] * (4 - len(rest))
    state, listitems, dictitems, setter = rest[:4]
    return rebuild, args, (state, setter, dict(attrs)), listitems, dictitems, restore_with_dict


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

_CHOOSES_STATE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _is_users_class(t: type) -> bool:
    from .install_paths import is_user_code_module

    return is_user_code_module(sys.modules.get(t.__module__))


def _reduce_for_key(obj: Any, protocol: int, registered: Any = None) -> Any:
    """*obj*'s own reduce (or the one *registered* for its type with
    ``copyreg.pickle``), with the instance attributes its state leaves out
    beside it (`chooses_its_state`). Only hashed, never loaded."""
    reduced = obj.__reduce_ex__(protocol) if registered is None else registered(obj)
    if isinstance(reduced, str):
        return reduced
    rebuild, args, *rest = reduced
    rest += [None] * (4 - len(rest))
    left_out = left_out_attrs(obj, rest[0])
    if not left_out:
        return reduced
    return rebuild, args, ("__cash_left_out__", rest[0], left_out), *rest[1:4]


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


def left_out_attrs(obj: Any, state: Any) -> dict:
    """The attributes of *obj* -- in its ``__dict__`` or its ``__slots__``
    -- that *state*, what its reduce saves, does not hold: by name, or
    under their name with another value (a ``__getstate__`` that saves a
    portable default in place of the live setting).

    One that cannot be pickled (a lock or a connection the class leaves out
    for that reason) is given as its type: its content is nothing pickle can
    read. Whether one pickles does not depend on what any object leaves
    out, as that is pickled or given as its type: an attribute that leads
    back to *obj* pickles when the rest of it does, as pickle stores a cycle.
    """
    keying = _keying()
    if keying is None:
        with one_keying():
            return left_out_attrs(obj, state)
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
    attrs = _instance_attrs(obj)
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
    if isinstance(obj, types.FunctionType) and isinstance(current, types.FunctionType):
        return current
    return None


def defined_again(then: Any = None, classes: list[type] | None = None) -> Callable[[Any], Any]:
    """A pickler's ``reducer_override`` for a value that holds a function
    whose cell ran again: its module names another function now, and pickle
    refuses the one held ("not the same object as __main__.column_ratio").

    When the one named now is the same definition (`same_definition`), the
    function is pickled by its name, and loads as the one that name holds
    then, as any function pickled by name does. When it is not, pickling
    raises PicklingError saying so: the value is not stored. *then* is the
    override to ask about everything else.

    A class is pickled by its name the same way when the one named now is
    the same definition (`same_class`): an instance of it then loads as an
    instance of the class defined now. Each such class is added to
    *classes*, for `by_name_as_then`; one being pickled under its plain
    name (`_PLAIN`) is left to the pickler.

    A notebook that repeats a cell defining ``column_ratio`` after building
    a pipeline on it held the first function: E6's fitted grid search was
    never stored, and ran again after every restart (~48 s).
    """
    plain = getattr(_PLAIN, "ids", None) or {}

    def override(obj: Any) -> Any:
        if type(obj) is types.FunctionType or isinstance(obj, type):
            if id(obj) in plain:
                return NotImplemented
            current = named_now(obj)
            if current is not None:
                same = same_class(obj, current) if isinstance(obj, type) else same_definition(obj, current)
                if not same:
                    name = obj.__qualname__
                    raise DefinedAgain(
                        f"it holds {name} as defined before its cell ran again with different code, "
                        f"and a stored copy could only load the {name} defined now"
                    )
                if classes is not None and isinstance(obj, type):
                    classes.append(obj)
                return by_name, (obj.__module__, obj.__qualname__)
        return NotImplemented if then is None else then(obj)

    return override


def by_name_as_then(dump: Callable[[Callable[[Any], Any]], Any], then: Any = None) -> Any:
    """``dump(override)``, *override* a `defined_again` asking *then* about
    everything else, with the classes whose cell ran again written under
    their plain names, as any class is.

    A class `defined_again` pickles by its name is written as a call that
    finds it (`by_name`), not as a plain name: a value holding it digested
    apart from the same value loaded after a restart, which holds the class
    defined now. So when the value holds one, *dump* runs again with pickle's
    Python pickler, which writes each of them as the plain name of the class
    defined now (`_PlainNaming`): the bytes of the value as it loads. The
    module's names are not touched."""
    classes: list[type] = []
    first = dump(defined_again(then, classes))
    if not classes:
        return first
    before = getattr(_PLAIN, "ids", None)
    _PLAIN.ids = {**(before or {}), **{id(cls): cls for cls in classes}}
    try:
        return dump(defined_again(then))
    finally:
        _PLAIN.ids = before


#: The classes a pickle on this thread writes under their plain names
#: although their module names another class now (`by_name_as_then`):
#: ``id -> class``.
_PLAIN = threading.local()

#: What the C pickler writes before it asks ``reducer_override``.
_SAVED_FIRST = frozenset(
    {type(None), bool, int, float, bytes, str, dict, set, frozenset, list, tuple, bytearray, pickle.PickleBuffer}
)


class _TableGet:
    """A dispatch table as the Python pickler reads it (``.get``), for one
    the C pickler reads by key (`_KeepDict`'s ``__missing__``)."""

    def __init__(self, table: Any) -> None:
        self.table = table

    def get(self, t: type, default: Any = None) -> Any:
        try:
            return self.table[t]
        except KeyError:
            return default


#: pickle's pure-Python pickler, which the stubs leave out.
_PyPickler: Any = getattr(pickle, "_Pickler")


class _PlainNaming(_PyPickler):
    """pickle's Python pickler, which writes the classes of `_PLAIN` under
    their plain names, where the C one refuses a class its module names
    another of now. Asks *override* about what the C pickler asks it
    (`_SAVED_FIRST`), so the bytes are the C pickler's."""

    override: Any = None
    plain: dict[int, type]

    def reducer_override(self, obj: Any) -> Any:
        if type(obj) in _SAVED_FIRST or self.override is None:
            return NotImplemented
        return self.override(obj)

    def save_global(self, obj: Any, name: Any = None) -> None:
        if id(obj) not in self.plain or self.proto < 4:
            super().save_global(obj, name)
            return
        module_name = obj.__module__
        qualname = obj.__qualname__ if name is None else name
        self.save(module_name)
        self.save(qualname)
        self.write(pickle.STACK_GLOBAL)
        self.memoize(obj)


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


def same_definition(old: types.FunctionType, new: types.FunctionType, classes: tuple[type, type] | None = None) -> bool:
    """Do *old* and *new* do the same, so that a value holding *old* can
    hold *new* instead: the same code (lines aside), the same globals, no
    closure, and alike defaults, attributes, docstring and annotations.

    Some closures are the same too (`_alike_cells`): methods of the classes
    *classes* (old, new) closing over their own class (``super()`` and
    ``__class__`` do), and a wrapper (``functools.wraps``: sklearn wraps a
    transformer's ``transform``) closing over the function it wraps, the
    same definition, and plain data."""
    from cash.decorator.function_identity import code_fingerprint

    if old.__globals__ is not new.__globals__:
        return False
    wrapped = old.__dict__.get("__wrapped__"), new.__dict__.get("__wrapped__")
    if (wrapped[0] is None) != (wrapped[1] is None):
        return False
    if wrapped[0] is not None and not (
        type(wrapped[0]) is types.FunctionType
        and type(wrapped[1]) is types.FunctionType
        and same_definition(wrapped[0], wrapped[1], classes)
    ):
        return False
    if (old.__closure__ is not None or new.__closure__ is not None) and not _alike_cells(old, new, classes, wrapped):
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
        (
            {k: v for k, v in old.__dict__.items() if k != "__wrapped__"},
            {k: v for k, v in new.__dict__.items() if k != "__wrapped__"},
        ),
        (old.__doc__, new.__doc__),
        (getattr(old, "__type_params__", ()), getattr(new, "__type_params__", ())),
    ]
    if not _alike_annotations(old, new):
        return False
    return all(_alike(x, y) for x, y in pairs) and code_fingerprint(a) == code_fingerprint(b)


def _alike_annotations(old: Any, new: Any) -> bool:
    """Do functions or classes *old* and *new* have alike annotations?
    From 3.14 they are made when asked for (closing over the class body of
    a method), and are read as names where a name is not defined yet
    (``annotationlib``'s FORWARDREF)."""
    try:
        return _alike(_annotations(old), _annotations(new))
    except Exception:  # noqa: BLE001 - an annotation that cannot be read: not alike
        return False


def _annotations(obj: Any) -> Any:
    """*obj*'s annotations, read without leaving them on it: asking a class
    for them stores them in its ``__dict__``, which what-ran traces digest."""
    own = vars(obj) if isinstance(obj, type) else None
    if sys.version_info < (3, 14):
        return own.get("__annotations__") if own is not None else obj.__annotations__
    import annotationlib

    annotate = (own.get("__annotate_func__") or own.get("__annotate__")) if own is not None else obj.__annotate__
    if annotate is None:
        return own.get("__annotations__") if own is not None else None
    return annotationlib.call_annotate_function(annotate, annotationlib.Format.FORWARDREF)


def _alike_cells(
    old: types.FunctionType, new: types.FunctionType, classes: tuple[type, type] | None, wrapped: tuple[Any, Any]
) -> bool:
    """Do *old* and *new* close over the same names, each holding their own
    class *classes* (old, new) as ``__class__``, or, for a wrapper, the
    function each wraps (*wrapped*, old and new) or alike plain data
    (`same_definition`)?"""
    if old.__closure__ is None or new.__closure__ is None or old.__code__.co_freevars != new.__code__.co_freevars:
        return False
    for name, x, y in zip(old.__code__.co_freevars, old.__closure__, new.__closure__):
        try:
            a, b = x.cell_contents, y.cell_contents
        except ValueError:  # an empty cell
            return False
        if name == "__class__" and classes is not None and a is classes[0] and b is classes[1]:
            continue
        if wrapped[0] is not None and a is wrapped[0] and b is wrapped[1]:
            continue
        if wrapped[0] is None or type(a) not in _ALIKE_ATOMS or not _alike(a, b):
            return False
    return True


#: What a class's ``__dict__`` holds that differs between two runs of one
#: cell: where it was defined, its own descriptors, the slot names copyreg
#: keeps once an instance was pickled, an ABC's registry, and from 3.14 what
#: makes its annotations (compared made, `_alike_annotations`).
_CLASS_PLACE = frozenset(
    {
        "__module__",
        "__qualname__",
        "__firstlineno__",
        "__dict__",
        "__weakref__",
        "__slotnames__",
        "_abc_impl",
        "__annotate_func__",
        "__annotate__",
        "__annotations_cache__",
        "__annotations__",
    }
)


def same_class(old: type, new: type) -> bool:
    """Do classes *old* and *new* do the same, so that a value holding *old*
    or an instance of it can hold *new* instead (`defined_again`): the same
    metaclass and bases (a base defined again counts when it is the same
    class), alike annotations, and in their bodies the same names, each
    bound to the same definition (`same_definition`, through
    ``staticmethod``, ``classmethod``, ``property`` and
    ``functools.cached_property``), slot, plain data, or an object of a
    library's Python class holding plain data. Anything else in a body (a
    dataclass's fields) is not compared, and the class counts as another
    one."""
    if type(old) is not type(new) or len(old.__bases__) != len(new.__bases__):
        return False
    for a, b in zip(old.__bases__, new.__bases__):
        if a is not b and not (named_now(a) is b and same_class(a, b)):
            return False
    ours, theirs = vars(old), vars(new)
    if ours.keys() - _CLASS_PLACE != theirs.keys() - _CLASS_PLACE:
        return False
    if not _alike_annotations(old, new):
        return False
    return all(_same_member(ours[k], theirs[k], (old, new)) for k in ours.keys() - _CLASS_PLACE)


def _same_member(a: Any, b: Any, classes: tuple[type, type]) -> bool:
    """Are *a* and *b*, bound to one name in the bodies of *classes* (old,
    new), the same definition (`same_class`)?"""
    t = type(a)
    if t is not type(b):
        return False
    if t is types.FunctionType:
        return same_definition(a, b, classes)
    if t in (staticmethod, classmethod):
        return _same_member(a.__func__, b.__func__, classes)
    if t is property:
        return all(
            (x is None and y is None) or (x is not None and y is not None and _same_member(x, y, classes))
            for x, y in ((a.fget, b.fget), (a.fset, b.fset), (a.fdel, b.fdel))
        ) and _alike(a.__doc__, b.__doc__)
    if t is functools.cached_property:
        return a.attrname == b.attrname and _same_member(a.func, b.func, classes) and _alike(a.__doc__, b.__doc__)
    if t is types.MemberDescriptorType:  # a slot
        return a.__name__ == b.__name__ and a.__objclass__ is classes[0] and b.__objclass__ is classes[1]
    if isinstance(a, type):
        return a is b or (named_now(a) is b and same_class(a, b))
    if _plain_library_object(a, classes[0].__module__):
        # sklearn's ``set_fit_request``: its class's code, and what it holds.
        return _alike(vars(a), vars(b))
    return _alike(a, b)


def _plain_library_object(obj: Any, module: str) -> bool:
    """Is *obj* an object of a Python class of a module other than
    *module*, kept wholly in its ``__dict__`` (no slots), so that its class's
    code and that dict say what it does (`_same_member`)?"""
    t = type(obj)
    return (
        bool(t.__flags__ & _HEAP_TYPE)
        and t.__module__ != module
        and "__slots__" not in vars(t)
        and type(getattr(obj, "__dict__", None)) is dict
        and all("__slots__" not in vars(base) for base in t.__mro__[1:-1])
    )


#: ``Py_TPFLAGS_HEAPTYPE``: a class made by a class statement, not in C.
_HEAP_TYPE = 1 << 9


def reduces_its_own(cls: type, module: str) -> bool:
    """Does a class of *module* in *cls*'s MRO define ``__reduce__`` or
    ``__reduce_ex__`` (`loads_otherwise`)?"""
    return any(
        getattr(base, "__module__", None) == module and ("__reduce__" in vars(base) or "__reduce_ex__" in vars(base))
        for base in cls.__mro__[:-1]
    )


class LoadsOtherwise(pickle.PicklingError):
    """A value holds an object whose class rebuilds it on load through code
    that has changed since it was made (`loads_otherwise`)."""


def loads_otherwise(value: Any, module: str) -> str | None:
    """The name of a class of *module*, with its own ``__reduce__``, of which
    *value* holds an object that loads back from its pickle with another
    state than it has; None when each loads back as it is, or when *value*
    does not pickle as plain pickle does.

    Such a class rebuilds its objects through code of its own when they
    load (``return (rebuild, (self.k,))``). Run again with other code since
    the object was made, ``rebuild`` makes another object: a stored copy
    loaded as ``k=4`` where the value held ``k=3``. The objects are compared
    by their ``__getstate__``, in the order pickle meets them."""

    def found(root: Any) -> list[Any]:
        met: list[Any] = []
        own: dict[type, bool] = {}

        class _Meeting(pickle.Pickler):
            def reducer_override(self, obj: Any) -> Any:
                t = type(obj)
                mine = own.get(t)
                if mine is None:
                    mine = own[t] = not isinstance(obj, type) and reduces_its_own(t, module)
                if mine:
                    met.append(obj)
                return NotImplemented

        stream = io.BytesIO()
        _Meeting(stream, protocol=pickle.HIGHEST_PROTOCOL).dump(root)
        met.append(stream.getvalue())
        return met

    try:
        before = found(value)
        if len(before) == 1:
            return None  # no such object
        after = found(pickle.loads(before[-1]))
        for old, new in zip(before[:-1], after[:-1]):
            if type(old).__qualname__ != type(new).__qualname__ or pickle.dumps(
                old.__getstate__(), pickle.HIGHEST_PROTOCOL
            ) != pickle.dumps(new.__getstate__(), pickle.HIGHEST_PROTOCOL):
                return type(old).__qualname__
    except Exception:  # noqa: BLE001 - not plain pickle's to judge: as before
        return None
    if len(before) != len(after):
        return type(before[0]).__qualname__
    return None


_ALIKE_ATOMS = (str, bytes, int, float, complex, bool, type(None))


def _alike(a: Any, b: Any) -> bool:
    """*a* is *b*, or both are equal plain data (`same_definition`)."""
    if a is b:
        return True
    t = type(a)
    if t is not type(b):
        return False
    if t in _ALIKE_ATOMS:
        return a == b
    if t in (tuple, list):
        return len(a) == len(b) and all(_alike(x, y) for x, y in zip(a, b))
    if t is dict:
        return a.keys() == b.keys() and all(_alike(v, b[k]) for k, v in a.items())
    if t in (set, frozenset):
        return all(type(x) in _ALIKE_ATOMS for x in a) and a == b
    if t.__name__ == "ForwardRef" and t.__module__ in ("annotationlib", "typing"):
        return bool(a == b)
    return False
