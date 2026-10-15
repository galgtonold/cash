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
    "FlatValue",
    "chooses_its_state",
    "dumps",
    "flatten",
    "left_out_attrs",
    "load_flat",
    "one_keying",
    "reducers",
    "restore_with_dict",
    "too_deep",
]


def dumps(*args: Any, **kwargs: Any) -> bytes:
    """`_dumps_value` inside `one_keying`; see there."""
    if _keying() is not None:
        return _dumps_value(*args, **kwargs)
    with one_keying():
        return _dumps_value(*args, **kwargs)


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
    are never pickled flat."""
    if sink is not None and not keyed:
        raise ValueError("dumps: a sink takes keyed bytes only")
    if FLAT_FIRST and not _surely_shallow(value):
        return _dumps_walked_first(value, protocol, fast, buffer_callback, extra, persistent_id, keyed, aliases, sink)
    try:
        return _dumps(value, protocol, fast, buffer_callback, extra, persistent_id, keyed, aliases, sink)
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
    *reducer_override* is the pickler's hook of that name."""
    if FLAT_FIRST:
        if sink is not None:
            raise RecursionError("pickled flat first on this Python: digested over its keyed bytes")
        stream = dumps(value, protocol, buffer_callback=buffer_callback, extra=extra, on_retry=on_retry)
        return stream, False
    differs: list[bool] = []
    table = _KeepDict(protocol, extra, watch=differs)
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
) -> bytes:
    """`dumps` where the pickler would run out of C stack before its
    recursion limit (`FLAT_FIRST`): the value is walked on the heap first,
    and handed to the pickler only when it is shallow enough
    (`_SAFE_DEPTH`). Deeper, it is pickled flat; for a key, as deep as
    pickle follows elsewhere, and past that it raises RecursionError as the
    pickler raises on other Pythons."""
    reducers, leaf = _KeepDict(protocol, extra, keyed), _flat_leaf(persistent_id)
    if _at_most_deep(value, _SAFE_DEPTH, reducers, leaf):
        return _dumps(value, protocol, fast, buffer_callback, extra, persistent_id, keyed, aliases, sink)
    deepest = sys.getrecursionlimit() // 2 if keyed else None
    flat = flatten(value, reducers, leaf, deepest)
    return _dumps(flat, protocol, fast, buffer_callback, extra, persistent_id, keyed, aliases, sink)


def _at_most_deep(value: Any, limit: int, reducers: Any, leaf: Any) -> bool:
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
        level = _next_level(level, reducers, leaf)
    return not level


#: How many objects `_at_most_deep` walks again before it leaves the value
#: to `flatten`: a few shared at every level (a class, a dtype) are not a
#: cycle.
_REWALKED_FREE = 1024


def _next_level(level: list, reducers: Any, leaf: Any) -> list:
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
            parts = _reduced_parts(obj, reducers, leaf)
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


def _reduced_parts(obj: Any, reducers: Any, leaf: Any) -> list:
    """What pickling *obj*, not a builtin container, saves inside it, as
    `flatten` follows it: nothing for an object pickled whole."""
    obj_type = type(obj)
    if leaf(obj) or obj_type in _WHOLE or issubclass(obj_type, type) or _flat_array(obj):
        return []
    reduced = reducers[obj_type](obj)
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
        pickler.persistent_id = key_aliases()
    pickler.dispatch_table = _KeepDict(protocol, extra, keyed) if table is None else table
    pickler.dump(value)
    return b"" if sink is not None else buf.getvalue()


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


def reducers(protocol: int = pickle.DEFAULT_PROTOCOL) -> dict:
    """Each type's reducer, as `dumps` reduces it: ``reducers(p)[type(obj)](obj)``
    is what pickling *obj* at protocol *p* saves (a class has none: it is
    pickled by name)."""
    return _KeepDict(protocol, None)


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

    def __missing__(self, t: type) -> Any:
        if issubclass(t, type):
            raise KeyError(t)  # a class: pickled by name, as pickle does
        registered = copyreg.dispatch_table.get(t)
        if self._watch is not None:
            reducer = _watched(self._store_reducer(t, registered), t, registered, self._watch)
            self[t] = reducer
            return reducer
        if self._keyed and t.__name__ == "ndarray" and t is getattr(sys.modules.get("numpy"), "ndarray", None):
            # A record array's padding bytes are whatever memory held:
            # keyed without them (`packed_records`).
            reducer = functools.partial(_reduce_array_for_key, protocol=self._protocol)
            self[t] = reducer
            return reducer
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
            if _cached_properties(t) and (self._keyed or _leaves_views_out(t)):
                which = None if self._keyed else stored_views
                reducer = functools.partial(_reduce_without_views, reducer=reducer, which=which)
        self[t] = reducer
        return reducer

    def _store_reducer(self, t: type, registered: Any) -> Any:
        """The reducer a store's pickle (not *keyed*) takes for *t*."""
        if registered is not None:
            return registered
        if _drops_dict(t):
            reducer = _reducer_with_dict(self._protocol)
        else:
            reducer = operator.methodcaller("__reduce_ex__", self._protocol)
        if _cached_properties(t) and _leaves_views_out(t):
            reducer = functools.partial(_reduce_without_views, reducer=reducer, which=stored_views)
        return reducer


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
    if registered is None and _cached_properties(t):

        def views(obj: Any) -> Any:
            reduced = reducer(obj)
            if not isinstance(reduced, str) and derived_views(obj) - stored_views(obj):
                watch.append(True)  # a view the stored bytes keep and a key leaves out
            return reduced

        return views
    return reducer


def _cached_properties(t: type) -> tuple[str, ...]:
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
    names = _cached_properties(type(obj))
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
    return _packed_array, (array.dtype.descr, packed)


def _packed_array(descr: Any, packed: Any) -> Any:
    """What `_reduce_array_for_key` names (key bytes are never loaded)."""
    import numpy as np

    return packed.astype(np.dtype(descr))


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
