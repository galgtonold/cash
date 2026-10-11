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

import copyreg
import functools
import io
import itertools
import operator
import pickle
import sys
import types
import weakref
from typing import Any

__all__ = [
    "FlatValue",
    "chooses_its_state",
    "dumps",
    "flatten",
    "left_out_attrs",
    "load_flat",
    "reducers",
    "restore_with_dict",
    "too_deep",
]


def dumps(
    value: Any,
    protocol: int = pickle.DEFAULT_PROTOCOL,
    *,
    fast: bool = False,
    buffer_callback: Any = None,
    extra: dict | None = None,
    persistent_id: Any = None,
    keyed: bool = False,
    on_retry: Any = None,
) -> bytes:
    """``pickle.dumps(value, protocol)``, keeping the instance attributes a C
    base's reduce drops. *fast* is the pickler's memo-less mode; *extra*
    are reducers by type that come first (a clock test double's);
    *persistent_id* is the pickler's hook of that name. *keyed*
    bytes are for hashing, never loaded: an instance of the user's own class
    also keeps the attributes its own reduce leaves out (`_reduce_for_key`).

    A value too deep for the pickler's recursion is pickled flat
    (`FlatValue`), which ``pickle.loads`` reads back as the value; when it
    is, *on_retry* is called first, as *buffer_callback* is called again
    for the buffers of the second attempt. *keyed* bytes are not: a value
    too deep to key raises RecursionError, as before."""
    if FLAT_FIRST and not _surely_shallow(value):
        return _dumps_walked_first(value, protocol, fast, buffer_callback, extra, persistent_id, keyed)
    try:
        return _dumps(value, protocol, fast, buffer_callback, extra, persistent_id, keyed)
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
    return _dumps(flat, protocol, fast, buffer_callback, extra, persistent_id, keyed)


def _dumps_walked_first(
    value: Any,
    protocol: int,
    fast: bool,
    buffer_callback: Any,
    extra: dict | None,
    persistent_id: Any,
    keyed: bool,
) -> bytes:
    """`dumps` where the pickler would run out of C stack before its
    recursion limit (`FLAT_FIRST`): the value is walked on the heap first,
    and handed to the pickler only when it is shallow enough
    (`_SAFE_DEPTH`). Deeper, it is pickled flat; for a key, as deep as
    pickle follows elsewhere, and past that it raises RecursionError as the
    pickler raises on other Pythons."""
    reducers, leaf = _KeepDict(protocol, extra, keyed), _flat_leaf(persistent_id)
    if _at_most_deep(value, _SAFE_DEPTH, reducers, leaf):
        return _dumps(value, protocol, fast, buffer_callback, extra, persistent_id, keyed)
    deepest = sys.getrecursionlimit() // 2 if keyed else None
    flat = flatten(value, reducers, leaf, deepest)
    return _dumps(flat, protocol, fast, buffer_callback, extra, persistent_id, keyed)


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
        again = len(level) - (len(seen) - before)
        if again:
            rewalked += again
            if rewalked > _REWALKED_FREE:
                # Much of it met again: a cycle, or objects shared all the
                # way down. `flatten` walks each object once.
                return False
            level = list(dict(zip(map(id, level), level)).values())  # each once in a level
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
) -> bytes:
    buf = io.BytesIO()
    pickler = pickle.Pickler(buf, protocol=protocol, buffer_callback=buffer_callback)
    pickler.fast = fast
    if persistent_id is not None:
        pickler.persistent_id = persistent_id
    pickler.dispatch_table = _KeepDict(protocol, extra, keyed)
    pickler.dump(value)
    return buf.getvalue()


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
        if state is not None:
            state = yield state
            steps.append(("build", slot, state, setter))
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

    def get(part: Any) -> Any:
        return made[part] if type(part) is _Slot else part

    for step in steps:
        kind = step[0]
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

    def __init__(self, protocol: int, extra: dict | None, keyed: bool = False) -> None:
        super().__init__(extra or ())
        self._protocol = protocol
        self._keyed = keyed

    def __missing__(self, t: type) -> Any:
        if issubclass(t, type):
            raise KeyError(t)  # a class: pickled by name, as pickle does
        registered = copyreg.dispatch_table.get(t)
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
        self[t] = reducer
        return reducer


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


def left_out_attrs(obj: Any, state: Any) -> dict:
    """The attributes of *obj* -- in its ``__dict__`` or its ``__slots__``
    -- that *state*, what its reduce saves, does not hold: by name, or
    under their name with another value (a ``__getstate__`` that saves a
    portable default in place of the live setting).

    One that cannot be pickled (a lock or a connection the class leaves out
    for that reason) is given as its type: its content is nothing pickle can
    read.
    """
    attrs = _instance_attrs(obj)
    if not attrs:
        return {}
    saved: dict = {}
    if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
        if isinstance(state[0], dict):
            saved.update(state[0])
        saved.update(state[1])  # (dict, slots)
    elif isinstance(state, dict):
        saved = state
    left_out = {}
    for name, value in attrs.items():
        if name in saved and _same(saved[name], value):
            continue
        try:
            dumps(value, keyed=True)
        except Exception:  # noqa: BLE001 - whatever pickle refuses
            value = ("__cash_unpicklable__", f"{type(value).__module__}.{type(value).__qualname__}")
        left_out[name] = value
    return left_out


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
