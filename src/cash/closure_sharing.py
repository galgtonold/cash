"""Functions in a value whose closure holds another part of it.

``return arr, lambda: arr``: a function from a module cannot be pickled, so
a stored value keeps it by reference while the rest is copied, and the
copy's ``arr`` is no longer the one the function returns. A write through a
view of one stops reaching the other, which no plain run does. Such a value
is not stored: in a notebook the statement runs again, and ``@cash.cache``
says STORE-PARTS-SHARED. A value too big to look through in the time the
check has (`closure_check_budget`) is not stored either
(STORE-PARTS-UNCHECKED).

No code of a user's class runs: what its ``__getstate__`` or
``__reduce__`` would hand pickle and its own attributes do not hold is not
seen.
"""

from __future__ import annotations

import bisect
import io
import operator
import pickle
import sys
import types
from collections.abc import Iterable, Mapping
from itertools import accumulate, chain, compress, count, islice, repeat
from typing import Any, NamedTuple

from cash._clock import CpuLimit
from cash._native import kernels as _KERNELS
from cash._plain_data import TREE_NODES, in_a_look, numpy_scalar_set, tree_facts, tree_levels
from cash._type_memo import MroMemo, TypeMemo, type_key
from cash._user_attrs import own_dict, own_dicts
from cash.install_paths import is_user_code_module
from cash.kept_state import reduce_counted
from cash.live_state import is_live_stream
from cash.value_types import IMMUTABLE_LEAF_TYPES

__all__ = ["ClosureCheck", "closure_check_budget", "closures_sharing"]

#: Hold nothing: not looked inside, and no place of the value's either.
_LEAVES = frozenset(IMMUTABLE_LEAF_TYPES) | {pickle.PickleBuffer, memoryview}


def _leaves() -> frozenset:
    """`_LEAVES` with numpy's exact number types (`numpy_scalar_set`),
    which hold a number and take no attribute: a list of a million
    ``np.float64`` (``list(arr)``) was read a scalar at a time, through its
    reduce, past the time and not stored. With numpy's ``datetime64`` and
    ``timedelta64`` as well, as atomic: ``list(frame['t'].diff())`` gave a
    list of 868,000 walked through two calls each, 27 s of a session."""
    global _WITH_SCALARS
    scalars = numpy_scalar_set()
    if _WITH_SCALARS[0] is not scalars:
        numpy = sys.modules.get("numpy")
        times = frozenset() if numpy is None else frozenset((numpy.datetime64, numpy.timedelta64))
        _WITH_SCALARS = (scalars, _LEAVES | scalars | times)
    return _WITH_SCALARS[1]


#: ``(numpy_scalar_set(), _leaves())`` as `_leaves` last found them.
_WITH_SCALARS: tuple[frozenset, frozenset] = (frozenset(), _LEAVES)
#: The builtin containers, read in bulk a level at a time.
_SEQUENCES = frozenset({tuple, list, set, frozenset})
#: The builtin containers whose subclasses' items `_object_parts` reads.
_BUILTIN_CONTAINERS = (dict, list, tuple, set, frozenset)
#: Not looked inside: pickled by reference, as a function is.
_OPAQUE = (type, types.ModuleType, types.FunctionType)
#: The check may take this many times what the statement or call took,
#: but not less than `CLOSURE_CHECK_FLOOR_S`: past it the value is not
#: stored, which a check that costs more than the work it saves earns.
CLOSURE_CHECK_FACTOR = 2.0
CLOSURE_CHECK_FLOOR_S = 1.0
#: Objects read one by one between two reads of the clock.
_CLOCK_EVERY = 4096
#: Items of a level taken in at once between two reads of the clock.
_CHUNK = 1 << 20


class ClosureCheck(NamedTuple):
    """What `closures_sharing` found: the *names* holding a function whose
    closure reaches another part of the value, and whether the walk
    *finished* within its time. Unfinished, nothing is known, and the
    value is not stored."""

    names: list[str]
    finished: bool = True


def closure_check_budget(cost: float) -> float:
    """Seconds `closures_sharing` may take on the outputs of work that took
    *cost* seconds."""
    return max(CLOSURE_CHECK_FLOOR_S, CLOSURE_CHECK_FACTOR * cost)


def closures_sharing(captured_vars: Mapping[str, Any], seconds: float) -> ClosureCheck:
    """The names among *captured_vars* (a statement's outputs, or one
    ``{"result": value}``) that are, or hold anywhere, a function whose
    closure reaches an object found elsewhere in the outputs (``return
    arr, lambda: box['a']`` with ``box = {'a': arr}``). Kept by reference,
    such a function would not share it with the restored copies.

    Walked whole, a level at a time, within *seconds* of this thread's CPU
    time (`CpuLimit`): a cap on objects let a big output use it up and pass
    the function behind it unseen, so past the time the answer is
    unfinished and the caller stores nothing. JSON-like data (exact dicts,
    lists and tuples over plain values, `tree_facts`) holds no function and
    is not walked unless a function elsewhere may reach into it, and
    nothing is walked when the outputs hold no function with a closure at
    all (`_holds_a_closure`)."""
    if _holds_a_closure(captured_vars, seconds) is False:
        return ClosureCheck([])
    limit = CpuLimit(seconds)
    plain: set[str] = set()
    for name, value in captured_vars.items():
        found = _plain(value, limit)
        if found is None:
            return ClosureCheck([], finished=False)
        if found:
            plain.add(name)
    # Each object is kept until the answer, so an id stands for one object
    # only: a reduce's temporary freed a level later gave its id to another,
    # which was then taken as met and not read.
    places: dict[int, Any] = {}
    functions: list[tuple[str, types.FunctionType]] = []
    for name, value in captured_vars.items():
        if name not in plain and not _walk(name, value, places, functions, limit):
            return ClosureCheck([], finished=False)
    if not functions:
        return ClosureCheck([])
    for name in plain:
        if not _walk(name, captured_vars[name], places, functions, limit):
            return ClosureCheck([], finished=False)
    names: set[str] = set()
    for name, fn in functions:
        if name in names:
            continue
        found = _reaches(fn, places, limit)
        if found is None:
            return ClosureCheck(sorted(names), finished=False)
        if found:
            names.add(name)
    return ClosureCheck(sorted(names))


def _plain(value: Any, limit: CpuLimit) -> bool | None:
    """Is *value* JSON-like data? None when *limit* ran out first. In a
    `one_look` the answer is the share check's, read once for both;
    outside one, the levels are read with the clock checked at each."""
    if type(value) not in TREE_NODES:
        return False
    try:
        if in_a_look():
            return tree_facts(value) is not None
        for _level in tree_levels(value):
            if limit.passed():
                return None
    except Exception:  # noqa: BLE001 - not plain (an object, an unhashable key): walked
        return False
    return True


def _holds_a_closure(captured_vars: Mapping[str, Any], seconds: float) -> bool | None:
    """Do the outputs hold, anywhere, a function with a closure? None when
    *seconds* of CPU time ran out first. Read as `_walk` reads them, JSON-
    like data (`_plain`) aside, but not below an Arrow array that holds no
    Python object of its own (`_arrow_type_without_code`), and every object
    read is kept until the answer, so an id stands for one object only: a
    frame concatenated from thousands of small ones keeps a chunk per
    piece, each read through its ``__reduce_ex__``, and none can hold a
    function. Without one, nothing is shared with a function and the walk
    is not needed; with one, or past the time, the walk runs as it would
    have, on a clock of its own."""
    limit = CpuLimit(seconds)
    values: list[Any] = []
    for value in captured_vars.values():
        found = _plain(value, limit)
        if found is None:
            return None
        if not found:
            values.append(value)
    seen: dict[int, Any] = {}
    level = _fresh([values], seen, limit)
    while level:
        held: list[Iterable[Any]] = []
        kinds = set(map(type, level))
        for kind in kinds:
            group = level if len(kinds) == 1 else _of_kind(level, kind)
            if kind in _SEQUENCES:
                held.append(chain.from_iterable(group))
            elif kind is dict:
                held.append(chain.from_iterable(group))  # the keys
                held.append(chain.from_iterable(map(dict.values, group)))
            elif kind is types.FunctionType:
                if any(fn.__closure__ for fn in group):
                    return True
            elif _pandas_time_kind(kind):
                if kind.__name__ == "Timestamp":
                    held.append(item.tzinfo for item in group if item.tzinfo is not None)
            elif not issubclass(kind, _OPAQUE):
                bulk = _group_parts(group, kind)
                if bulk is not None:
                    held.extend(bulk)
                    continue
                arrow = _arrow_array_kind(kind)
                for done, item in enumerate(group, 1):
                    if done % _CLOCK_EVERY == 0 and limit.passed():
                        return None
                    if not (arrow and _arrow_type_without_code(item)):
                        held.append(_object_parts(item, kind))
        if limit.passed():
            return None
        level = _fresh(held, seen, limit)
    return None if level is None else False


def _pandas_time_kind(kind: type) -> bool:
    """Is *kind* exactly pandas' ``Timestamp`` or ``Timedelta``? Pickled
    as numbers and, for a ``Timestamp``, its ``tzinfo``, read on: nothing
    else of Python's that could hold a function. An object column of a frame
    concatenated from frames with and without times holds one per row."""
    pandas = sys.modules.get("pandas")
    return pandas is not None and (kind is pandas.Timestamp or kind is pandas.Timedelta)


def _arrow_array_kind(kind: type) -> bool:
    """Is *kind* an Arrow array or chunked array class?"""
    pyarrow = sys.modules.get("pyarrow")
    return pyarrow is not None and issubclass(kind, (pyarrow.Array, pyarrow.ChunkedArray))


def _arrow_type_without_code(array: Any) -> bool:
    """Does the Arrow *array* hold no Python object but what Arrow makes
    for it (buffers, types, chunks), never a function? Not with an
    extension type anywhere in its type: a Python class of the user's or a
    library's, which can hold anything."""
    import pyarrow

    try:
        pending = [array.type]
        while pending:
            arrow_type = pending.pop()
            if isinstance(arrow_type, pyarrow.BaseExtensionType):
                return False
            if isinstance(arrow_type, pyarrow.DictionaryType):
                pending.extend((arrow_type.index_type, arrow_type.value_type))
                continue
            pending.extend(arrow_type.field(at).type for at in range(arrow_type.num_fields))
    except Exception:  # noqa: BLE001 - not known: read below it
        return False
    return True


def _walk(name: str, value: Any, places: dict[int, Any], functions: list, limit: CpuLimit) -> bool:
    """Add to *places* every object *value* holds, by its id, and to
    *functions* each function with a closure among them, with *name*;
    False when *limit* ran out first. A level's builtin containers are read
    in bulk; an atom is neither a place nor looked into."""
    level: list | None = _fresh([[value]], places, limit)
    if level is None:
        return False
    while level:
        held: list[Iterable[Any]] = []
        kinds = set(map(type, level))
        for kind in kinds:
            group = level if len(kinds) == 1 else _of_kind(level, kind)
            if kind in _SEQUENCES:
                held.append(chain.from_iterable(group))
            elif kind is dict:
                held.append(chain.from_iterable(group))  # the keys
                held.append(chain.from_iterable(map(dict.values, group)))
            elif kind is types.FunctionType:
                functions.extend((name, fn) for fn in group if fn.__closure__)
            elif not issubclass(kind, _OPAQUE):
                bulk = _group_parts(group, kind)
                if bulk is not None:
                    held.extend(bulk)
                    continue
                for done, item in enumerate(group, 1):
                    if done % _CLOCK_EVERY == 0 and limit.passed():
                        return False
                    held.append(_object_parts(item, kind))
        if limit.passed():
            return False
        level = _fresh(held, places, limit)
        if level is None:
            return False
    return True


def _fresh(held: list[Iterable[Any]], places: dict[int, Any], limit: CpuLimit) -> list | None:
    """What *held* yields that is not an atom and not in *places*, each
    once, now added to it; None when *limit* ran out first. Taken in
    chunks, each a few passes in C: a level of ten million strings is
    typed and dropped, never remembered one by one."""
    leaves = _leaves()
    items = chain.from_iterable(held)
    fresh: list = []
    while chunk := list(islice(items, _CHUNK)):
        if not leaves.issuperset(map(type, chunk)):
            chunk = _without_leaf_records(chunk)
            chunk = list(compress(chunk, map(operator.not_, map(leaves.__contains__, map(type, chunk)))))
            new = dict(zip(map(id, chunk), chunk))
            for key in places.keys() & new.keys():
                del new[key]
            places.update(new)
            fresh.extend(new.values())
        if limit.passed():
            return None
    return fresh


def _of_kind(level: list, kind: type) -> list:
    """The objects of *level* whose class is exactly *kind*, told in C."""
    return list(compress(level, map(operator.is_, map(type, level), repeat(kind))))


#: Per class, whether `_group_parts` reads its objects a list at a time.
_IN_BULK: TypeMemo[bool] = TypeMemo()
#: What makes an object pickle as other than its attributes and slots.
_REDUCE_HOOKS = ("__reduce_ex__", "__reduce__", "__getstate__", "__getnewargs_ex__", "__getnewargs__")
_HEAP_TYPE = 1 << 9
#: A class's ``__dict__`` as ``type`` reads it: no metaclass of the user's runs.
_CLASS_DICT = type.__dict__["__dict__"].__get__


def _read_in_bulk(kind: type) -> bool:
    """Does `_object_parts` read an object of *kind* as its attributes and
    its slots alone? So for a class of the user's, whose own methods are
    never asked, and for a library's whose every class but ``object`` a
    ``class`` statement made and defines no hook of pickle's
    (`_REDUCE_HOOKS`): its ``__reduce_ex__`` is ``object``'s, which hands
    pickle the class and those same attributes and slots. Not a builtin
    container's subclass, a bound method or a stream, read on their own."""
    found = _IN_BULK.get(type_key(kind))
    if found is None:
        mro = kind.__mro__
        found = not (
            kind is types.MethodType
            or issubclass(kind, _BUILTIN_CONTAINERS)
            or issubclass(kind, io.IOBase)
            or any("_cash_wrapped_stream" in _CLASS_DICT(klass) for klass in mro)
        ) and (
            _users(kind)
            or all(
                klass.__flags__ & _HEAP_TYPE and not any(hook in _CLASS_DICT(klass) for hook in _REDUCE_HOOKS)
                for klass in mro[:-1]
            )
        )
        found = _IN_BULK.put(kind, found)
    return found


def _group_parts(group: list, kind: type) -> list[Iterable[Any]] | None:
    """What `_object_parts` gives of each object of *group*, all of class
    *kind*, read a list at a time, or None when they are read one by one
    (`_read_in_bulk`): a trie of a few hundred thousand small nodes was
    read a node at a time, past the time, and never stored."""
    if not _read_in_bulk(kind):
        return None
    parts: list[Iterable[Any]] = [chain.from_iterable(map(dict.values, own_dicts(group, kind)))]
    for reader in slot_readers(kind):
        try:
            parts.append(list(map(reader.__get__, group)))
        except AttributeError:  # an empty slot somewhere: read one by one
            parts.append(_set_slots(reader, group))
    return parts


def _set_slots(reader: Any, group: list) -> list:
    """What the slot *reader* holds in each object of *group*, its empty
    ones left out."""
    found = []
    for item in group:
        try:
            found.append(reader.__get__(item))
        except AttributeError:
            continue
    return found


def _without_leaf_records(chunk: list) -> list:
    """*chunk* without those of its tuples that hold leaves alone: such a
    tuple holds no function and nothing that can change, so no identity of
    it a restore must keep, as a leaf has none. Told at C speed; a frame's
    two million ``(action, time)`` records were each made a place one by
    one, 2 s a statement (student_8). Each tuple is told on its own: the
    few that hold more (the state numpy's ``__reduce_ex__`` gives beside an
    object column's cells) kept every record of their chunk a place, 4 s a
    statement."""
    is_tuple = list(map(operator.is_, map(type, chunk), repeat(tuple)))
    if not any(is_tuple):
        return chunk
    every = all(is_tuple)
    tuples = chunk if every else list(compress(chunk, is_tuple))
    held = [
        start + at
        for start in range(0, len(tuples), _RECORDS_AT_ONCE)
        for at in _tuples_holding_more(tuples[start : start + _RECORDS_AT_ONCE])
    ]
    if every:
        return [tuples[at] for at in held]
    keep = list(map(operator.not_, is_tuple))
    position = list(compress(count(), is_tuple))
    for at in held:
        keep[position[at]] = True
    return list(compress(chunk, keep))


#: Tuples `_without_leaf_records` asks about at once: a run of records
#: holding leaves alone is dropped by one question, and the few holding
#: more are found among this many.
_RECORDS_AT_ONCE = 4096


def _tuples_holding_more(tuples: list) -> list[int]:
    """The positions in *tuples* of those holding anything but leaves."""
    if _KERNELS is not None and _KERNELS.leaf_items(tuples, 0, _LEAVES) is True:
        return []
    # Of the items of the tuples in a row, the positions of those that are
    # no leaf: each tells the tuple it is in.
    beyond = list(
        compress(count(), map(operator.not_, map(_LEAVES.__contains__, map(type, chain.from_iterable(tuples)))))
    )
    if not beyond:
        return []
    ends = list(accumulate(map(len, tuples)))
    return sorted({bisect.bisect_right(ends, position) for position in beyond})


#: Per class, held weakly: ``(index in its MRO, name)`` of each slot, not
#: the member descriptors, which refer to their class and would keep it alive.
_SLOTS: MroMemo = MroMemo()
_USERS: TypeMemo[bool] = TypeMemo()
_MODULE_OF = type.__dict__["__module__"].__get__


def slot_readers(kind: type) -> tuple:
    """The member descriptor of each ``__slots__`` entry along *kind*'s
    MRO: how pickle reads a slotted object, read without running its code."""
    found = _SLOTS.get_for(kind)
    if found is None:
        places = []
        for at, klass in enumerate(kind.__mro__):
            own = klass.__dict__
            if "__slots__" in own:
                places.extend((at, name) for name, m in own.items() if type(m) is types.MemberDescriptorType)
        found = _SLOTS.put(kind, tuple(places))
    if not found:
        return ()
    mro = kind.__mro__
    return tuple(vars(mro[at])[name] for at, name in found)


def _users(kind: type) -> bool:
    """Is *kind* defined in the user's code? Its own methods are never
    asked, where a library's (``__reduce_ex__``) are."""
    found = _USERS.get(type_key(kind))
    if found is None:
        module = _MODULE_OF(kind)  # type's own reading: no metaclass of the user's runs
        found = _USERS.put(kind, is_user_code_module(sys.modules.get(module)) if module else False)
    return found


def _object_parts(item: Any, kind: type) -> list:
    """What pickle writes with *item*, an object of *kind* that is no
    builtin container: its attributes, its slots, the items of a builtin
    container it subclasses, and, for a library's object with no attributes
    of its own (a numpy object array, a ``functools.partial``, a pandas
    block), what its own ``__reduce_ex__`` hands pickle. No code of a user's
    class runs."""
    if kind is types.MethodType:  # read by the type's own members, as pickle reads it
        return [item.__func__, item.__self__]
    if (issubclass(kind, io.IOBase) or hasattr(kind, "_cash_wrapped_stream")) and is_live_stream(item):
        return []  # pickled as the name it is bound to (`live_state`)
    parts: list = []
    attributes = own_dict(item)
    if attributes:
        parts.extend(attributes.values())
    for reader in slot_readers(kind):
        try:
            parts.append(reader.__get__(item, kind))
        except AttributeError:  # an empty slot
            continue
    if issubclass(kind, _BUILTIN_CONTAINERS):
        parts.extend(_items(item, kind))
    elif (not attributes or attributes.keys() == _CACHE_ONLY) and not _users(kind):
        # A pandas block keeps what it has worked out in a ``_cache`` of
        # its own (``cache_readonly``) once its dtype was read: its values
        # are still what pickle writes.
        parts.extend(_reduced(item, kind))
    return parts


#: The own attributes of a library's object that only caches (`_object_parts`).
_CACHE_ONLY = {"_cache"}


def _items(item: Any, kind: type) -> list:
    """The items of *item*, of a subclass *kind* of a builtin container (a
    named tuple, an ``OrderedDict``), by the builtin's own reading, not a
    method the class may define."""
    if issubclass(kind, dict):
        return [*dict.keys(item), *dict.values(item)]
    base = next(base for base in kind.__mro__ if base in _BUILTIN_CONTAINERS)
    return list(base.__iter__(item))


def _reduced(item: Any, kind: type) -> list:
    """What a library's *item* hands pickle (``__reduce_ex__``): the
    arguments, the state and the items of the value it reduces to. A numpy
    array holds its items only with object dtype (otherwise its bytes, which
    a copy may read through a view, so its base); reduced, an array that is
    not contiguous would be copied."""
    if kind.__name__ == "ndarray" and kind.__module__ == "numpy":
        if not item.dtype.hasobject:
            return [] if item.base is None else [item.base]
    try:
        reduced = reduce_counted(item, 5)
    except Exception:  # noqa: BLE001 - not picklable: stored by reference or not at all
        return []
    if not isinstance(reduced, tuple):
        return []
    parts: list = [*reduced[1:2], *reduced[2:3]]
    for extra in reduced[3:5]:
        if extra is not None:
            try:
                parts.extend(extra)
            except Exception:  # noqa: BLE001 - the library's own iterator
                continue
    return parts


def _reaches(fn: types.FunctionType, places: dict[int, Any], limit: CpuLimit) -> bool | None:
    """Does what *fn*'s closure holds reach an object in *places*? None
    when *limit* ran out first. Another function met on the way is not
    followed: its own check is its own."""
    held = []
    for cell in fn.__closure__ or ():
        try:
            held.append(cell.cell_contents)
        except ValueError:  # an empty cell
            continue
    reached: dict[int, Any] = {}
    level = _fresh([held], reached, limit)
    while level:
        if any(id(item) in places for item in level if not issubclass(type(item), _OPAQUE)):
            return True
        parts: list = []
        for done, item in enumerate(level, 1):
            if done % _CLOCK_EVERY == 0 and limit.passed():
                return None
            kind = type(item)
            if kind in _SEQUENCES:
                parts.append(item)
            elif kind is dict:
                parts.append(item)
                parts.append(item.values())
            elif not issubclass(kind, _OPAQUE):
                parts.append(_object_parts(item, kind))
        level = _fresh(parts, reached, limit)
    return None if level is None else False
