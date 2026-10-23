"""Keying one intercepted call: :func:`call_cache_key` and the tests it rests on.

A call's key is the statement path's key (:func:`compute_cache_key`, under the
``"call"`` namespace) extended with the parts a call needs that a statement
does not: its computed arguments, the enclosing loops' variables, the globals
the callee writes, and -- when everything the call reads is plain data or
code (:func:`_keys_by_content`) -- the values it receives instead of the
statement it sits in.
"""

from __future__ import annotations

import argparse
import builtins
import collections
import contextlib
import dataclasses
import datetime as _dt
import decimal as _decimal
import fractions as _fractions
import functools
import gc
import hashlib
import logging
import pathlib as _pathlib
import random
import reprlib
import sys
import types as _types
import weakref
from collections.abc import Callable, Mapping
from types import ModuleType as _ModuleType
from typing import TYPE_CHECKING, Any

from cash import _plain_data
from cash._memo import CODE_OBJECTS, LruMemo
from cash._paths import MAIN_MODULE_NAMES, is_pseudo_filename
from cash._user_attrs import attribute_or, has_attribute
from cash.analysis.bytecode import NAME_LOADS, argvals
from cash.analysis.callee_effects import source_global_mutations
from cash.analysis.namespace_effects import capturable_globals
from cash.exceptions import SOURCE_RETRIEVAL_ERRORS
from cash.install_paths import is_user_code_module, is_user_path
from cash.notebook import shadow as _shadow
from cash.notebook.cache_key import CacheKeyContext, compute_cache_key
from cash.notebook.call_effects import closure_cells, rebinds_its_closure
from cash.sizing import pandas_nbytes
from cash.source_reading import own_source, wrapper_layers
from cash.value_hash import compute_hash

if TYPE_CHECKING:
    from cash.notebook.call_interception import CallSite

logger = logging.getLogger(__name__)

__all__ = [
    "CallKeys",
    "call_cache_key",
    "callee_mutated_globals",
    "changes_its_closure",
    "global_digests",
    "global_names_reached",
    "TooMuchToLookThrough",
    "holds_a_closure_with_state",
]


def _is_dunder_loop_var(name: str) -> bool:
    """True if *name*'s own bare portion is dunder-prefixed.

    Handles both shapes `loop_vars` can arrive in: a bare name (`"x"` --
    every direct caller/test that predates depth-keying, e.g.
    `test_call_unit_key.py`'s hand-built dicts) and a depth-prefixed one
    (`"0:x"` -- the production path,
    `CallRouting.current_loop_vars_for_call_key`). A depth-prefixed
    dunder (`"0:__iterable_lineage__"`) no longer starts with `"__"` itself
    once the prefix is on -- checking the combined string, as this used to,
    would silently stop catching it and turn the enforced reorder guard back
    into a merely documented one. Splitting off the depth at the first colon
    before checking is what keeps the guard live for both shapes.
    """
    _, _, bare = name.partition(":")
    return (bare or name).startswith("__")


def _loop_var_digest(name: str, value: object, loop_var_digests: Mapping[str, str]) -> str:
    """The discriminating hash for one loop-var entry: its whole content.

    A loop variable is the per-iteration discriminator: a hash of a long
    list's ends would give two items agreeing there one entry (a wrong value
    on the first run). The fallback is ``compute_hash``, the digest
    ``for_handler`` records.

    A full hash of a large value can cost more than the call it keys, and is
    paid per call, so *loop_var_digests* -- the hashes ``for_handler`` took
    when it bound each loop variable, carried down ``CallRouting``'s
    ``loop_vars_scope`` stack -- are used when present. Not
    ``variable_lineage``: it is keyed by bare name and never popped, so after
    an inner loop reusing the outer loop's name it holds the inner value.
    """
    digest = loop_var_digests.get(name)
    return digest if digest is not None else compute_hash(value)


#: ``code object -> the globals its body mutates``, before the namespace filter.
#:
#: Keyed on the code object so that a hit skips ``inspect.getsource``, which is
#: the whole cost of the analysis and would otherwise be paid on every
#: intercepted call. A redefined function has a new code object, so an edit is
#: never served the old verdict. It is the code of the function ``getsource``
#: reads: a ``functools.wraps`` wrapper's own code, each layer read apart.
_GLOBAL_MUTATION_CACHE: LruMemo[Any, tuple[str, ...]] = LruMemo(CODE_OBJECTS)


def callee_mutated_globals(fn) -> tuple[str, ...]:
    """Names in *fn*'s own globals, or in its closure, that calling *fn*
    changes, sorted.

    The same analysis the statement path applies to a callee it finds by name
    (:func:`~cash.analysis.callee_effects.source_global_mutations`), read from
    the live function object instead. Never raises: a callee whose source
    cannot be read (a C builtin, an ``exec``'d string, a partial) yields
    ``()``, so its writes are not captured -- as for any call cash cannot see
    into.
    """
    globals_dict = attribute_or(fn, "__globals__", None)
    if not isinstance(globals_dict, dict):
        return ()
    # A name the callee reads from its closure is one too: `seen.append(x)`
    # on a list a factory made. Not in any namespace, so the filter below
    # drops it, and the second `add('a')` was served the first one's length.
    cells = closure_cells(fn)
    found: set[str] = set()
    # Every layer of a ``functools.wraps`` wrapper runs: a ``counted``
    # decorator's ``calls[name] += 1`` as much as the wrapped body.
    for layer in wrapper_layers(fn):
        layer_globals = attribute_or(layer, "__globals__", None)
        if not isinstance(layer_globals, dict):
            continue
        # The functions the layer reaches in the same namespace write those
        # globals too, at any depth: a hit skips them along with *fn*.
        cached = {
            name for reached in _functions_reached_in(layer_globals, layer) for name in _source_mutations(reached)
        }
        if not cached:
            continue
        # Filtered per call, not memoised: whether a name is bound (and not
        # a module) can change between calls, and the memo is about the
        # source.
        capturable = capturable_globals(cached, layer_globals)
        written = {n for n in cached if n in cells or n in capturable}
        if layer_globals is globals_dict:
            found |= written
        elif written:
            # Globals of another module, which the capture (reading *fn*'s
            # own globals) cannot put back: a name no namespace holds makes
            # the capture refuse, so the call is never stored.
            found.add(FOREIGN_GLOBALS)
    return tuple(sorted(found))


#: What `callee_mutated_globals` names for globals a wrapped layer writes in
#: another module's namespace: no namespace holds it, so no entry is stored.
FOREIGN_GLOBALS = "<globals of another module>"


#: ``code object -> _code_names(code)``, so a call does not disassemble its
#: callee on every invocation.
_CODE_NAMES: LruMemo[Any, frozenset[str]] = LruMemo(CODE_OBJECTS)


def _functions_reached_in(globals_dict: dict, fn) -> list:
    """*fn* and every function it calls by a global name, at any depth, that
    lives in *globals_dict* too."""
    reached = [fn]
    seen = {id(fn)}
    pending = [fn]
    while pending:
        code = attribute_or(pending.pop(), "__code__", None)
        if code is None:
            continue
        names = _CODE_NAMES.get(code)
        if names is None:
            names = _CODE_NAMES[code] = _code_names(code)
        for name in names:
            value = globals_dict.get(name)
            if type(value) is _types.FunctionType and value.__globals__ is globals_dict and id(value) not in seen:
                seen.add(id(value))
                reached.append(value)
                pending.append(value)
    return reached


def _source_mutations(fn) -> tuple[str, ...]:
    """The free names *fn*'s own source mutates in place, sorted, memoised
    on its code: a ``functools.wraps`` wrapper's own body, not the wrapped
    function's (`own_source`)."""
    memo_key = attribute_or(fn, "__code__", None)
    cached = _GLOBAL_MUTATION_CACHE.get(memo_key) if memo_key is not None else None
    if cached is None:
        try:
            cached = tuple(sorted(source_global_mutations(own_source(fn))))
        except SOURCE_RETRIEVAL_ERRORS:
            cached = ()
        if memo_key is not None:
            _GLOBAL_MUTATION_CACHE[memo_key] = cached
    return cached


def changes_its_closure(fn) -> bool:
    """Whether calling *fn* rebinds a variable of the function it was made
    in, or changes an object one of them holds: ``nonlocal count; count +=
    1``, ``seen.append(x)``."""
    if type(fn) is not _types.FunctionType:
        return False
    cells = closure_cells(fn)
    return (bool(cells) and (rebinds_its_closure(fn) or any(n in cells for n in _source_mutations(fn)))) or (
        holds_a_stream(fn)
    )


def holds_a_stream(fn) -> bool:
    """Whether a function holds a random generator or an iterator, however
    deeply, in its closure, its defaults (``def h(it=it)``,
    ``g=np.random.default_rng(0)``) or its attributes: calling it draws
    from it whatever method it calls (``g.normal()``), so the function
    changes what it holds. Stored, a hit handed back the function the last
    run made, its generator moved on by every draw since: after a restart
    ``x = h()`` gave the second number, not the first. Looked for in tuples,
    lists, sets and dicts (``st = {'it': iter(...)}``), in an object of the
    user's (``self.it``), and in another function's closure (a helper ``g``
    that holds it).

    Told by type, read off the class (`_stream_type`): no ``__iter__`` of
    the user's runs. Plain data costs no holder: a list of 30,000 objects
    of one class is one class looked into, its objects' attributes items.
    Past `_STREAM_HOLDERS_AT` functions and classes, or `_STREAM_ITEMS_AT`
    items, not known: taken as holding one."""
    if type(fn) is not _types.FunctionType:
        return False
    first = held_by(fn)
    if not first or _EXACT_ATOMS.issuperset(map(type, first)):
        return False
    stack = [fn]
    seen: set[int] = set()
    kinds: set[type] = set()
    ways: dict[type, Any] = {}
    holders = items = 0
    while stack:
        item = stack.pop()
        kind = type(item)
        way = ways.get(kind)
        if way is None:
            way = ways[kind] = _way_in(kind)
        if way is _RNG:
            if _rng_of_its_own(item):
                return True
            continue
        if way is _STREAM:
            return True
        if way is _NOTHING or id(item) in seen:
            continue
        seen.add(id(item))
        if way is _OBJECT:
            # A user's object, a namespace, a ``partial``: what it holds,
            # read without asking it. Its attributes as items, its class
            # once; one of numbers and text is not looked at further.
            for child in gc.get_referents(item):
                child_kind = type(child)
                if child_kind is dict:
                    items += len(child)
                    for part in (dict.values(child), dict.keys(child)):
                        if not _EXACT_ATOMS.issuperset(map(type, part)):
                            stack.extend(part)
                elif child_kind not in _EXACT_ATOMS:
                    stack.append(child)
            if kind not in kinds:
                kinds.add(kind)
                holders += 1
            if holders > _STREAM_HOLDERS_AT or items > _STREAM_ITEMS_AT:
                return True
            continue
        if way is _DICT:
            items += len(item)
            if items > _STREAM_ITEMS_AT:
                return True
            for part in (dict.values(item), dict.keys(item)):
                if not _EXACT_ATOMS.issuperset(map(type, part)):
                    stack.extend(part)
            continue
        if way is _SEQUENCE:
            children: Any = item
        else:  # a function, a method or a class: `held_by`
            children = held_by(item)
            if not children:
                continue
            holders += 1
            if holders > _STREAM_HOLDERS_AT:
                return True
        items += len(children)
        if items > _STREAM_ITEMS_AT:
            return True
        if not _EXACT_ATOMS.issuperset(map(type, children)):
            stack.extend(children)  # atoms among them are passed over as taken
    return False


#: How many functions and classes `holds_a_stream` looks into.
_STREAM_HOLDERS_AT = 10_000
#: How many items, all holders together, `holds_a_stream` reads.
_STREAM_ITEMS_AT = 2_000_000
_RNG, _STREAM, _NOTHING, _OBJECT, _SEQUENCE, _DICT, _HELD = (object() for _ in range(7))
#: Per type, `_stream_type`'s answer. Weak: a class a call made is not kept alive.
_STREAM_TYPES: weakref.WeakKeyDictionary[type, Any] = weakref.WeakKeyDictionary()


def _way_in(kind: type) -> Any:
    """How `holds_a_stream` reads an object of *kind*: ``_RNG`` or
    ``_STREAM`` (`_stream_type`), item by item (``_SEQUENCE``, ``_DICT``),
    by what it holds (``_OBJECT``, ``_HELD``: `held_by`), or not at all."""
    if kind in _EXACT_ATOMS:
        return _NOTHING
    try:
        found = _STREAM_TYPES.get(kind)
    except TypeError:  # not weakly referable
        found = None
    if found is None:
        found = _stream_type(kind)
        with contextlib.suppress(TypeError):
            _STREAM_TYPES[kind] = found
    if found is not _NOTHING:
        return found
    if kind in (tuple, list, set, frozenset):
        return _SEQUENCE
    if kind is dict:
        return _DICT
    if kind is _types.FunctionType or kind is _types.MethodType or issubclass(kind, type):
        return _HELD
    if issubclass(kind, (*_MEMBER_HOLDERS, *_PLAIN_HOLDERS)) or _users_type(kind):
        return _OBJECT
    return _NOTHING


def _stream_type(kind: type) -> Any:
    """``_RNG`` for a random generator's type, ``_STREAM`` for a stream's
    (an iterator, a generator, a queue: reading it moves it), ``_NOTHING``
    otherwise. Read off the class: a class with ``__iter__`` and
    ``__next__`` is taken as an iterator without asking its ``__iter__``
    whether it hands back itself, which ran the user's code at every
    object of the walk."""
    from cash.notebook.consumables import _GENERATOR_TYPES, _QUEUE_TYPES

    if issubclass(kind, type):
        return _NOTHING  # a class: its members are looked into, not its iteration
    if issubclass(kind, random.Random) or _numpy_rng_type(kind):
        return _RNG
    if issubclass(kind, (*_GENERATOR_TYPES, *_QUEUE_TYPES)):
        return _STREAM
    mro = _CLASS_MRO(kind)
    if any("__next__" in _CLASS_DICT(c) for c in mro) and any("__iter__" in _CLASS_DICT(c) for c in mro):
        return _STREAM
    return _NOTHING


def _numpy_rng_type(kind: type) -> bool:
    np_random = sys.modules.get("numpy.random")
    if np_random is None:
        return False
    rng_types = tuple(
        t for t in (np_random.__dict__.get(n) for n in ("Generator", "RandomState")) if isinstance(t, type)
    )
    return issubclass(kind, rng_types)


def _rng_of_its_own(item) -> bool:
    """A random generator that is not the module's own (``random``'s, or
    ``np.random``'s): those the module state replays."""
    if item is random.__dict__.get("_inst"):
        return False
    mtrand = sys.modules.get("numpy.random.mtrand")
    return mtrand is None or item is not mtrand.__dict__.get("_rand")


#: Values with nothing inside them to look at: skipped without a call.
_ATOMS = (int, float, complex, str, bytes, bool, type(None))

#: The same, by exact type, plus dates: one set lookup instead of a scan of `_ATOMS`.
_EXACT_ATOMS = frozenset({*_ATOMS, _dt.datetime, _dt.date, _dt.time, _dt.timedelta})

#: ``functools.cache`` / ``lru_cache``'s wrapper: a memo that fills as it is called.
#: A type where the C ``_functools`` is there; otherwise a function, and unused.
_MEMO_WRAPPER = functools.__dict__.get("_lru_cache_wrapper")
if not isinstance(_MEMO_WRAPPER, type):
    _MEMO_WRAPPER = None

#: How many functions, partials, methods, objects and classes the walk looks
#: into. Past it, what the rest holds is not known: the value is taken as
#: holding state, and not served (one rerun), rather than served unchecked.
_HOLDERS_LOOKED_INTO = 100_000
#: How many values the walk takes in all, plain records included: past it,
#: as past `_HOLDERS_LOOKED_INTO`, not known. A million records took 2.7 s
#: to the end; this many take about a second.
_STEPS_AT = 300_000

#: ``type -> whether it is a class of the user's`` (its module is user code).
#: Weak: a class a call made is not kept alive, nor the memo on it.
_USERS_TYPES: weakref.WeakKeyDictionary[type, bool] = weakref.WeakKeyDictionary()

#: Read through ``type``'s own descriptors: a metaclass's
#: ``__getattribute__`` is not asked.
_CLASS_DICT = type.__dict__["__dict__"].__get__
_CLASS_MRO = type.__dict__["__mro__"].__get__
_CLASS_MODULE = type.__dict__["__module__"].__get__
_CLASS_NAME = type.__dict__["__name__"].__get__

#: Plain attribute bags and containers of the library, read like the user's
#: objects: what they hold is what was put in them.
_PLAIN_HOLDERS = (
    _types.SimpleNamespace,
    argparse.Namespace,
    collections.deque,
    collections.UserDict,
    collections.UserList,
    collections.ChainMap,
)

#: What a class body holds a function in: read for what it holds.
_MEMBER_HOLDERS = (
    functools.partial,
    classmethod,
    staticmethod,
    property,
    functools.partialmethod,
    functools.cached_property,
)

#: The containers read item by item; a user's subclass of one is read as the
#: user's object (`held_by`), without asking it for its items.
_CONTAINERS = (tuple, list, set, frozenset, dict)
_EXACT_CONTAINERS = frozenset(_CONTAINERS)


def _repr_guards() -> frozenset[_types.CodeType]:
    """The code of the standard library's guard against a ``__repr__``
    calling itself (``reprlib.recursive_repr``, and dataclasses' own copy
    of it before 3.13): a closure over a set it adds to and takes from on
    each call, which leaves nothing behind once the call returns."""

    def plain(self):
        return ""

    guards = {reprlib.recursive_repr()(plain).__code__}
    own = dataclasses.__dict__.get("_recursive_repr")
    if callable(own):
        guards.add(own(plain).__code__)
    return frozenset(guards)


_REPR_GUARDS = _repr_guards()
#: By id, the codes kept alive above: a code object hashes its whole body.
_REPR_GUARD_IDS = frozenset(map(id, _REPR_GUARDS))


def _plain_record(attrs: dict, atoms: frozenset) -> bool:
    """Does an object's *attrs* hold only plain values, or lists and tuples
    of them: nothing inside to look at?"""
    for v in attrs.values():
        t = type(v)
        if t in atoms:
            continue
        if (t is list or t is tuple) and atoms.issuperset(map(type, v)):
            continue
        return False
    return True


#: How the walk reads a type (`_reading`): skipped, item by item, by its
#: referents, or one at a time (functions, methods, classes, memos).
_SKIP, _ITEMS, _REFERENTS, _ONE = range(4)


def _reading(kind: type) -> int:
    if kind in _EXACT_ATOMS or issubclass(kind, _ATOMS):
        return _SKIP
    if kind in _EXACT_CONTAINERS:
        return _ITEMS
    if kind is _types.FunctionType or kind is _types.MethodType or issubclass(kind, type):
        return _ONE
    if _MEMO_WRAPPER is not None and issubclass(kind, _MEMO_WRAPPER):
        return _ONE
    users = _users_type(kind)
    if issubclass(kind, _CONTAINERS) and not users:
        return _ITEMS
    if users or issubclass(kind, (*_MEMBER_HOLDERS, *_PLAIN_HOLDERS)):
        return _REFERENTS
    return _SKIP  # a library object: not looked into


class TooMuchToLookThrough(Exception):
    """`holds_a_closure_with_state` looked into `_HOLDERS_LOOKED_INTO`
    objects, or took `_STEPS_AT` values, and found more: what the rest
    holds is not known."""


def _plain_part(part: Any, ptype: type, atoms: frozenset) -> bool:
    """Is *part*, something a holder holds, a plain value or a list or
    tuple of them: nothing inside to look at? A record's attribute dict
    of them is too (`_plain_record`). On 3.12+ an instance hands over its
    attributes themselves, not a dict, until something asks for its
    ``__dict__``."""
    if ptype is list or ptype is tuple:
        return atoms.issuperset(map(type, part))
    return ptype is dict and _plain_record(part, atoms)


class _FlatRecords:
    """Which records of a walk hold only plain values (`record`).

    Plain is *leaves*: the atoms, and the types the walk does not look into
    (`_SKIP`: a numpy scalar, a ``Decimal``), added to as they are met. A
    record of them held by a record (a ``Point`` in a ``Segment``, an Enum
    member) is plain too: its class goes in *nested*, to be looked at once
    for its memos. *flat* holds the ids of the records so held, alive
    inside the value walked: an Enum member every row holds is judged once.
    """

    __slots__ = ("flat", "leaves", "nested", "readings")

    def __init__(self, atoms: frozenset, readings: dict[type, int]) -> None:
        self.leaves = set(atoms)
        self.readings = readings
        self.nested: set[type] = set()
        self.flat: set[int] = set()

    def record(self, item: Any, kind: type, top: bool = True) -> bool:
        """Does *item*, an object of *kind*, hold only plain values (its
        class aside): its attributes, or lists and tuples of them, or an
        attribute dict of them?"""
        leaves = self.leaves
        for part in gc.get_referents(item):
            ptype = type(part)
            if ptype in leaves or part is kind:
                continue
            if ptype is list or ptype is tuple:
                if leaves.issuperset(map(type, part)):
                    continue
                return False
            if ptype is dict:
                for v in part.values():
                    if type(v) not in leaves and not self._part(v, kind, top):
                        return False
            elif not self._part(part, kind, top):
                return False
        return True

    def _part(self, part: Any, kind: type, top: bool) -> bool:
        ptype = type(part)
        if part is kind or id(part) in self.flat:
            return True
        if ptype is list or ptype is tuple:
            return self.leaves.issuperset(map(type, part))
        reading = self.readings.get(ptype)
        if reading is None:
            reading = self.readings[ptype] = _reading(ptype)
        if reading == _SKIP:
            self.leaves.add(ptype)
            return True
        if reading == _REFERENTS and top and self.record(part, ptype, top=False):
            self.nested.add(ptype)
            self.flat.add(id(part))
            return True
        return False


def _memos_on(cls: type) -> list[tuple[str, Any, Any, bool]]:
    """``(name, memo, its wrapped function, reachable by name)`` for each
    ``functools.cache`` memo a class body member holds: itself, or inside
    a ``staticmethod`` or ``classmethod`` (reachable by its name), or a
    ``property`` or ``partialmethod`` (not). Read through ``type``'s
    descriptors and the wrappers' own slots, and kept per class while the
    members it holds are the same objects."""
    members = _CLASS_DICT(cls)
    stamp = tuple(map(id, members.values()))
    try:
        known = _CLASS_MEMOS.get(cls)
    except TypeError:  # not weakly referable
        known = None
    if known is not None and known[0] == stamp:
        return known[1]
    found = []
    for name, value in members.items():
        kind = type(value)
        named = True
        if kind is staticmethod or kind is classmethod:
            value = value.__func__
        elif kind is property:
            value, named = value.fget, False
        elif kind is functools.partialmethod:
            value, named = value.func, False
        if _MEMO_WRAPPER is not None and type(value) is _MEMO_WRAPPER:
            wrapped = value.__dict__.get("__wrapped__")
            found.append((name, value, wrapped, named))
    with contextlib.suppress(TypeError):
        _CLASS_MEMOS[cls] = (stamp, found)
    return found


#: ``class -> (the ids of its members, `_memos_on`)``.
_CLASS_MEMOS: weakref.WeakKeyDictionary[type, tuple[tuple[int, ...], list]] = weakref.WeakKeyDictionary()


def _named_class_memo_changed(
    cls: type, ran: Any, memos: list[dict] | None = None, before: Mapping[int, int] | None = None
) -> bool:
    """Is a memo on a class its module names (*cls* or a base of the
    user's) one a served result cannot stand for? Served, the next run
    finds the class's memo as the first run left it, where plain Python
    makes it again or fills it. Counted:

    - a memo whose function is not the class body's own: it was set on it
      from outside (``Holder.memo = staticmethod(cache(f))``);
    - one holding answers whose function ran in the call (*ran*, the codes
      its trace recorded): the call filled it, and a served entry would
      replay what that run printed where a run that finds it filled
      prints nothing;
    - one holding answers, with no trace, that *memos* has no record of
      and whose miss count moved since *before* (``id(memo) -> misses``
      before the call, `helper_memos.misses_before`), or that *before*
      did not know: the call may have filled it;
    - one holding answers, without *memos*: an earlier fill cannot be told
      from this call's.

    Otherwise a memo holding answers is recorded in *memos*
    (`helper_memos.class_memo_record`): the entry is served only while it
    holds them all again, so after a restart, with the memo empty, the
    call runs and fills it."""
    for base in _CLASS_MRO(cls):
        if not (_users_type(base) and _named(base)):
            continue
        prefix = type.__dict__["__qualname__"].__get__(base) + "."
        for name, memo, wrapped, named in _memos_on(base):
            if type(wrapped) is not _types.FunctionType or not wrapped.__qualname__.startswith(prefix):
                return True
            if not memo.cache_info().currsize:
                continue
            if memos is None or (ran is not None and wrapped.__code__ in ran):
                return True
            from cash.notebook import helper_memos

            record = helper_memos.class_memo_record(base, name, memo) if named else None
            if record is None:
                return True
            if any(known.get("where") == record["where"] for known in memos):
                continue  # read, and recorded, by `helper_memos.used`
            if ran is None and (before or {}).get(id(memo)) != memo.cache_info().misses:
                return True
            memos.append(record)
    return False


def _named(kind: type) -> bool:
    """Is *kind* the class its module holds under its name: a pickle or a
    copy of an instance comes back with this very class, so a memo on it is
    the one the next run uses as plain Python would, not one the result
    carries? A class a call made (``<locals>`` in its name) is not."""
    try:
        name = _CLASS_NAME(kind)
        qualname = type.__dict__["__qualname__"].__get__(kind)
        found: Any = sys.modules.get(_CLASS_MODULE(kind))
        if found is None or "<locals>" in qualname:
            return False
        for part in qualname.split("."):
            if issubclass(type(found), type):
                found = _CLASS_DICT(found).get(part)
            elif type(found) is _ModuleType:
                found = found.__dict__.get(part)
            else:
                return False
        return found is kind and bool(name)
    except Exception:  # noqa: BLE001 - not known to be named: looked into
        return False


_TYPE_FLAGS = type.__dict__["__flags__"].__get__
_HEAP_TYPE = 1 << 9  # Py_TPFLAGS_HEAPTYPE: made by a class statement or `type()`


def _users_type(kind: type) -> bool:
    try:
        known = _USERS_TYPES.get(kind)
    except TypeError:  # not weakly referable
        known = None
    if known is None:
        try:
            # A class made in a notebook cell: `__main__`, which under a
            # test runner is the runner's own module; or made by `exec` in a
            # namespace with no `__name__`, which names it `builtins`. Not
            # one of the interpreter's own types that ``builtins`` does not
            # name (``module``, ``function``): looked into, every statement
            # with an ``import`` was not stored, numpy's memos counted as its.
            module = _CLASS_MODULE(kind)
            known = (
                module in MAIN_MODULE_NAMES
                or (
                    module == "builtins"
                    and _TYPE_FLAGS(kind) & _HEAP_TYPE
                    and builtins.__dict__.get(_CLASS_NAME(kind)) is not kind
                )
                or (isinstance(module, str) and is_user_code_module(sys.modules.get(module)))
            )
        except Exception:  # noqa: BLE001 - not known to be the user's: not looked into
            known = False
        with contextlib.suppress(TypeError):
            _USERS_TYPES[kind] = known
    return known


def held_by(item) -> list | None:
    """What a holder that is not a container keeps a function or a memo in:
    a function's closure, defaults and attributes, a ``partial``'s function
    and arguments, a bound method's function and object, what a
    ``SimpleNamespace``, an ``argparse.Namespace``, a ``deque`` or a user's
    object holds (its class included), and the members of a user's class
    and of its user's bases. None for anything else: a library object is
    not looked into. Read without asking the object (``gc.get_referents``,
    ``type``'s own descriptors): an object's ``__getattr__``, a ``__dict__``
    property or a metaclass runs no code here."""
    kind = type(item)
    if kind is _types.FunctionType:
        parts = [cell.cell_contents for cell in item.__closure__ or () if _filled(cell)]
        parts.extend(item.__defaults__ or ())
        parts.extend((item.__kwdefaults__ or {}).values())
        return parts + list(item.__dict__.values())
    if kind is _types.MethodType:
        return [item.__func__, item.__self__]
    if issubclass(kind, type):
        # A class its module names: what is on it is the live class's, which
        # the next run uses in plain Python too (a module-level dataclass, a
        # class with a ``functools.cache`` method).
        if not _users_type(item) or _named(item):
            return None
        return [
            value
            for base in _CLASS_MRO(item)
            if _users_type(base) and not _named(base)
            for value in _CLASS_DICT(base).values()
        ]
    if issubclass(kind, (*_MEMBER_HOLDERS, *_PLAIN_HOLDERS)) or _users_type(kind):
        # A user's list or dict subclass included: its items and its own
        # attributes, without asking it (``__iter__``, ``values``).
        return gc.get_referents(item)
    return None


def _filled(cell) -> bool:
    try:
        cell.cell_contents  # an empty cell raises
    except ValueError:
        return False
    return True


def holds_a_closure_with_state(
    value, ran: Any = None, memos: list[dict] | None = None, before: Mapping[int, int] | None = None
) -> bool:
    """Whether *value* is, or holds however deeply nested, a function that
    changes what its closure holds when called (a counter or a log a factory
    returned), or a ``functools.cache`` memo. Looked for in tuples, lists,
    sets and dicts, and in what a function, a ``partial``, a bound method, a
    ``SimpleNamespace``, a user's object or a user's class holds
    (`held_by`); a library object is not looked into.

    Such a value cannot be served from the cache. The store keeps a function
    by reference, so a hit hands back the closure the last run made, with
    everything it has counted or collected since: ``counter =
    make_counter(0)`` run again gave a counter that went on from 2. A
    ``functools.cache`` memo is one too: ``functools.cache(g)`` stored, then
    filled later in the statement, came back already filled after a restart,
    and ``g`` did not run (or print) where plain Python ran it; held on an
    object (``SimpleNamespace(f=cache(g))``), in a wrapper's closure or in a
    ``partial``, a deep copy of the result still holds the same memo, as a
    function or a class copies as itself. Only two
    levels of containers were looked into, so ``{"on": {"click":
    [counter]}}`` was served; a seen set now ends a container that holds
    itself.

    A class its module names is looked into for its memos only: a copy of
    the result comes back with that very class, whose memo the next run
    uses in plain Python too (a dataclass, a class with a
    ``functools.cache`` method), unless the call put the memo there or
    filled it; one it only read is recorded in *memos* (what
    `helper_memos.used` recorded, added to) for the entry to be served only
    while the memo holds its answers again (`_named_class_memo_changed`;
    *ran* is the code the call's trace recorded, None without one, and
    *before* the memos' miss counts before it). A class
    a call made is looked into
    whole. The standard library's guard on a dataclass's ``__repr__`` keeps
    no state between calls. Past `_HOLDERS_LOOKED_INTO` objects with
    something inside to look at, or `_STEPS_AT` values in all, it raises
    `TooMuchToLookThrough`: not known, so not served.
    """
    if _plain_data.is_tree(value):
        # JSON-like data holds no function: answered a level at a time at C
        # speed, where the walk below takes a Python step per container --
        # seconds for a million parsed records.
        return False
    stack = [value]
    seen: set[int] = set()
    looked_into = 0
    atoms = _EXACT_ATOMS
    # How each type is read, for this walk: thousands of records share one.
    readings: dict[type, int] = {}
    record_kinds: set[type] = set()  # classes of flat records, pushed once
    flat_records = _FlatRecords(atoms, readings)
    nested = flat_records.nested
    steps = 0
    while stack:
        item = stack.pop()
        steps += 1
        if steps > _STEPS_AT:
            raise TooMuchToLookThrough
        # By type, not `isinstance`: that asks an object for its
        # `__class__`, which may be the user's code.
        kind = type(item)
        reading = readings.get(kind)
        if reading is None:
            reading = readings[kind] = _reading(kind)
        if reading == _SKIP:
            continue
        if reading == _ONE:
            if kind is _types.FunctionType:
                if (
                    item.__closure__ is None
                    and item.__defaults__ is None
                    and not item.__kwdefaults__
                    and not item.__dict__
                ):
                    continue  # holds nothing of its own: most functions
                if id(item.__code__) in _REPR_GUARD_IDS:
                    continue  # a dataclass's ``__repr__``: its set is empty between calls
                if changes_its_closure(item):
                    return True
            elif _MEMO_WRAPPER is not None and issubclass(kind, _MEMO_WRAPPER):
                return True
        if reading != _ITEMS:
            if id(item) in seen:
                continue
            seen.add(id(item))  # a class each record holds is looked at once
            if reading == _ONE and issubclass(kind, type) and _users_type(item):
                # Its bases its module names, and itself when it is one:
                # their memos only (`_named_class_memo_changed`).
                if _named_class_memo_changed(item, ran, memos, before):
                    return True
                if _named(item):
                    continue
            held = gc.get_referents(item) if reading == _REFERENTS else held_by(item)
            if held:
                pushed = False
                for part in held:
                    ptype = type(part)
                    if ptype in atoms or id(part) in seen:
                        continue
                    if _plain_part(part, ptype, atoms):
                        continue  # a record's attributes: nothing inside to look at
                    stack.append(part)
                    pushed = True
                if pushed:
                    # Only an object with something inside to look at counts:
                    # 100,000 records of plain values are looked through.
                    looked_into += 1
                    if looked_into > _HOLDERS_LOOKED_INTO:
                        raise TooMuchToLookThrough
            continue
        if id(item) in seen:
            continue
        children = item.values() if issubclass(kind, dict) else item
        # Only atoms, by exact type: asked at C speed, where the loop
        # below took a Python step per item (22 ms for 200,000 ints).
        if atoms.issuperset(map(type, children)):
            continue
        # A container holding only atoms (a million small tuples of a
        # record list) has nothing to look into and is not remembered.
        pushed = False
        for child in children:
            ctype = type(child)
            if ctype in atoms:
                continue
            if ctype is tuple:
                for part in child:
                    if type(part) not in atoms:
                        break
                else:
                    continue
            reading = readings.get(ctype)
            if reading is None:
                reading = readings[ctype] = _reading(ctype)
            if reading == _REFERENTS and (nested.clear() or flat_records.record(child, ctype)):
                # A record of plain values (a dataclass row): looked at here,
                # with no step of its own, so a million rows are not past
                # the bound. Its class, and those of the records it holds,
                # are looked at once, for their memos.
                nested.add(ctype)
                if not record_kinds.issuperset(nested):
                    stack.extend(nested - record_kinds)
                    record_kinds |= nested
                    pushed = True
                continue
            stack.append(child)
            pushed = True
        if pushed:
            seen.add(id(item))  # the containers stay alive inside *value*
    return False


def call_cache_key(
    site: CallSite,
    *,
    ctx: CacheKeyContext,
    arg_digests: list[str],
    loop_vars: dict[str, object],
    loop_var_digests: Mapping[str, str] | None = None,
    global_digests: Mapping[str, str] | None = None,
    by_content: bool = False,
    name_digests: Mapping[str, str] | None = None,
    code_names: frozenset[str] = frozenset(),
) -> str | None:
    """The cache key for one intercepted call, or ``None`` to refuse caching it.

    Built by :func:`compute_cache_key` (the only place a key is assembled) from
    the *call's* source and free names, under the ``"call"`` namespace. The
    callee is one of the free names, so editing it re-keys the call and the
    globals it reaches are folded in; bare-name arguments resolve through the
    lineage ladder, a dict lookup per call. Then, each hashed and
    ``|``-delimited so no two parts can run together:

    * *arg_digests*: a content hash of each argument that is not a bare name,
      one per ``site.computed_arg_positions``. Required: ``compute(next(it))``
      reads an ``it`` whose lineage never moves, so without it every
      iteration would share one entry. A count mismatch returns ``None``;
      the caller then runs the call uncached.
    * *loop_vars*: the enclosing loops' variables, by value, keyed
      ``"{depth}:{name}"`` so a name reused by a nested loop keeps both.
      Values, not the iteration context: they are order-independent and
      stable across runs, so reordering an iterable or re-running only a
      loop's tail keys correctly. Dunder entries (``__iterable_lineage__``,
      which changes for every iteration on a reorder) are filtered out here,
      past the depth prefix. *loop_var_digests* are their precomputed full
      hashes (see :func:`_loop_var_digest`); leaving it out is correct, only
      slower.
    * *global_digests*: the pre-call state of each global the callee writes,
      so a call entered with a different accumulator is not served another's
      end state.
    * ``site.stmt_identity``: the enclosing statement, so two statements
      making the same call (``fetch_next(conn)`` in two cells' loops) keep
      their own entries.

    *by_content* keys the call on what it receives: the names read only in
    computed arguments, and the bare names in *name_digests* (full hashes of
    those arguments), leave the base, and the statement's identity leaves the
    key, so fixing some of a frame's rows re-runs only the calls whose inputs
    changed. Only the caller can say that is safe: :class:`CallUnit` does
    when everything the call reads is plain data or code.

    *code_names* are the names read by the arguments whose value is a
    function (``fu.fit_ets`` in ``backtest(y, fu.fit_ets, cuts)``): under
    content keying they stay in the base, as the callee's names do, and the
    call keeps its spelled source, so a module read is narrowed to the
    functions the call names (``fit_ets`` among them) and an edit to one
    moves the key. A function's pickled bytes are its name, not its code.
    """
    free_names = set(site.free_names)
    source, occurrence = site.source, site.occurrence_index
    if by_content:
        free_names -= site.content_names - code_names
        free_names -= set(name_digests or ()) - code_names
        # The call's shape, not its spelling: the computed arguments' values
        # are in the key. And no occurrence: the same call on the same values
        # twice in a cell is the same result, which is what content keying
        # already asserts across statements.
        if code_names:
            occurrence = 0
        elif attribute_or(site, "content_source", ""):
            source, occurrence = site.content_source, 0
    keyed = compute_cache_key(
        source,
        free_names,
        ctx=ctx,
        occurrence_index=occurrence,
        namespace="call",
    )
    if keyed.matches_nothing:
        # No entry could ever match it: the call runs uncached, as its
        # statement does, and stores nothing.
        return None
    base = keyed.cache_key
    # A collapsed key is a wrong answer; an uncached call is merely slow.
    if len(arg_digests) != len(site.computed_arg_positions):
        return None

    filtered_loop_vars = {name: value for name, value in loop_vars.items() if not _is_dunder_loop_var(name)}

    if not arg_digests and not filtered_loop_vars and not site.stmt_identity and not global_digests and not by_content:
        return base
    # Length-prefixed and `|`-delimited, so two calls cannot collide on a join.
    parts = [f"{len(arg_digests)}"]
    parts.extend(arg_digests)
    resolved_digests: Mapping[str, str] = loop_var_digests or {}
    parts.extend(
        f"{name}={_loop_var_digest(name, value, resolved_digests)}"
        for name, value in sorted(filtered_loop_vars.items())
    )
    # `g:`, `n:`: a global or a name cannot take a loop variable's slot.
    if global_digests:
        parts.extend(f"g:{name}={digest}" for name, digest in sorted(global_digests.items()))
    if by_content:
        parts.extend(f"n:{name}={digest}" for name, digest in sorted((name_digests or {}).items()))
        # Marked, so a key without the statement can never equal one with it.
        parts.append("by=content")
    elif site.stmt_identity:
        # Hashed, so the statement's own text cannot break the delimiting.
        parts.append("stmt=" + hashlib.sha256(site.stmt_identity.encode("utf-8")).hexdigest())
    return "call:" + hashlib.sha256((base + "|" + "|".join(parts)).encode("utf-8")).hexdigest()


#: Values whose content is all there is to them: nothing about them can change
#: while their hash stays put. The test :func:`_keys_by_content` applies to
#: everything a call reads before keying it without its statement.
_PLAIN_ATOMS = (
    int,
    float,
    complex,
    str,
    bytes,
    type(None),
    _decimal.Decimal,
    _fractions.Fraction,
    _dt.date,
    _dt.time,
    _dt.timedelta,
    _pathlib.PurePath,
    range,
)
#: How many values one call may have looked at before the answer is "not
#: plain" -- a long list is keyed the old way rather than walked.
_PLAIN_BUDGET = 10_000
#: Computed arguments are hashed in full under content keying; past this many
#: bytes the hash could cost more than the call saves, so the old key stays.
_CONTENT_KEY_MAX_BYTES = 64 * 1024 * 1024
#: An argument passed by name is hashed per call only up to this size; a
#: bigger one keeps its lineage, a dict lookup.
_NAME_CONTENT_MAX_BYTES = 1024 * 1024


def _is_plain_data(value, budget: list[int]) -> bool:
    """True when *value* is data whose content hash covers all of its state.

    Scalars, strings, dates, paths; numpy arrays and scalars without Python
    objects inside; pandas frames, series and indexes (hashed cell by cell);
    builtin containers of those. Anything else -- a connection, an iterator,
    a model, a user class -- may hold state its key cannot see.
    """
    budget[0] -= 1
    if budget[0] < 0:
        return False
    if isinstance(value, _PLAIN_ATOMS):
        return True
    np = sys.modules.get("numpy")
    if np is not None:
        if isinstance(value, np.ndarray):
            return not value.dtype.hasobject
        if isinstance(value, np.generic):
            return not isinstance(value, np.object_)
    pd = sys.modules.get("pandas")
    if pd is not None and isinstance(value, (pd.DataFrame, pd.Series, pd.Index)):
        return True
    if type(value) in (list, tuple, set, frozenset):
        return all(_is_plain_data(item, budget) for item in value)
    if type(value) is dict:
        return all(_is_plain_data(k, budget) and _is_plain_data(v, budget) for k, v in value.items())
    return False


@functools.lru_cache(maxsize=1024)
def _code_names(code: _types.CodeType) -> frozenset[str]:
    """Global names *code* and the functions nested in it load.

    From the bytecode, not ``co_names``, which holds attribute names too:
    ``os.open`` would read as the global ``open`` -- cash's file-tracking
    wrapper in a notebook.
    """
    names = set(argvals(code, NAME_LOADS))
    for const in code.co_consts:
        if isinstance(const, _types.CodeType):
            names |= _code_names(const)
    return frozenset(names)


def _plain_or_code(value, seen: set[int], budget: list[int]) -> bool:
    if type(value) is _types.FunctionType:
        if attribute_or(value, "_is_file_tracker_patch", False) or attribute_or(value, "_cash_cached", False):
            return True  # cash's own: keyed or tracked by cash itself

        filename = attribute_or(value.__code__, "co_filename", "") or ""
        # A cell's code has a `<cash-...>` / `<ipython-...>` name: the user's.
        if filename and not is_pseudo_filename(filename) and not is_user_path(filename):
            # A library's function is code, as its classes are. Its module
            # state is no more in a lineage key than in a content key, and
            # walking it refused: sklearn's `normalize` is a validating
            # wrapper whose globals hold non-plain state, so `fit_vectors`
            # re-fitted on byte-identical text.
            return True
        return _callee_state_is_plain(value, seen, budget)
    if isinstance(value, (_ModuleType, type, _types.BuiltinFunctionType)):
        return True
    np = sys.modules.get("numpy")
    if np is not None and isinstance(value, np.ufunc):
        return True
    return _is_plain_data(value, budget)


def _callee_state_is_plain(fn, seen: set[int], budget: list[int]) -> bool:
    """True when every global, closure cell and default *fn* can reach --
    through the functions it calls too -- is plain data or code."""
    if id(fn) in seen:
        return True
    seen.add(id(fn))
    namespace = attribute_or(fn, "__globals__", None) or {}
    for name in _code_names(fn.__code__):
        if name in namespace and not _plain_or_code(namespace[name], seen, budget):
            return False
    for cell in fn.__closure__ or ():
        try:
            contents = cell.cell_contents
        except ValueError:  # an empty cell: nothing bound yet
            continue
        if not _plain_or_code(contents, seen, budget):
            return False
    defaults = (*(fn.__defaults__ or ()), *(fn.__kwdefaults__ or {}).values())
    return all(_is_plain_data(value, budget) for value in defaults)


def _nbytes(value) -> int:
    """Roughly how many bytes a full content hash of *value* reads."""
    np = sys.modules.get("numpy")
    if np is not None and isinstance(value, np.ndarray):
        return int(value.nbytes)
    pd = sys.modules.get("pandas")
    if pd is not None and isinstance(value, (pd.DataFrame, pd.Series)):
        sized = pandas_nbytes(value)
        if sized is not None:
            return int(sized)
        usage = value.memory_usage(index=True, deep=False)
        return int(usage.sum() if has_attribute(usage, "sum") else usage)
    if isinstance(value, (str, bytes)):
        return len(value)
    return 0


def global_names_reached(fn, seen: set[int] | None = None) -> set[str]:
    """Global names *fn* loads, and those of the functions and classes it
    reaches, however deep: a loop variable read seven calls down was not
    seen, so it was left out of the call's key and the call served the
    value from another iteration. *seen* ends cycles."""
    seen = set() if seen is None else seen
    names: set[str] = set()
    pending = [fn]
    while pending:
        current = pending.pop()
        code = attribute_or(current, "__code__", None)
        if code is None or id(current) in seen:
            continue
        seen.add(id(current))
        own = _code_names(code)
        names |= own
        namespace = attribute_or(current, "__globals__", None) or {}
        for name in own:
            value = namespace.get(name)
            if type(value) is _types.FunctionType:
                pending.append(value)
            elif isinstance(value, type) and id(value) not in seen:
                # A class the callee builds or calls into: its methods read globals too.
                seen.add(id(value))
                for member in vars(value).values():
                    member = attribute_or(member, "__func__", member)
                    if type(member) is _types.FunctionType:
                        pending.append(member)
    return names


def _user_global_names_reached(fn) -> set[str]:
    """`global_names_reached` through the user's own code only: a library's
    function reads its library's globals, never a notebook's loop variable,
    and walking a library's classes (``SARIMAX``'s methods, and theirs)
    cost seconds per call."""
    names: set[str] = set()
    seen: set[int] = set()
    pending = [fn]
    while pending:
        current = pending.pop()
        code = attribute_or(current, "__code__", None)
        if code is None or id(current) in seen:
            continue
        seen.add(id(current))
        filename = code.co_filename or ""
        if filename and not is_pseudo_filename(filename) and not is_user_path(filename):
            continue
        own = _code_names(code)
        names |= own
        namespace = attribute_or(current, "__globals__", None) or {}
        for name in own:
            value = namespace.get(name)
            if type(value) is _types.FunctionType:
                pending.append(value)
            elif isinstance(value, type) and id(value) not in seen:
                # A class it builds: its methods are followed as far as they
                # are the user's (the file test above).
                seen.add(id(value))
                pending.extend(
                    member
                    for member in (attribute_or(m, "__func__", m) for m in vars(value).values())
                    if type(member) is _types.FunctionType
                )
    return names


def _loop_vars_the_call_can_read(
    fn,
    site: CallSite,
    loop_vars: Mapping[str, object],
    name_digests: Mapping[str, str] | None,
    handed: list | tuple = (),
) -> dict[str, object]:
    """The enclosing loops' variables a content-keyed call can still read
    without them being in its key.

    A loop variable is in a call's key to close a channel the arguments do not
    cover: hidden state behind a name, or a global the callee reads. Keyed on
    what it receives, a call's arguments are hashed by value, so a loop
    variable passed in (`make_features(cleaned[mid], win)`) is already there,
    and one the call never reads cannot change its result. Kept anyway, it
    made the sweep's keys differ from the same call outside a loop: scoring
    with the chosen window re-fitted all 200 machines the sweep had just fitted.
    Kept: a loop variable the callee, or a function it
    calls, reads as a global; and one named in the call itself but not hashed
    by value (an argument too big to hash per call). Only a variable the
    arguments carry is dropped: one the call does not mention stays, since
    what the callee can reach is followed through functions and classes but
    not every path (a dispatch table, an object's attribute). The functions
    *handed* to the call as arguments run too: what they read counts.
    """
    hashed = set(name_digests or ()) | set(site.content_names)
    reached = None
    kept = {}
    for key, value in loop_vars.items():
        bare = key.split(":", 1)[1] if ":" in key else key
        if bare in hashed:
            if reached is None:
                reached = global_names_reached(fn)
                for other in handed:
                    reached |= _user_global_names_reached(other)
            if bare not in reached:
                continue
        kept[key] = value
    return kept


def _is_code_arg(value, budget: list[int]) -> bool:
    """True when *value*, an argument, is a function whose own state is plain
    (`_plain_or_code`): a fit function handed to a backtest. Only a plain
    function, a builtin or a ufunc; a class, a module or a bound method is
    not taken for code here."""
    if type(value) is _types.FunctionType:
        return _plain_or_code(value, set(), budget)
    if type(value) is _types.BuiltinFunctionType:
        return True
    np = sys.modules.get("numpy")
    return np is not None and isinstance(value, np.ufunc)


def code_arg_positions(site: CallSite, args: tuple, kwargs: dict) -> tuple[int, ...] | None:
    """Positions of the arguments that are functions (see `_is_code_arg`),
    or None when one is not a plain reference to a name
    (``CallSite.arg_names``): its code could not be keyed by name."""
    combined = (*args, *kwargs.values())
    np = sys.modules.get("numpy")
    positions = tuple(
        pos
        for pos, value in enumerate(combined)
        if issubclass(type(value), (_types.FunctionType, _types.BuiltinFunctionType))
        or (np is not None and isinstance(value, np.ufunc))
    )
    if not positions:
        return ()
    names = attribute_or(site, "arg_names", ())
    if len(names) != len(combined) or any(names[pos] is None for pos in positions):
        return None
    return positions


def _keys_by_content(
    fn,
    site: CallSite,
    args: tuple,
    kwargs: dict,
    loop_vars: Mapping[str, object],
    code_positions: tuple[int, ...] = (),
) -> bool:
    """Whether the call may be keyed on what it receives (see
    :func:`call_cache_key`'s *by_content*).

    Only when nothing it reads can change while its key stays put: every
    argument, every loop variable around it, and everything the callee
    reaches is plain data or code, and the arguments to hash in full are
    small enough to be worth it. An argument at one of *code_positions* may
    be a function (`_is_code_arg`): the key holds its code through its names
    (*code_names* of `call_cache_key`).
    """
    combined = (*args, *kwargs.values())
    budget = [_PLAIN_BUDGET]
    if not all(
        _is_code_arg(value, budget) if pos in code_positions else _is_plain_data(value, budget)
        for pos, value in enumerate(combined)
    ):
        return False
    if not all(_is_plain_data(value, budget) for name, value in loop_vars.items() if not _is_dunder_loop_var(name)):
        return False
    computed = sum(_nbytes(combined[pos]) for pos in site.computed_arg_positions if pos < len(combined))
    if computed > _CONTENT_KEY_MAX_BYTES:
        return False
    return _callee_state_is_plain(fn, set(), budget)


def global_digests(fn, names: tuple[str, ...]) -> dict[str, str]:
    """PRE-call content hashes of the globals *fn* writes, for the key.

    ``compute_hash``, every byte, for the same reason
    :func:`_loop_var_digest` documents: this IS the discriminator. Two
    accumulator states that agree at both ends must not key alike, and an
    accumulator is precisely the shape that grows in the middle.

    The cost this admits is real and bounded by how rare the case is: a
    callee that writes a global at all is uncommon, and the hash is over
    the accumulator, not over the arguments.

    A name that cannot be hashed at all is omitted, which makes the key
    LESS discriminating -- so ``call_effects.capture_globals`` independently
    refuses to store any entry whose capture is not sound, and the pair of
    them fails closed.
    """
    globals_dict = attribute_or(fn, "__globals__", None) or {}
    cells = closure_cells(fn)
    digests: dict[str, str] = {}
    for name in names:
        try:
            value = cells[name].cell_contents if name in cells else globals_dict[name]
            digests[name] = compute_hash(value)
        except Exception:  # noqa: BLE001 - a missing digest only widens the key
            logger.debug("call unit: could not digest global %r", name)
    return digests


#: Call sites (with their loop values, see `CallKeys._site_id`) whose key
#: parts are kept for the miss reason; the least recently keyed go first.
_SITE_PARTS_KEPT = 4096


class CallKeys:
    """Builds each intercepted call's key from the live processor state, and
    remembers what each call site was keyed on so a miss can say what moved.

    The providers are read at key time, not captured: one ``CallCache``
    serves every statement, and each brings its own lineage and loop.
    """

    def __init__(
        self,
        ctx_provider: Callable[[], CacheKeyContext],
        loop_vars_provider: Callable[[], dict[str, object]] | None = None,
        loop_var_digests_provider: Callable[[], Mapping[str, str]] | None = None,
    ):
        self._ctx_provider = ctx_provider
        # See `call_cache_key`'s `loop_vars` section. `None` means "no loop
        # context available", which degrades to `{}`: the key a call gets
        # outside any loop.
        self._loop_vars_provider = loop_vars_provider or (lambda: {})
        # See `call_cache_key`'s `loop_var_digests` section and
        # `_loop_var_digest`'s docstring. `None` is always CORRECT (every
        # entry falls through to a fresh `compute_hash`), only slower.
        self._loop_var_digests_provider = loop_var_digests_provider or (lambda: {})
        # Per call SITE: what its key was built from last time, and what moved
        # since -- the badge's answer to "why did this re-run?".
        self._site_parts: dict[tuple, tuple[str, dict]] = {}
        self._site_reason: dict[tuple, str | None] = {}
        # Sites already keyed in THIS statement run. A site called once per
        # loop item keys differently per item BY DESIGN -- that is the loop
        # variable doing its job, not something to explain. Only the first
        # call of each run is compared, against the first call of the last.
        self._keyed_this_run: set = set()

    def begin_statement(self) -> None:
        """A new statement run: each site's first call is compared again."""
        self._keyed_this_run.clear()

    def key(
        self, site: CallSite, args: tuple, kwargs: dict, global_digests: Mapping[str, str] | None = None, fn=None
    ) -> str | None:
        """The call's key. With *fn*, keyed on what it receives when
        :func:`_keys_by_content` allows it."""
        if site.has_unpacking and fn is not None:
            return self._build_unpacked_key(site, args, kwargs, fn)
        if site.has_unpacking:
            # `*args`/`**kwargs` unpacking means the call's live arity is not
            # statically known. `site.computed_arg_positions` is a STATIC
            # count (every position, fail-closed -- see
            # `_computed_arg_positions`), which need not match the RUNTIME
            # flattened `(*args, *kwargs.values())` length: `compute(*pair())`
            # has one static position but the pair unpacks to two live
            # arguments, and indexing only position 0 would hash the first
            # element and silently ignore the rest (reproduced as a second, DIFFERENT pair() result being served the
            # first call's cached value). Refuse the whole site rather than
            # mint a key that looks discriminated but isn't; an uncached call
            # is merely slow.
            return None
        try:
            loop_vars = self._current_loop_vars()
            code_positions = code_arg_positions(site, args, kwargs)
            by_content = (
                fn is not None
                and code_positions is not None
                and _keys_by_content(fn, site, args, kwargs, loop_vars, code_positions)
            )
            code_positions = code_positions or ()
            if attribute_or(site, "in_loop_unit", False) and not by_content:
                return None
            arg_digests = self._arg_digests(site, args, kwargs)
            code_names: frozenset[str] = frozenset()
            if by_content and code_positions:
                code_names = frozenset().union(*(site.arg_names[pos] for pos in code_positions))
            name_digests = self._name_digests(site, args, kwargs, code_names) if by_content else None
            if by_content and loop_vars:
                combined = (*args, *kwargs.values())
                loop_vars = _loop_vars_the_call_can_read(
                    fn, site, loop_vars, name_digests, [combined[pos] for pos in code_positions]
                )
            ctx = self._ctx_provider()
            key = call_cache_key(
                site,
                ctx=ctx,
                arg_digests=arg_digests,
                loop_vars=loop_vars,
                loop_var_digests=self._current_loop_var_digests(),
                global_digests=global_digests,
                by_content=by_content,
                name_digests=name_digests,
                code_names=code_names,
            )
            self._note_key_parts(site, key, ctx, arg_digests, global_digests)
            if _shadow.SHADOW is not None and key is not None:
                _shadow.SHADOW.note_call(
                    key,
                    site,
                    (arg_digests, name_digests, global_digests, self._current_loop_var_digests()),
                    by_content=set(name_digests or ()) | (set(site.content_names) if by_content else set()),
                )
            return key
        except Exception:  # noqa: BLE001 - never let keying break the call
            logger.debug("call unit: key build failed for %s", site.source)
            return None

    def _build_unpacked_key(self, site: CallSite, args: tuple, kwargs: dict, fn) -> str | None:
        """The key of a call with ``*``/``**`` unpacking, keyed on what it received.

        ``fit_series(g, **TUNED.get(dept, {}))`` in a comprehension ran uncached:
        positions written in the source say nothing about the values
        that arrive, so the site was refused (``compute(*pair())``
        had keyed the first of two values). What did arrive is in hand here:
        every positional value, and every keyword with its name, hashed in
        full. Only when the call may be keyed on content at all
        (:func:`_keys_by_content`); otherwise refused as before.
        """
        try:
            count = len(args) + len(kwargs)
            received = dataclasses.replace(
                site,
                computed_arg_positions=tuple(range(count)),
                local_arg_positions=tuple(range(count)),
                name_arg_positions=(),
                has_unpacking=False,
            )
            loop_vars = self._current_loop_vars()
            if not _keys_by_content(fn, received, args, kwargs, loop_vars):
                return None
            digests = [compute_hash(value) for value in args]
            digests.extend(f"{name}:{compute_hash(kwargs[name])}" for name in sorted(kwargs))
            if loop_vars:
                loop_vars = _loop_vars_the_call_can_read(fn, received, loop_vars, None)
            return call_cache_key(
                received,
                ctx=self._ctx_provider(),
                arg_digests=digests,
                loop_vars=loop_vars,
                loop_var_digests=self._current_loop_var_digests(),
                by_content=True,
            )
        except Exception:  # noqa: BLE001 - never let keying break the call
            logger.debug("call unit: key build failed for %s", site.source)
            return None

    def _current_loop_vars(self) -> dict[str, object]:
        """The live enclosing loop's non-dunder iteration vars, or ``{}``.

        Wired to ``CallRouting.current_loop_vars_for_call_key`` (see that class's
        ``_loop_vars`` stack, pushed/popped by
        ``ForLoopHandler._process_one_iteration`` around each iteration's body)
        via ``CallCache``'s ``loop_vars_provider``. Guarded independently of
        ``key``'s own try/except: a provider failure should degrade to
        "no loop discriminator" ``{}`` -- same as running outside a loop --
        not to refusing the key (and therefore the call's caching) entirely.
        """
        try:
            loop_vars = self._loop_vars_provider()
        except Exception:  # noqa: BLE001 - degrade, don't refuse the whole key
            logger.debug("call unit: loop_vars_provider failed for this call")
            return {}
        return loop_vars if isinstance(loop_vars, dict) else {}

    def _current_loop_var_digests(self) -> Mapping[str, str]:
        """The live enclosing loop's precomputed loop-var digests, or ``{}``.

        Wired to ``CallRouting.current_loop_var_digests_for_call_key`` (see that
        class's ``_loop_var_digests`` stack -- pushed/popped in
        lockstep with ``_loop_vars``, by the same
        ``loop_vars_scope`` call) via ``CallCache``'s
        ``loop_var_digests_provider``. Guarded independently of
        ``key``'s own try/except, same reasoning as
        ``_current_loop_vars``: a provider failure degrades to "no
        precomputed digest available" ``{}``, which ``_loop_var_digest``
        treats as "fall through to a fresh `compute_hash`" -- slower,
        never wrong -- not to refusing the key entirely.
        """
        try:
            digests = self._loop_var_digests_provider()
        except Exception:  # noqa: BLE001 - degrade, don't refuse the whole key
            logger.debug("call unit: loop_var_digests_provider failed for this call")
            return {}
        return digests if isinstance(digests, Mapping) else {}

    @staticmethod
    def _name_digests(
        site: CallSite, args: tuple, kwargs: dict, code_names: frozenset[str] = frozenset()
    ) -> dict[str, str]:
        """Full hashes of the small arguments passed as a bare name (see
        :func:`call_cache_key`'s *name_digests*). Past
        ``_NAME_CONTENT_MAX_BYTES`` a value keeps its lineage instead, and a
        function (one of *code_names*) is keyed by its name's code."""
        combined = (*args, *kwargs.values())
        digests = {}
        for name, pos in attribute_or(site, "name_arg_positions", ()):
            if name in code_names:
                continue
            if pos < len(combined) and _nbytes(combined[pos]) <= _NAME_CONTENT_MAX_BYTES:
                digests[name] = compute_hash(combined[pos])
        return digests

    def _arg_digests(self, site: CallSite, args: tuple, kwargs: dict) -> list[str]:
        """Content hashes of the live arguments at ``site.computed_arg_positions``.

        Positions are in ``(*args, *kwargs.values())`` order, matching how
        :func:`_computed_arg_positions` numbered them at rewrite time. A
        position beyond the live call's arity (the wrapped function called with
        a different shape than the site predicted) is simply not appended --
        the resulting length mismatch is caught by ``call_cache_key`` itself,
        which refuses rather than mint a key with a discriminator missing.
        """
        combined = (*args, *kwargs.values())
        return [compute_hash(combined[pos]) for pos in site.computed_arg_positions if pos < len(combined)]

    def _note_key_parts(self, site: CallSite, key, ctx, arg_digests, global_digests) -> None:
        """Remember what this call site was keyed on, and what moved since the
        last time it was keyed (read back by :meth:`why_missed`).

        In-memory and per site, like the statement guard's own components. A
        first run in a fresh kernel has nothing to compare against and says
        nothing -- the same deliberate silence a statement keeps, since "first
        time" is self-evident to someone running a cell for the first time.
        """
        if key is None:
            return
        try:
            site_id = self._site_id(site)
            if site_id in self._keyed_this_run:
                return
            self._keyed_this_run.add(site_id)
            lineages = attribute_or(ctx, "variable_lineage", None) or {}
            parts = {name: lineages.get(name) for name in site.free_names}
            for i, digest in enumerate(arg_digests or ()):
                parts[f"argument {i + 1}"] = digest
            for name, digest in (global_digests or {}).items():
                parts[name] = digest
            seen = self._site_parts.pop(site_id, None)
            self._site_parts[site_id] = (key, parts)
            while len(self._site_parts) > _SITE_PARTS_KEPT:
                oldest = next(iter(self._site_parts))
                del self._site_parts[oldest]
                self._site_reason.pop(oldest, None)
            if not seen or seen[0] == key:
                self._site_reason.pop(site_id, None)
                return
            moved = sorted(name for name in set(parts) | set(seen[1]) if parts.get(name) != seen[1].get(name))
            # The key moved with no named part of it moving: something else did
            # (a file the callee reads, the callee's own source). Naming
            # nothing beats naming the wrong thing.
            named = ", ".join(moved[:3]) + (", ..." if len(moved) > 3 else "")
            self._site_reason[site_id] = f"changed: {named}" if moved else None
        except Exception:  # noqa: BLE001 - attribution never breaks a call
            logger.debug("call unit: could not note key parts for %s", site.source)

    def _site_id(self, site: CallSite) -> tuple:
        """The call site, and the values of the loop variables around it.

        A loop's iterations call one site with other loop values by design:
        compared with the iteration before, every miss named the loop
        variable (``0/9 hit - changed: hfa`` for a grid whose ``hfa`` took
        the same three values as the run before). So a call is compared
        with the last call of the same site under the same loop values.
        """
        loop_vars = {name: value for name, value in self._current_loop_vars().items() if not _is_dunder_loop_var(name)}
        digests = self._current_loop_var_digests()
        values = tuple(sorted((name, _loop_var_digest(name, value, digests)) for name, value in loop_vars.items()))
        return (site.source, site.occurrence_index, attribute_or(site, "stmt_identity", ""), values)

    def why_missed(self, site: CallSite) -> str | None:
        """Which named part of this call's key moved since it was last keyed.

        A sweep re-ran and the badge said only "0/6 hit", so the user had to
        guess why -- and guessed wrong, then reported the re-run as a
        suspected bug.
        """
        try:
            return self._site_reason.get(self._site_id(site))
        except Exception:  # noqa: BLE001 - the reason is a hint; none is fine
            return None
