"""What a loop changed in place, restored into the object it changed.

``for f in files: ...; p.update()`` changes ``p``. A hit that bound ``p`` to
the restored copy put a new object where the old one was: ``weakref.ref(p)``
then gave None, a ``WeakValueDictionary`` lost its entry, and ``{id(p): ...}``
no longer found ``p``. None of those is a reference the share check counts,
and an ``id()`` key is no reference at all. So the copy's state goes into the
live object instead, which keeps its identity: a list's items, a dict's or a
set's, or the ``__dict__`` of an object pickled the default way.

Stored this way only when that restore is what plain Python gives
(`refusal`): an object of another kind, one whose pickling is its own
(``__getstate__`` or ``__reduce__``, which may leave out what the loop
added), or one that reaches itself, whose copy would point at the copy, runs
the loop again, as do other names or containers holding it beside its own
name (`holder_refusal`), and a weak reference the notebook holds to an
object it holds, or an ``id()`` of one kept in a container (a deque, a
``SimpleNamespace``, an object's slots, an integer NumPy array among them)
of the notebook (`_held_by_the_notebook`): what it holds comes back as
copies. A name bound
to the object itself is fine: it is bound to the live object again. An
``id()`` kept in a module, or below a container too big to walk, is not
seen. A hit asks again of the live objects, since a weak reference made
after the store finds a copy too.

An object of a kind no restore goes into, which the loop only assigns into
(``arr[i] = v``), is bound to the stored copy as any output is
(`bound_as_a_copy`), unless the notebook holds it by a weak reference or
its ``id()``.

A statement (not a loop) that writes into a variable (``arr[3] = 7``,
``d[k] = v``, ``o.x = 1``, ``arr += 1``) stores it as a plain output, which
a hit would bind to the stored copy. An ``id()`` key or a weak reference
taken after the entry was stored would then miss the object plain Python
keeps. A hit writes the stored state into the live object instead where it
can (`identity_kept`), and binds the copy only while the notebook holds
neither that object nor one it holds by identity; else the statement runs.
"""

from __future__ import annotations

import ast
import sys
import types
import weakref
from collections.abc import Iterable, Mapping
from itertools import chain, compress
from operator import not_
from typing import Any

from cash import rewrite_in_place
from cash._referents import get_referents
from cash._user_attrs import attribute_or, builtin_base, builtin_items, from_family, own_dict
from cash.closure_sharing import slot_readers
from cash.install_paths import is_user_code_module
from cash.live_state import is_live_stream
from cash.value_types import IMMUTABLE_LEAF_TYPES

from ... import _weak
from .identity_holders import IdentityHeld, library_parts

__all__ = [
    "bound_as_a_copy",
    "copies_kept",
    "forget_routed",
    "holder_refusal",
    "identity_kept",
    "memory_refusal",
    "ready",
    "refusal",
    "refused",
    "restorable_kind",
    "restore_into",
    "routed_as_a_copy",
    "written_into",
    "written_names",
]

#: Containers restored by their contents.
_CONTAINERS = (list, dict, set)
#: Most objects one check looks through: past it the loop runs again.
_MAX_SEEN = 200_000
#: A container with more items is not looked inside (`_held_by_the_notebook`).
_MAX_SCANNED = 10_000
#: What pickle's default handling of an object would be overridden by.
_OWN_PICKLING = ("__reduce__", "__reduce_ex__", "__getstate__", "__setstate__", "__getnewargs_ex__", "__getnewargs__")
#: Not looked into: what they hold is not the object's state as pickle sees it.
_OPAQUE = (type, types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.CodeType, types.FrameType)
_HEAP_TYPE = 1 << 9


def _default_object(kind: type) -> bool:
    """An object of a class of Python code, with a ``__dict__``, no slots of
    its own and pickled the default way: its ``__dict__`` is its state."""
    if not kind.__flags__ & _HEAP_TYPE or not kind.__dictoffset__:
        return False
    for klass in kind.__mro__:
        if klass is object:
            continue
        if not klass.__flags__ & _HEAP_TYPE or "__slots__" in vars(klass):
            return False
        if any(name in vars(klass) for name in _OWN_PICKLING):
            return False
    return True


def refusal(obj: Any, namespace: Mapping[str, Any] | None = None) -> str | None:
    """Why *obj* cannot be restored into, or None. What it holds comes back
    as copies, so a weak reference the notebook holds to one, or an
    ``id()`` of one kept in a container of *namespace*, refuses it too
    (`_held_by_the_notebook`)."""
    kind = type(obj)
    if rewrite_in_place.rewritable_kind(obj) and kind not in _CONTAINERS:
        # An array, a frame or a Series: written into, so it keeps its
        # identity, and what it holds is plain values (`rewrite_in_place`).
        return rewrite_in_place.refusal(obj)
    if not restorable_kind(obj):
        why = f"a {kind.__name__} is not restored into the object it is, and a new one would not be that object"
        # Only a loop's assignments into it (``arr[i] = v``) bring it here,
        # when the notebook holds it by identity (`bound_as_a_copy`): say
        # so, and warn.
        held = _held_itself(obj, namespace or {}) if namespace is not None else None
        return why if held is None else _identity_refusal(held, it=f"it, {why}")
    parts: set[int] = set()
    weakly: list[Any] = []
    found = _reaches(_state_without_own_methods(obj), obj, seen=parts, weakly=weakly)
    if found is None:
        return "it holds too many objects to check that it does not hold itself"
    if found:
        return "it holds itself, and a copy would hold the copy"
    # A weak reference the state holds itself comes back with it (pandas'
    # between the blocks of a frame).
    refs = [ref for item in weakly for ref in _users_refs(item) if id(ref) not in parts]
    held = _held_by_the_notebook(parts, refs, namespace or {})
    return None if held is None else _identity_refusal(held)


#: `IdentityHeld.what` of a refusal about the loop's object itself.
_ITSELF = "itself"


def _identity_refusal(held: str, it: str | None = None) -> IdentityHeld:
    """The refusal for what `_held_by_the_notebook` found: by default about
    an object the loop's object holds, else about *it*."""
    if held == _WEAK:
        if it is not None:
            return IdentityHeld(f"a weak reference points at {it}", "", "a weak reference", _ITSELF)
        return IdentityHeld(
            "a weak reference points at an object it holds, and a restore would hand it a copy", "", "a weak reference"
        )
    if held == _BY_ID:
        if it is not None:
            return IdentityHeld(f"a container keeps the id() of {it}", "", "an id()", _ITSELF)
        return IdentityHeld(
            "a container keeps the id() of an object it holds, and a restore would hand it a copy", "", "an id()"
        )
    name = attribute_or(held, "name", "")
    where = f" ('{name}')" if name else ""
    return IdentityHeld(
        f"the notebook holds containers too big to look through{where} for a weak reference to, or the id() "
        "of, an object it holds",
        name,
        "too many",
    )


def restorable_kind(obj: Any) -> bool:
    """Whether *obj* is of a kind a restore can go into: a list, dict or
    set, or an object whose ``__dict__`` is its state."""
    kind = type(obj)
    return kind in _CONTAINERS or _default_object(kind)


def written_into(obj: Any) -> bool:
    """Whether *obj* is restored by writing values into its memory: an
    array, a frame or a Series (`rewrite_in_place`), whose shape a restore
    must find unchanged. A list, a dict or a set is refilled whatever it
    holds, as a loop's always was."""
    return rewrite_in_place.rewritable_kind(obj) and type(obj) not in _CONTAINERS


def bound_as_a_copy(obj: Any, namespace: Mapping[str, Any]) -> bool:
    """Whether *obj*, which a loop only assigns into, may be bound to the
    stored copy as any output is (``arr[i] = v`` on a NumPy array): it is of
    a kind no restore goes into, not a list, dict or set of a subclass (whose
    own ``__setitem__`` a copy would call), and nothing holds it by identity,
    a weak reference or its ``id()``."""
    if restorable_kind(obj) or isinstance(obj, (list, dict, set)):
        return False
    return _held_itself(obj, namespace) is None


def written_names(tree: ast.AST | None, mutated: Iterable[str]) -> set[str]:
    """Of *mutated*, the names a statement changes in place, those it keeps
    bound to the same object: not one it binds anew (``arr = f(arr)``,
    where ``f`` changes its argument), except by augmented assignment
    (``arr += 1``), which keeps the object of any kind that is written
    into (`rewrite_in_place`)."""
    names = set(mutated)
    if tree is None or not names:
        return names
    rebound: set[str] = set()
    augmented: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            rebound.add(node.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            augmented.add(node.target.id)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            rebound.add(node.name)
        elif isinstance(node, ast.alias):
            rebound.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.Global | ast.Nonlocal):
            rebound.update(node.names)
    return names - (rebound - augmented)


def identity_kept(
    names: Iterable[str], restored: Mapping[str, Any], namespace: Mapping[str, Any]
) -> tuple[set[str], str | None]:
    """For a hit of a statement that wrote into the variables *names*
    (`written_names`), stored as plain outputs: those whose stored state
    goes into the live object, and why the hit must not be served, or None.

    Plain Python keeps the object a statement writes into, so whatever
    finds it by identity still finds it: ``reg = {id(arr): 'a'}`` or
    ``weakref.ref(arr)``, taken after the entry was stored. A copy bound in
    its place is another object. So the state is written into the live
    object whenever `rewrite_in_place.mismatch` allows it (an array, a
    frame or a Series, a list, dict or set of plain values, unchanged in
    shape), or a frame the statement only added columns to or dropped
    columns from (`_columns_fit`), and no other value the entry restores holds it
    (`holder_refusal`). Any other object is bound to the copy only while the notebook
    holds neither it nor an object it holds by identity (`_copy_refusal`);
    else the statement runs."""
    into: set[str] = set()
    for name in sorted(names):
        if name not in namespace or name not in restored:
            continue
        live = namespace[name]
        if type(live) in IMMUTABLE_LEAF_TYPES:
            # A number or a string is a new object in plain Python too.
            continue
        try:
            fits = (
                rewrite_in_place.mismatch(live, restored[name]) is None or _columns_fit(live, restored[name])
            ) and holder_refusal(name, restored[name], restored) is None
            why = None if fits else _copy_refusal(live, namespace, name)
        except Exception:  # noqa: BLE001 - an object whose own lookups raise: it runs
            forget_routed(live)
            return into, f"'{name}' could not be checked for what points at it, and a copy would not be that object"
        if fits:
            # Not when another value of the entry holds the stored copy
            # (``frames = [d1]`` stored with ``d1['z'] = ...``): restored, it
            # would hold the copy beside the live object.
            into.add(name)
            continue
        if why is not None:
            # A loop writing into it is routed afresh too (`routed_as_a_copy`).
            forget_routed(live)
            kind = type(live).__name__
            return into, f"{why}; a {kind} is not written into, and a copy would not be that object"
    return into, None


def copies_kept(
    names: Iterable[str], restored: Mapping[str, Any], namespace: Mapping[str, Any]
) -> tuple[set[str], str | None]:
    """For a hit of a loop that bound the variables *names* to the stored
    copy (``bound_copies``: ``arr[i] = v`` on a NumPy array): those whose
    stored state goes into the live object instead, and why the hit must
    not be served, or None.

    A copy stays bound while nothing holds the live object by identity
    (`bound_as_a_copy`). One the notebook holds now (``reg = {id(arr):
    'a'}``, taken after the entry was stored) is written into when
    `rewrite_in_place.mismatch` allows it (an array, a frame or a Series,
    unchanged in shape) and no other value the entry restores holds it
    (`holder_refusal`), as a statement's is (`identity_kept`): a loop whose
    passes ran one by one is served so, and so is one whose rest ran as one
    unit. Any other is refused, and its next loop routed afresh
    (`forget_routed`)."""
    into: set[str] = set()
    for name in sorted(names):
        if name not in namespace:
            continue
        live = namespace[name]
        if bound_as_a_copy(live, namespace):
            continue
        if name in restored and written_into(live):
            try:
                fits = (
                    rewrite_in_place.mismatch(live, restored[name]) is None
                    and holder_refusal(name, restored[name], restored) is None
                )
            except Exception:  # noqa: BLE001 - an object whose own lookups raise: it runs
                fits = False
            if fits:
                into.add(name)
                continue
        forget_routed(live)
        return into, f"something points at {name} now"
    return into, None


def _columns_fit(live: Any, stored: Any) -> bool:
    """Whether the frame *stored* can be written into the frame *live* by
    dropping the columns it lacks, writing the ones both have and adding
    its others at the end (`_rewrite_columns`): ``df['f'] = ...`` or
    ``del df['c']``, which `rewrite_in_place.mismatch` refuses as a change
    of shape. Same index, attrs and flags, unique column labels, columns
    a write in place takes (`rewrite_in_place.refusal`), and the columns
    both have in the same order, ahead of the added ones. A frame of these
    is never walked for what holds it by identity, which next to a big
    heap cost every hit seconds."""
    pd = sys.modules.get("pandas")
    if pd is None or type(live) is not pd.DataFrame or type(stored) is not pd.DataFrame:
        return False
    try:
        if rewrite_in_place.refusal(live) is not None or rewrite_in_place.refusal(stored) is not None:
            return False
        before, after = live.columns, stored.columns
        if not (before.is_unique and after.is_unique) or type(before) is not type(after):
            return False
        if list(before.names) != list(after.names):
            return False
        index = live.index
        if type(index) is not type(stored.index) or not index.equals(stored.index):
            return False
        if list(index.names) != list(stored.index.names) or index.dtype != stored.index.dtype:
            return False
        if live.flags.allows_duplicate_labels != stored.flags.allows_duplicate_labels:
            return False
        if not _same_attrs(live.attrs, stored.attrs):
            return False
        kept = [label for label in before if label in after]
        return list(after[: len(kept)]) == kept
    except Exception:  # noqa: BLE001 - not comparable: not written into this way
        return False


def _same_attrs(a: Any, b: Any) -> bool:
    try:
        return bool(dict(a) == dict(b))
    except Exception:  # noqa: BLE001 - an attrs value without a plain ==
        return False


def _rewrite_columns(live: Any, stored: Any) -> None:
    """Give the frame *live* the columns of *stored* (`_columns_fit`)."""
    for label in [label for label in live.columns if label not in stored.columns]:
        del live[label]
    width = live.shape[1]
    for position, label in enumerate(stored.columns):
        values = stored.iloc[:, position].array.copy()
        if position < width:
            live.isetitem(position, values)
        else:
            live[label] = values


def _copy_refusal(obj: Any, namespace: Mapping[str, Any], name: str) -> str | None:
    """Why *obj* may not be replaced by a copy: the notebook holds it, or an
    object it holds, by identity (a weak reference, its ``id()``), or holds
    more than can be looked through (`_held_by_the_notebook`); or *obj*
    holds more than `_MAX_SEEN` objects."""
    parts: set[int] = set()
    weakly: list[Any] = []
    if _reaches([obj], _NOTHING, seen=parts, weakly=weakly) is None:
        return f"'{name}' holds too many objects to check that nothing points at them"
    refs = [ref for item in weakly for ref in _users_refs(item) if id(ref) not in parts]
    held = _held_by_the_notebook(parts, refs, namespace)
    return None if held is None else str(_identity_refusal(held, it=f"'{name}' or an object it holds"))


def routed_as_a_copy(obj: Any, namespace: Mapping[str, Any]) -> bool:
    """`bound_as_a_copy`, as a loop's routing asks it before the loop runs:
    yes, with no walk of the notebook, unless a hit already refused the
    object (`forget_routed`).

    Only a route: a hit asks `bound_as_a_copy` of the live objects (a weak
    reference or an ``id()`` taken after the store must be seen there
    anyway), so routing a held object as a copy stores an entry no hit
    serves, never a wrong one. Walking the notebook before each loop cost
    a loop beside 2,000,000 objects 0.8 s a run (``df.loc[i, c] = v`` in
    a loop over a fresh frame, which no memo of the route spared). Once a
    hit refused the object, its next routes walk, so a held object is
    then stored to be written into, and an object no weak reference can
    be taken to, whose refusal cannot be remembered, is walked each time."""
    if restorable_kind(obj) or isinstance(obj, (list, dict, set)):
        return False
    if not refused(obj):
        try:
            _weak.ref(obj)
        except TypeError:  # its refusal could not be remembered: walked each time
            return _held_itself(obj, namespace) is None
        return True
    if _held_itself(obj, namespace) is not None:
        return False
    _REFUSED.pop(id(obj), None)
    return True


def forget_routed(obj: Any) -> None:
    """Remember that a hit found *obj* held by identity: its next loops
    walk the notebook before they are routed (`routed_as_a_copy`)."""
    key = id(obj)
    try:
        kept = _weak.ref(obj, lambda _gone, key=key: _forget_refusal(key, _gone))
    except TypeError:  # routed by a walk each time anyway
        return
    if len(_REFUSED) >= _MAX_REFUSED:
        _REFUSED.clear()
    _REFUSED[key] = kept


def refused(obj: Any) -> bool:
    """Whether a hit found *obj* held by identity (`forget_routed`)."""
    known = _REFUSED.get(id(obj))
    return known is not None and known() is obj


def _forget_refusal(key: int, gone: Any) -> None:
    if _REFUSED.get(key) is gone:
        del _REFUSED[key]


#: The objects a hit found held by identity (`forget_routed`), by their
#: ``id()``, each by a reference of cash's (`cash._weak.ref`), which no
#: walk counts as the notebook's.
_REFUSED: dict[int, weakref.ReferenceType] = {}
_MAX_REFUSED = 1024


def _users_refs(obj: Any) -> list[Any]:
    """The weak references to *obj* cash did not make (`cash._weak.is_cash_ref`):
    cash's own are never the notebook's."""
    try:
        return [ref for ref in weakref.getweakrefs(obj) if not _weak.is_cash_ref(ref)]
    except TypeError:
        return []


def _held_itself(obj: Any, namespace: Mapping[str, Any]) -> str | None:
    """How the notebook holds *obj* itself by identity (`_held_by_the_notebook`):
    a weak reference to it, or its ``id()``; None when it does not."""
    refs = _users_refs(obj)
    if refs:
        # One *obj* holds itself comes back with the copy (pandas keeps one
        # to the frame in its ``_flags``).
        inside: set[int] = set()
        if _reaches(get_referents(obj), _NOTHING, seen=inside) is not None:
            refs = [ref for ref in refs if id(ref) not in inside]
    return _held_by_the_notebook({id(obj)}, refs, namespace)


def _users_class(value: Any) -> bool:
    """Whether *value* is a class of the user's code."""
    if not isinstance(value, type):
        return False
    module = sys.modules.get(attribute_or(value, "__module__") or "")
    return module is not None and is_user_code_module(module)


def _users_function(func: Any) -> bool:
    """Whether *func* is a plain function of the user's code."""
    if type(func) is not types.FunctionType:
        return False
    module = sys.modules.get(attribute_or(func, "__module__") or "")
    return module is not None and is_user_code_module(module)


def _function_parts(func: types.FunctionType) -> list[Any] | None:
    """What a function of the user's holds: its closure cells' contents and
    its defaults. None for another function, which is not looked into."""
    if not _users_function(func):
        return None
    held: list[Any] = []
    for cell in func.__closure__ or ():
        try:
            held.append(cell.cell_contents)
        except ValueError:  # an empty cell
            continue
    held.extend(func.__defaults__ or ())
    held.extend((func.__kwdefaults__ or {}).values())
    return held or None


#: A target `_reaches` never finds: to collect what an object holds.
_NOTHING = object()


#: Packages whose objects `_held_by_the_notebook` does not look into: the
#: shell's ``exit`` reaches the shell, and through it cash's own records,
#: which key objects by their ``id()``.
_NOT_THE_NOTEBOOKS = frozenset({"IPython", "ipykernel", "jupyter_client", "traitlets", "zmq", "tornado", "cash"})

#: What `_held_by_the_notebook` finds.
_WEAK = "weak"
_BY_ID = "by id"
_TOO_MANY = "too many"
#: What a container too big to look inside may hold and keep nothing else.
_LEAVES = frozenset(IMMUTABLE_LEAF_TYPES)


def _held_by_the_notebook(parts: set[int], refs: list[Any], namespace: Mapping[str, Any]) -> str | None:
    """`_WEAK` when the notebook holds one of *refs*, the weak references to
    an object of *parts* (by ``id()``): in a variable or a container of
    *namespace* at any depth (dicts, lists, tuples, sets, an object's
    ``__dict__``, so a ``WeakValueDictionary`` too, but not an object of
    IPython's or cash's), or as a ``weakref.finalize``. Those cash keeps
    itself are not the notebook's.
    `_BY_ID` when one of those keeps the ``id()`` of one of *parts* as a
    key, a value or an item: ``reg = {id(p.seen): ...}``, ``{'seen':
    id(p.seen)}``, ``ids = [id(p.seen)]``. Else None.

    A number that only happens to equal an ``id()`` runs the loop again,
    which is always right. What a container of more than `_MAX_SCANNED`
    items holds is checked at C speed but not looked inside (walking each
    item of a list of a million tokens cost every restore 60 ms): one that
    holds anything but numbers, strings and the like gives `_TOO_MANY`, as
    do more objects than `_MAX_SEEN`, and the loop runs again."""
    if any(isinstance(attribute_or(ref, "__callback__"), weakref.finalize) for ref in refs):
        return _WEAK
    ref_ids = {id(ref) for ref in refs}
    # ``hex(id(p.seen))`` or ``str(id(p.seen))``: an id() as text.
    texts = _Texts(parts)
    budget = [_MAX_SEEN]
    seen: set[int] = set()
    ids = _Ids(parts)
    if parts.intersection(v for v in namespace.values() if type(v) is int):
        return _BY_ID
    # One variable at a time, so that a refusal can name the variable whose
    # value it could not look through.
    for name, value in list(namespace.items()):
        found = _held_below(value, parts, texts, ref_ids, ids, seen, budget)
        if found is None:
            continue
        if found == _TOO_MANY:
            return _TooMany(name)
        return found
    return None


class _TooMany(str):
    """`_TOO_MANY`, with the variable whose value could not be looked through."""

    name: str

    def __new__(cls, name: str) -> _TooMany:
        made = super().__new__(cls, _TOO_MANY)
        made.name = name
        return made


def _held_below(
    root: Any,
    parts: set[int],
    texts: _Texts,
    ref_ids: set[int],
    ids: _Ids,
    seen: set[int],
    budget: list[int],
) -> str | None:
    """`_held_by_the_notebook`'s answer for what *root* holds at any depth;
    *seen* and *budget* are shared across the variables."""
    level = [root]
    while level:
        if ref_ids and not ref_ids.isdisjoint(map(id, level)):
            return _WEAK
        nxt: list[Any] = []
        for value in level:
            if id(value) in seen:
                continue
            held = _what_it_holds(value, ids)
            if held is None:
                continue
            if type(held) is str:
                return held
            seen.add(id(value))
            if len(held) > _MAX_SCANNED:
                # Not looked inside one item at a time: its numbers and
                # strings are compared at C speed, and what any other item
                # holds is read a level at a time (`_read_in_bulk`).
                if set(map(type, held)) <= _LEAVES:
                    if _holds_one_of(held, parts, texts):
                        return _BY_ID
                    continue
                held = _read_in_bulk(held, parts, texts)
                if type(held) is str:
                    return held
            elif _holds_one_of(held, parts, texts):
                return _BY_ID
            budget[0] -= len(held) + 1
            if budget[0] < 0:
                return _TOO_MANY
            nxt.extend(held)
        level = [v for v in nxt if type(v) not in IMMUTABLE_LEAF_TYPES]
    return None


#: How many objects `_read_in_bulk` may meet, all levels together, and how
#: many levels below a too-big container it reads before giving up.
_MAX_BULK = 5_000_000
_BULK_LEVELS = 4
#: ``Py_TPFLAGS_HEAPTYPE``: a class made by a ``class`` statement (or by a
#: C extension's type spec, told apart by its module's file).
_HEAPTYPE = 1 << 9


def _gc_reads_it_all(kind: type) -> bool:
    """Whether the garbage collector's referents of an object of *kind* are
    all `_what_it_holds` reads of it: a plain container's items, and an
    object of a class written in Python its ``__dict__`` and slots (and
    its class). Not a C type's, whose fields may keep numbers unreported,
    nor one a library reads in its own way (`library_parts`)."""
    if kind in _PLAIN:
        return True
    found = _GC_READS.get(kind)
    if found is None:
        found = kind.__mro__[-1] is object and all(_written_in_python(klass) for klass in kind.__mro__[:-1])
        _GC_READS[kind] = found
    return found


def _written_in_python(klass: type) -> bool:
    """Whether *klass* was made by a ``class`` statement in a ``.py`` file."""
    if not klass.__flags__ & _HEAPTYPE:
        return False
    module = sys.modules.get(type.__dict__["__module__"].__get__(klass))
    path = attribute_or(module, "__file__", None) if module is not None else None
    return type(path) is str and path.endswith(".py")


_GC_READS: weakref.WeakKeyDictionary[type, bool] = _weak.WeakKeyDictionary()


def _holds_one_of(held: list[Any], parts: set[int], texts: _Texts) -> bool:
    """Whether a number or a text among *held* is one of *parts* or *texts*."""
    try:
        return not parts.isdisjoint(held) or texts.among(held)
    except TypeError:  # an unhashable item: the leaves alone
        leaves = [item for item in held if type(item) in _LEAVES]
        try:
            return not parts.isdisjoint(leaves) or texts.among(leaves)
        except TypeError:  # an unhashable leaf (a signalling NaN): one at a time
            spelled = texts.all()
            return any(type(item) in (int, str) and (item in parts or item in spelled) for item in leaves)


class _Texts:
    """The ``id()``s of *parts* as text, ``str`` and ``hex``
    (``hex(id(p.seen))``), spelled only when a text of their length is met:
    spelling the 170,000 ids of a list of frames cost each refusal 0.1 s,
    for dicts whose keys are short names."""

    def __init__(self, parts: set[int]) -> None:
        self.parts = parts
        self._spelled: set[str] | None = None
        self._lengths: set[int] | None = None
        self._plain: dict[type, bool] = {}

    def all(self) -> set[str]:
        """Every text looked for."""
        if self._spelled is None:
            self._spelled = {*map(str, self.parts), *map(hex, self.parts)}
        return self._spelled

    def among(self, items: list[Any]) -> bool:
        """Whether one of *items*, all hashable, is one of the texts: what
        ``not self.all().isdisjoint(items)`` gives. Only a ``str`` can equal
        one, unless its class compares in its own way; a ``str`` only when
        its length is one of theirs."""
        if self._spelled is not None:
            return not self._spelled.isdisjoint(items)
        kinds = set(map(type, items))
        odd = [kind for kind in kinds if kind is not str and not self._never_a_text(kind)]
        if odd:
            return not self.all().isdisjoint(items)
        if str not in kinds:
            return False
        if self._lengths is None:
            self._lengths = _text_lengths(self.parts)
        lengths = self._lengths
        words = list(compress(items, map(str.__instancecheck__, items)))
        near = list(compress(words, map(lengths.__contains__, map(len, words))))
        return bool(near) and not self.all().isdisjoint(near)

    def _never_a_text(self, kind: type) -> bool:
        found = self._plain.get(kind)
        if found is None:
            found = kind in _NEVER_TEXT or (
                not issubclass(kind, str) and attribute_or(kind, "__eq__", None) is _OBJECTS_EQ
            )
            self._plain[kind] = found
        return found


#: Never equal to a ``str``: the other leaves, and builtin containers.
_NEVER_TEXT = frozenset(IMMUTABLE_LEAF_TYPES) - {str} | {tuple, frozenset}
_OBJECTS_EQ = object.__eq__


def _text_lengths(parts: set[int]) -> set[int]:
    """The lengths of the ``str`` and ``hex`` of every number of *parts*."""
    if not parts:
        return set()
    low, high = min(parts), max(parts)
    lengths = set(range(len(str(low)), len(str(high)) + 1))
    lengths.update(range(len(hex(low)), len(hex(high)) + 1))
    return lengths


def _read_in_bulk(held: list[Any], parts: set[int], texts: _Texts) -> list[Any] | str:
    """What the items of *held*, a container too big to walk one item at a
    time, hold, read a level at a time at C speed: `_BY_ID` when a number
    or text among them is one of *parts* or *texts*; else the few objects
    left for `_held_below` to look at itself (classes, functions, weak
    references, an object of a C type...), or `_TOO_MANY` when they are
    many. A ``TarFile``'s 100,000 ``TarInfo`` members (mlbook ch08 keeps
    the archive it extracted) hold numbers, names and empty dicts.

    Each level is sorted by its items' classes, not looked at one item at
    a time: a list of 200,000 lists of words cost each loop 0.3 s. A
    dict's keys are read as well as its values, since the garbage
    collector does not report a dict's string keys (``{hex(id(p)): ...}``)."""
    left: dict[int, Any] = {}
    found = held
    met = 0
    for _ in range(_BULK_LEVELS):
        met += len(found)
        if met > _MAX_BULK:
            return _TOO_MANY
        kinds = list(map(type, found))
        present = set(kinds)
        if present <= _LEAVES:
            if _holds_one_of(found, parts, texts):
                return _BY_ID
            return list(left.values())
        if present.isdisjoint(_LEAVES):
            others, other_kinds = found, kinds
        else:
            leaf = list(map(_LEAVES.__contains__, kinds))
            if _holds_one_of(list(compress(found, leaf)), parts, texts):
                return _BY_ID
            others = list(compress(found, map(not_, leaf)))
            other_kinds = list(compress(kinds, map(not_, leaf)))
            present -= _LEAVES
        reads = {kind: _gc_reads_it_all(kind) for kind in present}
        if all(reads.values()):
            level = dict(zip(map(id, others), others))
        else:
            readable = list(map(reads.__getitem__, other_kinds))
            inside = list(compress(others, readable))
            level = dict(zip(map(id, inside), inside))
            rest = list(compress(others, map(not_, readable)))
            left.update(zip(map(id, rest), rest))
        if len(left) > _MAX_SCANNED:
            return _TOO_MANY
        if len(level) <= _MAX_SCANNED:
            left.update(level)
            return list(left.values())
        values = list(level.values())
        found = get_referents(*values)
        if dict in reads:
            # The keys too: the collector leaves out a dict's string keys.
            found.extend(chain.from_iterable(map(dict.keys, compress(values, map(dict.__instancecheck__, values)))))
    return _TOO_MANY


#: An integer array this small is read as Python numbers (`_Ids.in_array`).
_FEW_NUMBERS = 64


class _Ids:
    """The ``id()``s `_held_by_the_notebook` looks for, as a sorted NumPy
    array and its bounds, made once for every integer array it meets: a dict
    of 3,000 frames made it again for each frame's column."""

    def __init__(self, parts: set[int]) -> None:
        self.parts = parts
        self._sorted: Any = None

    def in_array(self, value: Any) -> bool:
        """Whether the integer NumPy array *value* holds one of them. A
        few signed numbers are looked up one by one: the sorted search cost
        a small frame's column a dozen NumPy calls."""
        if value.size <= _FEW_NUMBERS and value.dtype.kind == "i":
            return not self.parts.isdisjoint(value.ravel().tolist())
        numpy = sys.modules["numpy"]
        if self._sorted is None:
            self._sorted = numpy.sort(numpy.fromiter(self.parts, dtype=numpy.int64, count=len(self.parts)))
        if not self._sorted.size:
            return False
        flat = value.ravel()
        flat = flat[(flat >= self._sorted[0]) & (flat <= self._sorted[-1])]
        if not flat.size:
            return False
        at = numpy.searchsorted(self._sorted, flat).clip(max=self._sorted.size - 1)
        return bool((self._sorted[at] == flat).any())


def _what_it_holds(value: Any, ids: _Ids) -> list[Any] | str | None:
    """What *value* holds, for `_held_by_the_notebook`: a dict's keys and
    values both (``{id(p.seen): ...}``, ``{'seen': id(p.seen)}``,
    ``{weakref.ref(p.child): ...}``), the items of a list, tuple, set or
    deque or of a subclass of one (read through the builtin type's own
    methods, not the class's), the items of a NumPy array of objects, and
    an object's attributes and slots, and the closure cells and defaults
    of a function of the user's. `_BY_ID` for an integer NumPy array
    holding one of *ids*, `_TOO_MANY` for a container that cannot be
    read. None when nothing in it is looked at."""
    kind = type(value)
    if kind in _PLAIN:
        return [*value, *value.values()] if kind is dict else value
    if kind is types.FunctionType:
        # A function of the user's keeps what its closure cells and
        # defaults hold: ``r = weakref.ref(c); get = lambda: r().v``.
        return _function_parts(value)
    if kind is types.MethodType:
        return [value.__self__] if _users_function(value.__func__) else None
    if (
        (isinstance(value, _OPAQUE) and not _users_class(value))
        or type(kind.__module__) is not str
        or (kind.__module__.partition(".")[0] in _NOT_THE_NOTEBOOKS)
    ):
        return None
    held: list[Any] = []
    if isinstance(value, type):
        # A registry on a class of the user's: ``alive = weakref.WeakSet()``,
        # ``by_id = {}`` filled in __init__.
        state = attribute_or(value, "__dict__")
        return list(dict(state).values()) if type(state) is types.MappingProxyType else None
    if builtin_base(value) is not None:
        try:
            held.extend(builtin_items(value) or ())
        except Exception:  # noqa: BLE001 - not readable: what it holds is unknown
            return _TOO_MANY
        state = own_dict(value)
        if state:
            held.extend(state.values())
    elif kind.__name__ == "ndarray" and kind.__module__ == "numpy":
        if value.dtype.names and not value.dtype.hasobject:
            return [value[field] for field in value.dtype.names]
        if value.dtype.hasobject:
            return value.ravel().tolist()
        if value.dtype.kind in "iu" and value.dtype.itemsize == 8 and value.size:
            return _BY_ID if ids.in_array(value) else None
        return None
    else:
        try:
            parts = library_parts(value)
        except Exception:  # noqa: BLE001 - not readable: what it holds is unknown
            return _TOO_MANY
        if parts is not None:
            return parts or None
    state = own_dict(value)
    if state:
        held.extend(state.values())
    for reader in slot_readers(kind):
        try:
            held.append(reader.__get__(value, kind))
        except AttributeError:  # an empty slot
            continue
    return held or None


#: Opened as they are.
_PLAIN = frozenset({dict, list, tuple, set, frozenset})


def holder_refusal(name: str, obj: Any, values: Mapping[str, Any]) -> str | None:
    """Why the other *values* (outputs and holders, by name) stop *obj*,
    stored as *name*, from being restored into: one holds it inside, where
    a restore would put the copy. A value that is *obj* itself is bound to
    the live object again."""
    others = [value for other, value in values.items() if other != name and value is not obj]
    if not others:
        return None
    found = _reaches(others, obj)
    if found is None:
        return f"too many objects hold '{name}' to check them"
    if found:
        return f"another variable or container holds '{name}', and a restore would hand it a copy"
    return None


def memory_refusal(name: str, obj: Any, values: Mapping[str, Any]) -> str | None:
    """Why the other *values* (outputs and holders, by name) stop the array
    *obj*, stored as *name*, from being written into: one of them is an
    array sharing its memory (its base, or a view of it), which a restore
    would hand a copy, so that writes to one no longer reach the other.
    Looked for through what the values hold, up to `_MAX_SEEN` objects."""
    numpy = sys.modules.get("numpy")
    if numpy is None or not isinstance(obj, numpy.ndarray):
        return None
    module_dicts = {id(vars(m)) for m in list(sys.modules.values()) if isinstance(m, types.ModuleType)}
    seen: set[int] = {id(obj)}
    level = [value for other, value in values.items() if other != name]
    while level:
        nxt: list[Any] = []
        for item in level:
            key = id(item)
            if key in seen or key in module_dicts or isinstance(item, _OPAQUE) or type(item) in IMMUTABLE_LEAF_TYPES:
                continue
            seen.add(key)
            if len(seen) > _MAX_SEEN:
                return f"too many objects are stored with '{name}' to check that none shares its memory"
            if isinstance(item, numpy.ndarray):
                if numpy.may_share_memory(item, obj):
                    return f"another array stored with '{name}' shares its memory, and a restore would hand it a copy"
                if item.base is not None:
                    nxt.append(item.base)
                continue
            nxt.extend(get_referents(item))
        level = nxt
    return None


def _state(obj: Any) -> list[Any]:
    if type(obj) is dict:
        return [*obj.keys(), *obj.values()]
    if type(obj) in (list, set):
        return list(obj)
    return [obj.__dict__]


def _state_without_own_methods(obj: Any) -> list[Any]:
    """`_state`, with each attribute bound to a method of *obj* itself
    replaced by its function: ``self._out = self._no_stream`` (pyprind's
    bar) does not make a copy hold the copy, since `restore_into` binds
    such a method to the live object again."""
    if type(obj) in _CONTAINERS:
        return _state(obj)
    state = obj.__dict__
    return [*state.keys(), *(_unbound(value, obj) for value in state.values())]


def _unbound(value: Any, obj: Any) -> Any:
    if type(value) is types.MethodType and value.__self__ is obj:
        return value.__func__
    return value


def _reaches(
    roots: Iterable[Any], target: Any, seen: set[int] | None = None, weakly: list[Any] | None = None
) -> bool | None:
    """Whether *target* is among *roots* or what they hold; None past `_MAX_SEEN`.
    *seen* gets the ``id()`` of each object looked at, and *weakly* each one
    a weak reference points at.

    How an object is read is settled once per class (`_reach_mode`): a list
    of 3,000 small frames is 42,000 objects, and asking each of them whether
    it is a stream, a function or a pandas object cost a loop growing it
    most of its 0.35 s refusal."""
    module_dicts = {id(vars(m)) for m in list(sys.modules.values()) if isinstance(m, types.ModuleType)}
    if seen is None:
        seen = set()
    leaves = _LEAVES
    level = [item for item in roots if type(item) not in leaves]
    cells = [_MAX_CELLS]
    modes: dict[type, int] = {}
    streams = (type(sys.__stdout__), type(sys.__stderr__))
    pandas = sys.modules.get("pandas")
    weakcount = weakref.getweakrefcount
    add = seen.add
    while level:
        nxt: list[Any] = []
        extend = nxt.extend
        for item in level:
            if item is target:
                return True
            key = id(item)
            if key in seen:
                continue
            kind = type(item)
            mode = modes.get(kind)
            if mode is None:
                mode = modes[kind] = _reach_mode(kind, streams, pandas)
            if mode == _SKIP or key in module_dicts:
                continue
            if mode == _ASK and (isinstance(item, _OPAQUE) or is_live_stream(item)):
                # A standard stream comes back as itself (`live_state`).
                continue
            add(key)
            if weakly is not None and weakcount(item):
                weakly.append(item)
            if len(seen) > _MAX_SEEN:
                return None
            if mode == _GC:
                held = get_referents(item)
            else:
                held = _typed_referents(item, cells, None if mode == _ASK else mode)
                if held is None:
                    return None
            extend([ref for ref in held if type(ref) not in leaves])
        level = nxt
    return False


#: Most cells of object arrays one `_reaches` reads.
_MAX_CELLS = 2_000_000
#: What a pandas object's ``__dict__`` keeps that is not what it holds: the
#: block manager (its arrays are read by type below), caches of what it
#: holds, flags, and an index's copy-on-write bookkeeping.
_PANDAS_NOT_HELD = frozenset({"_mgr", "_item_cache", "_flags", "_cache", "_references", "_id"})


def _typed_referents(item: Any, cells: list[int], mode: int | None = None) -> list[Any] | None:
    """What *item* holds, for `_reaches`: ``get_referents``, except for a
    NumPy array and a pandas object, read by type. An array of objects is
    not tracked by the collector, so ``get_referents`` gave ``[]`` for a
    column holding the list that held its frame, and that list was restored
    into a copy holding a copy of itself. None past `_MAX_CELLS` cells,
    counted in *cells*. *mode* is how its class is read (`_reach_mode`),
    worked out here when not given."""
    kind = type(item)
    if mode is None:
        mode = _read_mode(kind, item)
    if mode == _ARRAY:
        base = [] if item.base is None else [item.base]
        if not item.dtype.hasobject:
            return base
        cells[0] -= item.size
        if cells[0] < 0:
            return None
        values = item.ravel().tolist()
        if item.dtype.names:  # records: a tuple of fields each
            values = [field for record in values for field in record]
        return values + base
    if mode in _PANDAS_MODES:
        # Its attributes (``attrs`` among them), its labels and its column
        # arrays, the last read by the array rule above.
        state = own_dict(item) or {}
        held = [value for name, value in state.items() if name not in _PANDAS_NOT_HELD]
        try:
            if mode == _FRAME:
                held += [item.index, item.columns, *item._mgr.arrays]
            elif mode == _SERIES:
                held += [item.index, item._values]
        except Exception:  # noqa: BLE001 - a pandas of another shape: as the collector sees it
            return get_referents(item)
        return held
    return get_referents(item)


#: How `_reaches` reads an object of a class (`_reach_mode`): passed over,
#: asked one object at a time as `_typed_referents` always did, through the
#: collector, or as a NumPy array, a frame, a series or an index.
_SKIP, _ASK, _GC, _ARRAY, _FRAME, _SERIES, _INDEX = range(7)
_PANDAS_MODES = frozenset({_FRAME, _SERIES, _INDEX})
#: Builtin classes whose attribute lookup is Python's own: ``__class__`` on
#: one of them, or on a class of Python code below one, is its type.
_PLAIN_LOOKUP = frozenset({object, dict, list, tuple, set, frozenset, bytearray})
_OBJECTS_CLASS = object.__dict__["__class__"]


def _read_mode(kind: type, item: Any = None, pandas: Any = None) -> int:
    """How `_typed_referents` reads an object of *kind*: as an array, a
    frame, a series, an index (*item* tested with ``isinstance`` when
    given, *kind* with ``issubclass`` when not), or through the collector."""
    if kind.__name__ == "ndarray" and from_family(kind, "numpy"):
        return _ARRAY
    if from_family(kind, "pandas") and kind.__name__ in ("DataFrame", "Series"):
        return _FRAME if kind.__name__ == "DataFrame" else _SERIES
    if item is not None:
        return _INDEX if _pandas_index(item) else _GC
    return _INDEX if pandas is not None and issubclass(kind, pandas.Index) else _GC


def _reach_mode(kind: type, streams: tuple[type, ...], pandas: Any) -> int:
    """How `_reaches` reads every object of *kind*, worked out once.
    ``isinstance`` answers by the object's ``__class__``, which a proxy or a
    class of the user's may change: such a class is asked one object at a
    time (`_ASK`), as is one whose objects may be a standard stream."""
    if issubclass(kind, _OPAQUE):
        return _SKIP
    try:
        lookup = next(klass for klass in kind.__mro__ if "__getattribute__" in vars(klass))
        plain = lookup in _PLAIN_LOOKUP and _static_class(kind) is _OBJECTS_CLASS
        if plain and (
            kind in streams
            or (kind.__name__ == "OutStream" and str(kind.__module__).startswith("ipykernel."))
            or getattr(kind, "_cash_wrapped_stream", None) is not None
        ):
            plain = False
    except Exception:  # noqa: BLE001 - a class that will not say: one object at a time
        return _ASK
    if not plain:
        return _ASK
    return _read_mode(kind, pandas=pandas)


def _static_class(kind: type) -> Any:
    """What ``__class__`` resolves to on *kind*'s objects, read off the MRO
    without running a descriptor."""
    for klass in kind.__mro__:
        found = vars(klass).get("__class__")
        if found is not None:
            return found
    return None


def _pandas_index(item: Any) -> bool:
    pd = sys.modules.get("pandas")
    return pd is not None and isinstance(item, pd.Index)


def ready(restored: Mapping[str, Any], names: Iterable[str], user_ns: Mapping[str, Any]) -> bool:
    """Whether each of *names* is bound to an object of the kind stored, so
    the restore can go into it."""
    return all(name in restored and type(user_ns.get(name)) is type(restored[name]) for name in names)


def _columns_changed(live: Any, restored: Any) -> bool:
    """Whether *restored* is a frame whose columns are not those of the
    frame *live* (`_columns_fit`)."""
    pd = sys.modules.get("pandas")
    return (
        pd is not None
        and type(live) is pd.DataFrame
        and type(restored) is pd.DataFrame
        and list(live.columns) != list(restored.columns)
    )


def restore_into(live: Any, restored: Any) -> None:
    """Give *live* the state of *restored*, an object of its kind (`ready`)."""
    kind = type(live)
    if kind is list:
        live[:] = restored
    elif rewrite_in_place.rewritable_kind(live):
        if _columns_changed(live, restored):
            _rewrite_columns(live, restored)
        else:
            rewrite_in_place.rewrite(live, restored)
    elif kind is dict or kind is set:
        live.clear()
        live.update(restored)
    else:
        state = live.__dict__
        state.clear()
        state.update(restored.__dict__)
        # A method of the copy itself is bound to the live object instead
        # (`_state_without_own_methods`).
        for name, value in list(state.items()):
            if type(value) is types.MethodType and value.__self__ is restored:
                state[name] = types.MethodType(value.__func__, live)
