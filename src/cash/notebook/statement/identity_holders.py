"""A value that finds another object by its identity, and so cannot be
restored on its own.

``wr = weakref.ref(p.child)`` and ``reg = {id(p.seen): 'seen'}`` are bound to
what they point at by identity, not by holding it. Restored from the cache,
such a value points at the object as it was when it was stored, while the
statement that binds ``p`` runs (or is restored) and binds a new one: the
weak reference finds nothing, or the old object, and the ``id()`` key finds
nothing, where plain Python finds ``p``'s part. Copied with the cache's RAM
tier, a weak reference even keeps pointing at the old object itself. So a
statement whose value holds either runs again, as a plain run does.

What is looked at:

- a weak reference (``weakref.ref``, a ``WeakValueDictionary``'s or a
  ``WeakSet``'s, a proxy) held anywhere in the value, whose object is alive
  and not inside the value itself (a restore rebuilds those together); a
  class, a module or a function comes back as itself and does not count;
- an ``int`` in the value equal to the ``id()`` of an object the value holds
  or the notebook's variables hold, at any depth; a number below every
  address this process uses (`_lowest_address`) is not one.

The value is walked through lists, tuples, dicts, sets and the ``__dict__``
of objects of the user's own classes, not into a library's objects (a
pandas frame keeps weak references between its own parts). A container of
more than `_MAX_SCANNED` items is not looked inside, and the walk stops at
`_MAX_SEEN` objects: walking a big value item by item cost each store more
than the check is worth. Such a container, and a pandas or NumPy object,
gives the ints among its first and last items and its index
(`_sampled_ints`): a registry keyed by ``id()`` keeps them all alike. When
the notebook holds more than the walk reads, the ids of every object the
garbage collector tracks rule a number in or out.
"""

from __future__ import annotations

import sys
import types
import weakref
from typing import Any

from cash._user_attrs import attribute_or, builtin_items
from cash.install_paths import is_user_code_module

__all__ = ["IdentityHeld", "identity_held_reason", "weakly_held_reason"]

#: A container with more items is not looked inside.
_MAX_SCANNED = 10_000
#: Most objects one walk looks at.
_MAX_SEEN = 200_000
#: Restored as themselves, by reference: a weak reference to one stays right.
_BY_REFERENCE = (type, types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType)
_REFERENCES = (weakref.ReferenceType, weakref.ProxyType, weakref.CallableProxyType)
_CONTAINERS = (list, tuple, dict, set, frozenset)
#: Hold their weak references in ``.data``.
_WEAK_CONTAINERS = (weakref.WeakValueDictionary, weakref.WeakKeyDictionary, weakref.WeakSet)
#: The smallest ``id()`` looked for: an object's address, aligned to 8.
_LOWEST_ADDRESS = 1 << 16


class IdentityHeld(str):
    """The refusal `identity_held_reason` gives: the reason, with the
    variable it names and how its value finds the object (*kind*: ``"a weak
    reference"`` or ``"an id()"``), for the warning (``NOTEBOOK-IDENTITY-HELD``)."""

    name: str
    kind: str
    #: What the warning says, when not that *name* holds *kind* of an object.
    what: str | None

    def __new__(cls, reason: str, name: str, kind: str, what: str | None = None) -> IdentityHeld:
        made = super().__new__(cls, reason)
        made.name, made.kind, made.what = name, kind, what
        return made


def identity_held_reason(name: str, value: Any, user_ns: dict[str, Any]) -> IdentityHeld | None:
    """Why *value*, bound to *name*, cannot be restored apart from what it
    points at by identity (see the module docstring), or while another
    variable finds it by one (`weakly_held_reason`); None when it can."""
    user_kinds: dict[type, bool] = {}
    seen: set[int] = set()
    refs: list[Any] = []
    numbers: set[int] = set()
    targets: list[Any] = []
    unread: list[Any] = []
    _walk([value], seen, user_kinds, refs, numbers, targets, unread=unread)
    numbers.update(_sampled_ints(unread))
    found = _held_by_id(name, refs, seen, numbers, user_kinds, user_ns, numbers_only=type(value) is int)
    if found is None and targets:
        found = _weakly_held(name, targets, user_ns)
    return found


def _held_by_id(
    name: str,
    refs: list[Any],
    seen: set[int],
    numbers: set[int],
    user_kinds: dict[type, bool],
    user_ns: dict[str, Any],
    numbers_only: bool = False,
) -> IdentityHeld | None:
    """`identity_held_reason`'s answer for what *value* itself holds, from
    its walk: *refs*, *seen* and *numbers*; *numbers_only* when the value is
    a bare ``int``."""
    for ref in refs:
        if not issubclass(type(ref), weakref.ReferenceType):
            if _alive(ref):
                return IdentityHeld(
                    f"'{name}' holds a weak proxy; restored, it would point at another object, so the statement "
                    "re-runs every time",
                    name,
                    "a weak reference",
                )
            continue
        target = ref()
        if target is not None and id(target) not in seen and not issubclass(type(target), _BY_REFERENCE):
            return IdentityHeld(
                f"'{name}' holds a weak reference to an object it does not hold; restored, it would "
                "point at another object, so the statement re-runs every time",
                name,
                "a weak reference",
            )
    lowest = max(_LOWEST_ADDRESS, _lowest_address(user_ns))
    candidates = {n for n in numbers if n >= lowest and not n % 8}
    if not candidates:
        return None
    if not candidates.isdisjoint(seen):
        return _holds_an_id(name)
    held = set(seen)
    from .output_refusals import is_history_name

    roots = [v for k, v in user_ns.items() if k != name and not is_history_name(k)]
    if _walk(roots, held, user_kinds, [], set(), wanted=candidates, unread=[]):
        return _holds_an_id(name) if not candidates.isdisjoint(held) else None
    if not candidates.isdisjoint(held):
        return _holds_an_id(name)
    # The walk left objects unread (a list of 20,000 objects, past
    # `_MAX_SEEN`): ask the collector for every object it tracks instead,
    # which the objects of the notebook's classes are among.
    if candidates.isdisjoint(_tracked_ids()):
        return None
    if numbers_only:
        # A bare number (``x = 1 << 20``) is far likelier a count than an
        # id(); only a value that keeps one beside other data is refused.
        return None
    return IdentityHeld(
        f"'{name}' holds a number that may be the id() of an object, and the notebook holds too many objects "
        "to rule that out; restored, the object would be another one, so the statement re-runs every time",
        name,
        "an id()",
        what=(
            f"'{name}' holds a number that may be the id() of an object, and cash could not look through all of "
            "the notebook's objects to rule that out. Restored from the cache, it would find another object, "
            "so this statement runs every time instead."
        ),
    )


def _holds_an_id(name: str) -> IdentityHeld:
    return IdentityHeld(
        f"'{name}' holds the id() of an object the notebook holds; restored, the object would be "
        "another one, so the statement re-runs every time",
        name,
        "an id()",
    )


def _tracked_ids() -> set[int]:
    """The ``id()`` of every object the garbage collector tracks."""
    import gc

    return {id(obj) for obj in gc.get_objects()}


#: How many items `_sampled_ints` reads from each end of a container it was
#: not looked inside.
_SAMPLE = 32


def _sampled_ints(unread: list[Any]) -> set[int]:
    """The ints among the first and last items (and keys) of each container
    in *unread* too big to look inside, and of the index and values of each
    pandas or NumPy object: a registry keyed by ``id()`` keeps them all
    alike, so a few find it."""
    import itertools

    out: set[int] = set()

    def ends(items: Any) -> list[Any]:
        if isinstance(items, (list, tuple)):
            return [*items[:_SAMPLE], *items[-_SAMPLE:]]
        return list(itertools.islice(items, _SAMPLE))

    def array_ends(array: Any) -> list[Any]:
        if attribute_or(attribute_or(array, "dtype", None), "kind", None) not in ("i", "u"):
            return []
        try:
            flat = array.ravel()
            return [int(n) for n in (*flat[:_SAMPLE], *flat[-_SAMPLE:])]
        except (TypeError, ValueError, AttributeError, IndexError):
            return []

    for obj in unread:
        if isinstance(obj, dict):
            picked = ends(dict.__iter__(obj)) + ends(dict.values(obj))
        elif isinstance(obj, (list, tuple, set, frozenset)):
            picked = ends(obj if isinstance(obj, (list, tuple)) else iter(obj))
        else:
            picked = array_ends(obj)
            for part in ("index", "columns"):
                picked += array_ends(attribute_or(attribute_or(obj, part, None), "values", None))
            if attribute_or(obj, "ndim", None) == 1:
                # A Series' values; a frame's would copy every column.
                picked += array_ends(attribute_or(obj, "values", None))
        out.update(n for n in picked if type(n) is int)
    return out


def _lowest_address(user_ns: dict[str, Any]) -> int:
    """Half the lowest address among a few objects of this process: its
    own (``None``), a small and a large one just made, and the notebook's
    namespace. An id() the notebook keeps is an address in the same heap;
    a number below this (``65536``, ``len(df)``) is not looked for."""
    return min(id(None), id(object()), id(bytearray(1 << 20)), id(user_ns)) // 2


def weakly_held_reason(name: str, value: Any, user_ns: dict[str, Any]) -> IdentityHeld | None:
    """Why *value*, bound to *name*, cannot be restored while another
    variable finds it, or an object in it, by a weak reference; None when it
    can (`_weakly_held`, with `identity_held_reason`'s walk)."""
    targets: list[Any] = []
    _walk([value], set(), {}, [], set(), targets)
    return _weakly_held(name, targets, user_ns) if targets else None


def _weakly_held(name: str, targets: list[Any], user_ns: dict[str, Any]) -> IdentityHeld | None:
    """Why *name*'s value, whose walk found *targets* with a weak reference
    to them, cannot be restored while another variable holds one.

    ``box = Box(BIG)`` keeping ``weakref.ref(BIG)``, then ``BIG[0].id = -5``
    restored: ``BIG`` is bound to the stored copy, the old object goes, and
    ``box.h()`` finds nothing (``TypeError: 'NoneType' object is not
    iterable`` where plain Python gives the sum). Asked only of objects that
    have a weak reference at all (`weakref.getweakrefcount`). The other
    variables are looked into as `_walk` does, `_WEAK_HOLDER_LEVELS` deep: a
    weak reference held deeper is not found. Not up the garbage collector's
    referrers: cash keeps weak references to the values it watches itself,
    and the read of the heap that tells those apart spent the share check's
    budget on every statement over them."""
    from .output_refusals import is_history_name

    wanted = {id(target) for target in targets}
    seen: set[int] = set()
    user_kinds: dict[type, bool] = {}
    cut: list[str] = []
    for var, held in list(user_ns.items()):
        if var == name or var.startswith("_") or is_history_name(var):
            continue
        refs: list[Any] = []
        if not _walk([held], seen, user_kinds, refs, set(), levels=_WEAK_HOLDER_LEVELS):
            cut.append(var)
        if any(issubclass(type(ref), weakref.ReferenceType) and id(ref()) in wanted for ref in refs):
            return IdentityHeld(
                f"'{var}' finds '{name}' by a weak reference; restored, '{name}' would be another object, "
                "so the statement re-runs every time",
                name,
                "a weak reference",
                what=(
                    f"'{var}' keeps a weak reference to '{name}', or to an object in it. Restored from the "
                    f"cache, '{name}' would be a copy and that reference would find nothing, so this "
                    "statement runs every time instead."
                ),
            )
    if cut and _weakly_found_by_others(targets):
        listed = ", ".join(f"'{var}'" for var in sorted(cut))
        return IdentityHeld(
            f"cash did not look all the way into {listed} for a weak reference to '{name}'; restored, "
            f"'{name}' would be another object, so the statement re-runs every time",
            name,
            "a weak reference",
            what=(
                f"An object in '{name}' has a weak reference to it, and cash looked only "
                f"{_WEAK_HOLDER_LEVELS} levels into {listed} for it. If one of them keeps it, a copy "
                f"of '{name}' restored from the cache would leave it finding nothing, so this statement "
                "runs every time instead."
            ),
        )
    return None


def _weakly_found_by_others(targets: list[Any]) -> bool:
    """Whether any of *targets* has a weak reference to it that is not
    cash's own: one whose callback is cash's code (cash's memos drop their
    entry when the object goes) is not a variable's."""
    for target in targets:
        for ref in weakref.getweakrefs(target):
            code = attribute_or(attribute_or(ref, "__callback__", None), "__code__", None)
            if code is None or not code.co_filename.startswith(_CASH_DIR):
                return True
    return False


_CASH_DIR = __file__.rsplit("notebook", 1)[0]


#: How deep `_weakly_held` looks into the other variables: ``box.h`` is two
#: levels below ``box``, ``reg.data['k']`` three below ``reg``.
_WEAK_HOLDER_LEVELS = 4


def _alive(proxy: Any) -> bool:
    """Whether the object a weak proxy points at is alive."""
    try:
        proxy.__class__  # raises once the object is gone
    except ReferenceError:
        return False
    except Exception:  # noqa: BLE001 - the object's own lookup raised: it is there
        return True
    return True


def _walk(
    roots: list[Any],
    seen: set[int],
    user_kinds: dict[type, bool],
    refs: list[Any],
    numbers: set[int],
    targets: list[Any] | None = None,
    levels: int | None = None,
    wanted: set[int] | None = None,
    unread: list[Any] | None = None,
) -> bool:
    """Look through *roots* (see the module docstring), adding the ``id()``
    of each object to *seen*, each weak reference to *refs* and each
    ``int`` to *numbers*, and with *targets*, each object some weak
    reference points at; with *levels*, no deeper than that below *roots*.
    False when it stopped at `_MAX_SEEN`, and with *levels*, also when it
    left an object to look into: below the last level, or in a container of
    over `_MAX_SCANNED` items. With *wanted*, it stops, complete, at the
    first level that sees one of those ids. With *unread*, each container it
    did not look inside and each pandas or NumPy object goes there, and the
    walk is not complete when it left a container unread."""
    level = roots
    complete = True
    while level:
        if levels is not None:
            if levels <= 0:
                return complete and not any(_looked_into(item, seen, user_kinds) for item in level)
            levels -= 1
        nxt: list[Any] = []
        for item in level:
            kind = type(item)
            if kind is int:
                numbers.add(item)
                continue
            if id(item) in seen:
                continue
            if targets is not None and weakref.getweakrefcount(item) and _notebook_own(kind, user_kinds):
                targets.append(item)
            if issubclass(kind, _REFERENCES):
                seen.add(id(item))
                refs.append(item)
                continue
            if kind in _CONTAINERS or issubclass(kind, (list, dict, set)):
                # By the builtin type's own methods: a subclass whose
                # ``keys()`` or ``__iter__`` raises is not asked.
                items: Any = builtin_items(item) or ()
            elif issubclass(kind, _WEAK_CONTAINERS):
                data = attribute_or(item, "data")
                items = (
                    list(data.keys()) + list(data.values())
                    if type(data) is dict
                    else list(data)
                    if type(data) is set
                    else []
                )
            elif _users(kind, user_kinds):
                state = attribute_or(item, "__dict__")
                if type(state) is not dict:
                    continue
                items = list(state.values())
            else:
                seen.add(id(item))
                if unread is not None and _an_array(kind):
                    unread.append(item)
                continue
            seen.add(id(item))
            if len(seen) > _MAX_SEEN:
                return False
            if len(items) > _MAX_SCANNED:
                if unread is not None:
                    unread.append(item)
                    complete = False
                complete = complete and levels is None
                continue
            nxt.extend(items)
        if wanted is not None and not wanted.isdisjoint(seen):
            return True
        level = nxt
    return complete


def _an_array(kind: type) -> bool:
    """Whether *kind* is a pandas or NumPy class, whose index or values may
    keep ids (``pd.Series(..., index=[id(p) for p in ps])``)."""
    module = attribute_or(kind, "__module__", None) or ""
    return module.split(".", 1)[0] in ("pandas", "numpy")


def _looked_into(item: Any, seen: set[int], user_kinds: dict[type, bool]) -> bool:
    """Whether `_walk` would look into *item*, or keep it as a reference."""
    kind = type(item)
    if kind is int or id(item) in seen:
        return False
    return (
        kind in _CONTAINERS
        or issubclass(kind, (list, dict, set, _REFERENCES, _WEAK_CONTAINERS))
        or _users(kind, user_kinds)
    )


def _notebook_own(kind: type, user_kinds: dict[type, bool]) -> bool:
    """Whether an object of *kind* is the notebook's own data: a list, dict
    or set, or an object of a class of the user's code. A library's object
    is not looked at for weak references to it: pandas keeps its own on a
    frame (``df.flags``), which no variable of the notebook finds it by."""
    return kind in _CONTAINERS or issubclass(kind, (list, dict, set)) or _users(kind, user_kinds)


def _users(kind: type, user_kinds: dict[type, bool]) -> bool:
    """Whether *kind* is a class of the user's code."""
    known = user_kinds.get(kind)
    if known is None:
        module = sys.modules.get(attribute_or(kind, "__module__") or "")
        known = user_kinds[kind] = module is not None and is_user_code_module(module)
    return known
