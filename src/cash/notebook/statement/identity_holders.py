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
  or the notebook's variables hold, at any depth; a number outside every
  writable mapping of the process (`_mapped`) is not one.

The value is walked through lists, tuples, dicts, sets and the ``__dict__``
of objects of the user's own classes, not into a library's objects (a
pandas frame keeps weak references between its own parts). A container of
more than `_MAX_SCANNED` items is not looked inside, and the walk stops at
`_MAX_SEEN` objects: walking a big value item by item cost each store more
than the check is worth. Such a container, and a pandas or NumPy object,
gives the ints among its first and last items and its index
(`_sampled_ints`): a registry keyed by ``id()`` keeps them all alike. When
the notebook holds more than the walk reads, a number it could not rule out
is refused, with the warning: not even the ids of every object the garbage
collector tracks rule one out, as it does not track a dict of plain values
or a str.
"""

from __future__ import annotations

import sys
import types
import weakref
from typing import Any

from cash._user_attrs import attribute_or, builtin_items, own_dict
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
    sampled, cut = _sampled_ints(unread, user_kinds)
    numbers.update(sampled)
    found = _held_by_id(name, refs, seen, numbers, user_kinds, user_ns)
    if found is None and cut:
        found = _sampling_cut(name)
    if found is None and targets:
        found = _weakly_held(name, targets, user_ns)
    return found


def _sampling_cut(name: str) -> IdentityHeld:
    """The refusal for a value whose sampling stopped with something it
    could not read (`_sampled_ints`): an id() may be kept there."""
    return IdentityHeld(
        f"cash could not sample all of '{name}' for a number that may be the id() of an object; restored, "
        "the object would be another one, so the statement re-runs every time",
        name,
        "an id()",
        what=(
            f"'{name}' holds containers nested deeper than cash samples, or more integer columns than it reads, "
            "so cash could not rule out that it keeps the id() of an object. Restored from the cache, such a "
            "number would find another object, so this statement runs every time instead."
        ),
    )


def _held_by_id(
    name: str,
    refs: list[Any],
    seen: set[int],
    numbers: set[int],
    user_kinds: dict[type, bool],
    user_ns: dict[str, Any],
) -> IdentityHeld | None:
    """`identity_held_reason`'s answer for what *value* itself holds, from
    its walk: *refs*, *seen* and *numbers*."""
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
    # Read the process's mappings only for a number that could be an id.
    candidates = {n for n in numbers if n >= _LOWEST_ADDRESS and not n % 8}
    if candidates:
        candidates = _mapped(candidates)
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
    # `_MAX_SEEN`). The collector's objects do not rule a number out: it
    # tracks no dict of plain values, no str. A bare one too: ``key =
    # id(p)`` is as much an id as one kept in a dict.
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


#: How many items `_sampled_ints` reads from each end of a container it was
#: not looked inside.
_SAMPLE = 32


def _sampled_ints(unread: list[Any], user_kinds: dict[type, bool]) -> tuple[set[int], bool]:
    """``(ints, cut)``: the ints among the first and last items (and keys)
    of each container in *unread* too big to look inside, and of the
    index, the values and each integer or object column of each pandas or
    NumPy object: a registry keyed by ``id()`` keeps them all alike, so a
    few find it. A sampled item that is a list, tuple, set, dict or object
    of the user's is sampled the same way, `_SAMPLE_LEVELS` deep: ``[{'pid':
    id(p)} for p in ps]`` keeps its ids a level down. *cut* when the
    sampling stopped with one such item unread, or a frame had more integer
    columns than `_SAMPLE_COLUMNS`: the caller refuses, as for a cut walk.
    The rows between the ends stay unread by design."""
    out: set[int] = set()
    cut: list[bool] = []
    for obj in unread:
        _sample_into(obj, out, _SAMPLE_LEVELS, user_kinds, cut)
    return out, bool(cut)


#: How many levels of containers `_sampled_ints` samples into.
_SAMPLE_LEVELS = 8

#: Most integer and object columns of one frame `_sampled_ints` reads.
_SAMPLE_COLUMNS = 1024


def _sample_into(obj: Any, out: set[int], levels: int, user_kinds: dict[type, bool], cut: list[bool]) -> None:
    import itertools

    def ends(items: Any) -> list[Any]:
        if isinstance(items, (list, tuple)):
            return [*items[:_SAMPLE], *items[-_SAMPLE:]]
        return list(itertools.islice(items, _SAMPLE))

    if isinstance(obj, dict):
        picked = ends(dict.__iter__(obj)) + ends(dict.values(obj))
    elif isinstance(obj, (list, tuple, set, frozenset)):
        picked = ends(obj if isinstance(obj, (list, tuple)) else iter(obj))
    else:
        picked = _array_ends(obj)
        for part in ("index", "columns"):
            picked += _array_ends(attribute_or(attribute_or(obj, part, None), "values", None))
        if attribute_or(obj, "ndim", None) == 1:
            # A Series' values; a frame's would copy every column.
            picked += _array_ends(attribute_or(obj, "values", None))
        elif attribute_or(obj, "ndim", None) == 2:
            picked += _frame_column_ends(obj, cut)
    for item in picked:
        kind = type(item)
        if kind is int:
            out.add(item)
            continue
        if kind in (dict, list, tuple, set, frozenset):
            inner = item
        elif _users(kind, user_kinds) and type(own_dict(item)) is dict:
            # ``self.pid = id(p)`` on each of a list of records.
            inner = own_dict(item)
        else:
            continue
        if levels > 1:
            _sample_into(inner, out, levels - 1, user_kinds, cut)
        else:
            cut.append(True)


def _array_ends(array: Any) -> list[int]:
    """The first and last `_SAMPLE` items of an integer NumPy array, and
    the ints among those of an object array."""
    kind = attribute_or(attribute_or(array, "dtype", None), "kind", None)
    if kind not in ("i", "u", "O"):
        return []
    try:
        flat = array.ravel()
        ends = (*flat[:_SAMPLE], *flat[-_SAMPLE:])
        if kind == "O":
            return [item for item in ends if type(item) is int]
        return [int(n) for n in ends]
    except (TypeError, ValueError, AttributeError, IndexError):
        return []


def _frame_column_ends(frame: Any, cut: list[bool]) -> list[int]:
    """The ends of each integer or object column of a pandas frame (a
    ``pid`` column of ids), read a column's ends at a time: the frame's
    ``values``, or a whole nullable column's, would be copied first. Past
    `_SAMPLE_COLUMNS` such columns, *cut* says the rest went unread."""
    dtypes = attribute_or(frame, "dtypes", None)
    iloc = attribute_or(frame, "iloc", None)
    if dtypes is None or iloc is None:
        return []
    picked: list[int] = []
    try:
        columns = [i for i, dtype in enumerate(dtypes) if attribute_or(dtype, "kind", None) in ("i", "u", "O")]
        if len(columns) > _SAMPLE_COLUMNS:
            cut.append(True)
        for position in columns[:_SAMPLE_COLUMNS]:
            for part in (iloc[:_SAMPLE, position], iloc[-_SAMPLE:, position]):
                # As objects: a nullable column with a missing value would
                # otherwise come back as floats.
                picked += _plain_ints(part.to_numpy(dtype=object))
    except (TypeError, ValueError, AttributeError, IndexError):
        return picked
    return picked


def _plain_ints(array: Any) -> list[int]:
    """The ints in *array*, a few items of a column: an integer column's
    all, an object or nullable one's ints (``pd.NA`` and strings left out)."""
    kind = attribute_or(attribute_or(array, "dtype", None), "kind", None)
    if kind in ("i", "u"):
        return [int(n) for n in array]
    if kind == "O":
        return [int(item) for item in array if type(item) is int or type(item).__name__ in _NUMPY_INTS]
    return []


_NUMPY_INTS = frozenset({"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"})


#: The lowest address a 64-bit macOS process maps: ``__PAGEZERO`` covers
#: the first 4 GB.
_MACOS_PAGEZERO = 1 << 32


def _lowest_address() -> int:
    """The lowest address this process has mapped, where it can be read:
    no object lies below it, so a number below it (``65536``, ``len(df)``)
    is no id(). On Linux, the first mapping in ``/proc/self/maps``, read
    each time (a mapping made since can be lower); on 64-bit macOS, the end
    of ``__PAGEZERO``. Elsewhere (Windows) nothing is known: 0, and only
    the first 64 KB, mapped by no process, is left out (`_LOWEST_ADDRESS`).
    Not the addresses of a few objects: how far below them the rest may
    lie depends on how Python was built."""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/maps", encoding="ascii") as maps:
                return min(int(line.split("-", 1)[0], 16) for line in maps if line.strip())
        except (OSError, ValueError):
            return 0
    if sys.platform == "darwin" and sys.maxsize > 2**32:
        return _MACOS_PAGEZERO
    return 0


def _mapped(numbers: set[int]) -> set[int]:
    """Those of *numbers* inside a mapping of this process that can be
    written, where every object lies: on Linux, from ``/proc/self/maps``,
    read again only for a number outside the mappings read before. A sum
    of 200,000 numbers (``19999900000``) lies in no mapping, so it is no
    id(). Elsewhere, those at or above `_lowest_address`; where the file
    cannot be read, all of them."""
    if not sys.platform.startswith("linux"):
        lowest = _lowest_address()
        return {n for n in numbers if n >= lowest}
    ranges = _MAPPINGS[0]
    inside = {n for n in numbers if ranges is not None and any(start <= n < end for start, end in ranges)}
    if len(inside) == len(numbers):
        return inside
    # One outside the mappings read before: a mapping made since may hold
    # it. One the process has since unmapped is kept, which only reruns.
    ranges = _MAPPINGS[0] = _read_mappings()
    if ranges is None:
        return numbers
    return {n for n in numbers if any(start <= n < end for start, end in ranges)}


def _read_mappings() -> list[tuple[int, int]] | None:
    """The writable mappings of ``/proc/self/maps``; None when it cannot be
    read."""
    try:
        with open("/proc/self/maps", encoding="ascii", errors="replace") as maps:
            ranges = []
            for line in maps:
                fields = line.split()
                if len(fields) > 1 and fields[1].startswith("rw"):
                    start, end = fields[0].split("-", 1)
                    ranges.append((int(start, 16), int(end, 16)))
    except (OSError, ValueError):
        return None
    return ranges


#: The mappings `_mapped` read last.
_MAPPINGS: list[list[tuple[int, int]] | None] = [None]


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
