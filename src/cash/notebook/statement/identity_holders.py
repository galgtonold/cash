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
  or the notebook's variables hold, at any depth, and an integer NumPy
  array holding one.

The value is walked through lists, tuples, dicts, sets, deques and their
subclasses (read through the builtin type's own methods, never the
class's), a NumPy array of objects, and the attributes and slots of
objects of the user's own classes and of a ``SimpleNamespace``, not into a
library's objects: a pandas frame keeps weak references between its own
parts, which a restore rebuilds together, and a stream the ident of its
thread, which no object of the notebook's is. A container of
more than `_MAX_SCANNED` numbers, strings and the like, or of tuples of
them (records), is read at C speed rather than item by item; one of the
notebook's holding other objects is checked for the ``id()`` of its items,
not of what they hold. A value the
walk cannot finish within `_MAX_SEEN` objects, or a container it cannot
read, could hold either, so the statement runs again with the warning.
"""

from __future__ import annotations

import sys
import types
import weakref
from itertools import chain
from typing import Any

from cash._user_attrs import attribute_or, builtin_base, builtin_items, own_dict
from cash.closure_sharing import _slot_readers
from cash.install_paths import is_user_code_module
from cash.value_types import IMMUTABLE_LEAF_TYPES

__all__ = ["IdentityHeld", "identity_held_reason"]

#: A container with more items is read at C speed when it holds only
#: leaves or records of them, and walked item by item otherwise.
_MAX_SCANNED = 10_000
#: Most objects one walk looks at.
_MAX_SEEN = 200_000
#: Restored as themselves, by reference: a weak reference to one stays right.
_BY_REFERENCE = (type, types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType)
_REFERENCES = (weakref.ReferenceType, weakref.ProxyType, weakref.CallableProxyType)
#: Hold their weak references in ``.data``.
_WEAK_CONTAINERS = (weakref.WeakValueDictionary, weakref.WeakKeyDictionary, weakref.WeakSet)
#: Hold nothing else; never the target of a registry's ``id()``.
_LEAVES = frozenset(IMMUTABLE_LEAF_TYPES)
#: The smallest ``id()`` looked for: an object's address. Nothing is mapped
#: below an executable's base, 0x400000 on Linux when not position
#: independent and far higher otherwise (macOS keeps the lowest 4 GiB
#: unmapped): ``N = 100_000`` or a ``range`` of row numbers is no ``id()``.
_LOWEST_ADDRESS = 1 << 22
#: Above any address a 64-bit process maps (48-bit virtual addresses): a
#: nanosecond or microsecond timestamp is no ``id()``.
_HIGHEST_ADDRESS = 1 << 48
#: What every object's address is a multiple of: a pointer's size. The heap
#: keeps 16 bytes on 64-bit builds, but an object compiled into a C image
#: keeps only 8: ``np.True_`` (which ``(arr >= 0).all()`` returns), ``()``,
#: and the builtin types on a clang build.
_ALIGNMENT = 8 if sys.maxsize > 1 << 32 else 4


class IdentityHeld(str):
    """The refusal `identity_held_reason` gives: the reason, with the
    variable it names and how its value finds the object (*kind*: ``"a weak
    reference"`` or ``"an id()"``), for the warning (``NOTEBOOK-IDENTITY-HELD``)."""

    name: str
    kind: str
    #: What the warning says happened, when not that *name* holds *kind*:
    #: a loop's refusal to restore into an object the notebook points at
    #: (`in_place.refusal`).
    what: str | None

    def __new__(cls, reason: str, name: str, kind: str, what: str | None = None) -> IdentityHeld:
        made = super().__new__(cls, reason)
        made.name, made.kind, made.what = name, kind, what
        return made


def identity_held_reason(name: str, value: Any, user_ns: dict[str, Any]) -> IdentityHeld | None:
    """Why *value*, bound to *name*, cannot be restored apart from what it
    points at by identity (see the module docstring); None when it can."""
    user_kinds: dict[type, bool] = {}
    walk = _Walk(user_kinds)
    walk.run([value])
    if walk.unread or walk.cut:
        what = (
            f"a {type(walk.unread[0]).__name__} cash cannot look inside"
            if walk.unread
            else "more objects than cash looks through"
        )
        return IdentityHeld(
            f"'{name}' holds {what}; it could hold a weak reference or an id(), so the statement re-runs every time",
            name,
            "a container cash cannot read",
            what=(
                f"'{name}' holds {what}, so cash cannot tell whether it finds another object by a weak "
                "reference or an id(). This statement runs every time instead."
            ),
        )
    for ref in walk.refs:
        if not issubclass(type(ref), weakref.ReferenceType):
            if _alive(ref):
                return IdentityHeld(
                    f"'{name}' holds a weak proxy; restored, it would point at another object, so the statement "
                    "re-runs every time",
                    name,
                    "a weak reference",
                )
            continue
        target = weakref.ref.__call__(ref)  # not a subclass's own __call__: the user's code
        if target is not None and id(target) not in walk.seen and not issubclass(type(target), _BY_REFERENCE):
            return IdentityHeld(
                f"'{name}' holds a weak reference to an object it does not hold; restored, it would "
                "point at another object, so the statement re-runs every time",
                name,
                "a weak reference",
            )
    candidates = {n for n in walk.numbers if _LOWEST_ADDRESS <= n < _HIGHEST_ADDRESS and not n % _ALIGNMENT}
    for array in walk.arrays:
        candidates.update(_addresses_in(array))
    if not candidates:
        return None
    from .output_refusals import is_history_name

    held = _Walk(user_kinds, seen=set(walk.seen), quiet=True, candidates=frozenset(candidates))
    held.run([v for k, v in user_ns.items() if k != name and not is_history_name(k)])
    if held.cut or held.found or not candidates.isdisjoint(held.seen):
        return IdentityHeld(
            f"'{name}' holds the id() of an object the notebook holds; restored, the object would be "
            "another one, so the statement re-runs every time",
            name,
            "an id()",
        )
    return None


def _alive(proxy: Any) -> bool:
    """Whether the object a weak proxy points at is alive."""
    try:
        proxy.__class__  # raises once the object is gone
    except ReferenceError:
        return False
    except Exception:  # noqa: BLE001 - the object's own lookup raised: it is there
        return True
    return True


def _addresses_in(array: Any) -> list[int]:
    """The numbers of an integer NumPy *array* that could be an ``id()``,
    read at C speed."""
    flat = array.ravel()
    found = flat[(flat >= _LOWEST_ADDRESS) & (flat < _HIGHEST_ADDRESS) & (flat % _ALIGNMENT == 0)]
    return found.tolist() if found.size else []


def _numpy_array(kind: type) -> bool:
    return kind.__name__ == "ndarray" and attribute_or(kind, "__module__") == "numpy"


class _Walk:
    """One look through what objects hold (see the module docstring): the
    ``id()`` of each object met in `seen`, each weak reference in `refs`,
    each ``int`` in `numbers`, each integer NumPy array in `arrays`, and
    each container that could not be read in `unread`. `cut` when it
    stopped at `_MAX_SEEN`. A *quiet* walk only gathers `seen`."""

    def __init__(
        self,
        user_kinds: dict[type, bool],
        seen: set[int] | None = None,
        quiet: bool = False,
        candidates: frozenset[int] = frozenset(),
    ) -> None:
        self.user_kinds = user_kinds
        self.seen: set[int] = set() if seen is None else seen
        self.budget = _MAX_SEEN
        self.refs: list[Any] = []
        self.numbers: set[int] = set()
        self.arrays: list[Any] = []
        self.unread: list[Any] = []
        self.cut = False
        self.quiet = quiet
        #: The ids looked for, by a *quiet* walk; `found` when one is the
        #: ``id()`` of an item of a big container met.
        self.candidates = candidates
        self.found = False

    def run(self, roots: list[Any]) -> None:
        level = list(roots)
        while level:
            into: list[Any] = []
            for item in level:
                self._open(item, into)
            if self.cut:
                return
            level = into

    def _open(self, item: Any, into: list[Any]) -> None:
        kind = type(item)
        if kind is int:
            self.numbers.add(item)
            return
        if kind in _LEAVES:
            return
        key = id(item)
        if key in self.seen:
            return
        self.seen.add(key)
        self.budget -= 1
        if self.budget < 0:
            self.cut = True
            return
        if issubclass(kind, _REFERENCES):
            if not self.quiet:
                self.refs.append(item)
            return
        if issubclass(kind, _BY_REFERENCE):
            return
        items: Any
        if builtin_base(item) is not None:
            # Through the builtin type's own methods: a dict subclass's
            # ``values()`` or a list subclass's ``__iter__`` is the user's
            # code, which plain Python never called here.
            try:
                items = builtin_items(item)
            except Exception:  # noqa: BLE001 - not readable: what it holds is unknown
                items = None
            if items is None:
                if not self.quiet:
                    self.unread.append(item)
                return
            # A subclass's own attributes: ``c = C([1]); c.meta = id(p.child)``.
            state = own_dict(item)
            if state:
                items.extend(state.values())
        elif issubclass(kind, _WEAK_CONTAINERS):
            state = own_dict(item)
            data = state.get("data") if state else None
            items = (
                [*dict.keys(data), *dict.values(data)]
                if type(data) is dict
                else list(data)
                if type(data) is set
                else []
            )
        elif _numpy_array(kind):
            if item.dtype.hasobject:
                items = item.ravel().tolist()
            else:
                if not self.quiet and item.dtype.kind in "iu" and item.dtype.itemsize == 8:
                    self.arrays.append(item)
                return
        elif issubclass(kind, types.SimpleNamespace) or self._users(kind):
            items = []
            state = own_dict(item)
            if state:
                items.extend(state.values())
            for reader in _slot_readers(kind):
                try:
                    items.append(reader.__get__(item, kind))
                except AttributeError:  # an empty slot
                    continue
        else:
            # A library's object: its own parts hold weak references to
            # each other (a pandas frame's), and numbers that are no id()
            # of the notebook's (a stream's thread ident).
            return
        if len(items) > _MAX_SCANNED:
            if self._read_flat(items):
                return
            if self.quiet:
                # The notebook's big list of records: an ``id()`` of one
                # of them is found, not one of what each holds, which
                # walking them all cost every check. At C speed, without
                # a set of their ids.
                if not self.found and not self.candidates.isdisjoint(map(id, items)):
                    self.found = True
                return
        into.extend(items)

    def _read_flat(self, items: list[Any]) -> bool:
        """Read a big container of leaves, or of tuples of leaves (records),
        at C speed: only its ``int``s matter. False when it holds anything
        else, to be walked item by item."""
        kinds = set(map(type, items))
        flat: Any = items
        if not kinds <= _LEAVES:
            if not kinds <= _LEAVES | {tuple}:
                return False
            flat = list(chain.from_iterable(item for item in items if type(item) is tuple))
            kinds = set(map(type, flat))
            if not kinds <= _LEAVES:
                return False
        if int not in kinds or self.quiet:
            return True
        if kinds == {int}:
            # At C speed: row numbers or counts all below any address.
            if max(flat) >= _LOWEST_ADDRESS and min(flat) < _HIGHEST_ADDRESS:
                self.numbers.update(flat)
        else:
            self.numbers.update(item for item in flat if type(item) is int and item >= _LOWEST_ADDRESS)
        return True

    def _users(self, kind: type) -> bool:
        """Whether *kind* is a class of the user's code."""
        known = self.user_kinds.get(kind)
        if known is None:
            module = sys.modules.get(attribute_or(kind, "__module__") or "")
            known = self.user_kinds[kind] = module is not None and is_user_code_module(module)
        return known
