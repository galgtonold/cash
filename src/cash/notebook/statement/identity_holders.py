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
  or the notebook's variables hold, at any depth, an integer NumPy array
  holding one, and one turned into text (``hex(id(x))``, `as_address`); a
  number outside every mapping this process can write (`_mapped`) is not
  one;
- a weak reference another variable keeps to the value, or to an object in
  it (`_weakly_held`).

The value is walked through lists, tuples, dicts, sets, deques and their
subclasses (read through the builtin type's own methods, never the
class's), a NumPy array of objects, and the attributes and slots of
objects of the user's own classes and of a ``SimpleNamespace``. A library's
object is read only as `library_parts` knows it (a pandas object's columns,
a ``functools.partial``'s arguments), never attribute by attribute: a
pandas frame keeps weak references between its own parts, which a restore
rebuilds together, and a stream the ident of its thread, which no object of
the notebook's is. One it does not know could hold an ``id()`` or a weak
reference too, so a value holding one next to either runs again. A
container of more than `_MAX_SCANNED` numbers, strings and the like, or of
tuples of them (records), is read at C speed rather than item by item; one
of the notebook's holding objects of one class of the user's with only such
attributes is checked for the ``id()`` of its items at C speed too. A value
the walk cannot finish within `_MAX_SEEN` objects, or a container it cannot
read, could hold either, and so could the notebook's when a candidate
``id()`` is looked for below more: the statement runs again with the
warning.
"""

from __future__ import annotations

import array
import functools
import operator
import re
import sys
import types
import weakref
from itertools import chain
from typing import Any

from cash._user_attrs import attribute_or, builtin_base, builtin_items, own_dict
from cash.closure_sharing import _slot_readers
from cash.install_paths import is_user_code_module
from cash.value_types import IMMUTABLE_LEAF_TYPES

__all__ = ["IdentityHeld", "as_address", "identity_held_reason", "library_parts"]

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
_DICT = operator.attrgetter("__dict__")
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


def identity_held_reason(
    name: str, value: Any, user_ns: dict[str, Any], weak_holders: bool = True
) -> IdentityHeld | None:
    """Why *value*, bound to *name*, cannot be restored apart from what it
    points at by identity (see the module docstring), or with *weak_holders*,
    while another variable finds it by one (`_weakly_held`); None when it
    can. Not *weak_holders* for a loop's object a hit restores into: it
    stays the object those references find."""
    user_kinds: dict[type, bool] = {}
    walk = _Walk(user_kinds, targets=[] if weak_holders else None)
    walk.run([value])
    found = _held_by_id(name, walk, user_kinds, user_ns)
    if found is None and walk.targets:
        found = _weakly_held(name, walk.targets, user_ns)
    return found


def _held_by_id(name: str, walk: _Walk, user_kinds: dict[type, bool], user_ns: dict[str, Any]) -> IdentityHeld | None:
    """`identity_held_reason`'s answer for what the value itself holds, from
    its *walk*."""
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
    if candidates:
        # The process's mappings are read only for a number that could be
        # an id.
        lowest = _lowest_address()
        candidates = _mapped({n for n in candidates if n >= lowest})
    # An array's are kept as arrays: a column of millions of numbers is
    # checked at C speed, never made a set.
    in_arrays = [found for found in map(_addresses_in, walk.arrays) if found.size]
    if candidates or in_arrays:
        from .output_refusals import is_history_name

        held = _Walk(user_kinds, seen=set(walk.seen), quiet=True, candidates=frozenset(candidates))
        held.id_arrays = [] if in_arrays else None
        held.run([v for k, v in user_ns.items() if k != name and not is_history_name(k)])
        if held.cut and not held.found:
            # Not known to be an id(): the notebook's objects were not all
            # looked through.
            return IdentityHeld(
                f"'{name}' holds a number that could be an id(), and the notebook holds more objects than cash "
                "looks through, so the statement re-runs every time",
                name,
                "a container cash cannot read",
                what=(
                    f"'{name}' holds a number that could be the id() of an object, and the notebook holds more "
                    "objects than cash looks through, so cash cannot tell whether it is one. This statement "
                    "runs every time instead."
                ),
            )
        if held.found or not candidates.isdisjoint(held.seen) or (in_arrays and _any_held(in_arrays, held)):
            return IdentityHeld(
                f"'{name}' holds the id() of an object the notebook holds; restored, the object would be "
                "another one, so the statement re-runs every time",
                name,
                "an id()",
            )
    if walk.closed and (candidates or in_arrays or walk.refs):
        # A value that keeps ids or weak references, next to a library's
        # object cash does not look inside: that object could keep one too.
        what = f"a {type(walk.closed[0]).__name__} cash does not look inside"
        return IdentityHeld(
            f"'{name}' holds {what}; it could hold a weak reference or an id(), so the statement re-runs every time",
            name,
            "a container cash cannot read",
            what=(
                f"'{name}' holds {what}, so cash cannot tell whether it finds another object by a weak "
                "reference or an id(). This statement runs every time instead."
            ),
        )
    return None


#: Every digit as ``0``: a run of seven digits is then ``"0000000"``.
_DIGITS_AS_ZERO = str.maketrans("123456789", "000000000")
#: The lines of text `as_address` may read as a number.
_NUMBER_LINES = re.compile(r"^(?:0[xX][0-9a-fA-F]{5,16}|[0-9]{7,18})$", re.MULTILINE)


def as_address(text: str) -> int | None:
    """The number *text* spells when it could be an ``id()`` turned into
    text, ``hex(id(x))`` or ``str(id(x))``; None otherwise."""
    if not 6 < len(text) < 19:
        return None
    try:
        if text[:2] in ("0x", "0X"):
            return int(text, 16)
        return int(text) if text.isdigit() else None
    except ValueError:
        return None


def library_parts(value: Any) -> list[Any] | None:
    """What a library's object of a kind cash knows keeps the user's data
    in: a ``functools.partial``'s function and arguments, an
    ``argparse.Namespace``'s attributes, an ``array.array``'s numbers, a
    structured NumPy array's fields, and a pandas frame's, series' or
    index's columns as NumPy arrays. None for any other object: what it
    holds is not known."""
    if isinstance(value, functools.partial):
        keywords = functools.partial.keywords.__get__(value)
        parts = [functools.partial.func.__get__(value), *functools.partial.args.__get__(value)]
        parts.extend(dict.values(keywords) if type(keywords) is dict else ())
        state = own_dict(value)
        if state:
            parts.extend(state.values())
        return parts
    kind = type(value)
    if kind is array.array:
        return array.array.tolist(value) if value.typecode in "lLqQ" else []
    if _numpy_array(kind) and value.dtype.names:
        return [value[field] for field in value.dtype.names]
    argparse = sys.modules.get("argparse")
    if argparse is not None and isinstance(value, argparse.Namespace):
        state = own_dict(value)
        return list(state.values()) if state else []
    pandas = sys.modules.get("pandas")
    if pandas is not None and isinstance(value, (pandas.DataFrame, pandas.Series, pandas.Index)):
        return _pandas_parts(pandas, value)
    return None


def _pandas_parts(pandas: Any, value: Any) -> list[Any]:
    """A pandas object's labels and columns that could keep an ``id()`` or
    a weak reference: as NumPy arrays of 8-byte integers or of objects, a
    frame's read a dtype at a time (2D)."""
    if isinstance(value, pandas.RangeIndex):
        return []
    if isinstance(value, pandas.DataFrame):
        parts: list[Any] = [value.columns, value.index]
        by_dtype: dict[Any, list[int]] = {}
        for at, dtype in enumerate(value.dtypes):
            by_dtype.setdefault(dtype, []).append(at)
        for dtype, columns in by_dtype.items():
            column = _pandas_column(value.iloc[:, columns], dtype)
            if column is not None:
                parts.append(column)
        return parts
    if isinstance(value, pandas.MultiIndex):
        return list(value.levels)
    column = _pandas_column(value, value.dtype)
    parts = [] if column is None else [column]
    if isinstance(value, pandas.Series):
        parts.append(value.index)
    return parts


def _pandas_column(column: Any, dtype: Any) -> Any:
    """The values of *column* (a series, an index, or a frame's columns of
    one *dtype*) as a NumPy array when they could keep an ``id()`` or a
    weak reference: 8-byte integers, or objects. None otherwise."""
    kind = attribute_or(dtype, "kind")
    if kind == "O":
        return column.to_numpy(dtype=object)
    if kind in ("i", "u") and attribute_or(dtype, "itemsize") == 8:
        numpy = sys.modules.get("numpy")
        if numpy is not None and isinstance(dtype, numpy.dtype):
            return column.to_numpy()
        return column.to_numpy(dtype="int64", na_value=0)  # a nullable Int64 column
    return None


#: The lowest address a 64-bit macOS process maps: ``__PAGEZERO`` covers
#: the first 4 GB.
_MACOS_PAGEZERO = 1 << 32


def _lowest_address() -> int:
    """The lowest address this process has mapped, where it can be read:
    no object lies below it, so a number below it is no id(). On Linux, the
    first mapping in ``/proc/self/maps``, read each time (a mapping made
    since can be lower); on 64-bit macOS, the end of ``__PAGEZERO``.
    Elsewhere (Windows) nothing is known: 0, and `_LOWEST_ADDRESS` holds."""
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
    id(). Elsewhere, or where the file cannot be read, all of them."""
    if not sys.platform.startswith("linux"):
        return numbers
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


def _weakly_held(name: str, targets: list[Any], user_ns: dict[str, Any]) -> IdentityHeld | None:
    """Why *name*'s value, whose walk found *targets* with a weak reference
    to them, cannot be restored while another variable holds one.

    ``box = Box(BIG)`` keeping ``weakref.ref(BIG)``, then ``BIG[0].id = -5``
    restored: ``BIG`` is bound to the stored copy, the old object goes, and
    ``box.h()`` finds nothing (``TypeError: 'NoneType' object is not
    iterable`` where plain Python gives the sum). Asked only of objects that
    have a weak reference at all (`weakref.getweakrefcount`). The other
    variables are looked into as `_Walk` does, `_WEAK_HOLDER_LEVELS` deep: a
    weak reference held deeper is not found. Not up the garbage collector's
    referrers: cash keeps weak references to the values it watches itself,
    and the read of the heap that tells those apart spent the share check's
    budget on every statement over them."""
    from .output_refusals import is_history_name

    wanted = {id(target) for target in targets}
    # A proxy finds its object as a reference does, but has no call to ask.
    wanted_refs = {id(ref) for target in targets for ref in weakref.getweakrefs(target)}
    seen: set[int] = set()
    user_kinds: dict[type, bool] = {}
    cut: list[str] = []
    for var, held in list(user_ns.items()):
        if var == name or var.startswith("_") or is_history_name(var):
            continue
        walk = _Walk(user_kinds, seen=seen, quiet=True, keep_refs=True, levels=_WEAK_HOLDER_LEVELS)
        walk.run([held])
        if walk.cut or walk.unread:
            cut.append(var)
        if any(
            id(ref) in wanted_refs
            or (issubclass(type(ref), weakref.ReferenceType) and id(weakref.ref.__call__(ref)) in wanted)
            for ref in walk.refs
        ):
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


def _addresses_in(array: Any) -> Any:
    """The numbers of an integer NumPy *array* that could be an ``id()``,
    as an array, read at C speed."""
    flat = array.ravel()
    return flat[(flat >= _LOWEST_ADDRESS) & (flat < _HIGHEST_ADDRESS) & (flat % _ALIGNMENT == 0)]


def _any_held(in_arrays: list[Any], held: _Walk) -> bool:
    """Whether a number of *in_arrays* is the ``id()`` of an object *held*
    met, at C speed: only the numbers between the lowest and the highest
    of those ids are looked up."""
    numpy = sys.modules["numpy"]
    ids = numpy.concatenate(
        [numpy.fromiter(held.seen, dtype=numpy.int64, count=len(held.seen)), *(held.id_arrays or ())]
    )
    if not ids.size:
        return False
    low, high = ids.min(), ids.max()
    for found in in_arrays:
        found = found[(found >= low) & (found <= high)]
        if found.size and numpy.isin(found, ids).any():
            return True
    return False


def _code_parts(item: Any) -> list[Any]:
    """What a class or a function of the user's code holds: a class's own
    attributes but Python's ``__dunder__`` ones, a function's closure cells
    and defaults, a bound method's object. Nothing for a library's."""
    if isinstance(item, types.MethodType):
        return [item.__self__] if _users_code(item.__func__) else []
    if not _users_code(item):
        return []
    if isinstance(item, type):
        return [v for k, v in vars(item).items() if not (k.startswith("__") and k.endswith("__"))]
    if type(item) is types.FunctionType:
        held: list[Any] = []
        for cell in item.__closure__ or ():
            try:
                held.append(cell.cell_contents)
            except ValueError:  # an empty cell
                continue
        held.extend(item.__defaults__ or ())
        held.extend((item.__kwdefaults__ or {}).values())
        return held
    return []


def _users_code(item: Any) -> bool:
    """Whether *item*, a class or a function, is of the user's code."""
    module = sys.modules.get(attribute_or(item, "__module__") or "")
    return module is not None and is_user_code_module(module)


def _numpy_array(kind: type) -> bool:
    return kind.__name__ == "ndarray" and attribute_or(kind, "__module__") == "numpy"


class _Walk:
    """One look through what objects hold (see the module docstring): the
    ``id()`` of each object met in `seen`, each weak reference in `refs`,
    each ``int`` in `numbers`, each integer NumPy array in `arrays`, and
    each container that could not be read in `unread`. `cut` when it
    stopped at `_MAX_SEEN`, or with *levels*, when it left an object to look
    into below the last of them. A *quiet* walk only gathers `seen`, and
    with *keep_refs* `refs`. With *targets*, each object of the notebook's
    own (`_notebook_own`) some weak reference points at goes there."""

    def __init__(
        self,
        user_kinds: dict[type, bool],
        seen: set[int] | None = None,
        quiet: bool = False,
        candidates: frozenset[int] = frozenset(),
        targets: list[Any] | None = None,
        keep_refs: bool = False,
        levels: int | None = None,
    ) -> None:
        self.user_kinds = user_kinds
        self.targets = targets
        self.keep_refs = keep_refs
        self.levels = levels
        self.seen: set[int] = set() if seen is None else seen
        self.budget = _MAX_SEEN
        self.refs: list[Any] = []
        self.numbers: set[int] = set()
        self.arrays: list[Any] = []
        self.unread: list[Any] = []
        #: The library's objects met and not looked inside (`library_parts`).
        self.closed: list[Any] = []
        self.cut = False
        self.quiet = quiet
        #: The ids looked for, by a *quiet* walk; `found` when one is the
        #: ``id()`` of an item of a big container met.
        self.candidates = candidates
        self.found = False
        #: The ids of the items of each big container a *quiet* walk met,
        #: as NumPy arrays, when an array's numbers are looked for; None
        #: otherwise.
        self.id_arrays: list[Any] | None = None

    def run(self, roots: list[Any]) -> None:
        level = list(roots)
        levels = self.levels
        while level:
            if levels is not None:
                if levels <= 0:
                    self.cut = any(self._opens(item) for item in level)
                    return
                levels -= 1
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
        if kind is str:
            # ``hex(id(x))`` or ``str(id(x))``: an id() as well.
            if not self.quiet:
                number = as_address(item)
                if number is not None:
                    self.numbers.add(number)
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
        if self.targets is not None and weakref.getweakrefcount(item) and self._notebook_own(kind):
            self.targets.append(item)
        if issubclass(kind, _REFERENCES):
            if not self.quiet or self.keep_refs:
                self.refs.append(item)
            return
        if issubclass(kind, _BY_REFERENCE):
            # Restored as itself. Looking through the notebook (*quiet*),
            # a class or a function of the user's still holds objects: an
            # ``id()`` of ``Reg.items[0]``, or ``r = weakref.ref(o)`` in a
            # closure, finds them there.
            if self.quiet:
                into.extend(_code_parts(item))
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
        elif _numpy_array(kind) and item.dtype.names and not item.dtype.hasobject:
            items = [item[field] for field in item.dtype.names]
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
            # A library's object: read as `library_parts` knows it, never
            # attribute by attribute. Its own parts hold weak references to
            # each other (a pandas frame's), and numbers that are no id()
            # of the notebook's (a stream's thread ident).
            try:
                items = library_parts(item)
            except Exception:  # noqa: BLE001 - not readable: what it holds is unknown
                items = None
            if items is None:
                if not self.quiet:
                    self.closed.append(item)
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
                self._note_ids(items)
                if self.found or self._lone_records(items):
                    return
                # What each holds is looked through, within the budget:
                # ``reg = {id(objs[7].child): ...}``.
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
        if self.quiet:
            return True
        if str in kinds:
            # The text spelling a number, found by one search of it all.
            texts = flat if kinds == {str} else [item for item in flat if type(item) is str]
            text = "\n".join(texts)
            # Most text has no run of seven digits and no "0x": found by
            # plain searches, far quicker than the pattern's.
            if "0x" in text or "0X" in text or "0000000" in text.translate(_DIGITS_AS_ZERO):
                self.numbers.update(map(as_address, _NUMBER_LINES.findall(text)))
                self.numbers.discard(None)
        if int not in kinds:
            return True
        if kinds == {int}:
            # At C speed: row numbers or counts all below any address.
            if max(flat) >= _LOWEST_ADDRESS and min(flat) < _HIGHEST_ADDRESS:
                self.numbers.update(flat)
        else:
            self.numbers.update(item for item in flat if type(item) is int and item >= _LOWEST_ADDRESS)
        return True

    def _note_ids(self, items: Any) -> None:
        """Keep the ids of *items* for an array's numbers, when looked for."""
        if self.id_arrays is not None:
            numpy = sys.modules["numpy"]
            self.id_arrays.append(numpy.fromiter(map(id, items), dtype=numpy.int64, count=len(items)))

    def _lone_records(self, items: list[Any]) -> bool:
        """Whether the notebook's big container *items* holds objects of
        one class of the user's, whose attributes are all numbers, strings
        and the like: nothing below them is an object an ``id()`` could
        find but their attribute dicts, checked here at C speed (a loop
        over 200,000 points). False, with nothing checked, otherwise."""
        kinds = set(map(type, items))
        if len(kinds) != 1:
            return False
        (kind,) = kinds
        if (
            not self._users(kind)
            or _slot_readers(kind)
            or own_dict(items[0]) is None
            or attribute_or(kind, "__getattribute__") is not object.__getattribute__
        ):
            return False
        # Python's own slot, read at C speed: the class has no lookup of its
        # own, and `own_dict` found its ``__dict__`` is that slot.
        states = list(map(_DICT, items))
        if set(map(type, states)) != {dict}:
            return False
        if not _LEAVES.issuperset(map(type, chain.from_iterable(map(dict.values, states)))):
            return False
        if not self.candidates.isdisjoint(map(id, states)):
            self.found = True
        self._note_ids(states)
        return True

    def _opens(self, item: Any) -> bool:
        """Whether `_open` would look into *item*, or keep it as a weak
        reference, had the walk gone a level deeper."""
        kind = type(item)
        if kind in (int, str) or kind in _LEAVES or id(item) in self.seen:
            return False
        if issubclass(kind, _BY_REFERENCE):
            return self.quiet and bool(_code_parts(item))
        return (
            issubclass(kind, (*_REFERENCES, *_WEAK_CONTAINERS, types.SimpleNamespace))
            or builtin_base(item) is not None
            or (_numpy_array(kind) and item.dtype.hasobject)
            or self._users(kind)
        )

    def _notebook_own(self, kind: type) -> bool:
        """Whether an object of *kind* is the notebook's own data: a list,
        dict or set, or an object of a class of the user's code. A library's
        object is not looked at for weak references to it: pandas keeps its
        own on a frame (``df.flags``), which no variable of the notebook
        finds it by."""
        return issubclass(kind, (list, dict, set)) or self._users(kind)

    def _users(self, kind: type) -> bool:
        """Whether *kind* is a class of the user's code."""
        known = self.user_kinds.get(kind)
        if known is None:
            module = sys.modules.get(attribute_or(kind, "__module__") or "")
            known = self.user_kinds[kind] = module is not None and is_user_code_module(module)
        return known
