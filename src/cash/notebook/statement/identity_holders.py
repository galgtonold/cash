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
  holding one, and one spelled in text, alone or anywhere inside a longer
  one (``hex(id(x))``, ``f"run-{id(x)}"``, an object's default ``repr``:
  `_spelled_in`); a number outside every mapping this process can write
  (`_mapped`) is not one;
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
import bisect
import collections
import contextlib
import functools
import operator
import re
import struct
import sys
import types
import weakref
from collections.abc import Callable, Iterable, Iterator
from itertools import accumulate, chain, compress, repeat
from typing import Any

from cash import _native, _weak
from cash._plain_data import tree_walk, whole_tree_facts
from cash._user_attrs import attribute_or, builtin_base, builtin_items, has_attribute, own_dict, own_dicts
from cash._weak import callback_of, is_cash_ref
from cash.closure_sharing import slot_readers
from cash.install_paths import is_user_code_module
from cash.notebook.shared_objects import library_exact_values, walk_budget, walk_time_left
from cash.value_types import IMMUTABLE_LEAF_TYPES

from ... import _pandas_compat
from ..._type_memo import class_dict, class_flags, class_mro, class_name
from ...loaded_modules import loaded_module

__all__ = [
    "ID_KIND",
    "IdentityHeld",
    "end_cell_reads",
    "forget_flat_reads",
    "identity_held_reason",
    "library_parts",
    "reading_every_spelling",
    "returned_identity_reason",
]

#: A container with more items is read at C speed when it holds only
#: leaves or records of them, and walked item by item otherwise.
_MAX_SCANNED = 10_000
#: Outside the notebook's walk, a container with at least this many items
#: is tried at C speed too (`_Walk._read_flat`).
_SMALL_FLAT = 8
# Lists and tuples `_read_flat` reads through, a level at a time, and how deep.
_NESTED = frozenset({list, tuple})
_NESTED_LEVELS = 8
_JUST_TUPLES = frozenset({tuple})
#: Most objects one walk looks at, outside a share check's time
#: (`walk_time_left`); within it, the walk goes on `_MORE_SEEN` objects at
#: a time while the time lasts.
_MAX_SEEN = 200_000
#: Objects a walk past `_MAX_SEEN` looks at between two reads of the clock.
_MORE_SEEN = 50_000
#: Restored as themselves, by reference: a weak reference to one stays right.
_BY_REFERENCE = (type, types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType)
_REFERENCES = (weakref.ReferenceType, weakref.ProxyType, weakref.CallableProxyType)
#: Hold their weak references in ``.data``.
_WEAK_CONTAINERS = (weakref.WeakValueDictionary, weakref.WeakKeyDictionary, weakref.WeakSet)
#: Hold nothing else; never the target of a registry's ``id()``.
_LEAVES = frozenset(IMMUTABLE_LEAF_TYPES)
_DICT = operator.attrgetter("__dict__")
_PLAIN_NODES = frozenset({dict, list, tuple})
#: The containers `_Walk._read_records` reads through, by exact type.
_RECORD_NODES = frozenset({dict, list, tuple, set, frozenset})
#: Of those, the ones no weak reference can point at.
_UNREFERABLE = frozenset({dict, list, tuple})
#: Fewest items of a container `_Walk._read_records` reads once however
#: often it is held.
_FLAT_RECORDS_MIN = 32
#: ``Py_TPFLAGS_HEAPTYPE``: a class defined in Python.
_HEAPTYPE = 1 << 9


def _plain_user_class(kind: type) -> bool:
    """Whether `_Walk._open` reads an object of *kind*, a class of the
    user's, by its ``__dict__``'s values alone, the dict Python's own slot
    holds: no ``__slots__``, no ``__getattribute__``, ``__getattr__``,
    ``__class__`` or ``__dict__`` of its own, nothing but classes defined
    in Python above it but ``object``, and none of the kinds `_open` asks
    about first."""
    try:
        if issubclass(kind, (*_REFERENCES, *_WEAK_CONTAINERS, weakref.finalize, *_BY_REFERENCE)):
            return False
        if slot_readers(kind) or attribute_or(kind, "__getattribute__") is not object.__getattribute__:
            return False
        for klass in class_mro(kind)[:-1]:
            if not class_flags(klass) & _HEAPTYPE:
                return False
            own = class_dict(klass)
            if "__class__" in own or "__getattr__" in own or "__slots__" in own:
                return False
            held = own.get("__dict__")
            if held is not None and type(held) is not types.GetSetDescriptorType:
                return False
        return True
    except Exception:  # noqa: BLE001 - a class that cannot be asked is walked an object at a time
        return False


def _value_kinds() -> tuple[frozenset[type], frozenset[type]]:
    """`_LEAVES` with the library's exact value types
    (`library_exact_values`: numpy's scalars and, while none had an
    attribute set, pandas' ``Timestamp`` and ``Timedelta``), and those of
    them that are numpy integers, each read as the ``int`` it is. Not
    numpy's ``str_`` and ``bytes_``, read as any object is: text spelling
    an ``id()`` is looked for in exact strings alone. ``list(series.diff())``
    over times gives a list of a million Timedeltas, which counted against
    `_MAX_SEEN` one by one, so a loop filling one ran again every time.

    ``timedelta64`` subclasses ``signedinteger`` but is a time, not an
    integer: ``int()`` of one in seconds is a ``TypeError`` (its ``.item()``
    is a ``datetime.timedelta``), which broke the user's loop collecting
    them. It and ``datetime64`` are leaves and never read as an id()."""
    global _VALUE_KINDS
    values = library_exact_values()
    if _VALUE_KINDS[0] is not values:
        numpy = loaded_module("numpy")
        extra = frozenset(kind for kind in values if not issubclass(kind, (str, bytes)))
        ints = frozenset(
            kind
            for kind in extra
            if numpy is not None and issubclass(kind, numpy.integer) and not issubclass(kind, numpy.timedelta64)
        )
        _VALUE_KINDS = (values, _LEAVES | extra, ints)
    return _VALUE_KINDS[1], _VALUE_KINDS[2]


#: ``(library_exact_values(), leaves, integer leaves)`` as `_value_kinds`
#: last found them.
_VALUE_KINDS: tuple[frozenset, frozenset[type], frozenset[type]] = (frozenset(), _LEAVES, frozenset())
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


@functools.cache
def _fixed_image() -> tuple[int, int] | None:
    """Where the interpreter's executable is mapped, ``(start, end)``, when
    it is mapped at a fixed address: a Linux build that is not position
    independent (an ELF ``ET_EXEC``). The objects compiled into it (the
    small ints, the one-character and interned strings, ``None``, the
    builtin types) are then at the same address in every process, so an
    id() of one is still that object after a restart; and a column of
    ordinary numbers meets their addresses often (the image sits near
    20 MB: ``np.arange(8_000_000) * 12`` holds ``id(7)``). None anywhere
    else: an image that moves leaves no address fixed.

    Read from the executable's own load segments (``PT_LOAD``: their
    addresses and sizes, its zero-filled data included), not from the
    mappings ``/proc/self/maps`` names it in: the file mapped again later
    (by a library reading its symbols) sits next to the heap's arenas, and
    a span up to there took every object's address for one compiled in,
    so no ``id()`` was looked for at all."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        with open("/proc/self/exe", "rb") as image:
            header = image.read(64)
            if len(header) < 52 or header[:4] != b"\x7fELF" or header[4] not in (1, 2) or header[5] not in (1, 2):
                return None
            wide, order = header[4] == 2, "<" if header[5] == 1 else ">"
            if struct.unpack_from(order + "H", header, 16)[0] != 2:  # ET_EXEC
                return None
            if wide:
                table, size, count = (
                    struct.unpack_from(order + "Q", header, 32)[0],
                    *struct.unpack_from(order + "HH", header, 54),
                )
                entry, fields = order + "IIQQQQQQ", (0, 3, 6)  # p_type, p_vaddr, p_memsz
            else:
                table, size, count = (
                    struct.unpack_from(order + "I", header, 28)[0],
                    *struct.unpack_from(order + "HH", header, 42),
                )
                entry, fields = order + "IIIIIIII", (0, 2, 5)
            if size < struct.calcsize(entry) or not 0 < count < 1 << 12:
                return None
            image.seek(table)
            headers = image.read(size * count)
    except (OSError, ValueError, struct.error):
        return None
    spans = []
    for at in range(0, len(headers) - size + 1, size):
        values = struct.unpack_from(entry, headers, at)
        kind, start, length = (values[i] for i in fields)
        if kind == 1 and length:  # PT_LOAD
            spans.append((start, start + length))
    if not spans:
        return None
    low, high = min(start for start, _ in spans), max(end for _, end in spans)
    # The objects compiled in must be in it, or this is not the image that runs.
    return (low, high) if low <= id(None) < high else None


#: The C kernel of `_Walk._read_nested`, None without it: a frame's column of
#: two million ``(action, time)`` records took a second a statement in
#: Python (student_8).
_NESTED_LEAVES = attribute_or(_native.kernels, "nested_leaves", None)

#: What `_Walk._read_flat` found in a big container of leaves this cell,
#: by its length: the very items read (held, so none of their ids is
#: reused while kept) and the numbers they spelled. A loop that writes one
#: column of a wide frame per pass checks every column after each pass;
#: the columns it left are the same strings, found here by identity at C
#: speed instead of read again (student_5: 113 s for one pass over 60
#: columns of 20,000 rows). Cleared when the cell ends (`end_cell_reads`).
_FLAT_READS: dict[int, list[tuple[list[Any], frozenset[int]]]] = {}
#: The leaves `_FLAT_READS` keeps: plain values every reference count of
#: cash's leaves out, so keeping them changes no other check's answer.
_KEPT_KINDS = frozenset({str, int, float, bool, type(None)})
#: Most items `_FLAT_READS` keeps, all entries together.
_MAX_FLAT_KEPT = 5_000_000
_flat_kept = [0]


#: A fingerprint of a big container's content (``shape_witness``), None
#: without the C kernels: the addresses, types and nesting of everything
#: in it, and the value of each str and int.
_SHAPE_WITNESS = attribute_or(_native.kernels, "shape_witness", None)
#: ``tree_id_leaves``: the ints and texts of a JSON-like value that could
#: be an ``id()``, read in one pass (`_plain_tree_without_ids`).
_TREE_ID_LEAVES = attribute_or(_native.kernels, "tree_id_leaves", None)

#: What `_Walk._read_flat` found in a big container this cell, by the
#: witness of its content: the numbers it spelled, the lists it counted
#: against the budget. A frame changed by each statement of a cell (a
#: column added) is checked after each one, and its columns it left alone
#: were read again each time: two million ``(action, time)`` records,
#: student_8's exploded sessions. The witness is taken again, at C speed
#: and holding nothing (`_SHAPE_WITNESS`), so the memo keeps no object
#: alive and adds no reference any other check counts; a container changed
#: in place, or freed and another built at its address, has another
#: witness. Kept from cell to cell (`end_cell_reads`): a frame's column the
#: next cell leaves alone has the witness it had, and was read again by
#: each cell's first statement on the frame, half a second for student_8's
#: sessions.
_SHAPE_READS: dict[tuple[int, int], tuple[frozenset[int], int, bool]] = {}
#: Most entries `_SHAPE_READS` keeps.
_MAX_SHAPE_READS = 1024
#: Most numbers an entry of `_SHAPE_READS` may hold to be kept past its
#: cell: a column of records holding millions of ints keeps them only
#: while its cell runs.
_KEPT_SHAPE_NUMBERS = 1024


def forget_flat_reads() -> None:
    """Drop what `_FLAT_READS`, `_SHAPE_READS` and `_ARROW_READS` keep."""
    _FLAT_READS.clear()
    _SHAPE_READS.clear()
    _ARROW_READS.clear()
    _ARROW_SIGNED.clear()
    _flat_kept[0] = 0


def end_cell_reads() -> None:
    """The cell ended: drop what `_FLAT_READS` keeps, which holds the very
    objects read, the entries of `_ARROW_READS` holding many numbers, and
    those of `_SHAPE_READS`; the rest of `_SHAPE_READS`, which holds no
    object, is kept for the next cell until it is half full. The rest of
    `_ARROW_READS` holds no object either, and each entry goes with the
    last Arrow array it was read from."""
    _FLAT_READS.clear()
    _flat_kept[0] = 0
    for signature, read in list(_ARROW_READS.items()):  # a copy: an entry may go as it is read
        if read.numbers > _KEPT_ARROW_NUMBERS and _ARROW_READS.get(signature) is read:
            del _ARROW_READS[signature]
            read.drop()
    if len(_SHAPE_READS) > _MAX_SHAPE_READS // 2:
        _SHAPE_READS.clear()
        return
    for witness in [witness for witness, read in _SHAPE_READS.items() if len(read[0]) > _KEPT_SHAPE_NUMBERS]:
        del _SHAPE_READS[witness]


def _flat_read_before(items: list[Any]) -> frozenset[int] | None:
    """The numbers a read of these very *items* found earlier this cell, or
    None when they were not read as leaves (or not as this read reads them:
    `reading_every_spelling`)."""
    if _EVERY_SPELLING[0]:
        return None
    for kept, numbers in _FLAT_READS.get(len(items), ()):
        if all(map(operator.is_, kept, items)):
            return numbers
    return None


def _keep_flat_read(items: list[Any], numbers: frozenset[int]) -> None:
    if _EVERY_SPELLING[0]:
        return
    if _flat_kept[0] + len(items) > _MAX_FLAT_KEPT:
        return
    _flat_kept[0] += len(items)
    _FLAT_READS.setdefault(len(items), []).append((list(items), numbers))


#: `IdentityHeld.kind` of a value holding the ``id()`` of an object the
#: notebook holds (or one the check could not look through for one).
ID_KIND = "an id()"


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
    name: str,
    value: Any,
    user_ns: dict[str, Any],
    weak_holders: bool = True,
    more: int = 0,
    id_free: bool | None = None,
) -> IdentityHeld | None:
    """Why *value*, bound to *name*, cannot be restored apart from what it
    points at by identity (see the module docstring), or with *weak_holders*,
    while another variable finds it by one (`_weakly_held`); None when it
    can. Not *weak_holders* for a loop's object a hit restores into: it
    stays the object those references find. *more*: objects the walk looks
    at beyond `_MAX_SEEN` (the answers a statement put into a helper memo,
    as many as the memo holds). *id_free*: what `holds_no_ids` already
    answered for *value*, so its leaves are not read again; None to ask."""
    if id_free is None:
        id_free = _plain_tree_without_ids(value)
    if id_free:
        return None
    try:
        user_kinds: dict[type, bool] = {}
        walk = _Walk(user_kinds, targets=[] if weak_holders else None)
        walk.budget += more
        walk.allowance = walk.budget
        walk.run([value])
        found = _held_by_id(name, value, walk, user_kinds, user_ns)
        if found is None and walk.targets:
            found = _weakly_held(name, walk.targets, user_ns)
        return found
    except Exception as error:  # noqa: BLE001 - cash's own bookkeeping never breaks the user's cell
        # Not known whether it holds an id(): not stored, and said so.
        what = (
            f"cash could not look through '{name}' for an object it points at by identity "
            f"({class_name(type(error))}: {error}), so this statement runs every time instead."
        )
        return IdentityHeld(f"'{name}' could not be looked through", name, ID_KIND, what)


def returned_identity_reason(
    value: Any, holders: dict[str, Any] | Callable[[], dict[str, Any]], seconds: float
) -> IdentityHeld | None:
    """For ``@cash.cache``: why *value*, what a call returned, must not be
    served from the cache, as `identity_held_reason` tells it for a
    statement's value; None when it may.

    *holders* stands in for the notebook's variables: the call's arguments,
    the function's closure and defaults, and the module globals it can
    read. An ``int`` of the value, or one spelled in its text
    (``f"run-{id(x)}"``, a default ``repr``), is an ``id()`` when it is that
    of an object the value or one of *holders* holds. Served in another
    process, it would name the old process's address where a plain run
    names the new one's; in this one, an object that may be gone. A weak
    reference to an object outside the value, or one a holder keeps to the
    value, is refused as well: a hit hands back a copy.

    The walk may take *seconds* of this thread's CPU time past `_MAX_SEEN`
    objects (`walk_budget`); a value it cannot finish is refused, as it
    could hold either. The memo of big containers read in a cell
    (`_FLAT_READS`), which keeps the very objects it read until the cell
    ends, is left as it was: outside a notebook no cell ends to clear it."""
    kept = {length: list(reads) for length, reads in _FLAT_READS.items()}
    count = _flat_kept[0]
    try:
        # A lone number or text is read as a list of it would be.
        if _plain_tree_without_ids((value,) if type(value) in _PLAIN_LEAVES else value):
            return None
        if callable(holders):
            # Given as a function: asked only now that they are looked through.
            holders = holders()
        with walk_budget(seconds):
            return identity_held_reason("the result", value, holders, id_free=False)
    finally:
        _FLAT_READS.clear()
        _FLAT_READS.update(kept)
        _flat_kept[0] = count


def _held_by_id(
    name: str, value: Any, walk: _Walk, user_kinds: dict[type, bool], user_ns: dict[str, Any]
) -> IdentityHeld | None:
    """`identity_held_reason`'s answer for what *value* itself holds, from
    its *walk*."""
    if walk.unread or walk.cut:
        what = (
            f"a {class_name(type(walk.unread[0]))} cash cannot look inside"
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
        if target is not None and id(target) not in walk.seen_all() and not issubclass(type(target), _BY_REFERENCE):
            return IdentityHeld(
                f"'{name}' holds a weak reference to an object it does not hold; restored, it would "
                "point at another object, so the statement re-runs every time",
                name,
                "a weak reference",
            )
    candidates = _candidate_ids(walk.numbers)
    # An array's are kept as arrays: a column of millions of numbers is
    # checked at C speed, never made a set.
    in_arrays = [found for found in map(_addresses_in, walk.arrays) if found.size]
    if candidates or in_arrays:
        from .output_refusals import is_history_name

        held = _Walk(user_kinds, seen=set(walk.seen_all()), quiet=True, candidates=frozenset(candidates))
        held.id_arrays = [] if in_arrays else None
        held.run([v for k, v in user_ns.items() if k != name and not is_history_name(k)])
        known = held.found or not candidates.isdisjoint(held.seen) or bool(in_arrays and _any_held(in_arrays, held))
        if not known and not held.cut:
            # The value's own strings and numbers: its walk keeps no leaf's
            # id, and ``pair = (s, id(s))`` with *s* nowhere else is its
            # own. Looked through again only now, for the numbers it holds,
            # with the leaves and the count its own walk had: a library's
            # scalars cost nothing, and a share check's time counts too.
            own = _Walk(user_kinds, quiet=True, candidates=frozenset(candidates))
            own.leaves = _value_kinds()[0]
            own.budget = walk.allowance + walk.granted
            own.id_arrays = [] if in_arrays else None
            own.run([value])
            # Cut short, not known to hold none: as the notebook's look is.
            held.found, held.cut = own.found, own.cut
            if own.id_arrays is not None:
                held.id_arrays = [*(held.id_arrays or ()), *own.id_arrays]
                held.leaf_ids.extend(own.leaf_ids)
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
                ID_KIND,
            )
    if walk.closed and (candidates or in_arrays or walk.refs):
        # A value that keeps ids or weak references, next to a library's
        # object cash does not look inside: that object could keep one too.
        what = f"a {class_name(type(walk.closed[0]))} cash does not look inside"
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


def _candidate_ids(numbers: set[int]) -> set[int]:
    """Those of *numbers* that could be the ``id()`` of an object: aligned,
    within the addresses a process maps, and inside a mapping of this one
    (`_mapped`)."""
    candidates = {n for n in numbers if _LOWEST_ADDRESS <= n < _HIGHEST_ADDRESS and not n % _ALIGNMENT}
    fixed = _fixed_image()
    if fixed is not None:
        candidates = {n for n in candidates if not fixed[0] <= n < fixed[1]}
    if candidates:
        # The process's mappings are read only for a number that could be
        # an id.
        lowest = _lowest_address()
        candidates = _mapped({n for n in candidates if n >= lowest})
    return candidates


#: What `_plain_tree_without_ids` takes: exact dicts, lists and tuples over
#: immutable leaves. None of them can be weakly referenced, and none is an
#: object of the user's code: such a value holds no weak reference, none
#: points at it or into it, and only its numbers and strings could find an
#: object by its ``id()``.
_PLAIN_TREE_TYPES = frozenset((dict, list, tuple, *IMMUTABLE_LEAF_TYPES))
#: Of `_PLAIN_TREE_TYPES`, the leaves.
_PLAIN_LEAVES = frozenset(IMMUTABLE_LEAF_TYPES)
#: Of `_PLAIN_TREE_TYPES`, the leaves an ``id()`` can be read from.
_ID_LEAVES = frozenset({int, str})


def holds_no_ids(value: Any) -> bool:
    """Whether *value* is plain data that holds nothing finding another
    object by identity (`_plain_tree_without_ids`): `identity_held_reason`
    answers None for it without a walk."""
    return _plain_tree_without_ids(value)


def _plain_tree_without_ids(value: Any) -> bool:
    """Whether *value* is JSON-like data (`_plain_data.tree_walk`) over
    immutable leaves alone (`_PLAIN_TREE_TYPES`: no ``bytearray``, no fake
    clock's types), none of whose ints and strings, nor its dicts' keys,
    could be an ``id()`` (`_candidate_ids`): then it holds nothing that
    finds another object by identity, whatever its size, and the walk
    object by object (`_Walk`) is not needed. False otherwise, the full walk
    then deciding: parsed sessions of a million records cost half a second
    a statement walked one object at a time, and past `_MAX_SEEN` objects
    the statement ran every time."""
    if type(value) not in _PLAIN_NODES:
        return False
    try:
        if _TREE_ID_LEAVES is not None:
            # Its facts (which the look keeps for the checks after this
            # one) and then its ints and texts, each a pass at C speed that
            # makes no list of the value's levels.
            facts = whole_tree_facts(value)
            if facts is None or not facts.types <= _PLAIN_TREE_TYPES:
                return False
            if facts.types.isdisjoint(_ID_LEAVES):
                return True
            read = _TREE_ID_LEAVES(value, _LEAVES, _LOWEST_ADDRESS, _HIGHEST_ADDRESS, _SHORTEST_RUN)
            if read is not NotImplemented:
                reader = _Walk({})
                reader.numbers = set(read[0])
                reader._read_texts(read[1])
                if not _candidate_ids(reader.numbers):
                    return True
                _keep_plain_read(value, reader.numbers)
                return False
        levels = tree_walk(value)
        if levels is None:
            return False
        reader = _Walk({})
        for level, flat, kinds, key_kinds in levels:
            if not (kinds <= _PLAIN_TREE_TYPES and key_kinds <= _PLAIN_TREE_TYPES):
                return False
            if not kinds.isdisjoint(_ID_LEAVES):
                reader._read_leaves(flat, kinds)
            if not key_kinds.isdisjoint(_ID_LEAVES):
                keys = list(chain.from_iterable(compress(level, map(operator.is_, map(type, level), repeat(dict)))))
                reader._read_leaves(keys, key_kinds)
            if reader.numbers:
                if _candidate_ids(reader.numbers):
                    # A list of leaves alone has this one level.
                    _keep_plain_read(value, reader.numbers)
                    return False
                reader.numbers = set()
        return True
    except Exception:  # noqa: BLE001 - not known: the full walk decides
        return False


def _keep_plain_read(value: Any, numbers: set[int]) -> None:
    """Keep what `_plain_tree_without_ids` read in *value* (*numbers*)
    when it is a long list of texts and numbers alone, which the walk it
    leaves the answer to would read a second time: that walk finds this
    read by the very items (`_Walk._read_flat`)."""
    if type(value) is list and len(value) > _MAX_SCANNED and set(map(type, value)) <= _KEPT_KINDS:
        _keep_flat_read(value, frozenset(numbers))


#: Every digit as ``0``: a run of seven digits is then ``"0000000"``.
_DIGITS_AS_ZERO = str.maketrans("123456789", "000000000")
#: The fewest digits an address is spelled with in decimal: `_LOWEST_ADDRESS`
#: has seven. Not more: an interpreter that is not position independent
#: keeps its heap right above its image, at 8-digit addresses (``0x12c5000``).
_SHORTEST_RUN = len(str(_LOWEST_ADDRESS))
#: The most: `_HIGHEST_ADDRESS` has fifteen. A longer run of digits holds
#: one only as a part of it (``f"{id(a)}{id(b)}"``), read by `_windows`,
#: as a shorter one can (``'v2' + str(id(x))``).
_ADDRESS_DIGITS = len(str(_HIGHEST_ADDRESS))
#: The longest run of digits also read as one number, as `_spelled_in`
#: read a text that is all digits before reading inside texts.
_LONGEST_NUMBER = 18
#: A run of decimal digits that could spell an address, anywhere in a text.
_DECIMAL_RUN = re.compile(r"[0-9]{%d,}" % _SHORTEST_RUN)
#: ``0x`` and the hexadecimal digits after it, anywhere in a text: read as
#: a number when 5 to 16 digits long, as ``hex(id(x))`` is.
_HEX_RUN = re.compile(r"0[xX]([0-9a-fA-F]{5,})")
_SHORTEST_HEX = 5
_LONGEST_HEX = 16
#: The longest run after ``0x`` whose starts are read as an address with
#: hexadecimal digits glued after it: not a hash (``0x`` and 40 or 64
#: digits), whose every start would be read for nothing.
_GLUED_HEX = 24
#: Texts `_Walk._read_texts` joins into one string to search, and the
#: characters, so a column of long texts is not joined into one huge string.
_TEXTS_AT_ONCE = 100_000
_CHARS_AT_ONCE = 1 << 24

assert 1000 % _ALIGNMENT == 0  # an address's last three digits tell its alignment


def _spelled_in(text: str, mapped_only: bool = False) -> set[int]:
    """The numbers *text* spells that could be an ``id()`` turned into text,
    alone or inside it: ``str(id(x))``, ``'k' + str(id(x))``,
    ``f"run-{id(x)}"``, ``hex(id(x))``, ``<Q object at 0x7f...>``. Each run
    of decimal digits of 7 to 18, and the parts of a longer one
    (`_windows`); each ``0x`` followed by 5 to 16 hexadecimal digits.
    Inside `reading_every_spelling`, also the parts of a shorter run
    (``'v2' + str(id(x))``) and the starts of a run of up to `_GLUED_HEX`
    after ``0x`` (``hex(id(x)) + 'ff'``).
    Restored, a value holding one spells the old process's address, where
    plain Python spells the new one's. Which of them is an object's address
    `_candidate_ids` decides.

    With *mapped_only*, the parts of many long runs are only those inside a
    mapping of this process now (`_mapped_windows`), for Python strings read
    as they are (`_Walk._read_texts`): a string is never rewritten, so one
    spelling the ``id()`` of an object alive at a later check spelled it
    while that object was alive, and its address was mapped at this read.
    Not for a column's buffers, whose read is kept by the buffers' address
    and is every number, whatever the mappings."""
    numbers: set[int] = set()
    runs = _DECIMAL_RUN.findall(text)
    if runs:
        lengths = list(map(len, runs))
        numbers.update(map(int, compress(runs, map(operator.ge, repeat(_LONGEST_NUMBER), lengths))))
        shortest = _SHORTEST_RUN if _EVERY_SPELLING[0] else _ADDRESS_DIGITS
        if max(lengths) > shortest:
            longer = list(compress(runs, map(operator.lt, repeat(shortest), lengths)))
            found = _mapped_windows(longer) if mapped_only and len(longer) >= _MANY_RUNS else None
            if found is None:
                for run in longer:
                    numbers.update(_windows(run))
            else:
                numbers.update(found)
    if "0x" in text or "0X" in text:
        for digits in _HEX_RUN.findall(text):
            if len(digits) <= _LONGEST_HEX:
                numbers.add(int(digits, 16))
            if _EVERY_SPELLING[0] and _SHORTEST_HEX < len(digits) <= _GLUED_HEX:
                # ``hex(id(x)) + 'ff'``: an address at the run's start.
                ends = range(_SHORTEST_HEX, min(len(digits) - 1, _LONGEST_HEX) + 1)
                numbers.update(int(digits[:end], 16) for end in ends)
    return numbers


#: Whether `_spelled_in` reads the spellings `reading_every_spelling` adds.
_EVERY_SPELLING = [False]


@contextlib.contextmanager
def reading_every_spelling() -> Iterator[None]:
    """Inside the block, `_spelled_in` reads an address glued to other
    digits too (``'v2' + str(id(x))``, ``hex(id(x)) + 'ff'``), and no read
    is taken from or kept in the memos of reads made without it.

    For a statement or call whose code reads ``id`` or ``repr``, the only
    code that spells an address: the windows of every run of 8 to 15
    digits made a column of a million ``'T' + 13 digits`` texts four times
    slower to read (0.8 s to 3.6 s). A value built from one is keyed by its
    lineage, which is keyed by that value."""
    before = _EVERY_SPELLING[0]
    _EVERY_SPELLING[0] = True
    try:
        yield
    finally:
        _EVERY_SPELLING[0] = before


def _windows(run: str) -> list[int]:
    """The numbers inside *run*, a run of more digits than the shortest
    address, that could be one: 7 to 15 digits, not starting with ``0``
    (``str()`` writes none), and a multiple of `_ALIGNMENT`, which its last
    three digits tell. ``f"{id(a)}{id(b)}"`` spells both, and ``'v2' +
    str(id(x))`` a 14-digit address in a run of 15."""
    found = []
    for end in range(_SHORTEST_RUN, len(run) + 1):
        if int(run[end - 3 : end]) % _ALIGNMENT:
            continue
        for start in range(max(0, end - _ADDRESS_DIGITS), end - _SHORTEST_RUN + 1):
            if run[start] != "0":
                found.append(int(run[start:end]))
    return found


#: From how many runs of digits on `_spelled_in` reads their parts by
#: `_mapped_windows`.
_MANY_RUNS = 256
#: The digits `_mapped_windows` searches a run for, at most, and how many
#: such starts it takes for the addresses of one length.
_START_DIGITS = 6
_STARTS = 24
#: Three digits that end a multiple of `_ALIGNMENT`, as a pattern.
_ALIGNED_END = (
    "(?:[02468](?:00|08|16|24|32|40|48|56|64|72|80|88|96)|[13579](?:04|12|20|28|36|44|52|60|68|76|84|92))"
    if _ALIGNMENT == 8
    else "[0-9](?:[02468][048]|[13579][26])"
)


def _window_starts(ranges: list[tuple[int, int]]) -> dict[int, list[str]]:
    """For each count of digits an address inside one of *ranges* can
    have, the leading digits (all of one length, and before its last
    three) every such number starts with: as many digits as leave at most
    `_STARTS` of them, down to the first digit alone."""
    starts: dict[int, list[str]] = {}
    for digits in range(_SHORTEST_RUN, _ADDRESS_DIGITS + 1):
        low, high = 10 ** (digits - 1), 10**digits
        spans = [(max(start, low), min(end, high) - 1) for start, end in ranges if start < high and end > low]
        if not spans:
            continue
        for lead in range(min(digits - 3, _START_DIGITS), 0, -1):
            unit = 10 ** (digits - lead)
            # The spans as leading digits, those that touch joined: a
            # thousand small mappings next to each other share a few.
            joined: list[list[int]] = []
            for first, last in sorted((first // unit, last // unit) for first, last in spans):
                if joined and first <= joined[-1][1] + 1:
                    joined[-1][1] = max(joined[-1][1], last)
                else:
                    joined.append([first, last])
            if sum(last - first + 1 for first, last in joined) <= _STARTS:
                starts[digits] = [str(n) for first, last in joined for n in range(first, last + 1)]
                break
    return starts


def _window_patterns(starts: dict[int, list[str]]) -> list[tuple[str, Any]]:
    """`_window_starts` as patterns: for each count of digits and first
    digit, that digit (which the search jumps to) and a pattern matching
    it where the rest of an aligned number of that many digits, starting
    with one of the leads, follows; the rest is its one group. A match
    takes the first digit only, so a number starting inside another is
    found as well."""
    patterns = []
    for digits, leads in starts.items():
        for first in sorted({lead[0] for lead in leads}):
            rests = "|".join(lead[1:] for lead in leads if lead[0] == first)
            between = digits - 3 - len(leads[0])
            patterns.append((first, re.compile(f"{first}(?=((?:{rests})[0-9]{{{between}}}{_ALIGNED_END}))")))
    return patterns


def _mapped_windows(runs: list[str]) -> set[int] | None:
    """What `_windows` finds in each of *runs* that lies inside a mapping
    of this process (`_read_mappings`, read now), the only ones
    `_candidate_ids` keeps, and at most a few it drops: found by searching
    the runs, joined, for the leading digits such an address has, so the
    numbers that are no address are never made. A list of 300,000 texts of
    18 digits spelled a million numbers, 6,000 of them inside a mapping,
    and took 5 s to read where the function took 40 ms. None where the
    mappings cannot be read: `_windows` reads the runs then."""
    if not sys.platform.startswith("linux"):
        return None
    ranges = _read_mappings()
    if not ranges:
        return None
    starts = _window_starts(ranges)
    # A line break between the runs: no window reaches over it.
    text = "\n".join(runs)
    found: set[int] = set()
    for first, pattern in _window_patterns(starts):
        found.update(map(int, map(first.__add__, pattern.findall(text))))
    return found


def _may_spell(text: str) -> bool:
    """Whether *text* could hold what `_spelled_in` reads, told by plain
    searches, far quicker than its patterns: most text has no run of seven
    digits and no ``0x``."""
    return "0x" in text or "0X" in text or "0" * _SHORTEST_RUN in text.translate(_DIGITS_AS_ZERO)


def _joined(texts: list[str]) -> Any:
    """*texts* joined by line breaks, at most `_TEXTS_AT_ONCE` of them and
    about `_CHARS_AT_ONCE` characters at once (a longer text alone)."""
    ends = list(accumulate(map(len, texts)))
    start = 0
    while start < len(texts):
        before = ends[start - 1] if start else 0
        stop = min(len(texts), start + _TEXTS_AT_ONCE, bisect.bisect_right(ends, before + _CHARS_AT_ONCE))
        stop = max(stop, start + 1)
        yield "\n".join(texts[start:stop])
        start = stop


def library_parts(value: Any) -> list[Any] | None:
    """What a library's object of a kind cash knows keeps the user's data
    in: a ``functools.partial``'s function and arguments, an
    ``argparse.Namespace``'s attributes, an ``array.array``'s numbers, a
    structured NumPy array's fields, a pandas frame's, series' or index's
    columns as NumPy arrays, and a pandas array's values
    (`_pandas_array_parts`). None for any other object: what it holds is
    not known."""
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
    argparse = loaded_module("argparse")
    if argparse is not None and isinstance(value, argparse.Namespace):
        state = own_dict(value)
        return list(state.values()) if state else []
    pandas = loaded_module("pandas")
    if pandas is not None and isinstance(value, (pandas.DataFrame, pandas.Series, pandas.Index)):
        return _pandas_parts(pandas, value)
    if pandas is not None and isinstance(value, pandas.api.extensions.ExtensionArray):
        return _pandas_array_parts(value)
    return None


def _pandas_object(kind: type) -> bool:
    """Whether *kind* is a pandas frame or series class (a user's subclass
    of one, read attribute by attribute, `_state_held`)."""
    pandas = loaded_module("pandas")
    return pandas is not None and issubclass(kind, (pandas.DataFrame, pandas.Series))


def _state_held(kind: type, state: dict[str, Any]) -> Iterable[Any]:
    """The values of *state*, an object of *kind*'s own ``__dict__``, that
    could keep an ``id()`` or a weak reference of the user's.

    Not pandas 2's ``_is_copy`` of a frame or series subclass (``df[df.a >
    1]`` on a ``GeoDataFrame``): a weak reference to the frame it was taken
    from, which only ``NDFrame._check_setitem_copy`` reads, to warn
    (``SettingWithCopyWarning``) on a later write to the slice; no value of
    either frame depends on it. Pickling drops it (``NDFrame.__getstate__``),
    as it does for a plain ``DataFrame``, whose ``_is_copy`` is never read
    here (`library_parts`). Kept, and so refused, while
    ``mode.chained_assignment`` is ``"raise"`` without copy-on-write: then a
    write to the slice raises ``SettingWithCopyError`` in plain Python and
    would not on a hit. pandas 3 has no ``_is_copy``. A series' ``_cacher``
    is kept: through it a write to the series changes its parent frame
    (``DataFrame._maybe_cache_changed``)."""
    held = state.get("_is_copy")
    if type(held) is not weakref.ReferenceType or not _pandas_object(kind) or _copy_check_raises():
        return state.values()
    return [value for key, value in state.items() if key != "_is_copy"]


def _copy_check_raises() -> bool:
    """Whether pandas' ``_check_setitem_copy`` raises: ``mode.chained_assignment``
    is ``"raise"`` and copy-on-write is off (``True`` or ``"warn"`` returns
    before the check). True when it cannot be read."""
    pandas = sys.modules["pandas"]
    try:
        if pandas.get_option("mode.copy_on_write") is not False:
            return False
        return pandas.get_option("mode.chained_assignment") == "raise"
    except Exception:  # noqa: BLE001 - an option this pandas does not have: assume it raises
        return True


def _pandas_array_parts(values: Any) -> list[Any]:
    """A pandas array's values that could keep an ``id()`` or a weak
    reference, as a column of it is read (`_pandas_column`): Arrow's text
    by `_arrow_text_numbers`, other objects as a NumPy array of them, and
    8-byte integers as one of those. ``pd.array(keys, dtype='string')`` on
    its own, a call's result cash keeps apart from the frame made of it,
    was read as nothing, and served with the old process's ``str(id(x))``."""
    spelled = _arrow_text_numbers(values)
    if spelled is not None:
        return spelled
    dtype = values.dtype
    kind = attribute_or(dtype, "kind")
    pandas = loaded_module("pandas")
    if kind == "O" and pandas is not None and isinstance(dtype, pandas.CategoricalDtype):
        return [dtype.categories]  # every value is one of them (`_pandas_column`)
    if kind == "O":
        return [values.to_numpy(dtype=object)]
    if kind in ("i", "u") and attribute_or(dtype, "itemsize") == 8:
        return [values.to_numpy(dtype="int64", na_value=0)]
    return []


def _pandas_parts(pandas: Any, value: Any) -> list[Any]:
    """A pandas object's labels and columns that could keep an ``id()`` or
    a weak reference: as NumPy arrays of 8-byte integers or of objects, a
    frame's read a dtype at a time (2D)."""
    if isinstance(value, pandas.RangeIndex):
        return []
    if isinstance(value, pandas.DataFrame):
        parts: list[Any] = [value.columns, value.index]
        # The frame's own arrays, and their dtypes: a column, or the
        # ``dtypes`` series, made for each took longer than reading it, for
        # a list of thousands of small frames.
        arrays = _column_arrays(value)
        by_dtype: dict[Any, list[int]] = {}
        for at, dtype in enumerate([array.dtype for array in arrays] if arrays else value.dtypes):
            by_dtype.setdefault(dtype, []).append(at)
        for dtype, columns in by_dtype.items():
            if attribute_or(dtype, "kind") == "O":
                # A column at a time: one a loop rewrote is read again, the
                # others are the very strings read before (`_FLAT_READS`);
                # Arrow's text straight from the frame's array.
                for at in columns:
                    spelled = _arrow_text_numbers(arrays[at]) if arrays else None
                    if spelled is None and arrays and _plain_array(arrays[at], dtype):
                        # A NumPy column of objects: the very array the
                        # column's ``to_numpy`` gives, without a column made
                        # for it (``iloc``), most of the read of a small frame.
                        spelled = arrays[at]
                    parts.append(spelled if spelled is not None else _pandas_column(value.iloc[:, at], dtype))
                continue
            if not _may_keep_an_id(dtype):
                continue  # not taken out of the frame only to be passed over
            if arrays and all(_plain_array(arrays[at], dtype) for at in columns):
                # The frame's own NumPy arrays of 8-byte integers, as they
                # are: the same numbers ``iloc`` would copy out of them.
                parts.extend(arrays[at] for at in columns)
                continue
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


def _plain_array(array: Any, dtype: Any) -> bool:
    """Whether *array*, a frame's own array for one column, is a 1-D NumPy
    array of *dtype*: what reading the column through ``iloc`` would give
    (as `_pandas_column` reads it) holds the same values."""
    return _numpy_array(type(array)) and array.ndim == 1 and array.dtype == dtype


def _may_keep_an_id(dtype: Any) -> bool:
    """Could values of *dtype* keep an ``id()`` or a weak reference, as
    `_pandas_column` reads them: objects, or 8-byte integers?"""
    kind = attribute_or(dtype, "kind")
    return kind == "O" or (kind in ("i", "u") and attribute_or(dtype, "itemsize") == 8)


def _pandas_column(column: Any, dtype: Any) -> Any:
    """The values of *column* (a series, an index, or a frame's columns of
    one *dtype*) as a NumPy array when they could keep an ``id()`` or a
    weak reference: 8-byte integers, or objects. None otherwise."""
    kind = attribute_or(dtype, "kind")
    pandas = loaded_module("pandas")
    if kind == "O" and pandas is not None and isinstance(dtype, pandas.CategoricalDtype):
        # Every value is one of its categories, the very object: read
        # those, not a million values made objects one by one.
        categories = dtype.categories
        return _pandas_column(categories, categories.dtype)
    if kind == "O":
        spelled = _arrow_text_spelling_numbers(column)
        if spelled is not None:
            return spelled
        values = attribute_or(column, "_values", None)
        if (
            attribute_or(column, "ndim", 0) == 1
            and has_attribute(values, "to_numpy")
            and not has_attribute(values, "flags")
        ):
            # An extension array (strings held as Python objects): read
            # from it, not through ``column.to_numpy()``, which goes by
            # ``.array`` -- a handle the frame memo records, so the frame
            # was hashed afresh after every statement (and an index's
            # cache kept the array as one more reference).
            return values.to_numpy(dtype=object)
        return column.to_numpy(dtype=object)
    if kind in ("i", "u") and attribute_or(dtype, "itemsize") == 8:
        numpy = loaded_module("numpy")
        if numpy is not None and isinstance(dtype, numpy.dtype):
            return column.to_numpy()
        return column.to_numpy(dtype="int64", na_value=0)  # a nullable Int64 column
    return None


def _column_arrays(frame: Any) -> list[Any]:
    """The arrays *frame* keeps its columns in, by position, read without
    making a column of each (pandas' own ``_iter_column_arrays``, for
    reading only); empty where this pandas has no such reader."""
    try:
        reader = type(frame)._iter_column_arrays
    except AttributeError:
        return []
    try:
        arrays = list(reader(frame))
    except Exception:  # noqa: BLE001 - read a column at a time instead
        return []
    return arrays if len(arrays) == frame.shape[1] else []


def _arrow_text_spelling_numbers(column: Any) -> list[Any] | None:
    """Of a column of Arrow-backed text (pandas' default ``str``), the only
    thing that could matter: the numbers its values spell as `_spelled_in`
    reads them, the values that could spell one picked at C speed -- a value
    that is all digits parsed by Arrow, into one NumPy array of 8-byte
    integers (checked as an integer column is), the others' numbers as
    `_spelled_parts` gives them. Converting the column to Python strings
    made new ones each time, a million for a wide frame, after every
    statement of a loop writing it. Text holds no object. None for any
    other column, or where Arrow is not at hand."""
    # ``_values``, not ``.array``: the frame memo records every ``.array``
    # handed out as a way to write the column (`watch_array_handles`), so
    # reading it here made the memo hash the frame afresh after each pass.
    return _arrow_text_numbers(attribute_or(column, "_values", None))


def _arrow_text_numbers(values: Any) -> list[Any] | None:
    """`_arrow_text_spelling_numbers` of a column's own array (*values*):
    None unless it is Arrow-backed text."""
    chunked = attribute_or(values, _pandas_compat.ARROW_SLOT, None)
    pyarrow = loaded_module("pyarrow")
    compute = loaded_module("pyarrow.compute")
    if chunked is None or pyarrow is None or compute is None:
        return None
    # Read as every spelling is (`reading_every_spelling`): not the memo's.
    every = _EVERY_SPELLING[0]
    known = None if every else _arrow_read_of(chunked)
    if known is not None:
        return known
    try:
        if not (pyarrow.types.is_string(chunked.type) or pyarrow.types.is_large_string(chunked.type)):
            return None
        if every:
            parts: list[Any] = []
            for piece in _joined_runs(pyarrow, chunked):
                _spelled_numbers(pyarrow, compute, piece, parts)
            return parts
        signature = _arrow_signature(chunked)
        read = _ARROW_READS.get(signature)
        if read is not None and read.alive():
            read.add(chunked)
            return read.parts
        if len(chunked) <= _FEW_TEXTS:
            few = _few_texts_spelling_numbers(chunked.to_pylist())
            _ArrowRead(signature, few).add(chunked)
            return few
        # Only a value holding a run of 7 digits or ``0x`` and 5 hexadecimal
        # digits can spell a number: those that could are picked from the
        # text's bytes (`_could_spell`), and only they matched. A merged
        # frame's text column came in thousands of chunks, and two patterns
        # over all of it took 6 s after one `merge` (p18).
        parts = []
        for piece in _joined_runs(pyarrow, chunked):
            _spelled_numbers(pyarrow, compute, piece, parts)
        _ArrowRead(signature, parts).add(chunked)
        return parts
    except Exception:  # noqa: BLE001 - read as before, through NumPy
        return None


def _arrow_read_of(chunked: Any) -> list[Any] | None:
    """What `_arrow_text_numbers` found in this very Arrow array before,
    while it is alive; None when it was not read, or not this one."""
    key = id(chunked)
    signature = _ARROW_SIGNED.get(key)
    if signature is None:
        return None
    read = _ARROW_READS.get(signature)
    if read is None:
        return None
    ref = read.refs.get(key)
    return read.parts if ref is not None and ref() is chunked else None


def _arrow_signature(chunked: Any) -> tuple:
    """What names the text an Arrow array (*chunked*, a ChunkedArray)
    holds while it is alive: its type, and each chunk's offset, length and
    the address and size of each of its buffers -- as the frame digests
    name an Arrow column's memory (`column_memory_key`). Arrow never writes
    a buffer it handed out, and pandas wraps the same buffers anew in each
    frame sharing a column (``labs = clean(labs)``, a merge's untouched
    side). Past `_SIGNED_CHUNKS` chunks the array names only itself: their
    buffers took longer to list than the read they spare."""
    if chunked.num_chunks > _SIGNED_CHUNKS:
        return ("array", id(chunked))
    return (
        str(chunked.type),
        tuple(
            (
                chunk.offset,
                len(chunk),
                tuple((buf.address, buf.size) if buf is not None else None for buf in chunk.buffers()),
            )
            for chunk in chunked.iterchunks()
        ),
    )


class _ArrowRead:
    """An entry of `_ARROW_READS`: the numbers (*parts*) the text named by
    *signature* spells, and a weak reference to each Arrow array read with
    that signature (by its id). The entry lasts while one of them is alive,
    which keeps the buffers the signature names from being freed and their
    addresses used again; it holds no object of its own."""

    __slots__ = ("numbers", "parts", "refs", "signature")

    def __init__(self, signature: tuple, parts: list[Any]) -> None:
        self.signature = signature
        self.parts = parts
        self.numbers = sum(1 if type(part) is int else len(part) for part in parts)
        self.refs: dict[int, Any] = {}
        old = _ARROW_READS.get(signature)
        _ARROW_READS[signature] = self
        if old is not None:
            old.drop()

    def alive(self) -> bool:
        return any(ref() is not None for ref in list(self.refs.values()))

    def add(self, chunked: Any) -> None:
        key = id(chunked)
        if key in self.refs:
            return
        signature = self.signature
        # The two tables are bound here, not looked up when the callback
        # runs: an array freed while the interpreter shuts down calls it
        # after this module's globals were set to None.
        signed, reads = _ARROW_SIGNED, _ARROW_READS

        def gone(_ref: Any) -> None:
            # Runs as the array is freed, before its id can be used again.
            if signed.get(key) == signature:
                signed.pop(key, None)
            if reads.get(signature) is self:
                self.refs.pop(key, None)
                if not self.refs:
                    reads.pop(signature, None)

        self.refs[key] = _weak.ref(chunked, gone)
        _ARROW_SIGNED[key] = signature

    def drop(self) -> None:
        """Out of `_ARROW_READS`, with the arrays it was found by."""
        for key in list(self.refs):
            if _ARROW_SIGNED.get(key) == self.signature:
                _ARROW_SIGNED.pop(key, None)
        self.refs = {}


#: A column of chunks this many rows long on average, or shorter, is read
#: by `_joined_runs` a run of chunks at a time.
_SMALL_CHUNK_ROWS = 1024
#: The rows of one run.
_RUN_ROWS = 1 << 16


def _joined_runs(pyarrow: Any, chunked: Any) -> list[Any]:
    """*chunked* as runs of its chunks joined into one Array each, when
    its chunks are many and small; else *chunked* itself, alone.

    Each of Arrow's kernels costs a few microseconds per chunk: a text
    column of 6,000 chunks of five rows, which ``df = pd.concat([df,
    new])`` in a loop grows, took 28 ms to read, against 6 ms for joining
    its chunks and reading the one array. A run ends once it reaches
    `_RUN_ROWS` rows, so about that much is copied at once; a run Arrow
    cannot join (text past 2 GiB) fails the read, which then goes through
    NumPy as before."""
    chunks = chunked.num_chunks
    if chunks < 2 * _FEW_TEXTS or len(chunked) > chunks * _SMALL_CHUNK_ROWS:
        return [chunked]
    runs: list[Any] = []
    run: list[Any] = []
    rows = 0
    for chunk in chunked.iterchunks():
        run.append(chunk)
        rows += len(chunk)
        if rows >= _RUN_ROWS:
            runs.append(pyarrow.concat_arrays(run))
            run, rows = [], 0
    if run:
        runs.append(pyarrow.concat_arrays(run))
    return runs


def _spelled_numbers(pyarrow: Any, compute: Any, chunked: Any, parts: list[Any]) -> None:
    """Append to *parts* the numbers the text of *chunked* (an Array or a
    ChunkedArray) spells, as `_arrow_text_numbers` reads them: the values
    that are 7 to 15 digits, cast by Arrow, then those of the others that
    hold a spelling (`_SPELLING_TEXT`), read by `_spelled_in`."""
    narrowed = _could_spell(pyarrow, chunked)
    if narrowed is not None:
        if not len(narrowed):
            return
        chunked = narrowed
    length = compute.binary_length(chunked)
    # Read whole, at C speed; inside `reading_every_spelling` only the
    # values of the shortest run, the rest read for the parts of their run.
    longest = _SHORTEST_RUN if _EVERY_SPELLING[0] else _ADDRESS_DIGITS
    whole = compute.and_(
        compute.and_(compute.greater_equal(length, _SHORTEST_RUN), compute.less_equal(length, longest)),
        compute.ascii_is_decimal(chunked),
    )
    decimal = compute.filter(chunked, whole)
    if len(decimal):
        parts.append(compute.cast(decimal, pyarrow.int64()).to_numpy(zero_copy_only=False))
    rest = compute.filter(chunked, compute.invert(whole))
    if len(rest):
        rest = compute.filter(
            rest, compute.match_substring_regex(rest, _EVERY_SPELLING_TEXT if _EVERY_SPELLING[0] else _SPELLING_TEXT)
        )
    if len(rest):
        numbers: set[int] = set()
        texts = [text for text in rest.to_pylist() if text is not None]
        for text in _joined(texts):
            numbers |= _spelled_in(text)
        parts.extend(_spelled_parts(numbers))


def _spelled_parts(numbers: set[int]) -> list[Any]:
    """*numbers* as `_arrow_text_numbers` hands them on: those an 8-byte
    integer holds as one NumPy array, in order, then any larger as ints."""
    import numpy

    small = sorted(n for n in numbers if n < 1 << 63)
    return [numpy.array(small, dtype=numpy.int64), *sorted(n for n in numbers if n >= 1 << 63)]


def _spelling_bytes(numpy: Any, found: Any) -> Any:
    """For each byte of *found* (a NumPy array of them), whether it can be
    part of a spelling `_spelled_in` reads: a digit, ``x`` or a hexadecimal
    letter, in either case. By compares, each a pass at C speed: a table
    looked up took four times as long over p18's 5.6 million values."""
    lower = found | numpy.uint8(0x20)
    fits = found - numpy.uint8(0x30)
    fits = fits < 10
    letter = lower - numpy.uint8(0x61)
    fits |= letter < 6
    fits |= lower == 0x78
    return fits


#: The fewest rows a chunk of Arrow text averages for `_could_spell` to
#: pick its candidates: below that, picking chunk by chunk costs more than
#: Arrow reading all of it (even at 512 rows a chunk, measured).
_NARROWED_CHUNK_ROWS = 512


def _could_spell(pyarrow: Any, chunked: Any) -> Any:
    """The values of *chunked* (Arrow text: an Array or a ChunkedArray)
    that could spell a number as `_spelled_numbers` reads them, in order,
    as one ChunkedArray; a missing value among them stays missing, as
    Arrow's patterns drop it.

    A spelling is at least 7 bytes of digits, ``x`` and hexadecimal letters
    (`_spelling_bytes`): 7 digits, or ``0x`` and 5 hexadecimal digits. So a
    value shorter than 7 bytes spells none; in a value of 7 to 13 bytes
    every spelling covers its bytes ``length - 7`` to ``6`` alike, each
    looked at, the middle one first, for the values still left; and in a longer
    one every spelling covers one of its bytes 6, 13, 20 and so on. The
    rest are left out: they spell nothing.

    Read from the arrays' offsets and bytes through NumPy, without a copy
    of the text: Arrow's length, mask and filter over every value took
    most of the read, 0.13 s for a column of 5.6 million ids (``S011-00001``,
    the ``-`` ruling each out) after each of p18's statements making a new
    frame of lab rows. None when a chunk's buffers are not laid out as
    Arrow's ``string`` or ``large_string`` lay them out: Arrow reads all of
    *chunked*, as before.

    None too when *chunked*'s chunks average fewer than
    `_NARROWED_CHUNK_ROWS` rows: picking costs some 15 microseconds a
    chunk, and 10,000 chunks of one row took 0.19 s against Arrow's 0.03 s
    for all of them. A chunk at least two thirds of whose values are
    candidates is kept whole rather than copied (2 million digit strings):
    its other values spell nothing, which Arrow's patterns find as well,
    so the numbers found are the same."""
    numpy = loaded_module("numpy")
    if numpy is None:
        return None
    if isinstance(chunked, pyarrow.ChunkedArray):
        if len(chunked) < chunked.num_chunks * _NARROWED_CHUNK_ROWS:
            return None
        chunks = chunked.chunks
    else:
        chunks = [chunked]
    kind = chunked.type
    if pyarrow.types.is_string(kind):
        offset_type = numpy.int32
    elif pyarrow.types.is_large_string(kind):
        offset_type = numpy.int64
    else:
        return None
    span = _SHORTEST_RUN
    width = numpy.dtype(offset_type).itemsize
    taken: list[Any] = []
    for chunk in chunks:
        rows = len(chunk)
        if not rows:
            continue
        buffers = chunk.buffers()
        if len(buffers) != 3 or buffers[1] is None:
            return None
        offsets = numpy.frombuffer(buffers[1], dtype=offset_type, count=rows + 1, offset=chunk.offset * width)
        if offsets[-1] == offsets[0]:
            continue  # all empty (or missing): nothing fits
        if buffers[2] is None:
            return None
        text = numpy.frombuffer(buffers[2], dtype=numpy.uint8)
        starts, ends = offsets[:-1], offsets[1:]
        lengths = ends - starts
        if lengths.max() < span:
            # No value is long enough to spell one: a pass over the lengths
            # instead of the five below and a read of the text (p18's codes
            # ``V1``, ``ALT``, ``U/L``: 40-55 ms each per 5.6 million rows).
            continue
        kept: list[Any] = []
        longer = numpy.flatnonzero(lengths >= 2 * span)
        # 7 to 13 bytes: the bytes every spelling covers, `count` of them
        # from `low`. The middle one first, for every value at once (a
        # separator, as in ``S011-00001``, rules most out); the others for
        # the values it leaves. The lengths become the middles in place:
        # each array of 5.6 million offsets made costs its pages afresh.
        fits = lengths >= span
        fits &= lengths < 2 * span
        middle = lengths
        middle -= 1
        middle >>= 1
        middle += starts  # each value's byte ``(length - 1) // 2``
        fits &= _spelling_bytes(numpy, text.take(middle, mode="clip"))
        del middle, lengths
        at = numpy.flatnonzero(fits)
        low = ends[at].astype(numpy.int64) - span
        count = 2 * span - (ends[at] - starts[at])
        for step in range(span):
            more = count > step
            if not more.all():
                kept.append(at[~more])
                at, low, count = at[more], low[more], count[more]
            if not at.size:
                break
            fits = _spelling_bytes(numpy, text[low + step])
            at, low, count = at[fits], low[fits], count[fits]
        kept.append(at)
        # Longer: a byte every 7, from the 7th.
        at = longer
        if at.size:
            count = ((ends[at] - starts[at]) // span).astype(numpy.int64)
            first = numpy.cumsum(count) - count
            spots = numpy.repeat(starts[at] + (span - 1) - span * first, count) + span * numpy.arange(int(count.sum()))
            spots = spots.astype(numpy.int64, copy=False)
            kept.append(at[numpy.logical_or.reduceat(_spelling_bytes(numpy, text[spots]), first)])
        at = numpy.sort(numpy.concatenate(kept))
        if 3 * at.size >= 2 * rows:
            taken.append(chunk)  # most of it: Arrow's patterns pick them
        elif at.size:
            taken.append(chunk.take(pyarrow.array(at, type=pyarrow.int64())))
    return pyarrow.chunked_array(taken, type=kind)


#: What `_arrow_text_spelling_numbers` found, by what names the text read
#: (`_arrow_signature`): Arrow's arrays never change, a loop writing one
#: column of a wide frame leaves the others the same arrays, and a frame
#: carried from cell to cell (``labs_raw`` to ``labs`` to ``chg``) keeps
#: its text columns' buffers, read once instead of after each statement
#: and cell that outputs them (p18: 2.5 s of reading, 1.1 s now). An entry goes with the last
#: array it was read from (`_ArrowRead`); one holding more than
#: `_KEPT_ARROW_NUMBERS` numbers goes when its cell ends (`end_cell_reads`).
_ARROW_READS: dict[tuple, _ArrowRead] = {}
#: The signature of each Arrow array `_ARROW_READS` was read from, by its
#: id, while it is alive.
_ARROW_SIGNED: dict[int, tuple] = {}
#: Most chunks an Arrow array may have to be named by its buffers.
_SIGNED_CHUNKS = 64
#: Most numbers an entry of `_ARROW_READS` may hold to be kept past its
#: cell: a column of numbers kept as text keeps one per value.
_KEPT_ARROW_NUMBERS = 1 << 16
#: What a value `_spelled_in` reads a number in holds, for Arrow's
#: patterns (RE2): a run of 7 digits, or ``0x`` and 5 to 16 hexadecimal
#: digits ending there. Missed by none of its spellings, and matching no
#: column of longer hashes (``0x`` and 64 digits), read in Python for nothing.
_SPELLING_TEXT = r"[0-9]{7}|0[xX][0-9a-fA-F]{5,16}(?:[^0-9a-fA-F]|$)"
#: The same inside `reading_every_spelling`: up to `_GLUED_HEX` digits.
_EVERY_SPELLING_TEXT = r"[0-9]{7}|0[xX][0-9a-fA-F]{5,24}(?:[^0-9a-fA-F]|$)"
#: An Arrow text array of at most this many values is read in Python: each
#: Arrow call costs tens of microseconds however short the array, and a
#: list of thousands of small frames made four per column.
_FEW_TEXTS = 64


def _few_texts_spelling_numbers(texts: list[Any]) -> list[Any]:
    """What `_arrow_text_numbers` finds in a few *texts* (None for a
    missing value): the numbers they spell (`_spelled_in`), searched
    joined, as `_spelled_parts` hands them on; nothing when they spell
    none."""
    joined = "\n".join(text for text in texts if text is not None)
    numbers = _spelled_in(joined) if _may_spell(joined) else None
    return _spelled_parts(numbers) if numbers else []


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
    inside = _inside(numbers, ranges) if ranges is not None else set()
    if len(inside) == len(numbers):
        return inside
    # One outside the mappings read before: a mapping made since may hold
    # it. One the process has since unmapped is kept, which only reruns.
    ranges = _MAPPINGS[0] = _read_mappings()
    if ranges is None:
        return numbers
    return _inside(numbers, ranges)


def _inside(numbers: set[int], ranges: list[tuple[int, int]]) -> set[int]:
    """Those of *numbers* inside one of *ranges* (sorted by start, apart):
    found by a binary search each, not by asking every range (a frame of
    digits spells a million numbers, and a process maps a thousand ranges)."""
    starts = [start for start, _ in ranges]
    return {n for n in numbers if (at := bisect.bisect_right(starts, n) - 1) >= 0 and n < ranges[at][1]}


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
    return sorted(ranges)


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

    finalized = _finalized(targets)
    if finalized:
        return IdentityHeld(
            f"a weakref.finalize ({finalized}) is bound to '{name}' or an object in it; restored, '{name}' "
            "would be another object, so the statement re-runs every time",
            name,
            "a weak reference",
            what=(
                f"A weakref.finalize ({finalized}) watches '{name}', or an object in it, whichever variable "
                f"holds it, or none. Restored from the cache, '{name}' would be a copy: the finalizer would "
                "go on watching the object that was there, so this statement runs every time instead."
            ),
        )
    # Every live weak reference to an object is in its own list
    # (``weakref.getweakrefs``): one only cash made there means no variable
    # of the notebook can hold one, and the other variables are not walked
    # (a list of 200,000 lists of words cost each statement 0.4 s).
    users = [ref for target in targets for ref in weakref.getweakrefs(target) if not is_cash_ref(ref)]
    if not users:
        return None
    wanted = {id(target) for target in targets}
    # A proxy finds its object as a reference does, but has no call to ask.
    wanted_refs = {id(ref) for ref in users}
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
    if cut:
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


def _finalize_info(item: Any) -> Any:
    """The registry entry of the ``weakref.finalize`` *item*, while alive:
    its ``weakref``, ``func``, ``args`` and ``kwargs``."""
    return weakref.finalize._registry.get(item)  # type: ignore[attr-defined]


def _finalized(targets: list[Any]) -> str | None:
    """The callback of a ``weakref.finalize`` of the user's (not cash's,
    `cash._weak.is_cash_ref`) bound to one of *targets*, as text; None when
    there is none. Found from the object: the finalizer is kept alive by
    ``weakref.finalize``'s own registry, so no variable need hold it."""
    for target in targets:
        for ref in weakref.getweakrefs(target):
            if not isinstance(ref, weakref.ReferenceType):
                continue
            callback = callback_of(ref)
            if isinstance(callback, weakref.finalize) and not is_cash_ref(ref):
                info = _finalize_info(callback)
                if info is not None:
                    return _callback_name(info.func)
    return None


def _callback_name(func: Any) -> str:
    """*func* as a finalizer's refusal names it."""
    name = attribute_or(func, "__qualname__", None)
    return f"{name}()" if isinstance(name, str) else class_name(type(func))


def _weakly_found_by_others(targets: list[Any]) -> bool:
    """Whether any of *targets* has a weak reference to it that is not
    cash's own (`cash._weak.is_cash_ref`): cash's memos keep weak
    references to the user's objects, and whether one is there at the
    moment must not decide whether the statement is stored."""
    return any(not is_cash_ref(ref) for target in targets for ref in weakref.getweakrefs(target))


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


#: The bits an ``id()`` leaves clear, both powers of two: those at and above
#: `_HIGHEST_ADDRESS` (the sign's too) and those below `_ALIGNMENT`.
_NO_ADDRESS_BITS = -_HIGHEST_ADDRESS | (_ALIGNMENT - 1)


def _addresses_in(array: Any) -> Any:
    """The numbers of an integer NumPy *array* that could be an ``id()``,
    as an ``int64`` array, read at C speed.

    The share check has `share_check_budget` for this: a column of 8
    million numbers took 0.1 s of it read with ``%`` and a boolean index,
    the floor of a budget the 60 ms statement that made it is given, so
    whether it was stored came down to a few milliseconds. One mask of the
    bits an address leaves clear and `numpy.compress` take ~0.04 s."""
    numpy = sys.modules["numpy"]
    # An unsigned number past the sign bit reads negative: no address.
    flat = array.ravel().astype(numpy.int64, copy=False)
    if flat.size >= _BOUNDS_FIRST and not _mapped_between(int(flat.min()), int(flat.max())):
        # Two passes, not five: ``np.arange(10_000_000)`` lies below every
        # mapping an object can be in.
        return flat[:0]
    keep = numpy.bitwise_and(flat, _NO_ADDRESS_BITS) == 0
    numpy.logical_and(keep, flat >= _LOWEST_ADDRESS, out=keep)
    return numpy.compress(keep, flat)


#: Fewest numbers of an array `_addresses_in` first tells by its lowest and
#: highest whether any could be an address at all.
_BOUNDS_FIRST = 100_000


def _mapped_between(low: int, high: int) -> bool:
    """Whether a writable mapping of this process where an object can lie
    (not the executable's own image when it is at a fixed address:
    `_fixed_image`) overlaps the numbers from *low* to *high*. True where
    the mappings cannot be read (not Linux), unless the numbers lie wholly
    below `_LOWEST_ADDRESS` or from `_HIGHEST_ADDRESS` up, where no process
    maps an object."""
    if high < _LOWEST_ADDRESS or low >= _HIGHEST_ADDRESS:
        return False
    ranges = _read_mappings()
    if ranges is None:
        return True
    fixed = _fixed_image()
    return any(
        start <= high and low < end and not (fixed is not None and fixed[0] <= start and end <= fixed[1])
        for start, end in ranges
    )


def _any_held(in_arrays: list[Any], held: _Walk) -> bool:
    """Whether a number of *in_arrays* is the ``id()`` of an object *held*
    met, at C speed: only the numbers between the lowest and the highest
    of those ids are looked up."""
    numpy = sys.modules["numpy"]
    ids = numpy.concatenate(
        [
            numpy.fromiter(held.seen, dtype=numpy.int64, count=len(held.seen)),
            numpy.array(held.leaf_ids, dtype=numpy.int64),
            *(held.id_arrays or ()),
        ]
    )
    fixed = _fixed_image()
    if fixed is not None and ids.size:
        # An object compiled into the executable keeps its id() across a
        # restart.
        ids = ids[(ids < fixed[0]) | (ids >= fixed[1])]
    if not ids.size:
        return False
    low, high = ids.min(), ids.max()
    for found in in_arrays:
        keep = found >= low
        numpy.logical_and(keep, found <= high, out=keep)
        found = numpy.compress(keep, found)
        if found.size and numpy.isin(found, ids).any():
            return True
    return False


def _code_parts(item: Any) -> list[Any]:
    """What a class or a function of the user's code holds: a class's own
    attributes but Python's ``__dunder__`` ones, a function's closure cells
    and defaults, a bound method's object. Nothing for a library's."""
    if issubclass(type(item), types.MethodType):
        return [item.__self__] if _users_code(item.__func__) else []
    if not _users_code(item):
        return []
    if isinstance(item, type):
        return [v for k, v in class_dict(item).items() if not (k.startswith("__") and k.endswith("__"))]
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


def _type_tells(kind: type) -> bool:
    """Whether `library_parts` answers alike for every object of *kind*:
    it asks ``isinstance``, which an object can answer apart from its type
    through a ``__class__`` of its own (a proxy's); a NumPy array's answer
    is its dtype's."""
    return not _numpy_array(kind) and not any("__class__" in vars(base) for base in class_mro(kind)[:-1])


def _numpy_array(kind: type) -> bool:
    return class_name(kind) == "ndarray" and attribute_or(kind, "__module__") == "numpy"


def _nodes_read(items: list[Any]) -> set[int]:
    """The ids of the containers `_Walk._read_flat` keeps of *items* for
    `_Walk.seen_all`, read again: what a memo's answer for *items* leaves
    out."""
    walk = _Walk({})
    walk._read_flat_now(items)
    return walk.seen_all()


#: Fewest objects in a level for `_Walk._open_in_bulk` to read them a class
#: at a time.
_BULK_MIN = 64
#: Fewer items than a dict `_Walk._open_in_bulk` reads with the others.
_BULK_DICT = 32
#: The builtin containers `builtin_base` tells; a subclass of one is
#: opened one by one.
_BUILTIN_CONTAINERS = (dict, list, tuple, set, frozenset, collections.deque)
_NEW_IN = attribute_or(_native.kernels, "new_in", None)


_SPLIT_LEVEL = attribute_or(_native.kernels, "split_level", None)


def _by_type(level: list[Any]) -> list[tuple[type, list[Any]]]:
    """*level*'s objects grouped by their exact type, each group in order."""
    if _SPLIT_LEVEL is not None:
        groups = _SPLIT_LEVEL(level, set(), set())
        if groups is not NotImplemented:
            return groups
    found: dict[type, list[Any]] = {}
    for item in level:
        found.setdefault(type(item), []).append(item)
    return list(found.items())


def _is_leaf(kind: type) -> bool:
    """Whether `_Walk._open` reads an object of *kind* as a leaf."""
    return kind is int or kind is str or kind in _LEAVES


def _new_objects(group: list[Any], seen: set[int]) -> list[Any]:
    """The objects of *group* not in *seen* (by id), each once, in the order
    met; *seen* gets their ids."""
    if _NEW_IN is not None:
        return _NEW_IN(group, False, seen)
    found = dict(zip(map(id, group), group))
    for key in seen.intersection(found):
        del found[key]
    seen.update(found)
    return list(found.values())


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
        #: Per class, `_in_bulk`'s answer.
        self.bulk_kinds: dict[type, bool] = {}
        #: Per class, `_plain_record`'s answer.
        self.record_kinds: dict[type, bool] = {}
        #: What the value's own walk reads as a leaf (`_value_kinds`), and
        #: which of those are numpy's integers, read as the ``int`` each
        #: is. Looking through the notebook (*quiet*), a library's scalar
        #: is an object like any other: a registry may hold its ``id()``.
        self.leaves, self.int_leaves = (_LEAVES, frozenset()) if quiet else _value_kinds()
        self.targets = targets
        self.keep_refs = keep_refs
        self.levels = levels
        self.seen: set[int] = set() if seen is None else seen
        #: Containers `_read_nested` read, not yet in *seen* (`seen_all`),
        #: or a function giving them.
        self._unseen: list[Any] = []
        self.budget = _MAX_SEEN
        #: What `_more` added to *budget*.
        self.granted = 0
        #: *budget* as the walk started, `identity_held_reason`'s extra included.
        self.allowance = _MAX_SEEN
        #: The ids queued for the next level that will count against the
        #: budget, while `run` counts ahead (`_count_ahead`); None otherwise.
        self._queued: set[int] | None = None
        self.refs: list[Any] = []
        self.numbers: set[int] = set()
        self.arrays: list[Any] = []
        self.unread: list[Any] = []
        #: The library's objects met and not looked inside (`library_parts`).
        self.closed: list[Any] = []
        #: The kinds of `closed`'s objects, each told closed by its type.
        self.closed_kinds: set[type] = set()
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
        #: The ids of the strings, numbers and other leaves a *quiet* walk
        #: met, kept only when an array's numbers are looked for
        #: (`id_arrays`); otherwise each is only looked up in `candidates`.
        self.leaf_ids: list[int] = []

    def run(self, roots: list[Any]) -> None:
        level = list(roots)
        levels = self.levels
        # The value's own walk counts what it queued for the next level
        # (`_count_ahead`): each of those objects costs the budget once when
        # opened, so once they outnumber what is left of it, the walk is cut
        # all the same, and stops here. A list of 87,000 small frames was
        # opened frame by frame, every text column read, to pass the budget
        # on their parts a level down; counted ahead, it stops a third of
        # the way through the frames with the same answer.
        ahead = levels is None and not self.quiet
        while level:
            if levels is not None:
                if levels <= 0:
                    self.cut = any(self._opens(item) for item in level)
                    return
                levels -= 1
            into: list[Any] = []
            queued: set[int] = set()
            self._queued = queued if ahead else None
            if len(level) >= _BULK_MIN:
                level = self._open_in_bulk(level, into)
                if self.cut or (ahead and into and self._count_ahead(into, 0, queued)):
                    self.cut = True
                    return
            for item in level:
                start = len(into)
                self._open(item, into)
                if self.cut:
                    # Past the budget the answer is known: the rest of the
                    # level was only counted (a list of 1.3 million
                    # Timedeltas was, six times over the budget).
                    return
                if ahead and len(into) > start and self._count_ahead(into, start, queued):
                    self.cut = True
                    return
            level = into
        self._queued = None

    def _open_in_bulk(self, level: list[Any], into: list[Any]) -> list[Any]:
        """Open the objects of *level* that `_open` would read as a few items
        and no more, a class at a time: exact dicts (of fewer than
        `_BULK_DICT` items), lists and tuples, and
        the objects of a class of the user's or a ``SimpleNamespace`` (each
        with fewer than `_SMALL_FLAT` items, so none is one `_open` would
        read at C speed or as records). What each holds goes to *into*, as
        `_open` puts it there; the rest of *level* is returned, to be opened
        one by one. A trie of 285,000 slotted nodes and their child dicts
        took 1 s opened an object at a time, past its budget."""
        groups = _by_type(level)
        if len(groups) == 1 and not self._in_bulk(groups[0][0]) and not _is_leaf(groups[0][0]):
            return level
        rest: list[Any] = []
        for kind, group in groups:
            if _is_leaf(kind):
                self._leaves_in_bulk(group, kind)
                continue
            if not self._in_bulk(kind):
                rest.extend(group)
                continue
            if kind is dict:
                # A dict of up to `_BULK_DICT` items too: `_open` would try
                # reading it at C speed (`_read_flat`), which a dict of
                # objects fails; its leaves are read a class at a time here.
                small = list(map(operator.gt, repeat(_BULK_DICT), map(len, group)))
            elif kind in _NESTED:
                small = list(map(operator.gt, repeat(_SMALL_FLAT), map(len, group)))
            else:
                small = None
                readers = slot_readers(kind)
                dicts = own_dicts(group, kind)
                if max(map(len, dicts), default=0) + len(readers) >= _SMALL_FLAT:
                    rest.extend(group)
                    continue
            if small is not None and not all(small):
                rest.extend(compress(group, map(operator.not_, small)))
                group = list(compress(group, small))
            new = _new_objects(group, self.seen)
            if not new:
                continue
            if self._queued:
                self._queued.difference_update(map(id, new))
            self.budget -= len(new)
            if self.budget < 0 and not self._more(-self.budget):
                self.cut = True
                return []
            if self.targets is not None and self._notebook_own(kind):
                self.targets.extend(compress(new, map(weakref.getweakrefcount, new)))
            if kind is dict:
                into.extend(chain.from_iterable(map(dict.keys, new)))
                into.extend(chain.from_iterable(map(dict.values, new)))
            elif kind in _NESTED:
                into.extend(chain.from_iterable(new))
            else:
                into.extend(chain.from_iterable(map(dict.values, own_dicts(new, kind))))
                for reader in readers:
                    try:
                        into.extend(list(map(reader.__get__, new)))
                    except AttributeError:  # an empty slot somewhere: read one by one
                        for item in new:
                            try:
                                into.append(reader.__get__(item, kind))
                            except AttributeError:
                                continue
        return rest

    def _leaves_in_bulk(self, group: list[Any], kind: type) -> None:
        """`_open` for each leaf of *group*, all of *kind*: an ``int`` is a
        number, a text read for the numbers it spells, the rest holds none;
        a *quiet* walk notes each one's id."""
        if self.quiet:
            self._leaves_held(group)
        elif kind is int:
            self.numbers.update(group)
        elif kind is str:
            self._read_texts(list(compress(group, map(operator.le, repeat(_SHORTEST_RUN), map(len, group)))))

    def _in_bulk(self, kind: type) -> bool:
        """Whether `_open_in_bulk` may read objects of *kind*: `_open` reads
        one as its builtin items (an exact dict, list or tuple) or as its
        attributes and slots alone (a class of the user's, a
        ``SimpleNamespace``), with nothing else asked of it."""
        found = self.bulk_kinds.get(kind)
        if found is None:
            if kind is dict or kind in _NESTED:
                found = True
            else:
                found = (
                    kind not in self.leaves
                    and kind not in self.closed_kinds
                    and not issubclass(
                        kind,
                        (*_REFERENCES, *_WEAK_CONTAINERS, weakref.finalize, *_BY_REFERENCE, *_BUILTIN_CONTAINERS),
                    )
                    and not _numpy_array(kind)
                    and not _pandas_object(kind)
                    and (issubclass(kind, types.SimpleNamespace) or self._users(kind))
                )
            self.bulk_kinds[kind] = found
        return found

    def _count_ahead(self, into: list[Any], start: int, queued: set[int]) -> bool:
        """Add to *queued* the ids of the objects *into* got from *start*
        on that opening them will count against the budget: neither a leaf
        nor seen already. True when they are more than the budget has left:
        the walk is certain to be cut a level down. An object queued that
        this level opens before then is taken out of *queued* when seen
        (`_open`), so none is counted that the budget would not pay for."""
        added = into[start:] if start else into
        if len(added) < _BULK_MIN:
            seen = self.seen
            for item in added:
                kind = type(item)
                if kind is int or kind is str or kind in self.leaves:
                    continue
                key = id(item)
                if key not in seen:
                    queued.add(key)
        elif walk_time_left():
            # The answer is False while the time lasts; a level read in bulk
            # is not counted for it (`run` only stops a walk sooner by it).
            return False
        else:
            # At C speed: a level read in bulk puts hundreds of thousands here.
            leaves = self.leaves | _ID_LEAVES
            opened = compress(added, map(operator.not_, map(leaves.__contains__, map(type, added))))
            queued.update(set(map(id, opened)).difference(self.seen))
        return len(queued) > self.budget and not walk_time_left()

    def _more(self, needed: int) -> bool:
        """Past the budget by *needed* objects: True, with the budget raised
        past them, while the share check this walk runs in has time left
        (`walk_time_left`). A loop that ran for minutes filling a list of a
        million records is worth the seconds its walk takes; at a fixed
        count the walk was cut, and the loop ran again every time."""
        if self.quiet or not walk_time_left():
            return False
        self.budget += needed + _MORE_SEEN
        self.granted += needed + _MORE_SEEN
        return True

    def _open(self, item: Any, into: list[Any]) -> None:
        kind = type(item)
        if kind is int or kind is str or kind in _LEAVES:
            if self.quiet:
                # Looking for an id(): a string or a float is an object an
                # ``id()`` names as well, at a new address after a restart.
                self._leaf_held(item)
            elif kind is int:
                self.numbers.add(item)
            elif kind is str:
                # ``hex(id(x))``, ``str(id(x))``, ``f"run-{id(x)}"``: an id() as well.
                if len(item) >= _SHORTEST_RUN:
                    self.numbers.update(_spelled_in(item))
            return
        if kind in self.leaves:
            # A library's scalar, read so by the value's own walk alone (a
            # *quiet* walk's leaves are `_LEAVES`): holds nothing, but an
            # id() of it is the value's own all the same. Costs no budget.
            self.seen.add(id(item))
            if kind in self.int_leaves:
                self.numbers.add(int(item))
            return
        key = id(item)
        if key in self.seen:
            return
        self.seen.add(key)
        if self._queued:
            self._queued.discard(key)
        self.budget -= 1
        if self.budget < 0 and not self._more(-self.budget):
            self.cut = True
            return
        if self.targets is not None and weakref.getweakrefcount(item) and self._notebook_own(kind):
            self.targets.append(item)
        if kind in self.closed_kinds:
            # As the first of its kind was: a library's object not looked
            # inside, told by its type alone (a list of 1.3 million
            # Timedeltas each asked every question below).
            if not self.quiet:
                self.closed.append(item)
            return
        if issubclass(kind, _REFERENCES):
            if not self.quiet or self.keep_refs:
                self.refs.append(item)
            return
        if issubclass(kind, weakref.finalize):
            # Finds its object by the weak reference it keeps out of
            # sight, in ``weakref.finalize._registry``: ``f.peek()``.
            info = _finalize_info(item)
            if info is not None and (not self.quiet or self.keep_refs):
                self.refs.append(info.weakref)
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
                # A frame's column of text, read whole whatever its length.
                if items and self._read_flat(items):
                    return
            else:
                if not self.quiet and item.dtype.kind in "iu" and item.dtype.itemsize == 8:
                    self.arrays.append(item)
                return
        elif issubclass(kind, types.SimpleNamespace) or self._users(kind):
            items = []
            state = own_dict(item)
            if state:
                items.extend(_state_held(kind, state))
            for reader in slot_readers(kind):
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
            else:
                if items is None and _type_tells(kind):
                    self.closed_kinds.add(kind)
            if items is None:
                if not self.quiet:
                    self.closed.append(item)
                return
        if len(items) > _MAX_SCANNED or (not self.quiet and len(items) >= _SMALL_FLAT):
            # A small one too, outside the notebook's walk: a column of
            # 60,000 lists of 20 records each was walked a tuple and a
            # string at a time after every statement (student_8).
            if self._read_flat(items):
                return
        if len(items) > _MAX_SCANNED:
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
            elif self._read_records(items):
                return
        into.extend(items)

    def _read_flat(self, items: list[Any]) -> bool:
        """Read a big container of leaves, or of tuples of leaves (records),
        at C speed: only its ``int``s matter. False when it holds anything
        else, to be walked item by item. The very items read before in this
        cell (`_FLAT_READS`) are leaves again, spelling the same numbers,
        and so is a container of the same content (`_SHAPE_READS`)."""
        if self.quiet or len(items) <= _MAX_SCANNED:
            return self._read_flat_now(items)
        known = _flat_read_before(items)
        if known is not None:
            self.numbers.update(known)
            return True
        # Lists and tuples only: a column of leaves is known by its very
        # items (`_FLAT_READS`), and its witness would cost each first read.
        witness = (
            _SHAPE_WITNESS(items, _NESTED_LEVELS)
            if _SHAPE_WITNESS is not None and type(items) is list and type(items[0]) in _NESTED
            else None
        )
        if witness is not None and _EVERY_SPELLING[0]:
            witness = None
        if witness is not None:
            read = _SHAPE_READS.get(witness)
            if read is not None and read[1] <= self.budget:
                numbers, lists, nodes = read
                self.numbers.update(numbers)
                self.budget -= lists
                if nodes:
                    # The containers read, as the read would have kept them.
                    self._unseen.append(functools.partial(_nodes_read, items))
                return True
        # Read into a set of its own: what it finds, all of it, is what it
        # keeps, numbers met before in this walk too.
        numbers, budget, unseen = self.numbers, self.budget - self.granted, len(self._unseen)
        self.numbers = set()
        try:
            done = self._read_flat_now(items)
        finally:
            found, self.numbers = self.numbers, numbers
            numbers |= found
        if done and not self.cut and witness is not None and len(_SHAPE_READS) < _MAX_SHAPE_READS:
            spent = budget - (self.budget - self.granted)
            _SHAPE_READS[witness] = (frozenset(found), spent, len(self._unseen) > unseen)
        return done

    def _read_flat_now(self, items: list[Any]) -> bool:
        """`_read_flat`, without its memos."""
        big = len(items) > _MAX_SCANNED
        kinds = set(map(type, items))
        flat: Any = items
        if not kinds.isdisjoint(self.leaves - _LEAVES) and kinds <= self.leaves:
            # A library's scalars, each seen as `_open` sees it: their ids
            # only when asked (`seen_all`).
            self._unseen.append(items if kinds.isdisjoint(_LEAVES) else [i for i in items if type(i) not in _LEAVES])
        elif not kinds <= _LEAVES:
            # A record's library scalars are read as any object is.
            leaves = _LEAVES
            if not kinds <= leaves | {tuple}:
                if self.quiet or not kinds <= leaves | _NESTED:
                    return False
                return self._read_nested(items)
            if kinds == _JUST_TUPLES and not self.quiet and _NESTED_LEAVES is not None:
                # Records alone: what reading them flat finds, `_read_nested`
                # finds too, and at C speed (student_8's two million
                # ``(action, time)`` records, a second a statement).
                return self._read_nested(items)
            records = items if kinds == _JUST_TUPLES else [item for item in items if type(item) is tuple]
            flat = list(chain.from_iterable(records))
            kinds = set(map(type, flat))
            if not kinds <= leaves:
                if not self.quiet and kinds <= leaves | _NESTED:
                    # Records holding lists of records (``(user, [(action,
                    # time), ...])``, student_8's 87,000 sessions): read a
                    # level at a time, where each list took a read of its own.
                    return self._read_nested(items)
                return False
            if not self.quiet:
                # Walked, each would be seen: an id() of one is the value's
                # own. Their ids only when asked (`seen_all`), as
                # `_read_nested` keeps them: a set of two million ids was a
                # third of a frame's read.
                self._unseen.append(records)
        if self.quiet:
            # The leaves read at once, and the records holding them: an
            # ``id()`` of one is looked for as `_open` looks for one.
            if flat is not items:
                self._leaves_held(records)
            self._leaves_held(flat)
            return True
        if not big or flat is not items or not kinds <= _KEPT_KINDS:
            self._read_leaves(flat, kinds)
            return True
        before = set(self.numbers)
        self._read_leaves(flat, kinds)
        _keep_flat_read(items, frozenset(self.numbers - before))
        return True

    def _read_nested(self, items: list[Any]) -> bool:
        """`_read_flat` for lists and tuples of leaves, nested: an object
        column of 30,000 lists of ``(name, time)`` records (student_8),
        read a level at a time instead of a list at a time. Each list and
        tuple is seen, as walking it would see it, and counts against the
        budget; False, with nothing seen, when a level holds anything else."""
        if _NESTED_LEAVES is not None and type(items) is list:
            read = _NESTED_LEAVES(items, _LEAVES, _NESTED_LEVELS, _SHORTEST_RUN)
            if read is False:
                return False
            if read is not NotImplemented:
                nodes, texts, ints, leaf_kinds, lists = read
                if lists > self.budget and not self._more(lists - self.budget):
                    # Against all this walk has seen, the containers an
                    # earlier read only noted (`seen_all`) too: walked
                    # again, those cost nothing.
                    if len(set(map(id, nodes)) - self.seen_all()) > self.budget:
                        # Walked a list at a time, each would count against
                        # the budget, and the walk would stop past it: told
                        # here, at C speed (a frame's column of 300,000
                        # lists took 0.3 s a statement to reach the cut).
                        self.cut = True
                        return True
                    return False
                self.budget -= lists
                self._unseen.append(nodes)
                self._read_texts(texts)
                self._read_ints(ints, leaf_kinds == {int})
                return True
        level: list[Any] = items
        nodes: list[Any] = []
        leaves: list[Any] = []
        # The leaves' types, as each level's are found: a level of two
        # million records is not passed over again for them (student_8).
        leaf_kinds: set[type] = set()
        lists = 0
        for _ in range(_NESTED_LEVELS):
            kinds = set(map(type, level))
            if kinds <= _LEAVES:
                leaves.extend(level)
                leaf_kinds |= kinds
                break
            if not kinds <= _LEAVES | _NESTED:
                return False
            if kinds <= _NESTED:
                inner = level
            else:
                is_node = list(map(_NESTED.__contains__, map(type, level)))
                inner = list(compress(level, is_node))
                leaves.extend(compress(level, map(operator.not_, is_node)))
                leaf_kinds |= kinds - _NESTED
            nodes.extend(inner)
            # A list counts against the budget as walking it would; a tuple
            # of leaves is read whole, as `_read_flat` reads records. Told
            # from the level's types when it holds lists or tuples alone.
            if tuple not in kinds:
                lists += len(inner)
            elif list in kinds:
                lists += sum(map(operator.is_, map(type, inner), repeat(list)))
            level = list(chain.from_iterable(inner))
        else:
            return False
        if lists > self.budget and not self._more(lists - self.budget):
            # As the C read above tells it, read to the end: more containers
            # than the budget, none seen before, is the walk's cut, known
            # without opening them one at a time (Python 3.10 and 3.11 have
            # no C read, and a column of 250,000 lists was opened a list at
            # a time up to the budget).
            if len(set(map(id, nodes)) - self.seen_all()) > self.budget:
                self.cut = True
                return True
            return False
        self.budget -= lists
        # Their ids only when asked (`seen_all`): most values hold no
        # weak reference and no number an id() could be, and a set of
        # 600,000 ids was half of what reading them cost.
        self._unseen.append(nodes)
        self._read_leaves(leaves, leaf_kinds)
        return True

    def _read_records(self, items: list[Any]) -> bool:
        """`_read_nested` for a big container of records: objects of a class
        the walk reads by its attributes alone (`_plain_record`), and
        dicts, lists, tuples, sets and frozensets, over leaves, read a level
        at a time. A list of a million ``SimpleNamespace`` records, or of
        dicts each holding a set, was opened an object at a time until the
        share check's time ran out, and the statement ran every time. Each
        object read is seen and counts against the budget, as walking it
        would; False, with nothing seen, for anything else, or for one a
        weak reference points at (`targets`), which the walk keeps."""
        level: list[Any] = items
        nodes: list[Any] = []
        leaves: list[Any] = []
        leaf_kinds: set[type] = set()
        for _ in range(_NESTED_LEVELS):
            kinds = set(map(type, level))
            if kinds <= _LEAVES:
                leaves.extend(level)
                leaf_kinds |= kinds
                break
            inner_kinds = kinds - _LEAVES
            if not all(kind in _RECORD_NODES or self._plain_record(kind) for kind in inner_kinds):
                return False
            if inner_kinds == kinds:
                inner = level
            else:
                is_node = list(map(inner_kinds.__contains__, map(type, level)))
                inner = list(compress(level, is_node))
                leaves.extend(compress(level, map(operator.not_, is_node)))
                leaf_kinds |= kinds - inner_kinds
            # A long container held more than once is read once, as the
            # walk opens it once.
            long = [item for item in inner if type(item) in _RECORD_NODES and len(item) >= _FLAT_RECORDS_MIN]
            if len(set(map(id, long))) < len(long):
                inner = list({id(item): item for item in inner}.values())
            del long
            parts: list[Any] = []
            for kind in inner_kinds:
                group = inner if len(inner_kinds) == 1 else [item for item in inner if type(item) is kind]
                if self.targets is not None and kind not in _UNREFERABLE and any(map(weakref.getweakrefcount, group)):
                    return False
                if kind is dict:
                    parts.append(chain.from_iterable(group))
                    parts.append(chain.from_iterable(map(dict.values, group)))
                elif kind in _RECORD_NODES:
                    parts.append(chain.from_iterable(group))
                else:
                    # Python's own slot, read at C speed, as `_lone_records` reads it.
                    states = list(map(_DICT, group))
                    if set(map(type, states)) != {dict}:
                        return False
                    parts.append(chain.from_iterable(map(dict.values, states)))
            nodes.extend(inner)
            level = list(chain.from_iterable(parts))
        else:
            return False
        if len(nodes) > self.budget and not self._more(len(nodes) - self.budget):
            # As `_read_nested` tells the walk's cut.
            if len(set(map(id, nodes)) - self.seen_all()) > self.budget:
                self.cut = True
                return True
            return False
        self.budget -= len(nodes)
        self._unseen.append(nodes)
        self._read_leaves(leaves, leaf_kinds)
        return True

    def _plain_record(self, kind: type) -> bool:
        """Whether `_open` reads an object of *kind* by its ``__dict__``'s
        values alone, and Python's own slot holds that dict: a
        ``SimpleNamespace``, or a class of the user's code with no slots,
        no lookup, ``__class__`` or ``__dict__`` of its own and no C base
        but ``object``."""
        known = self.record_kinds.get(kind)
        if known is None:
            known = self.record_kinds[kind] = kind is types.SimpleNamespace or (
                self._users(kind) and _plain_user_class(kind)
            )
        return known

    def seen_all(self) -> set[int]:
        """*seen*, with the containers `_read_nested` read."""
        unseen = self._unseen
        while unseen:
            found = unseen.pop()
            self.seen.update(found() if callable(found) else map(id, found))
        return self.seen

    def _read_leaves(self, flat: Any, kinds: set[type]) -> None:
        """The numbers *flat*, leaves of *kinds*, holds or spells."""
        if str in kinds:
            # The text spelling a number, found by one search of it all.
            texts = flat if kinds == {str} else compress(flat, map(operator.is_, map(type, flat), repeat(str)))
            # Each text once: a column of 5 million cells repeating 8,000 log
            # lines of 1 KB each was joined into one 5 GB string, and the
            # kernel ran out of memory (student_3's ``explode``). Read
            # whatever its length: an id() can be spelled inside any text.
            texts = list(dict.fromkeys(texts))
            self._read_texts(texts)
        if int in kinds:
            self._read_ints(flat, kinds == {int})
        numpy_ints = kinds & self.int_leaves
        if numpy_ints:
            # Each as the ``int`` it is: ``np.int64(id(x))`` is an id() too.
            kind_of = map(type, flat)
            self._read_ints(list(map(int, compress(flat, map(numpy_ints.__contains__, kind_of)))), True)

    def _read_texts(self, texts: list[str]) -> None:
        """The numbers *texts*, each one once, spell anywhere in them
        (`_spelled_in`), searched joined a chunk at a time."""
        for text in _joined(texts):
            if _may_spell(text):
                self.numbers.update(_spelled_in(text, mapped_only=True))

    def _read_ints(self, flat: Any, only: bool) -> None:
        """The ints of *flat* that could be an id(); with *only*, *flat*
        holds nothing but ints."""
        if only:
            # At C speed: row numbers or counts all below any address.
            if flat and max(flat) >= _LOWEST_ADDRESS and min(flat) < _HIGHEST_ADDRESS:
                self.numbers.update(flat)
        else:
            # The ints picked out at C speed, and looked through one by one
            # only when one could be an address: the values level of a
            # million ``{'i': i, 'v': [...]}`` records was a quarter second.
            ints = list(compress(flat, map(operator.is_, map(type, flat), repeat(int))))
            if ints and max(ints) >= _LOWEST_ADDRESS:
                self.numbers.update(item for item in ints if item >= _LOWEST_ADDRESS)

    def _leaf_held(self, item: Any) -> None:
        """Note a leaf a *quiet* walk met, for the ids looked for."""
        if self.id_arrays is not None:
            self.leaf_ids.append(id(item))
        if self.candidates and id(item) in self.candidates:
            self.found = True

    def _leaves_held(self, items: list[Any]) -> None:
        """`_leaf_held` for each of *items*, at C speed."""
        if self.id_arrays is not None:
            self.leaf_ids.extend(map(id, items))
        if self.candidates and not self.candidates.isdisjoint(map(id, items)):
            self.found = True

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
            or slot_readers(kind)
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
        if self.candidates or self.id_arrays is not None:
            # Their attributes' strings and numbers too: ``id(pts[7].name)``.
            self._leaves_held(list(chain.from_iterable(map(dict.values, states))))
        return True

    def _opens(self, item: Any) -> bool:
        """Whether `_open` would look into *item*, or keep it as a weak
        reference, had the walk gone a level deeper."""
        kind = type(item)
        if kind in (int, str) or kind in self.leaves or id(item) in self.seen_all():
            return False
        if issubclass(kind, _BY_REFERENCE):
            return self.quiet and bool(_code_parts(item))
        return (
            issubclass(kind, (*_REFERENCES, *_WEAK_CONTAINERS, weakref.finalize, types.SimpleNamespace))
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
