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
import bisect
import functools
import operator
import re
import sys
import types
import weakref
from itertools import chain, compress, repeat
from typing import Any

from cash import _native
from cash._user_attrs import attribute_or, builtin_base, builtin_items, has_attribute, own_dict
from cash._weak import callback_of, is_cash_ref
from cash.closure_sharing import slot_readers
from cash.install_paths import is_user_code_module
from cash.value_types import IMMUTABLE_LEAF_TYPES

__all__ = ["IdentityHeld", "as_address", "end_cell_reads", "forget_flat_reads", "identity_held_reason", "library_parts"]

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
    _flat_kept[0] = 0


def end_cell_reads() -> None:
    """The cell ended: drop what `_FLAT_READS` and `_ARROW_READS` keep,
    which hold the very objects read, and the entries of `_SHAPE_READS`
    holding many numbers; the rest of it, which holds no object, is kept
    for the next cell until it is half full."""
    _FLAT_READS.clear()
    _ARROW_READS.clear()
    _flat_kept[0] = 0
    if len(_SHAPE_READS) > _MAX_SHAPE_READS // 2:
        _SHAPE_READS.clear()
        return
    for witness in [witness for witness, read in _SHAPE_READS.items() if len(read[0]) > _KEPT_SHAPE_NUMBERS]:
        del _SHAPE_READS[witness]


def _flat_read_before(items: list[Any]) -> frozenset[int] | None:
    """The numbers a read of these very *items* found earlier this cell, or
    None when they were not read as leaves."""
    for kept, numbers in _FLAT_READS.get(len(items), ()):
        if all(map(operator.is_, kept, items)):
            return numbers
    return None


def _keep_flat_read(items: list[Any], numbers: frozenset[int]) -> None:
    if _flat_kept[0] + len(items) > _MAX_FLAT_KEPT:
        return
    _flat_kept[0] += len(items)
    _FLAT_READS.setdefault(len(items), []).append((list(items), numbers))


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
    name: str, value: Any, user_ns: dict[str, Any], weak_holders: bool = True, more: int = 0
) -> IdentityHeld | None:
    """Why *value*, bound to *name*, cannot be restored apart from what it
    points at by identity (see the module docstring), or with *weak_holders*,
    while another variable finds it by one (`_weakly_held`); None when it
    can. Not *weak_holders* for a loop's object a hit restores into: it
    stays the object those references find. *more*: objects the walk looks
    at beyond `_MAX_SEEN` (the answers a statement put into a helper memo,
    as many as the memo holds)."""
    user_kinds: dict[type, bool] = {}
    walk = _Walk(user_kinds, targets=[] if weak_holders else None)
    walk.budget += more
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
        if target is not None and id(target) not in walk.seen_all() and not issubclass(type(target), _BY_REFERENCE):
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

        held = _Walk(user_kinds, seen=set(walk.seen_all()), quiet=True, candidates=frozenset(candidates))
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

#: The longest line `_NUMBER_LINES` matches.
_LONGEST_NUMBER = 18

#: Texts `_Walk._read_leaves` joins into one string to search.
_TEXTS_AT_ONCE = 100_000


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
        numpy = sys.modules.get("numpy")
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
    thing that could matter: the numbers its values spell as `as_address`
    reads them, picked and parsed by Arrow at C speed -- the decimal ones as
    one NumPy array of 8-byte integers (checked as an integer column is),
    the hexadecimal ones as ints. Converting the column to Python strings
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
    chunked = attribute_or(values, "_pa_array", None)
    pyarrow = sys.modules.get("pyarrow")
    compute = sys.modules.get("pyarrow.compute")
    if chunked is None or pyarrow is None or compute is None:
        return None
    known = _ARROW_READS.get(id(chunked))
    if known is not None and known[0] is chunked:
        return known[1]
    try:
        if not (pyarrow.types.is_string(chunked.type) or pyarrow.types.is_large_string(chunked.type)):
            return None
        if len(chunked) <= _FEW_TEXTS:
            few = _few_texts_spelling_numbers(chunked.to_pylist())
            _ARROW_READS[id(chunked)] = (chunked, few)
            return few
        # Only a value of 7 to 18 bytes can spell either number, all digits
        # or starting ``0x``: each is picked by masks Arrow computes without
        # copying the text, and only those taken out. A merged frame's text
        # column came in thousands of chunks, and two patterns over all of
        # it took 6 s after one `merge` (p18). The patterns are matched on
        # those values alone, and the prefix as two plain ones: matched
        # ignoring case, Arrow ran a regular expression over every value.
        length = compute.binary_length(chunked)
        fits = compute.and_(compute.greater_equal(length, 7), compute.less_equal(length, 18))
        parts: list[Any] = []
        candidates = compute.filter(chunked, fits)
        decimal = compute.filter(candidates, compute.ascii_is_decimal(candidates))
        if len(decimal):
            parts.append(compute.cast(decimal, pyarrow.int64()).to_numpy(zero_copy_only=False))
        prefixed = compute.filter(
            candidates,
            compute.or_(compute.starts_with(candidates, "0x"), compute.starts_with(candidates, "0X")),
        )
        if len(prefixed):
            hexadecimal = compute.filter(prefixed, compute.match_substring_regex(prefixed, _HEX_TEXT))
            parts.extend(int(text, 16) for text in hexadecimal.to_pylist() if text is not None)
        _ARROW_READS[id(chunked)] = (chunked, parts)
        return parts
    except Exception:  # noqa: BLE001 - read as before, through NumPy
        return None


#: What `_arrow_text_spelling_numbers` found this cell, by the Arrow array
#: read (kept, so its id is not reused): Arrow's arrays never change, and a
#: loop writing one column of a wide frame leaves the others the same
#: arrays. Cleared when the cell ends (`end_cell_reads`).
_ARROW_READS: dict[int, tuple[Any, list[Any]]] = {}
#: `_NUMBER_LINES`' hexadecimal spelling, a whole value (the decimal one,
#: 7 to 18 digits, is picked by length and `ascii_is_decimal`).
_HEX_TEXT = r"^0[xX][0-9a-fA-F]{5,16}$"
#: The same two, for Python's ``re`` (``fullmatch``: Arrow's ``$`` is the
#: end of the text, never before a last newline).
_DECIMAL_WHOLE = re.compile(r"[0-9]{7,18}")
_HEX_WHOLE = re.compile(r"0[xX][0-9a-fA-F]{5,16}")
#: The same, each a whole line of texts joined by line breaks, and a line
#: of a number's length (7 to 18 characters, of ASCII text: bytes).
_DECIMAL_LINE = re.compile(r"^[0-9]{7,18}$", re.MULTILINE)
_HEX_LINE = re.compile(r"^0[xX][0-9a-fA-F]{5,16}$", re.MULTILINE)
_FITS_LINE = re.compile(r"^.{7,18}$", re.MULTILINE)
#: An Arrow text array of at most this many values is read in Python: each
#: Arrow call costs tens of microseconds however short the array, and a
#: list of thousands of small frames made four per column.
_FEW_TEXTS = 64


def _few_texts_spelling_numbers(texts: list[Any]) -> list[Any]:
    """What `_arrow_text_numbers` finds in a few *texts* (None for a
    missing value), the same way: nothing when no value is 7 to 18 bytes
    long, else the decimal numbers as one NumPy array of 8-byte integers,
    then the hexadecimal ones as ints. ASCII texts with no line break are
    searched joined, a line each, in three calls."""
    import numpy

    present = [text for text in texts if text is not None]
    joined = "\n".join(present)
    if joined.isascii() and joined.count("\n") == len(present) - 1:
        if not _FITS_LINE.search(joined):
            return []
        decimal = _DECIMAL_LINE.findall(joined)
        hexadecimal = _HEX_LINE.findall(joined)
    else:
        if not any(7 <= len(text.encode()) <= 18 for text in present):
            return []
        decimal = [text for text in present if _DECIMAL_WHOLE.fullmatch(text)]
        hexadecimal = [text for text in present if _HEX_WHOLE.fullmatch(text)]
    # An array of its own each time, as Arrow's read makes: each is one
    # object the walk counts against its budget (`_MAX_SEEN`).
    parts: list[Any] = [numpy.array([int(text) for text in decimal], dtype=numpy.int64)]
    parts.extend(int(text, 16) for text in hexadecimal)
    return parts


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
    return f"{name}()" if isinstance(name, str) else type(func).__name__


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


def _type_tells(kind: type) -> bool:
    """Whether `library_parts` answers alike for every object of *kind*:
    it asks ``isinstance``, which an object can answer apart from its type
    through a ``__class__`` of its own (a proxy's); a NumPy array's answer
    is its dtype's."""
    return not _numpy_array(kind) and not any("__class__" in vars(base) for base in kind.__mro__[:-1])


def _numpy_array(kind: type) -> bool:
    return kind.__name__ == "ndarray" and attribute_or(kind, "__module__") == "numpy"


def _nodes_read(items: list[Any]) -> set[int]:
    """The ids of the containers `_Walk._read_flat` keeps of *items* for
    `_Walk.seen_all`, read again: what a memo's answer for *items* leaves
    out."""
    walk = _Walk({})
    walk._read_flat_now(items)
    return walk.seen_all()


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
        #: Containers `_read_nested` read, not yet in *seen* (`seen_all`),
        #: or a function giving them.
        self._unseen: list[Any] = []
        self.budget = _MAX_SEEN
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

    def _count_ahead(self, into: list[Any], start: int, queued: set[int]) -> bool:
        """Add to *queued* the ids of the objects *into* got from *start*
        on that opening them will count against the budget: neither a leaf
        nor seen already. True when they are more than the budget has left:
        the walk is certain to be cut a level down. An object queued that
        this level opens before then is taken out of *queued* when seen
        (`_open`), so none is counted that the budget would not pay for."""
        seen = self.seen
        for item in into[start:]:
            kind = type(item)
            if kind is int or kind is str or kind in _LEAVES:
                continue
            key = id(item)
            if key not in seen:
                queued.add(key)
        return len(queued) > self.budget

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
        if self._queued:
            self._queued.discard(key)
        self.budget -= 1
        if self.budget < 0:
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
                items.extend(state.values())
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
        numbers, budget, unseen = self.numbers, self.budget, len(self._unseen)
        self.numbers = set()
        try:
            done = self._read_flat_now(items)
        finally:
            found, self.numbers = self.numbers, numbers
            numbers |= found
        if done and not self.cut and witness is not None and len(_SHAPE_READS) < _MAX_SHAPE_READS:
            _SHAPE_READS[witness] = (frozenset(found), budget - self.budget, len(self._unseen) > unseen)
        return done

    def _read_flat_now(self, items: list[Any]) -> bool:
        """`_read_flat`, without its memos."""
        big = len(items) > _MAX_SCANNED
        kinds = set(map(type, items))
        flat: Any = items
        if not kinds <= _LEAVES:
            if not kinds <= _LEAVES | {tuple}:
                if self.quiet or not kinds <= _LEAVES | _NESTED:
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
            if not kinds <= _LEAVES:
                if not self.quiet and kinds <= _LEAVES | _NESTED:
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
            read = _NESTED_LEAVES(items, _LEAVES, _NESTED_LEVELS, _LONGEST_NUMBER)
            if read is False:
                return False
            if read is not NotImplemented:
                nodes, texts, ints, leaf_kinds, lists = read
                if lists > self.budget:
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
            if lists > self.budget:
                return False
            level = list(chain.from_iterable(inner))
        else:
            return False
        self.budget -= lists
        # Their ids only when asked (`seen_all`): most values hold no
        # weak reference and no number an id() could be, and a set of
        # 600,000 ids was half of what reading them cost.
        self._unseen.append(nodes)
        self._read_leaves(leaves, leaf_kinds)
        return True

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
            # Each text once, and only one a line of could spell a number:
            # a column of 5 million cells repeating 8,000 log lines of 1 KB
            # each was joined into one 5 GB string, and the kernel ran out
            # of memory (student_3's ``explode``).
            unique = list(dict.fromkeys(texts))
            texts = list(
                compress(
                    unique,
                    map(
                        operator.or_,
                        map(operator.ge, repeat(_LONGEST_NUMBER), map(len, unique)),
                        map(operator.contains, unique, repeat("\n")),
                    ),
                )
            )
            del unique
            self._read_texts(texts)
        if int in kinds:
            self._read_ints(flat, kinds == {int})

    def _read_texts(self, texts: list[str]) -> None:
        """The numbers *texts*, each one once and none longer than a number
        unless it holds a line break, spell (`as_address`)."""
        for start in range(0, len(texts), _TEXTS_AT_ONCE):
            text = "\n".join(texts[start : start + _TEXTS_AT_ONCE])
            # Most text has no run of seven digits and no "0x": found by
            # plain searches, far quicker than the pattern's.
            if "0x" in text or "0X" in text or "0000000" in text.translate(_DIGITS_AS_ZERO):
                self.numbers.update(map(as_address, _NUMBER_LINES.findall(text)))
                self.numbers.discard(None)

    def _read_ints(self, flat: Any, only: bool) -> None:
        """The ints of *flat* that could be an id(); with *only*, *flat*
        holds nothing but ints."""
        if only:
            # At C speed: row numbers or counts all below any address.
            if flat and max(flat) >= _LOWEST_ADDRESS and min(flat) < _HIGHEST_ADDRESS:
                self.numbers.update(flat)
        else:
            self.numbers.update(item for item in flat if type(item) is int and item >= _LOWEST_ADDRESS)

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
        return True

    def _opens(self, item: Any) -> bool:
        """Whether `_open` would look into *item*, or keep it as a weak
        reference, had the walk gone a level deeper."""
        kind = type(item)
        if kind in (int, str) or kind in _LEAVES or id(item) in self.seen_all():
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
