"""The canonical form a cache key pickles a value in (`stable_key_repr`,
`canonical_bytes`): equal values give equal bytes, in any process, however
they were built.

Sets are put in a stable order, every container is tagged with its type, a
container met twice in one value is marked as shared, and an object holding
a set is opened up so the set can be ordered. A frame, array or table inside
the value becomes its content hash; the caller says how to read one
(`ContentHashing`): ``cash.content_hashers.BUILTIN_CONTENT`` reads every
byte, the decorator puts its copy-on-write memo in front.

A key's form depends on the value and the arguments alone. The module keeps
no state between calls but one per-type memo (`_holds_native_state`), a
fact about a class that never changes. Nothing here knows about ``Cash``,
IPython or lineage: a caller that needs context passes it in.
"""

from __future__ import annotations

import copyreg
import fractions
import logging
import sys
import types
import uuid
from collections.abc import Callable
from typing import Any, NamedTuple

from . import _plain_data
from .sizing import SPARSE_PARTS, pandas_nbytes
from .value_types import CODELESS_PRIMS, LEAF_TYPES, PARSED_VALUE_TYPES


class CyclicValueError(TypeError):
    """A value whose object graph loops back on itself and holds a set."""


class ContentHashing(NamedTuple):
    """How a key walk reads the frames, arrays and tables it meets.

    *family* names the content hasher that claims a type, or None;
    *digest* is a claimed value's content hash, or None when it has none.
    *memoable*, when given, says the caller can check a value instead of
    reading it again: the decorator's copy-on-write memo for an unchanged
    pandas frame. Such a value is always keyed on its own inside an object
    (`holds_content_data`), however small. The caller passes this in, so a
    walk's result depends on its arguments alone.
    """

    family: Callable[[type], str | None]
    digest: Callable[[Any], str | None]
    memoable: Callable[[Any], bool] | None = None


def object_state(value: Any) -> dict:
    """Return an object's instance state as a name -> value dict, covering both
    ``__dict__`` and ``__slots__`` (collected across the MRO so slots declared
    on base classes are included). Builtins and leaf values yield ``{}``. Used
    so set-canonicalisation reaches a set buried inside a ``__slots__`` object,
    not just a ``__dict__``-backed one.
    """
    state: dict = {}
    obj_dict = getattr(value, "__dict__", None)
    if isinstance(obj_dict, dict):
        state.update(obj_dict)
    for klass in type(value).__mro__:
        slots = getattr(klass, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name in ("__dict__", "__weakref__") or name in state:
                continue
            try:
                state[name] = getattr(value, name)
            except AttributeError:
                pass  # slot declared but never assigned
    return state


#: The tag a builtin container is keyed under: one string object per type, so a
#: key's pickle stores it once however many containers it holds.
_BUILTIN_CONTAINER_TAGS = {t: t.__qualname__ for t in (dict, list, tuple, set, frozenset)}


def _typed(value: Any, canon: Any, walk: _Walk) -> tuple:
    """*canon*, a container's canonical items, tagged with the container's type.

    Every container carries its type, so containers holding equal items key
    apart when their types differ: a list and a tuple, a set and a frozenset,
    or ``P(1, 2)`` and ``Q(1, 2)`` from two namedtuple types, which otherwise
    shared one entry and were served each other's results.

    A subclass also brings the state it holds beside its items, which its
    items drop: ``defaultdict(list)`` and ``defaultdict(set)`` shared one
    entry, and a ``dict`` subclass holding ``self.source`` served the first
    caller's answer for every source. The state is read from ``__dict__``
    and ``__slots__`` both (`object_state`): a subclass declaring slots kept
    its values out of the key. The state is walked as part of the same
    *walk*, so a list it shares with the items is marked as
    shared and a loop back to the container is caught. Nothing is caught
    here: a part that cannot be read is not left out of the key, it makes
    the call unkeyable (run uncached, with a warning).
    """
    t = type(value)
    tag = _BUILTIN_CONTAINER_TAGS.get(t)
    if tag is not None:
        return ("__cash_type__", tag, canon)
    state: Any = ()
    factory = getattr(value, "default_factory", None)
    own = {k: v for k, v in object_state(value).items() if not k.startswith("__")}
    if factory is not None:
        state += (("default_factory", getattr(factory, "__qualname__", repr(factory))),)
    if own:
        state += tuple(sorted((k, _walk(v, walk)) for k, v in own.items()))
    tag = f"{t.__module__}.{t.__qualname__}"
    return ("__cash_type__", tag, canon, state) if state else ("__cash_type__", tag, canon)


def stable_key_repr(
    value: Any,
    content: ContentHashing,
    *,
    seen: dict | None = None,
    hook: Callable[[Any], Any] | None = None,
    left: list | None = None,
) -> Any:
    """The form a cache key hashes *value* in: equal values pickle to equal
    bytes, in any process.

    * Every dict, list, tuple, set and frozenset becomes a tuple tagged with its
      type (`_typed`), so containers of different types never key alike.
    * The items of a set are sorted by their pickled bytes: a set of strings
      iterates in an order PYTHONHASHSEED picks, and nothing can read a
      set's order back. A dict keeps its insertion order, which code reads:
      ``pd.DataFrame(d)`` orders its columns by it, ``json.dumps`` and
      ``csv.DictWriter`` write in it. Sorted, ``{"name": ..., "score": ...}``
      and its reordering shared an entry and the second call got the first
      one's column order. Equal dicts built in two orders now cost a miss.
    * An object with a set somewhere inside becomes its type and the
      canonicalised parts pickle would store for it (`_pickled_state`), so
      that set is sorted too. Any other object is left to pickle, which stores it as it asks to
      be stored (its ``__reduce__``) and keeps the loops in its graph.

    * A value a content hasher claims (a frame, an array, a table) becomes
      that hash (*content*, `ContentHashing`), wherever it sits, as it does
      as an argument of its own. *hook*, when given, is asked first about every
      value that is not a primitive: it returns the value's stand-in, or
      `NOT_HOOKED` -- how ``cash.register_hasher`` reaches a value inside a
      list or dict.
    * A list, dict or set met a second time in one walk becomes a marker
      naming where it was first met. Rebuilt as tuples, ``[[0] * 3] * 3``
      -- one row three times -- and three separate rows keyed alike, and a
      function that copied the grid and set one cell was served the aliased
      grid's answer.

    A container graph that loops back on itself raises `CyclicValueError` (a
    TypeError, so the value is reported as unhashable and the call runs
    uncached): a form that stood in for the loop could make two different
    graphs key alike, which would be a wrong answer. A value nested deeper
    than the walk can follow raises `TooDeepValueError`, a TypeError too.
    There is no depth limit below that: a part cut off at a fixed depth and
    left to pickle keeps its sets in the order PYTHONHASHSEED picks, and the
    key changes from process to process.

    *seen*, when given, is filled with the writable containers the walk met.
    *left*, when given, gets an entry for each object left to pickle whole
    (`canonical_bytes` reads it): only then can the form hold an object
    pickled twice, whose identity only pickle's memo records.
    """
    try:
        return _walk(value, _Walk(content, {} if seen is None else seen, hook, left))
    except RecursionError:
        raise TooDeepValueError(_too_deep(value)) from None


class TooDeepValueError(TypeError):
    """A value nested deeper than Python's recursion limit lets cash walk it,
    such as a long linked list. A TypeError, so the value is reported as
    unhashable and the call runs uncached: a key over part of it could serve
    one value's result for another."""


def _too_deep(value: Any) -> str:
    return f"a {type(value).__qualname__} nested too deeply to key (deeper than the recursion limit)"


class _Walk:
    """One `stable_key_repr` walk: the path walked so far (*stack*), the
    containers met so far (*seen*), and the caller's *content*, *hook* and
    *left* (see `stable_key_repr`)."""

    __slots__ = ("content", "hook", "left", "seen", "stack")

    def __init__(
        self, content: ContentHashing, seen: dict, hook: Callable[[Any], Any] | None, left: list | None
    ) -> None:
        self.content = content
        self.seen = seen
        self.hook = hook
        self.left = left
        self.stack: set[int] = set()


def _walk(value: Any, walk: _Walk) -> Any:
    """`stable_key_repr` of one value, as part of *walk*."""
    seen = walk.seen
    if type(value) in CODELESS_PRIMS:
        if type(value) is bytearray:
            # Written into, one bytearray held twice changes in two places.
            first = seen.get(id(value))
            if first is not None:
                return ("__cash_alias__", first[0])
            seen[id(value)] = (len(seen), value)
        return value
    if walk.hook is not None:
        stand_in = walk.hook(value)
        if stand_in is not NOT_HOOKED:
            return stand_in
    if type(value) in _plain_data.numpy_scalar_set():
        # A number: pickled by value, nothing inside to order.
        return ("__cash_np__", value.dtype.char, value.tobytes())
    family = walk.content.family(type(value))
    if family is not None:
        if family in _WRITABLE_FAMILIES:
            # One array held twice changes in two places when written.
            first = seen.get(id(value))
            if first is not None:
                return ("__cash_alias__", first[0])
        digest = walk.content.digest(value)
        if digest is not None:
            if family in _WRITABLE_FAMILIES:
                seen[id(value)] = (len(seen), value)
            return ("__cash_content__", family, digest)
    if id(value) in walk.stack:
        raise CyclicValueError(f"a {type(value).__qualname__} that contains itself has no stable form to key on")
    if isinstance(value, _MUTABLE_CONTAINERS):
        first = seen.get(id(value))
        if first is not None:
            return ("__cash_alias__", first[0])
        # The value is held, so its id cannot be reused by another
        # container while the walk lasts.
        seen[id(value)] = (len(seen), value)
    walk.stack.add(id(value))
    try:
        return _walk_object(value, walk)
    finally:
        walk.stack.discard(id(value))


#: Containers whose identity code can observe by writing through one
#: reference and reading through another.
_MUTABLE_CONTAINERS = (list, dict, set)

#: Content-hashed families whose values can be written into in place.
_WRITABLE_FAMILIES = frozenset({"numpy", "pandas", "scipy.sparse"})

#: Leaves that pickle as their value, whatever else refers to them.
_VALUE_LEAVES = (*PARSED_VALUE_TYPES, fractions.Fraction, uuid.UUID)

#: Pickled by name, not by content.
_BY_NAME = (type, types.FunctionType, types.BuiltinFunctionType, types.ModuleType)


def canonical_bytes(
    value: Any, content: ContentHashing, hook: Callable[[Any], Any] | None = None, seen: dict | None = None
) -> bytes:
    """*value*'s canonical form (`stable_key_repr`), pickled: equal values give
    equal bytes, in any process, however they were built.

    Pickled without the memo when the form is all tuples and leaves: through
    the memo, a second reference to one string, date or Decimal is written
    as a back-reference, so ``{"start": s, "end": s}`` and an equal dict
    parsed from JSON, whose values are two strings, keyed apart. Sharing
    that matters -- a list, dict, set or array held twice -- is spelled out
    in the form. Only when an object is left to pickle whole, which may
    share state inside, is the memo kept. *seen*, when given, is filled with
    the writable containers the walk met (`stable_key_repr`'s *seen*).
    """
    left: list = []
    form = stable_key_repr(value, content, seen=seen, hook=hook, left=left)
    try:
        if left:
            return b"m" + _plain_data.key_dumps(form)
        return b"u" + _plain_data.content_dumps(form)
    except RecursionError:
        # An object left to pickle whole can be deeper than pickle follows.
        raise TooDeepValueError(_too_deep(value)) from None


#: Exact types `canonical_call_bytes` writes straight into the form: each is
#: its own canonical form (`stable_key_repr` returns it as it is).
_FORM_PRIMS = frozenset((str, int, float, bool, type(None), bytes, complex))


def canonical_call_bytes(args: tuple, kwargs: dict) -> bytes | None:
    """``canonical_bytes((args, kwargs))`` for a call whose every argument is
    a primitive (`_FORM_PRIMS`), built without the walk; None for any other.

    The same bytes, at a tenth of the cost: a primitive is its own form, and
    the tuple and dict around them are tagged as `_typed` tags them. Hooks do
    not apply to primitives, and none of them is a container another
    argument could share.
    """
    prims = _FORM_PRIMS
    for a in args:
        if type(a) not in prims:
            return None
    for k, v in kwargs.items():
        if type(v) not in prims or type(k) is not str:
            return None
    return _call_bytes(tuple(args), tuple(kwargs.items()))


def canonical_marker_bytes(marker: tuple) -> bytes:
    """``canonical_bytes(((marker,), {}))`` for a *marker* that is a tuple of
    strings: the one argument an argument hash reduced to a digest of its own
    (`arg_hashing.plain_key_part`)."""
    return _call_bytes((("__cash_type__", _BUILTIN_CONTAINER_TAGS[tuple], marker),), ())


def _call_bytes(arg_forms: tuple, kwarg_forms: tuple) -> bytes:
    """The canonical bytes of ``(args, kwargs)`` from their items' forms."""
    tuple_tag = _BUILTIN_CONTAINER_TAGS[tuple]
    form = (
        "__cash_type__",
        tuple_tag,
        (
            ("__cash_type__", tuple_tag, arg_forms),
            ("__cash_type__", _BUILTIN_CONTAINER_TAGS[dict], kwarg_forms),
        ),
    )
    return b"u" + _plain_data.content_dumps(form)


#: What a `stable_key_repr` hook returns for a value it has no stand-in for.
NOT_HOOKED = object()

#: Items looked at per container attribute by `holds_content_data`.
_HOLDS_SCAN_ITEMS = 256


#: An array, frame or table at least this big makes the object holding it
#: keyed part by part (`holds_content_data`). Below it, pickling the object
#: whole is cheaper than hashing each part on its own: a fitted random forest
#: holds a few small arrays per tree, and opened up took 11 ms a hit where
#: its pickle took 4.
OPEN_UP_BYTES = 1 << 20


def holds_content_data(value: Any, content: ContentHashing) -> bool:
    """Does the object *value* hold a frame, array or table worth keying on
    its own (`_worth_opening`), in an attribute or in a list, tuple or dict
    an attribute holds? *content* says what counts as one.

    ``__dict__`` alone, not `object_state`: this is asked of every object a
    key walk leaves to pickle, and the slots walk up the MRO made keying a
    list of records markedly slower. An object holding its frames in slots is
    pickled whole.
    """
    state = getattr(value, "__dict__", None)
    if type(state) is not dict:
        return False
    family = content.family
    for v in state.values():
        t = type(v)
        if family(t) is not None:
            if _worth_opening(v, content):
                return True
            continue
        if t is list or t is tuple:
            items: Any = v[:_HOLDS_SCAN_ITEMS]
        elif t is dict:
            items = [x for _, x in zip(range(_HOLDS_SCAN_ITEMS), v.values())]
        else:
            continue
        if any(family(type(x)) is not None and _worth_opening(x, content) for x in items):
            return True
    return False


def _worth_opening(value: Any, content: ContentHashing) -> bool:
    """Is *value*, a frame, array or table inside an object, cheaper keyed on
    its own than pickled with the object? When the key's caller can check it
    instead of reading it (``content.memoable``: an unchanged copy-on-write
    pandas frame), or when it is big (`OPEN_UP_BYTES`)."""
    if content.memoable is not None and content.memoable(value):
        return True
    family = content.family(type(value))
    try:
        if family == "pandas":
            size = pandas_nbytes(value)
        elif family == "polars":
            size = value.estimated_size()
        elif family in ("numpy", "pyarrow"):
            size = value.nbytes
        elif family == "scipy.sparse":
            size = sum(getattr(getattr(value, part, None), "nbytes", 0) for part in SPARSE_PARTS)
        else:
            # A lazy collection (dask, modin) is keyed by its own token,
            # which costs less than computing it to pickle it.
            return True
    except Exception:  # noqa: BLE001 - a size we cannot read: pickle it whole, as before
        return False
    return isinstance(size, int) and size >= OPEN_UP_BYTES


def _walk_object(value: Any, walk: _Walk) -> Any:
    """`_walk` of a container or object, once it is on ``walk.stack``."""

    def sub(v: Any) -> Any:
        return _walk(v, walk)

    if isinstance(value, (set, frozenset)):
        items = [sub(v) for v in value]
        items.sort(key=_plain_data.content_dumps)
        return _typed(value, tuple(items), walk)
    if isinstance(value, dict):
        return _typed(value, tuple((sub(k), sub(v)) for k, v in value.items()), walk)
    if isinstance(value, (list, tuple)):
        return _typed(value, tuple(sub(v) for v in value), walk)
    t = type(value)
    if contains_set(value):
        return ("__cash_obj__", f"{t.__module__}.{t.__qualname__}", sub(_pickled_state(value)))
    if holds_content_data(value, walk.content):
        # Pickled whole, every frame inside was serialised and hashed on each
        # call. Opened up, each goes through the caller's content hasher
        # (``walk.content``, the decorator's memo), so an unchanged frame is
        # checked rather than read again. An object that reaches itself is
        # left to pickle, which keeps the loop.
        try:
            return ("__cash_obj__", f"{t.__module__}.{t.__qualname__}", sub(_pickled_state(value)))
        except CyclicValueError:
            pass
    left = walk.left
    if left is not None and not (
        type(value) in _VALUE_LEAVES or isinstance(value, _BY_NAME) or type(value) in _plain_data.fake_clock()[0]
    ):
        left.append(value)
    return value


#: The reconstructors ``object.__reduce_ex__`` names: their state is the
#: instance's own (``__getstate__``), so nothing is left beside it.
_COPYREG_REBUILDERS = (copyreg.__newobj__, copyreg.__newobj_ex__, copyreg._reconstructor)


def _pickled_state(value: Any) -> Any:
    """What pickle stores for *value*, as parts `stable_key_repr` can order.

    An object holding a set is opened up so the set can be sorted. Its
    ``__dict__`` alone is not all it holds: the value of a builtin or C base
    -- an ndarray subclass's data, a ``str`` subclass's text, a ``deque``
    subclass's items -- lives outside it, and two tagged arrays over
    ``[1, 2]`` and ``[30, 40]`` keyed alike. So the parts are what
    ``__reduce_ex__`` hands pickle: the constructor arguments, the state
    and any items. The class inside the arguments is replaced by a marker,
    the type being keyed beside them, so a class pickle cannot find by name
    still keys. A class whose own reduce leaves its instance dict out
    (``ndarray``'s) has that added. A value that refuses to be reduced (a
    module, a function) keeps its instance state alone.
    """
    try:
        reduced = value.__reduce_ex__(4)
    except TypeError:
        return object_state(value)
    if isinstance(reduced, str):
        return reduced  # pickled by name
    t = type(value)
    rebuild, args, *rest = reduced
    parts: list = [tuple("__cash_self_type__" if a is t else a for a in args or ())]
    for i, part in enumerate(rest):
        # State, then the iterators of list items and dict items.
        parts.append(list(part) if i and part is not None else part)
    if rebuild not in _COPYREG_REBUILDERS:
        parts.append(object_state(value))
    return tuple(parts)


def contains_set(value: Any, _seen: set[int] | None = None) -> bool:
    """True if *value* contains a set/frozenset anywhere (recursively, including
    inside objects). `stable_key_repr` opens an object up only when it holds
    one; any other object is left to pickle.

    Each container or object is looked at once per walk, which is what ends
    the walk on a cyclic graph: a module-level ``logger`` reaches the logging
    manager, whose dict of every logger reaches the manager again. A node
    seen before is either still being walked (its other branches answer for
    it) or was walked and held no set, or the walk would have stopped there.
    There is no depth limit: a set below one would be left to pickle, in the
    order PYTHONHASHSEED picks. A value deeper than the recursion limit
    raises RecursionError, which `stable_key_repr` reports as
    `TooDeepValueError`.
    """
    if _seen is None:
        _seen = set()
    # An exact builtin primitive cannot contain anything, so it cannot contain
    # a set. Without this the fall-through below called ``object_state`` on
    # EVERY element -- which walks ``type(value).__mro__`` looking for
    # ``__slots__`` -- so hashing a 10k-element list of ints made 10k such
    # walks per cache hit. Exact-type test, matching ``CODELESS_PRIMS``'s own
    # contract: a str/int SUBCLASS can carry a ``__dict__`` holding a set and
    # must still be walked.
    if type(value) in CODELESS_PRIMS:
        return False
    if isinstance(value, (set, frozenset)):
        return True
    if id(value) in _seen:
        return False
    if isinstance(value, logging.Logger):
        # Pickled by NAME (`Logger.__reduce__`), so nothing inside it reaches
        # the key -- and walking it means walking every logger in the process,
        # 270 us on each call of any function that reads a module `logger`.
        return False
    if type(value) in _plain_data.numpy_scalar_set():
        return False  # a number; without this, an MRO walk per scalar
    _seen.add(id(value))
    if isinstance(value, dict):
        return any(contains_set(k, _seen) or contains_set(v, _seen) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(contains_set(v, _seen) for v in value)
    obj_state = object_state(value)
    if obj_state and any(contains_set(v, _seen) for v in obj_state.values()):
        return True
    if _holds_native_state(type(value)):
        try:
            parts = _pickled_state(value)
        except Exception:  # noqa: BLE001 - what cannot be reduced fails in the pickle, with its own error
            return False
        return contains_set(parts, _seen)
    return False


#: ``Py_TPFLAGS_IMMUTABLETYPE``: set on a class written in C, never on one
#: a ``class`` statement makes.
_IMMUTABLE_TYPE_FLAG = 1 << 8

#: Types pickled by name, or leaves: nothing inside them is pickled by value.
_NO_NATIVE_STATE = (type, types.FunctionType, types.BuiltinFunctionType, types.ModuleType)


def _holds_native_state(type_: type) -> bool:
    """Can a *type_* instance hold values outside its ``__dict__`` and slots?

    A standard-library class written in C keeps them in C: a
    ``functools.partial``'s arguments, a ``deque``'s items, what a list
    iterator has left. `object_state` does not see them, so a set among them
    was left to pickle, which writes it in the order PYTHONHASHSEED picks,
    and the key changed from process to process. Third-party C types are
    left alone: reducing one can serialise all its data (a tensor's
    storage) on every call.
    """
    try:
        return _NATIVE_STATE[type_]
    except KeyError:
        pass
    except TypeError:  # a class whose metaclass makes it unhashable
        return False
    top = (getattr(type_, "__module__", "") or "").split(".", 1)[0]
    found = (
        not issubclass(type_, _NO_NATIVE_STATE)
        and type_ not in LEAF_TYPES
        and (top == "builtins" or top in sys.stdlib_module_names)
        and any(k.__flags__ & _IMMUTABLE_TYPE_FLAG for k in type_.__mro__[:-1])
    )
    _NATIVE_STATE[type_] = found
    return found


_NATIVE_STATE: dict[type, bool] = {}
