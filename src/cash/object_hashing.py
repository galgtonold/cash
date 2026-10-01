"""Pure functions for hashing arbitrary Python values.

One module for every value hash cash takes, so the decorator and the notebook
cannot drift apart on what makes two values the same:

* ``builtin_hash`` and the per-library hashers under it (pandas, numpy,
  polars, PyArrow, modin, dask) read every byte of a value together with its
  schema -- column names, dtypes, an array's memory layout. The decorator keys
  arguments on them; ``compute_hash`` uses them for every value the notebook
  hashes.
* ``stable_key_repr`` is the canonical form a key pickles a value in: sets and
  dicts in a stable order, every container tagged with its type.
* ``compute_hash`` is the notebook's one value hash: a value's whole
  content, a collection of frames item by item. It is the ``compute_hash_fn``
  seam threaded into ``StatementProcessor`` and ``UpstreamChecker``, what
  ``Restorer`` checks a restored object against, and the digest of a loop
  variable, a call's arguments and the globals a call writes. No value hash
  here samples: a sample decides "unchanged" for an edit outside it.

**Anti-god-class rule (load-bearing):** this module is *pure functions*.
No state, no class, no IPython, no ``Cash`` dependency. If a caller
needs context (e.g. "hash relative to lineage X"), the context stays
in the caller — do not grow this module into a class with options.
"""

from __future__ import annotations

import copyreg
import fractions
import hashlib
import logging
import pickle
import sys
import types
import uuid
from collections.abc import Callable
from typing import Any, NamedTuple

from . import _plain_data
from .sizing import pandas_nbytes
from .value_types import CODELESS_PRIMS, LEAF_TYPES, PARSED_VALUE_TYPES

logger = logging.getLogger(__name__)

#: What a hash that cannot read a value raises. RecursionError: a value nested
#: deeper than pickle or a walk follows, such as a linked list of a few
#: hundred objects, has no content hash either.
_HASH_ERRORS = (TypeError, ValueError, AttributeError, pickle.PicklingError, RecursionError)


# ---------------------------------------------------------------------------
# The canonical form a key pickles a value in
# ---------------------------------------------------------------------------


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
            size = sum(getattr(getattr(value, part, None), "nbytes", 0) for part in _SPARSE_PARTS)
        else:
            # A lazy collection (dask, modin) is keyed by its own token,
            # which costs less than computing it to pickle it.
            return True
    except Exception:  # noqa: BLE001 - a size we cannot read: pickle it whole, as before
        return False
    return isinstance(size, int) and size >= OPEN_UP_BYTES


def builtin_family_of(type_: type) -> str | None:
    """`builtin_hash_family`, remembered per type: asked of every value a
    key walks."""
    try:
        return _FAMILIES[type_]
    except KeyError:
        family = _FAMILIES[type_] = builtin_hash_family(type_)
        return family
    except TypeError:  # a class whose metaclass makes it unhashable
        return builtin_hash_family(type_)


_FAMILIES: dict[type, str | None] = {}


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


# ---------------------------------------------------------------------------
# Content hashers: one per library, shared by the decorator's argument keys and
# the notebook's value hash (`compute_hash`)
# ---------------------------------------------------------------------------


def builtin_hash_family(type_: type) -> str | None:
    """Name the built-in content hasher that claims *type_*, or ``None``.

    Asked of a bare TYPE as well as of a value: ``register_hasher`` needs it
    to tell a user that the hasher they are registering would never run.

    Matched on module PREFIX, so a user's own subclass defined in their own
    module is deliberately not claimed -- registering a hasher for it has
    always worked and still does.
    """
    type_name = getattr(type_, "__name__", "")
    module = getattr(type_, "__module__", "") or ""
    if module.startswith("pandas") and type_name in ("DataFrame", "Series"):
        return "pandas"
    if type_name == "ndarray" and module.startswith("numpy"):
        return "numpy"
    if module.startswith("polars"):
        return "polars"
    if module.startswith("pyarrow"):
        return "pyarrow"
    if module.startswith("modin"):
        return "modin"
    if module.startswith("dask"):
        return "dask"
    if module.startswith("scipy.sparse") and hasattr(type_, "tocsr") and hasattr(type_, "format"):
        return "scipy.sparse"
    return None


def builtin_hash(value: Any) -> str | None:
    """Every byte of *value*, with its schema, for the library types cash knows.

    A hex digest for pandas, numpy, polars, PyArrow, modin and dask values, or
    ``None`` when no built-in hasher claims *value*'s type or hashing it failed
    -- a normal answer: the caller falls through to its own next step.

    Every hasher folds in what the values alone do not say: column and index
    names, dtypes, the memory layout of an array. The same values under two
    dtypes are two different objects to the code reading them.
    """
    family = builtin_hash_family(type(value))
    if family == "pandas":
        return hash_pandas(value)
    if family == "numpy":
        return hash_numpy(value)
    if family == "polars":
        return hash_polars(value)
    if family == "pyarrow":
        return hash_pyarrow(value)
    if family == "modin":
        return hash_modin(value)
    if family == "dask":
        return hash_dask(value)
    if family == "scipy.sparse":
        return hash_sparse(value)
    return None


#: How a key walk reads a frame, array or table with no memo in front:
#: every byte, through `builtin_hash`.
BUILTIN_CONTENT = ContentHashing(builtin_family_of, builtin_hash)


def hash_pandas(value: Any) -> str | None:
    """Hash a pandas DataFrame or Series over values AND schema.

    ``hash_pandas_object`` covers row values + index values but NOT the
    schema labels: column names, ``Series.name``, and index name(s) are
    invisible to it, so ``df.rename(columns=...)`` (or an empty frame of any
    shape) collided with the original and returned its cached result. Fold
    the labels in as a digest prefix.

    The dtypes go in for the same reason, and it is the sharper one: the same
    values under two dtypes are two different objects to the body. A tz-naive
    and a tz-aware series collided, and the tz-aware call was served the naive
    one's ``TypeError: Cannot convert tz-naive timestamps``; so did
    ``int64``/``Int64`` (pd.NA semantics), ``int64``/``int32`` and a
    categorical against an object column.

    The values are hashed column by column and index level by index level
    (`_fold_pandas_values`), each in the form that keeps it exact.
    """
    try:
        import pandas as pd

        index = value.index
        index_dtypes = [_pandas_dtype_key(dt) for dt in getattr(index, "dtypes", [index.dtype])]
        # What the labels and dtypes leave out, and code reads: the column
        # axis's own name (``melt``, ``stack`` and ``reset_index`` name
        # columns after it), an index's ``freq`` (``shift(freq=...)``,
        # ``asfreq``) and ``attrs``.
        axes = f"{list(index.names)!r}:{index_dtypes!r}:{getattr(index, 'freq', None)!r}:"
        if type(value).__name__ == "DataFrame":
            dtypes = [_pandas_dtype_key(dt) for dt in value.dtypes]
            schema = f"{list(value.columns)!r}:{list(value.columns.names)!r}:{dtypes!r}:{axes}"
        else:  # Series
            schema = f"{value.name!r}:{_pandas_dtype_key(value.dtype)!r}:{axes}"
        h = hashlib.sha256(schema.encode("utf-8"))
        if value.attrs:
            h.update(canonical_bytes(value.attrs, BUILTIN_CONTENT))
        _fold_pandas_values(h, value, pd)
        return h.hexdigest()
    except (ImportError, TypeError, ValueError, AttributeError, pickle.PicklingError):
        logger.debug("Failed to hash pandas %s via hash_pandas_object", type(value).__name__)
        return None


def _fold_pandas_values(h: Any, value: Any, pd: Any) -> None:
    """Fold a frame's or series' values, then its index's, into *h*.

    ``hash_pandas_object`` keys an array of Python objects by ``str()`` of
    each: when one value is not a string it stringifies the whole column, so
    ``1`` and ``'1'``, ``True`` and ``'True'``, ``b'a'`` and ``'a'``, or a date
    and its ISO string hashed alike, and ``s * 2`` was served ``2`` for
    ``'11'``. The same held for an index of them. Such an array -- object
    dtype, and pandas' string dtypes, whose values are Python strings -- is
    keyed by its items' pickled form instead (`_object_items_bytes`), which is
    also several times faster than ``hash_pandas_object`` on strings. A
    categorical is keyed by its codes, the categories being in the schema.
    Everything else keeps ``hash_pandas_object``'s per-value hash, which reads
    the raw bits of numbers and dates.
    """
    if type(value).__name__ == "DataFrame":
        rest = []
        for pos, dtype in enumerate(value.dtypes):
            if _pandas_value_route(dtype, pd) is None:
                rest.append(pos)
            else:
                h.update(f"|{pos}|".encode())
                _fold_pandas_array(h, value.iloc[:, pos], pd)
        if rest:
            others = value if len(rest) == value.shape[1] else value.iloc[:, rest]
            h.update(_np_bytes(pd.util.hash_pandas_object(others, index=False)))
    else:
        _fold_pandas_array(h, value, pd)
    index = value.index
    if isinstance(index, pd.MultiIndex):
        # Each level's distinct values, then which one each row holds.
        for level, codes in zip(index.levels, index.codes):
            h.update(b"|level|")
            _fold_pandas_array(h, level, pd)
            h.update(codes.tobytes())
    elif isinstance(index, pd.RangeIndex):
        h.update(f"|range({index.start}, {index.stop}, {index.step})".encode())
    else:
        h.update(b"|index|")
        _fold_pandas_array(h, index, pd)


def _pandas_value_route(dtype: Any, pd: Any) -> str | None:
    """How `_fold_pandas_array` keys values of *dtype*: ``"objects"``,
    ``"codes"``, or None for ``hash_pandas_object``."""
    if isinstance(dtype, pd.CategoricalDtype):
        return "codes"
    if str(dtype) == "object" or isinstance(dtype, pd.StringDtype):
        return "objects"
    return None


def _fold_pandas_array(h: Any, values: Any, pd: Any) -> None:
    """Fold one Series' or Index's values into *h* (`_fold_pandas_values`)."""
    route = _pandas_value_route(values.dtype, pd)
    if route == "codes":
        h.update(_np_bytes(values.codes if isinstance(values, pd.Index) else values.cat.codes))
    elif route == "objects":
        h.update(_object_items_bytes(_np_array(values, object).tolist()))
    else:
        h.update(_np_bytes(pd.util.hash_pandas_object(values, index=False)))


def _np_array(values: Any, dtype: Any = None) -> Any:
    """``np.asarray(values)``: a pandas value's array through ``__array__``,
    not ``to_numpy()``, which the decorator's frame memo watches for handles
    a caller could write through (`arg_hashing.watch_writable_handles`)."""
    import numpy as np

    return np.asarray(values, dtype=dtype)


def _np_bytes(values: Any) -> bytes:
    return _np_array(values).tobytes()


def _object_items_bytes(items: list) -> bytes:
    """The bytes a list of Python objects keys on: plain data pickled as it
    is (`_plain_data`, C speed), anything else in its canonical form
    (`stable_key_repr`), so sets and dicts inside are in a stable order."""
    if _plain_data.is_plain(items):
        return b"P" + _plain_data.pickle_unshared(items)
    # The array or column holds each item besides *items*.
    tree = _plain_data.sharing(items, tree=True, held_twice_at=0)
    if tree is not None:
        return b"T" + _plain_data.pickle_unshared((items, tree[0]))
    return b"S" + canonical_bytes(items, BUILTIN_CONTENT)


def _pandas_dtype_key(dtype: Any) -> str:
    """A pandas dtype as a key reads it: ``repr``, and all of a categorical.

    ``str`` is ``'category'`` for every categorical, and ``repr`` elides
    a long list of categories, so the categories (every one, by content) and
    the ``ordered`` flag are spelled out: ``get_dummies``, ``value_counts`` and ``cat.codes``
    read them, and two series of the same values over different categories
    were served each other's columns.
    """
    categories = getattr(dtype, "categories", None)
    if categories is None or getattr(dtype, "name", None) != "category":
        return repr(dtype)
    listed = hashlib.sha256(_object_items_bytes(categories.tolist())).hexdigest()
    return f"category:{dtype.ordered!r}:{_pandas_dtype_key(categories.dtype)}:{listed}"


def array_layout(value: Any) -> str:
    """The order *value*'s axes are laid out in memory: ``C``, ``F`` or ``K…``.

    The key used to fold in the raw strides (0fd2cb5), which separated C-
    from F-ordered arrays -- the point -- but also a strided VIEW from its
    contiguous copy. Those hold the same values in the same memory order,
    so no order-reading callee (``ravel(order='A'/'K')``, ``reshape``) tells
    them apart -- only ``.flags`` does, and a result computed FROM
    contiguity now shares an entry between the two, knowingly. What did
    tell them apart, on every run, was the cache itself. A function that
    returned ``arr[:, 0]`` handed its caller a view on the computing run and
    a contiguous copy on every restored one, so the caller's key changed
    between the two and its expensive step ran twice after every upstream
    edit.

    So: the axes of length > 1, ordered by |stride| from outermost in, with
    a broadcast (zero-stride) axis outermost. Identity is ``C``, reversed is
    ``F``; anything else spells the permutation. Stride MAGNITUDE and sign
    do not change what a memory-order read returns, so they stay out.

    Except for one flag. ``order='A'`` (``ravel``, ``reshape``, ``tobytes``,
    ``copy``) reads in Fortran order only when the array is F-CONTIGUOUS,
    and C order otherwise -- so an F-like strided view (``a.T[::2]``) and
    its F-contiguous copy read differently, and sharing ``F`` handed one the
    other's result. ``Fs`` is the F-like array that is not
    F-contiguous. Nothing else needs the flag: with two or more axes longer
    than 1, only an F-like layout can be F-contiguous, and ``order='A'``
    reads everything else in C order, as ``C`` and ``K…`` already imply.

    One case still re-keys once: an F-like but non-contiguous view is stored
    by pickle as a C-ordered copy, and it genuinely ravels differently from
    one, so the restored value must key apart. Safe direction.
    """
    axes = [(axis, stride) for axis, (n, stride) in enumerate(zip(value.shape, value.strides)) if n > 1]
    if len(axes) <= 1:
        return "C"
    outer_first = sorted(axes, key=lambda a: (-abs(a[1]) if a[1] else float("-inf"), a[0]))
    perm = tuple(axis for axis, _ in outer_first)
    natural = tuple(axis for axis, _ in axes)
    if perm == natural:
        return "C"
    if perm == natural[::-1]:
        return "F" if value.flags.f_contiguous else "Fs"
    return "K" + ",".join(map(str, perm))


def hash_numpy(value: Any) -> str | None:
    """Hash a numpy ndarray over its FULL contents.

    Correctness requires hashing every byte, not a sample: two large arrays
    that differ only outside a sampled window would otherwise collide and
    return a wrong cached result (a silent data-corruption bug, especially for
    the large ML/data arrays caching targets). Shape and dtype are folded in
    so a reshape or retype of the same bytes does not collide. Uses a
    zero-copy ``memoryview`` for contiguous arrays and falls back to
    ``tobytes()`` (C-order copy) otherwise.

    The LAYOUT is folded in too -- the order the axes sit in memory, see
    `array_layout` -- because the C-order fallback above erases it. Without
    it a C-ordered and an F-ordered array holding equal values hash
    identically, and a layout-sensitive callee is served the other one's
    result: measured, ``np.ravel(x, order='A')`` returned ``[0, 1, 2, …]``
    for an F-ordered input whose true answer is ``[0, 4, 8, 1, …]``.
    Normalising to C-order is right for value EQUALITY and wrong for a KEY.
    """
    try:
        h = hashlib.sha256(f"{value.shape}:{value.dtype}:{array_layout(value)}:".encode())
        if getattr(value.dtype, "hasobject", False):
            # object-dtype arrays: the buffer holds raw PyObject *pointers*,
            # not content, so tobytes() hashes memory addresses - identical
            # content in fresh objects never collides (permanent misses,
            # cross-process-unstable) and address reuse could alias distinct
            # content onto one key. Hash the elements' stable representation
            # instead (canonicalising nested sets/dicts so the key is order-
            # and PYTHONHASHSEED-independent).
            h.update(_object_items_bytes(value.tolist()))
            return h.hexdigest()
        try:
            h.update(memoryview(value).cast("B"))  # no copy if C-contiguous
        except (TypeError, ValueError):
            h.update(value.tobytes())  # non-contiguous / odd layout
        return h.hexdigest()
    except (TypeError, ValueError, AttributeError, MemoryError, pickle.PicklingError):
        logger.debug("Failed to hash numpy ndarray")
        return None


def hash_polars(value: Any) -> str | None:
    """Hash a polars DataFrame, Series, or LazyFrame, schema included.

    ``hash_rows()`` and ``hash()`` see the values only: an ``Int32`` and an
    ``Int64`` column holding the same numbers, or a renamed column, would
    collide. The schema -- names and dtypes -- is folded in ahead of them.

    An ``Object`` column is keyed by its items' content
    (`_object_items_bytes`). Polars hashes Object values with Python's
    ``hash()``, which panics on an unhashable one (an ndarray, a dict) and is
    no content key: ``hash(-1) == hash(-2)``, ``1``, ``1.0`` and ``True``
    share a hash, and a string's changes with ``PYTHONHASHSEED``.

    A ``LazyFrame`` is identified by ``serialize()``, never by ``explain()``.
    ``explain()`` renders the human-readable QUERY PLAN, and two frames over
    different in-memory data print identically -- both
    ``pl.DataFrame({"x": [1, 2, 3]}).lazy()`` and the same over
    ``[10, 20, 30]`` are ``DF ["x"]; PROJECT */1 COLUMNS``, so the second
    call was served the first's result. ``serialize()`` carries the plan
    *and* the data the plan closes over, and is byte-identical across
    processes, so persisted entries still hit after a restart. A plan
    ``serialize()`` refuses gets no built-in hash at all.

    KNOWN GAP: a plan that reads from an external source
    (``scan_csv``/``scan_parquet``/...) serializes the PATH, not the file's
    contents, so editing that file in place does not move the key. Closing
    that would mean collecting the frame to build a cache key, which defeats
    the point of a LazyFrame and can be arbitrarily expensive. Collect before
    passing, or name the file with ``file_depends_on=``.
    """
    try:
        import polars as pl

        if isinstance(value, pl.DataFrame):
            h = hashlib.sha256(f"{value.schema}:{value.height}:".encode("utf-8"))
            objects = [name for name, dtype in value.schema.items() if dtype == pl.Object]
            rest = value.drop(objects) if objects else value
            if rest.width:
                h.update(rest.hash_rows().to_numpy().tobytes())
            for name in objects:
                h.update(f"|{name!r}|".encode("utf-8"))
                h.update(_object_items_bytes(value.get_column(name).to_list()))
            return h.hexdigest()
        if isinstance(value, pl.Series):
            h = hashlib.sha256(f"{value.name!r}:{value.dtype}:{len(value)}:".encode("utf-8"))
            if value.dtype == pl.Object:
                h.update(_object_items_bytes(value.to_list()))
            else:
                h.update(value.hash().to_numpy().tobytes())
            return h.hexdigest()
        if isinstance(value, pl.LazyFrame):
            try:
                return hashlib.sha256(value.serialize()).hexdigest()
            except Exception:  # noqa: BLE001 - polars raises its own types
                logger.debug("polars LazyFrame serialize() failed; no built-in hash for it")
                return None
    except (ImportError, TypeError, ValueError, AttributeError):
        logger.debug("Failed to hash polars %s", type(value).__name__)
    except BaseException as exc:
        if not is_native_panic(exc):
            raise
        logger.debug("polars panicked hashing a %s: %s", type(value).__name__, exc)
    return None


def held_objects(value: Any) -> list | None:
    """The Python objects a library value keeps in object storage, other
    than plain leaves; None when it keeps none.

    A numpy ``object`` array, a pandas ``object`` column and a polars
    ``Object`` column hold arbitrary objects -- a model, a function -- where
    no attribute walk reaches them: in the array buffer, in the blocks. Their
    content hashers pickle such an object by reference, so its code is in no
    key unless the code search is handed them here. Strings, numbers and
    dates, which fill most object columns, are left out at C speed.
    """
    family = builtin_family_of(type(value))
    columns: list = []
    try:
        if family == "numpy":
            if getattr(value.dtype, "hasobject", False):
                columns.append(value.ravel(order="K"))
        elif family == "pandas":
            frame = type(value).__name__ == "DataFrame"
            for pos, dtype in enumerate(value.dtypes if frame else [value.dtype]):
                if str(dtype) == "object":
                    columns.append(_np_array(value.iloc[:, pos] if frame else value, object))
        elif family == "polars":
            import polars as pl

            if isinstance(value, pl.DataFrame):
                columns.extend(value.get_column(n).to_list() for n, dt in value.schema.items() if dt == pl.Object)
            elif isinstance(value, pl.Series) and value.dtype == pl.Object:
                columns.append(value.to_list())
    except Exception:  # a library's internals changed: nothing found
        logger.debug("Could not list the objects a %s holds", type(value).__name__, exc_info=True)
        return None
    found: list = []
    for column in columns:
        if not all(k in LEAF_TYPES for k in set(map(type, column))):
            found.extend(v for v in column if type(v) not in LEAF_TYPES)
    return found or None


def is_native_panic(exc: BaseException) -> bool:
    """Is *exc* a panic raised out of a Rust extension (``pyo3``), such as
    polars'? It derives from ``BaseException``, so no ``except Exception``
    stops it, and a panic while building a key ended the user's call."""
    return type(exc).__name__ == "PanicException"


class _HashSink:
    """A write-only file that feeds what is written to it into a hash."""

    closed = False

    def __init__(self, h: Any) -> None:
        self._h = h

    def write(self, data: Any) -> int:
        self._h.update(data)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def hash_pyarrow(value: Any) -> str | None:
    """Hash a PyArrow Table or RecordBatch by what it holds, not its buffers.

    The table is written as an Arrow IPC stream into the hash, never into
    memory. The raw buffers are not the content: a slice (``t.slice(2, 2)``,
    each batch of ``to_batches()``) shares its parent's buffers and differs
    only in an offset they do not show, so every equal-length slice of one
    table hashed alike and was served the first one's result. A dictionary
    column's buffers hold only the indices, so ``["red", "blue"]`` and
    ``["cat", "dog"]`` hashed alike too. The IPC stream carries the schema,
    each batch's rows from its offset, and every dictionary.
    """
    try:
        import pyarrow as pa

        if isinstance(value, (pa.Table, pa.RecordBatch)):
            h = hashlib.sha256(f"{type(value).__name__}:{value.num_rows}:".encode())
            sink = pa.PythonFile(_HashSink(h), mode="w")
            with pa.ipc.new_stream(sink, value.schema) as writer:
                writer.write(value)
            return h.hexdigest()
    except (ImportError, TypeError, ValueError, AttributeError, MemoryError, NotImplementedError):
        logger.debug("Failed to hash PyArrow %s", type(value).__name__)
    return None


def hash_modin(value: Any) -> str | None:
    """Hash a modin DataFrame or Series as the pandas one it converts to --
    schema included (`hash_pandas`)."""
    try:
        return hash_pandas(value._to_pandas() if hasattr(value, "_to_pandas") else value)
    except (TypeError, ValueError, AttributeError):
        logger.debug("Failed to hash modin %s", type(value).__name__)
        return None


def hash_dask(value: Any) -> str | None:
    """Hash a dask collection by its task-graph keys, plus its schema.

    The keys carry a data-derived token. The schema is the collection's
    ``_meta``, the empty pandas frame or numpy array that stands for its
    columns and dtypes, hashed as that type is.
    """
    try:
        h = hashlib.sha256(str(value.__dask_keys__()).encode("utf-8"))
        meta = getattr(value, "_meta", None)
        if meta is not None:
            h.update(f":{builtin_hash(meta)}".encode("utf-8"))
        return h.hexdigest()
    except (TypeError, ValueError, AttributeError):
        logger.debug("Failed to hash dask object via __dask_keys__")
        return None


#: The arrays a scipy sparse matrix is, by format: everything its values
#: and their positions are stored in.
_SPARSE_PARTS = ("data", "indices", "indptr", "offsets")


def hash_sparse(value: Any) -> str | None:
    """Hash a scipy sparse matrix or array by its content, format included.

    The type, shape and dtype, then the arrays its format keeps (`_SPARSE_PARTS`,
    and a COO's coordinates), each hashed as an array (`hash_numpy`), so an
    explicit zero, an unsorted index or an ``int32`` against an ``int64``
    index keys apart: code reading ``.data`` or ``.indices`` sees them. A
    DOK or LIL matrix, whose entries live in Python dicts and lists, is keyed
    as the CSR matrix it converts to. Without this every sparse argument --
    a TF-IDF matrix, a one-hot encoding -- ran uncached.
    """
    try:
        t = type(value)
        h = hashlib.sha256(f"{t.__module__}.{t.__qualname__}:{value.shape}:{value.dtype}:{value.format}:".encode())
        if value.format in ("dok", "lil"):
            canon = value.tocsr()
            canon.sort_indices()
            arrays = [(name, getattr(canon, name)) for name in ("data", "indices", "indptr")]
        else:
            arrays = [(name, getattr(value, name)) for name in _SPARSE_PARTS if hasattr(value, name)]
            coords = getattr(value, "coords", None)
            if coords is None and value.format == "coo":
                coords = (value.row, value.col)
            if coords is not None:
                arrays.extend((f"coords{i}", c) for i, c in enumerate(coords))
        for name, array in arrays:
            digest = hash_numpy(array)
            if digest is None:
                return None
            h.update(f"|{name}:{digest}".encode())
        return h.hexdigest()
    except (TypeError, ValueError, AttributeError, MemoryError):
        logger.debug("Failed to hash scipy sparse %s", type(value).__name__)
        return None


# ---------------------------------------------------------------------------
# Content hashes
# ---------------------------------------------------------------------------


_BULKY_TYPE_NAMES = frozenset({"DataFrame", "Series", "ndarray"})


def _is_bulky(value: Any) -> bool:
    """A frame, array or table: hashed on its own rather than pickled with
    the collection that holds it."""
    t = type(value)
    return t.__name__ in _BULKY_TYPE_NAMES or builtin_hash_family(t) is not None


def _hash_collection(obj: Any) -> str:
    """Hash a list/tuple/dict/set/frozenset over every item it holds.

    A few frames in a dict (`blocks = {w: net_returns(orders, w) ...}`) are
    hashed element by element, each element as ``compute_hash`` hashes it
    alone, rather than pickled whole with the dict. Any other collection is
    pickled whole, whatever its size: a hash of its ends and its length gave
    two lists that differ in the middle one hash, and an in-place edit there
    read as no change.
    """
    items = list(obj.items()) if isinstance(obj, dict) else None
    values = [v for _, v in items] if items is not None else (list(obj) if isinstance(obj, (list, tuple)) else [])
    if any(_is_bulky(v) for v in values):
        parts = [f"{type(obj).__name__}:{len(obj)}"]
        if items is not None:
            parts.extend(f"{compute_hash(k)}={compute_hash(v)}" for k, v in items)
        else:
            parts.extend(compute_hash(v) for v in values)
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return hashlib.sha256(pickle.dumps(obj)).hexdigest()


def identity_hash(obj: Any) -> str:
    """``compute_hash``'s tier-3 fallback formula, factored out so a caller can
    recognise when a hash it received IS this fallback (see
    ``is_identity_fallback_hash``) rather than a real content hash.

    Deliberately id-based, not content-based: this is the tier ``compute_hash``
    reaches only once pickling itself has failed, so there is no content
    signal left to hash. It "always succeeds" in the sense that ``id()`` never
    raises -- not in the sense that it reflects the object's content. An
    object hashed this way that is mutated in place produces the SAME hash
    before and after, because ``id()`` does not change under mutation.
    """
    return hashlib.sha256(str(id(obj)).encode("utf-8")).hexdigest()


def is_identity_fallback_hash(obj: Any, hash_value: str) -> bool:
    """True when *hash_value* -- assumed to be ``compute_hash(obj)``'s result
    for THIS *obj* -- is the tier-3 identity fallback rather than a real
    content hash.

    Exists for callers that need to know whether a ``compute_hash`` result
    can be trusted to change when the object's *content* changes -- e.g.
    before/after mutation detection (``hash_args`` in ``notebook/call_effects.py``). Content-hashed
    results reflect the object's data; an identity-hashed result reflects only
    ``id(obj)``, which is invariant across an in-place mutation, so a caller
    diffing two ``compute_hash`` snapshots across a mutation would otherwise
    see no change and wrongly conclude the object was untouched.

    Recomputes ``identity_hash(obj)`` and compares against *hash_value*
    rather than re-deriving "did compute_hash take the fallback path" some
    other way, so this can never drift out of sync with what ``compute_hash``
    actually did -- it asks the same question ``compute_hash`` answered,
    using the same formula, not a parallel guess at it. A genuine content hash
    coincidentally colliding with ``sha256(str(id(obj)))`` is a second-preimage
    event on SHA-256 and not a practical concern.
    """
    return hash_value == identity_hash(obj)


_COLLECTIONS = frozenset((list, tuple, dict, set, frozenset))


def compute_hash(obj: Any) -> str:
    """Hash *obj* over its whole content, with explicit fallbacks.

    Strategy order:
    1. A frame, array or table through its built-in content hasher
       (`builtin_hash`), every byte and its schema; a collection item by item
       (`_hash_collection`)
    2. Generic pickle hash
    3. Identity hash (always succeeds) -- see ``identity_hash`` /
       ``is_identity_fallback_hash`` for why this tier is content-BLIND, not
       merely a cruder content hash.

    Never a sample. This is the notebook's one value hash: a variable with no
    lineage, a loop variable, a call's arguments and the globals a call
    writes are keyed on it, and every "did this value change?" check (a
    restored input, a loop's mutated variables) compares two of its digests.
    A digest of a frame's first rows or a list's ends let an edit elsewhere
    read as no change.

    A library value goes through ``builtin_hash``, the hasher the decorator
    keys arguments on, so it carries the value's schema as well: an ``int64``
    and an ``Int64`` column, a tz-naive and a tz-aware one, or a C- and an
    F-ordered array holding equal values hash apart.
    """
    type_name = type(obj).__name__

    try:
        digest = builtin_hash(obj)
        if digest is not None:
            return digest
        if isinstance(obj, tuple) and isinstance(getattr(type(obj), "_fields", None), tuple):
            # A namedtuple is its name, its fields and its values. Pickling it
            # pickles its CLASS by reference, which fails for a class made on
            # the spot -- as `df.itertuples()` makes one per call -- and the
            # identity tier below then keyed every row on `id(row)`, new on
            # every run: a loop over itertuples() never restored.
            fields = type(obj)._fields
            return hashlib.sha256(
                f"namedtuple:{type(obj).__name__}:{fields!r}:{_hash_collection(tuple(obj))}".encode("utf-8")
            ).hexdigest()
        if type(obj) in _COLLECTIONS:
            # Exact types: a subclass is pickled whole, with the attributes
            # it holds beside its items.
            return _hash_collection(obj)
        return hashlib.sha256(pickle.dumps(obj)).hexdigest()
    except _HASH_ERRORS as exc:
        logger.debug("Primary hash failed for %s: %s", type_name, exc)
    except BaseException as exc:
        if not is_native_panic(exc):
            raise
        return identity_hash(obj)

    try:
        return hashlib.sha256(pickle.dumps(obj)).hexdigest()
    except (TypeError, pickle.PicklingError, RecursionError):
        pass
    except BaseException as exc:
        if not is_native_panic(exc):
            raise

    return identity_hash(obj)


def mutation_fingerprint(obj: Any) -> str | None:
    """A digest that changes when *obj* is changed IN PLACE, or ``None``.

    Reads the whole value: `builtin_hash` for the library types (pandas,
    numpy, polars, ...), and for an AnnData-like object every part scanpy
    writes to -- the ``obs``/``var`` frames, ``X``, and each value of
    ``uns``/``obsm``/``varm``/``obsp``/``varp``/``layers`` -- each by its own
    content hash. A checksum of ``X`` and the key names alone missed a
    reordering of ``X`` and a column rewritten in place. Taken only around a
    statement that is actually executing, twice, so its O(n) cost is paid
    next to real work.

    ``None`` when the value cannot be observed (it, or a part of it, cannot be
    pickled, so its only hash would be its ``id``, which no in-place change
    moves).
    """
    h = hashlib.sha256()
    t = type(obj)
    h.update(f"{t.__module__}.{t.__qualname__}".encode("utf-8"))
    digest = builtin_hash(obj)
    if digest is not None:
        h.update(digest.encode("utf-8"))
        return h.hexdigest()
    try:
        is_anndata = all(hasattr(obj, a) for a in ("obs", "var", "uns", "X"))
    except Exception:  # noqa: BLE001 - a property that raises: not AnnData-like
        is_anndata = False
    if is_anndata:
        try:
            parts: list[Any] = [repr(getattr(obj, "shape", None))]
            for slot in ("obs", "var", "X", "uns"):
                parts.append(f"{slot}={_part_digest(getattr(obj, slot))}")
            for slot in ("obsm", "varm", "obsp", "varp", "layers"):
                mapping = getattr(obj, slot, None)
                if mapping is None:
                    parts.append(f"{slot}=None")
                    continue
                for key in sorted(mapping.keys(), key=str):
                    parts.append(f"{slot}[{key!r}]={_part_digest(mapping[key])}")
        except (_Unobservable, *_HASH_ERRORS):
            return None
        h.update("|".join(parts).encode("utf-8"))
        return h.hexdigest()
    digest = compute_hash(obj)
    if digest == identity_hash(obj):
        return None
    return digest


class _Unobservable(Exception):
    """A part of a value whose only hash is its identity."""


def _part_digest(value: Any) -> str:
    """`compute_hash` of one part of a value, or `_Unobservable`."""
    if value is None:
        return "None"
    digest = compute_hash(value)
    if digest == identity_hash(value):
        raise _Unobservable(type(value).__name__)
    return digest
