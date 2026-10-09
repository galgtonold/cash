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
import datetime
import decimal
import fractions
import hashlib
import logging
import operator
import sys
import types
import uuid
import weakref
from collections.abc import Callable, Generator
from itertools import chain
from typing import Any, NamedTuple

from . import _plain_data, kept_state
from .sizing import SPARSE_PARTS, pandas_nbytes
from .value_types import CODELESS_PRIMS, IMMUTABLE_PRIMS, LEAF_TYPES, PARSED_VALUE_TYPES


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


def _typed(value: Any, canon: Any, walk: _Walk) -> Generator[Any, Any, tuple]:
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
    *walk* (each value yielded to `_walk`, see `_object_steps`), so a list it
    shares with the items is marked as shared and a loop back to the
    container is caught. Nothing is caught
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
        walked = []
        for k, v in own.items():
            walked.append((k, (yield v)))
        state += tuple(sorted(walked))
    tag = f"{t.__module__}.{t.__qualname__}"
    return ("__cash_type__", tag, canon, state) if state else ("__cash_type__", tag, canon)


def stable_key_repr(
    value: Any,
    content: ContentHashing,
    *,
    seen: dict | None = None,
    hook: Callable[[Any], Any] | None = None,
    left: list | None = None,
    fielded: list | None = None,
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
    graphs key alike, which would be a wrong answer. There is no depth
    limit: a part cut off at a fixed depth and left to pickle keeps its sets
    in the order PYTHONHASHSEED picks, and the key changes from process to
    process. The walk keeps its path on the heap, not the C stack (`_walk`),
    so however deep a value nests it cannot overflow the stack. Only pickling
    the finished form recurses; a form deeper than pickle follows raises
    `TooDeepValueError`, a TypeError too, and the whole value is refused.

    *seen*, when given, is filled with the writable containers the walk met.
    *left*, when given, gets an entry for each object left to pickle whole
    (`canonical_bytes` reads it): only then can the form hold an object
    pickled twice, whose identity only pickle's memo records.
    *fielded*, when given, lets an object of a plain class be keyed by its
    fields (`_fields_form`) and gets the id of each one that is; the caller
    must then check that none of them is held twice (`canonical_bytes`).
    """
    try:
        return _walk(value, _Walk(content, {} if seen is None else seen, hook, left, fielded))
    except RecursionError:
        raise TooDeepValueError(_too_deep(value)) from None


class TooDeepValueError(TypeError):
    """A value nested deeper than pickle follows (Python's recursion limit,
    or the C recursion limit on 3.12 and later), such as a long linked list.
    A TypeError, so the value is reported as unhashable and the call runs
    uncached: a key over part of it could serve one value's result for
    another."""


def _too_deep(value: Any) -> str:
    return f"a {type(value).__qualname__} nested too deeply to key"


class _Walk:
    """One `stable_key_repr` walk: the path walked so far (*stack*), the
    containers met so far (*seen*), and the caller's *content*, *hook* and
    *left* (see `stable_key_repr`)."""

    __slots__ = ("content", "fielded", "holds_set", "hook", "left", "seen", "stack")

    def __init__(
        self,
        content: ContentHashing,
        seen: dict,
        hook: Callable[[Any], Any] | None,
        left: list | None,
        fielded: list | None = None,
    ) -> None:
        self.content = content
        self.seen = seen
        self.hook = hook
        self.left = left
        #: The ids of the objects keyed by their fields (`_fields_form`),
        #: None when none may be.
        self.fielded = fielded
        self.stack: set[int] = set()
        #: Objects known to hold a set (`contains_set`'s *found*).
        self.holds_set: dict[int, Any] = {}


def _walk(value: Any, walk: _Walk) -> Any:
    """`stable_key_repr` of one value, as part of *walk*.

    Iterative: each container or object being walked is a generator
    (`_object_steps`) that yields the parts it needs walked and is sent back
    their forms, and this loop keeps those generators on a list, so the C
    stack the walk takes does not grow with the value's depth. A recursive
    walk takes C stack for every level -- on Python 3.10 every Python call
    does -- and a value nested a few hundred deep overflows Windows' 2 MB
    main thread stack, killing the process, before Python's recursion limit
    is reached. An exception a part raises is thrown into the generator that
    asked for it, as a recursive call would raise it there, so
    ``_object_steps`` can still catch one.
    """
    entered = _enter(value, walk)
    if type(entered) is not _Descend:
        return entered
    steps = [entered.steps]
    sent: Any = None
    error: BaseException | None = None
    while True:
        current = steps[-1]
        try:
            if error is None:
                part = current.send(sent)
            else:
                part = current.throw(error)
        except StopIteration as done:
            steps.pop()
            if not steps:
                return done.value
            sent, error = done.value, None
            continue
        except BaseException as exc:
            steps.pop()
            if not steps:
                raise
            sent, error = None, exc
            continue
        error = None
        try:
            entered = _enter(part, walk)
        except BaseException as exc:  # noqa: BLE001 - raised in the step that asked for the part, as a call would
            sent, error = None, exc
            continue
        if type(entered) is _Descend:
            steps.append(entered.steps)
            sent = None
        else:
            sent = entered


class _Descend:
    """`_enter`'s answer for a value whose form needs its parts walked first."""

    __slots__ = ("steps",)

    def __init__(self, steps: Generator[Any, Any, Any]) -> None:
        self.steps = steps


def _enter(value: Any, walk: _Walk) -> Any:
    """*value*'s form when it needs no walk below it (a primitive, a hooked
    or content-hashed value, a container met before), else a `_Descend`
    holding the generator that walks it."""
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
    return _Descend(_object_steps(value, walk))


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
    fielded: list = []
    walked: dict = {} if seen is None else seen
    form = stable_key_repr(value, content, seen=walked, hook=hook, left=left, fielded=fielded)
    if fielded and (left or len(set(fielded)) != len(fielded) or not walked.keys().isdisjoint(fielded)):
        # An object keyed by its fields, or a list or dict inside one that
        # something else holds too, is one pickle's memo no longer sees:
        # met again -- in another argument, inside an object left to
        # pickle, walked -- it would key like an equal copy. Keyed the old
        # way.
        del form
        left = []
        walked.clear()
        form = stable_key_repr(value, content, seen=walked, hook=hook, left=left)
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


def _object_steps(value: Any, walk: _Walk) -> Generator[Any, Any, Any]:
    """The form of a container or object: a generator `_walk` drives, which
    yields each part to walk and is sent its form. *value* is on
    ``walk.stack`` while its parts are walked, so a part that is *value*
    again is a loop. A primitive part is its own form and is not yielded."""
    walk.stack.add(id(value))
    try:
        prims = _IMMUTABLE_PRIMS
        items: list = []
        if isinstance(value, (set, frozenset)):
            for v in value:
                items.append(v if type(v) in prims else (yield v))
            items.sort(key=_plain_data.content_dumps)
            return (yield from _typed(value, tuple(items), walk))
        if isinstance(value, dict):
            for k, v in value.items():
                k_form = k if type(k) in prims else (yield k)
                items.append((k_form, v if type(v) in prims else (yield v)))
            return (yield from _typed(value, tuple(items), walk))
        if isinstance(value, (list, tuple)):
            if walk.fielded is not None and len(value) >= RECORDS_FROM:
                by_fields = _records_by_fields(value, walk)
                if by_fields is not None:
                    return by_fields
            records = record_class(value, walk.content.family) if len(value) >= RECORDS_FROM else None
            record_form = _record_former(records, walk) if records is not None else None
            for v in value:
                if type(v) in prims:
                    items.append(v)
                elif type(v) is records:
                    items.append(record_form(v))
                else:
                    items.append((yield v))
            return (yield from _typed(value, tuple(items), walk))
        t = type(value)
        if walk.fielded is not None:
            by_fields = _object_by_fields(value, walk)
            if by_fields is not None:
                return by_fields
        if contains_set(value, walk.holds_set):
            return ("__cash_obj__", f"{t.__module__}.{t.__qualname__}", (yield _pickled_state(value)))
        if holds_content_data(value, walk.content):
            # Pickled whole, every frame inside was serialised and hashed on each
            # call. Opened up, each goes through the caller's content hasher
            # (``walk.content``, the decorator's memo), so an unchanged frame is
            # checked rather than read again. An object that reaches itself is
            # left to pickle, which keeps the loop.
            try:
                return ("__cash_obj__", f"{t.__module__}.{t.__qualname__}", (yield _pickled_state(value)))
            except CyclicValueError:
                pass
        left = walk.left
        if left is not None and not (
            type(value) in _VALUE_LEAVES or isinstance(value, _BY_NAME) or type(value) in _plain_data.fake_clock()[0]
        ):
            left.append(value)
        return value
    finally:
        walk.stack.discard(id(value))


#: What the fields of records keyed by their fields may hold
#: (`_records_by_fields`): values that cannot change, with no set, frame or
#: code inside -- a ``datetime`` or ``time`` only with no ``tzinfo``
#: (`_CLOCKS`), which is any object. Not a ``bytearray``: one held by two
#: records changes in both, which only pickle's memo recorded.
_FIELD_LEAVES = frozenset((*IMMUTABLE_PRIMS, *PARSED_VALUE_TYPES))
_CLOCKS = frozenset((datetime.datetime, datetime.time))

_STR_ONLY = frozenset((str,))

#: The methods by which a class says what pickle stores for its instances.
_PICKLE_HOOKS = ("__reduce__", "__reduce_ex__", "__getstate__", "__getnewargs__", "__getnewargs_ex__", "__cash_key__")


class _PlainProbe:
    pass


#: The instance size of a class written in Python with no slots.
_PLAIN_SIZE = (_PlainProbe.__basicsize__, _PlainProbe.__itemsize__)


def fields_class(t: type) -> bool:
    """Is *t* a class whose instances pickle as their class and ``__dict__``
    alone, as ``object`` pickles them? Then an instance is keyed by its
    fields (`_fields_form`) and needs no pickle of its own.

    Written in Python with no slots and no C base (`_record_kind`), with no
    metaclass -- remembered per class, as none of it changes -- and, asked
    each time, with no method saying what pickle stores or how to key it
    (`_PICKLE_HOOKS`), of its own or registered with ``copyreg``.
    """
    try:
        fixed = _FIELDS_CLASSES[t]
    except KeyError:
        fixed = _FIELDS_CLASSES[t] = _decide_fields_class(t)
    except TypeError:  # a class whose metaclass makes it unhashable
        return False
    if not fixed or t in copyreg.dispatch_table or kept_state.chooses_its_state(t):
        return False
    return all(getattr(t, name, None) is getattr(object, name, None) for name in _PICKLE_HOOKS)


def _decide_fields_class(t: type) -> bool:
    if type(t) is not type or _record_kind(t) is not _OBJECT_RECORD:
        return False
    return (t.__basicsize__, t.__itemsize__) == _PLAIN_SIZE and bool(t.__dictoffset__)


_FIELDS_CLASSES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _found_by_name(t: type) -> bool:
    """Does pickle find *t* by its module and qualified name, as it must to
    pickle an instance? A class made in a function, or one a cell defined
    again, is not: its instances are unkeyable, as pickled."""
    module = sys.modules.get(t.__module__)
    found: Any = module
    for part in t.__qualname__.split("."):
        found = getattr(found, part, None)
        if found is None:
            return False
    return found is t


def _fields_form(kind: str, t: type, digest: str) -> tuple:
    """The form of objects keyed by their fields: a tuple no value's form
    can be, since every container's form is tagged (`_typed`)."""
    return ("__cash_fields__", kind, f"{t.__module__}.{t.__qualname__}", digest)


def _records_by_fields(value: list | tuple, walk: _Walk) -> Any:
    """The form of a list or tuple of records of one class keyed by their
    fields, or None to walk it as before.

    A million records cost about 5 us each a hit: each one asked of the
    hook, left to pickle whole -- an object's reduce in Python per record --
    and pickled with a memo entry per object. Every record's ``__dict__``
    holding values alone (`_FIELD_LEAVES`) under string names, the
    records are their class and those dicts, pickled at C speed without a
    memo. That no record is held twice -- in *value* or anywhere else in
    the key -- is the caller's check (*walk*'s ``fielded``).
    """
    if type(value) not in (list, tuple):
        return None
    kinds = set(map(type, value))
    t = kinds.pop()
    if kinds or walk.content.family(t) is not None or not fields_class(t):
        return None
    if walk.hook is not None and walk.hook(value[0]) is not NOT_HOOKED:
        return None  # the hook answers by type: it keys every record
    if not _found_by_name(t):
        return None
    states = list(map(_STATE_OF, value))
    if set(map(type, states)) != {dict} or not _STR_ONLY.issuperset(map(type, chain.from_iterable(states))):
        return None
    kinds = set(map(type, chain.from_iterable(map(dict.values, states))))
    if not _FIELD_LEAVES.issuperset(kinds):
        return None
    if not kinds.isdisjoint(_CLOCKS) and any(
        type(v) in _CLOCKS and v.tzinfo is not None for v in chain.from_iterable(map(dict.values, states))
    ):
        return None
    digest = hashlib.sha256(_plain_data.pickle_unshared(states)).hexdigest()
    walk.fielded.extend(map(id, value))  # type: ignore[union-attr]
    return ("__cash_type__", _BUILTIN_CONTAINER_TAGS[type(value)], (_fields_form("records", t, digest),))


def _object_by_fields(value: Any, walk: _Walk) -> Any:
    """The form of one object keyed by its fields, or None to key it as
    before: an object of a plain class (`fields_class`) whose ``__dict__``
    holds JSON-like data (`_plain_data.tree_levels`) with a container of
    `RECORDS_FROM` items or more in it.

    ``self.data = records`` keyed a method's every call by a search of all
    the records for a set, then a pickle of the object whole, where the
    same records as an argument are keyed by one read of their content at
    C speed (`_plain_data.sharing`): 0.77 s a hit for 100,000 records. The
    lists and dicts held twice inside are part of that digest, as pickle's
    memo kept them. That nothing outside holds the object too is the
    caller's check (*walk*'s ``fielded``).
    """
    t = type(value)
    if not fields_class(t):
        return None
    state = getattr(value, "__dict__", None)
    if type(state) is not dict or not _STR_ONLY.issuperset(map(type, state)):
        return None
    if not any(type(v) in _plain_data.TREE_NODES and len(v) >= RECORDS_FROM for v in state.values()):
        return None  # small: pickled whole costs no more
    if not _found_by_name(t):
        return None
    shape = _plain_data.sharing(state, tree=True)
    if shape is None:
        return None
    digest = hashlib.sha256(_plain_data.pickle_unshared((state, shape[0]))).hexdigest()
    # Its lists and dicts something else holds too: met again in the key --
    # in another object keyed by its fields, or walked -- they are one
    # object two parts of the key share, which pickle's memo recorded.
    walk.fielded.append(id(value))  # type: ignore[union-attr]
    walk.fielded.extend(shape[1])
    del shape
    return _fields_form("object", t, digest)


#: A list or tuple at least this long is checked for records (`record_class`).
RECORDS_FROM = 8

#: What a record object's attributes may hold below their lists and dicts
#: (`record_class`): values with no set, frame or code inside. Not a
#: ``datetime`` or ``time``, whose ``tzinfo`` is any object.
_RECORD_LEAVES = (*CODELESS_PRIMS, datetime.date, datetime.timedelta, decimal.Decimal)

_OBJECT_RECORD = "object"
_NAMEDTUPLE_RECORD = "namedtuple"


def record_class(value: list | tuple, family: Callable[[type], str | None]) -> type | None:
    """The class of the records in *value*, or None.

    A list of records -- parsed rows as dataclasses, plain objects or
    namedtuples -- was walked one object at a time: each searched for a set
    (`contains_set`) and for frames (`holds_content_data`), and by the
    decorator for code its attributes hold. That was 15-25 us an element, 13x
    the same records as dicts. Answered for all of them at once instead:
    every item that is not a primitive is an instance of one class, and what
    they hold is JSON-like data (`_plain_data.is_tree`) -- an object's
    ``__dict__`` over `_RECORD_LEAVES`, a namedtuple's fields over immutable
    primitives -- so none holds a set, a frame or code. Each record's form is
    then what the walk gives it (`_record_former`), and the class is the only
    code they carry. *family* is ``ContentHashing.family``.
    """
    kinds = set(map(type, value))
    kinds.difference_update(CODELESS_PRIMS)
    if len(kinds) != 1:
        return None
    t = kinds.pop()
    kind = _record_kind(t)
    if kind is None or family(t) is not None or t in _plain_data.numpy_scalar_set() or t in _plain_data.fake_clock()[0]:
        return None
    members = value if len(value) == 1 or type(value[0]) is t and len(set(map(type, value))) == 1 else None
    if members is None:
        members = [v for v in value if type(v) is t]
    if kind is _OBJECT_RECORD:
        return t if _plain_data.is_tree(list(map(_STATE_OF, members)), _RECORD_LEAVES) else None
    return t if _plain_data.is_tree(list(map(tuple, members)), IMMUTABLE_PRIMS) else None


def _record_former(records: type, walk: _Walk) -> Callable[[Any], Any]:
    """What makes a record's form (`record_class`): what `_enter` and
    `_object_steps` make of it. A hooked value is its stand-in; an object,
    holding no set and no frame, is left to pickle; a namedtuple is its
    tagged fields."""
    hook, left = walk.hook, walk.left
    if _record_kind(records) is _NAMEDTUPLE_RECORD:
        tag = f"{records.__module__}.{records.__qualname__}"
        prims = _IMMUTABLE_PRIMS

        def form(value: Any) -> Any:
            if hook is not None:
                stand_in = hook(value)
                if stand_in is not NOT_HOOKED:
                    return stand_in
            return ("__cash_type__", tag, tuple([v if type(v) in prims else _tree_form(v, walk) for v in value]))

        return form

    def left_to_pickle(value: Any) -> Any:
        if hook is not None:
            stand_in = hook(value)
            if stand_in is not NOT_HOOKED:
                return stand_in
        if left is not None:
            left.append(value)
        return value

    return left_to_pickle


def _tree_form(value: Any, walk: _Walk) -> Any:
    """The form of an exact dict, list or tuple over immutable primitives, as
    `_enter` and `_object_steps` make it: a list or dict met before in the
    walk is an alias, and is recorded as met."""
    if walk.hook is not None:
        stand_in = walk.hook(value)
        if stand_in is not NOT_HOOKED:
            return stand_in
    t = type(value)
    if t is not tuple:
        seen = walk.seen
        first = seen.get(id(value))
        if first is not None:
            return ("__cash_alias__", first[0])
        seen[id(value)] = (len(seen), value)
    prims = _IMMUTABLE_PRIMS
    if t is dict:
        canon = [(k, v if type(v) in prims else _tree_form(v, walk)) for k, v in value.items()]
    else:
        canon = [v if type(v) in prims else _tree_form(v, walk) for v in value]
    return ("__cash_type__", _BUILTIN_CONTAINER_TAGS[t], tuple(canon))


def _record_kind(t: type) -> str | None:
    """Whether a *t* instance can be a record (`record_class`), and which kind;
    remembered per class."""
    try:
        return _RECORD_KINDS[t]
    except KeyError:
        pass
    except TypeError:  # a class whose metaclass makes it unhashable
        return None
    kind = _RECORD_KINDS[t] = _decide_record_kind(t)
    return kind


def _decide_record_kind(t: type) -> str | None:
    """An object keeps its state in its ``__dict__`` alone, a namedtuple in
    its fields alone (`object_state` finds nothing else, `_typed` adds
    nothing), and neither holds state in C (`_holds_native_state`)."""
    if t in CODELESS_PRIMS or t in _VALUE_LEAVES or issubclass(t, (dict, list, set, frozenset, *_BY_NAME)):
        return None
    if _holds_native_state(t):
        return None
    for klass in t.__mro__:
        slots = klass.__dict__.get("__slots__", ())
        if any(name not in ("__dict__", "__weakref__") for name in ((slots,) if isinstance(slots, str) else slots)):
            return None
    if issubclass(t, tuple):
        if (
            isinstance(getattr(t, "_fields", None), tuple)
            and not t.__dictoffset__
            and not hasattr(t, "default_factory")
        ):
            return _NAMEDTUPLE_RECORD
        return None
    return _OBJECT_RECORD if t.__dictoffset__ else None


_RECORD_KINDS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

_STATE_OF = operator.attrgetter("__dict__")


#: Exact types that are their own form: `_enter` returns them as they are,
#: so `_object_steps` takes them without a step. ``bytearray`` is left to
#: `_enter`, which marks one met twice.
_IMMUTABLE_PRIMS = frozenset(IMMUTABLE_PRIMS)


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
    elif kept_state.chooses_its_state(t):
        parts.append(kept_state.left_out_attrs(value, rest[0] if rest else None))
    return tuple(parts)


def contains_set(value: Any, found: dict[int, Any] | None = None) -> bool:
    """True if *value* contains a set/frozenset anywhere (recursively, including
    inside objects). `stable_key_repr` opens an object up only when it holds
    one; any other object is left to pickle.

    Each container or object is looked at once, which is what ends the
    search on a cyclic graph: a module-level ``logger`` reaches the logging
    manager, whose dict of every logger reaches the manager again. A node
    seen before is either still being searched (its other branches answer
    for it) or was searched and held no set, or the search would have
    stopped there. There is no depth limit: a set below one would be left to
    pickle, in the order PYTHONHASHSEED picks. The nodes still to look at are
    kept on a list, not the C stack, in the order a recursive search takes
    them, so a value nested however deep cannot overflow the stack.

    *found*, when given, is one walk's record of the nodes known to hold a
    set: every node on the path to a set found is added, and a node in it
    answers at once. The walk asks again of each object it opens below one
    that holds a set, and without the record a linked list with a set at
    its end was searched to the end once per node.
    """
    seen: set[int] = set()
    # Every node looked at is held until the search ends, so the id of one
    # made on the way (the parts `_pickled_state` builds) is not reused by
    # another and taken for it.
    held: list = []
    path: list = []
    todo = [value]
    while todo:
        node = todo.pop()
        kind = type(node)
        if kind is _Leave:
            path.pop()
            continue
        if kind is _NativeParts:
            try:
                todo.append(_pickled_state(node.value))
            except Exception:  # noqa: BLE001 - what cannot be reduced fails in the pickle, with its own error
                pass
            continue
        # An exact builtin primitive cannot contain anything, so it cannot
        # contain a set. Without this the fall-through below called
        # ``object_state`` on EVERY element -- which walks
        # ``type(value).__mro__`` looking for ``__slots__`` -- so hashing a
        # 10k-element list of ints made 10k such walks per cache hit.
        # Exact-type test, matching ``CODELESS_PRIMS``'s own contract: a
        # str/int SUBCLASS can carry a ``__dict__`` holding a set and must
        # still be walked.
        if kind in CODELESS_PRIMS:
            continue
        if isinstance(node, (set, frozenset)) or (found is not None and id(node) in found):
            if found is not None:
                for on_path in path:
                    found[id(on_path)] = on_path
            return True
        if id(node) in seen:
            continue
        if isinstance(node, logging.Logger):
            # Pickled by NAME (`Logger.__reduce__`), so nothing inside it
            # reaches the key -- and walking it means walking every logger in
            # the process, 270 us on each call of any function that reads a
            # module `logger`.
            continue
        if kind in _plain_data.numpy_scalar_set():
            continue  # a number; without this, an MRO walk per scalar
        seen.add(id(node))
        held.append(node)
        if isinstance(node, dict):
            parts: list = []
            for k, v in node.items():
                parts.append(k)
                parts.append(v)
        elif isinstance(node, (list, tuple)):
            parts = list(node)
        else:
            parts = list(object_state(node).values())
            if _holds_native_state(kind):
                parts.append(_NativeParts(node))
        path.append(node)
        todo.append(_LEAVE)
        parts.reverse()
        todo.extend(parts)
    return False


class _Leave:
    """`contains_set`'s step for leaving a node: its parts are all looked at."""

    __slots__ = ()


_LEAVE = _Leave()


class _NativeParts:
    """`contains_set`'s step for the parts pickle stores for *value* beside
    its ``__dict__`` and slots (`_holds_native_state`), taken after them."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value


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
