"""Putting a stored state back into the live object a computation changed.

``W, H = fit_nmf(X, k)`` where ``fit_nmf`` writes into ``X``: the result alone
is not what the computation did, so it ran every time. It is stored with the
state ``X`` was left in, and a hit hands the result back and writes that
state into the object ``X`` is bound to, which keeps its identity: another
name, a container or a view holding the same object sees the change, as in
a plain run.

Only objects whose whole state a write in place reaches are taken
(`refusal`): an exact NumPy array of a non-object dtype, an exact pandas
``DataFrame`` or ``Series`` whose cells are plain values, and an exact
``list``, ``dict`` or ``set`` of immutable leaves. A class instance, a model,
a memory map, an array of objects, a list of lists (a restore would hand
back copies of the inner lists that something else may hold) run every
time, as before.

A hit writes only into an object of the same shape as the stored state
(`mismatch`): an array of the same shape and dtype that can be written; a
frame or Series with the same index, columns, dtypes and names (a column the
computation added cannot be put back into the frame in place). Everything
is checked before anything is written, so a refused hit leaves the object as
it was and the computation runs on it.

An instance of a plain class is not written into this way: a call that only
binds, rebinds or deletes its attributes (``est.fit(X)`` setting
``self.coef_``) leaves an `AttrState` instead, which a hit replays on the
object handed in (`AttrRecording`, `attribute_refusal`). Each value bound is
stored with the result, or, when it was one of the call's arguments, the
object itself or a module global, named so a hit binds that very object.
"""

from __future__ import annotations

import io
import pickle
import sys
import types
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple

from cash._user_attrs import attribute_or
from cash.value_types import IMMUTABLE_LEAF_TYPES

from . import _weak as cash_weak
from ._native import cwatch

__all__ = [
    "ABSENT",
    "WRITTEN_BACK",
    "ArgumentsLeft",
    "AttrRecording",
    "BoundBefore",
    "AttrState",
    "Relink",
    "attribute_refusal",
    "put_back",
    "FrameState",
    "HeldByCash",
    "KeyedBefore",
    "ResultIsArgument",
    "Shape",
    "before_call",
    "Columns",
    "release_frame",
    "mismatch",
    "refusal",
    "rewritable_kind",
    "rewrite",
    "shape_change",
    "shape_of",
    "snapshot",
]

#: The kind of effect a call's observer records for the arguments whose
#: state after the call its entry stores, and a hit writes back.
WRITTEN_BACK = "argument written back"

#: Containers rewritten by their contents.
_CONTAINERS = (list, dict, set)
_LEAVES = frozenset(IMMUTABLE_LEAF_TYPES)


def _numpy_array(obj: Any) -> bool:
    kind = type(obj)
    return kind.__name__ == "ndarray" and kind.__module__ == "numpy"


def _pandas_kind(obj: Any) -> str | None:
    """``"frame"`` or ``"series"`` for an exact pandas DataFrame or Series."""
    pd = sys.modules.get("pandas")
    if pd is None:
        return None
    kind = type(obj)
    if kind is pd.DataFrame:
        return "frame"
    if kind is pd.Series:
        return "series"
    return None


def rewritable_kind(obj: Any) -> bool:
    """Whether *obj* is of a kind `rewrite` writes into (the checks of
    `refusal` aside)."""
    return type(obj) in _CONTAINERS or _numpy_array(obj) or _pandas_kind(obj) is not None


def _leaves_only(items: Any) -> bool:
    return set(map(type, items)) <= _LEAVES


def _plain_numpy_dtype(dtype: Any) -> bool:
    np = sys.modules.get("numpy")
    return np is not None and isinstance(dtype, np.dtype) and not dtype.hasobject


def _column_refusal(values: Any, dtype: Any, frame: bool) -> str | None:
    """Why a column (or a Series) of *dtype* cannot be rewritten, or None."""
    if _plain_numpy_dtype(dtype):
        return None
    np = sys.modules.get("numpy")
    if np is not None and isinstance(dtype, np.dtype):
        # An object column: its cells are copies after a restore, which is
        # the same only for values nothing changes in place.
        return None if _leaves_only(values) else "a column holds objects that can change in place"
    if frame:
        # An extension column is put back whole, with its dtype.
        return None
    return f"a Series of dtype {dtype} is not put back in place"


def refusal(obj: Any) -> str | None:
    """Why *obj* cannot have a stored state written back into it, or None."""
    if type(obj) in _CONTAINERS:
        items = [*obj.keys(), *obj.values()] if type(obj) is dict else obj
        if not _leaves_only(items):
            return f"a {type(obj).__name__} holding objects that can change in place is not put back in place"
        return None
    if _numpy_array(obj):
        if obj.dtype.hasobject:
            return "an array of objects is not put back in place"
        if not obj.flags.writeable:
            return "a read-only array is not put back in place"
        return None
    kind = _pandas_kind(obj)
    if kind == "frame":
        try:
            for _, column in obj.items():
                why = _column_refusal(column.array, column.dtype, frame=True)
                if why is not None:
                    return why
        except Exception:  # noqa: BLE001 - a frame that cannot be read is not written
            return "the frame could not be read"
        return None
    if kind == "series":
        try:
            return _column_refusal(obj.array, obj.dtype, frame=False)
        except Exception:  # noqa: BLE001 - a Series that cannot be read is not written
            return "the Series could not be read"
    return f"a {type(obj).__name__} is not put back in place, and a new one would not be that object"


def _same_axis(a: Any, b: Any) -> bool:
    """Whether two pandas indexes are the same labels, dtype, names and type."""
    return (
        type(a) is type(b)
        and a.dtype == b.dtype
        and list(a.names) == list(b.names)
        and attribute_or(a, "freq", None) == attribute_or(b, "freq", None)
        and a.equals(b)
    )


class Shape:
    """What of an object a write in place cannot change: an array's shape
    and dtype, a frame's or Series' axes, dtypes, name, attrs and flags
    (`shape_of`). A list, dict or set has none."""

    __slots__ = ("arrays", "kind", "parts")

    def __init__(self, kind: type, parts: tuple) -> None:
        self.kind = kind
        self.parts = parts
        #: A frame's `Columns`, when taken (`before_call`).
        self.arrays: Columns | None = None


def before_call(obj: Any, release: bool = True) -> Shape:
    """*obj*'s `Shape` before a call that may change it, with a frame's
    `Columns`, which `snapshot` compares after it.

    With *release*, a frame's entry in the frame-hash memo is dropped
    first: the memo's shallow copy holds the frame's blocks, and while it
    does, pandas copies a block before any write to it. The call then runs
    on the frame as the caller holds it, and which columns it writes into
    and which it gives a new array is what plain pandas does with that
    frame. Without, a frame the memo holds is noted as such
    (`Columns.cash_held`), and a call that gives one of its columns a new
    array is not stored (`snapshot`): pandas may have copied it only
    because of the memo."""
    if release:
        release_frame(obj)
    shape = shape_of(obj)
    if _pandas_kind(obj) == "frame":
        shape.arrays = Columns.of(obj)
    return shape


def release_frame(obj: Any) -> None:
    """Drop what cash holds of *obj*'s blocks (`FrameDigests.forget`)."""
    if _pandas_kind(obj) != "frame":
        return
    hashing = sys.modules.get("cash.decorator.arg_hashing")
    if hashing is not None:
        hashing.FRAME_DIGESTS.forget(obj)


def forget_moved_lists(values: Iterable[Any]) -> None:
    """Drop the frame memo's entry of each frame among *values* whose
    block of lists was replaced since it was hashed
    (`FrameDigests.forget_moved_lists`)."""
    hashing = sys.modules.get("cash.decorator.arg_hashing")
    if hashing is None:
        return
    for value in values:
        if _pandas_kind(value) == "frame":
            hashing.FRAME_DIGESTS.forget_moved_lists(value)


def _held_by_cash(obj: Any) -> bool:
    hashing = sys.modules.get("cash.decorator.arg_hashing")
    return hashing is not None and id(obj) in hashing.FRAME_DIGESTS


def shape_of(obj: Any) -> Shape:
    """*obj*'s `Shape`. Holds its index objects, which pandas never changes
    in place, and a copy of its attrs."""
    if type(obj) in _CONTAINERS:
        return Shape(type(obj), ())
    if _numpy_array(obj):
        return Shape(type(obj), (obj.shape, obj.dtype))
    flags = obj.flags.allows_duplicate_labels
    if _pandas_kind(obj) == "frame":
        return Shape(type(obj), (obj.columns, obj.index, tuple(obj.dtypes), dict(obj.attrs), flags))
    return Shape(type(obj), (obj.index, obj.dtype, obj.name, dict(obj.attrs), flags))


def _weak(obj: Any) -> Callable[[], Any]:
    """A weak reference to *obj*, or, for an object that takes none, a
    callable that returns it."""
    try:
        return cash_weak.ref(obj)
    except TypeError:
        return lambda: obj


def _root(array: Any) -> Any:
    """What owns an ndarray's memory: the end of its ``base`` chain."""
    np = sys.modules["numpy"]
    while isinstance(array, np.ndarray) and array.base is not None:
        array = array.base
    return array


class Columns(NamedTuple):
    """Which array holds each column of a frame (*tokens*: a NumPy column's
    memory, address and strides, and its row in its block; an extension
    column's array object and the address ranges of its memory), whether
    another pandas object shares each column's block (*shared*, None where
    pandas does not say, False throughout where pandas does not copy on
    write), and, per column, weak references to the array
    object and to what owns its memory (*held*): memory that is gone was
    replaced, whatever array took its address since. Strong ones would
    count as names for the arrays, which the frame-hash memo reads as the
    frame's data being borrowed."""

    tokens: tuple
    shared: tuple
    held: tuple
    #: The frame-hash memo held the frame's blocks (`before_call`).
    cash_held: bool = False

    @classmethod
    def of(cls, frame: Any) -> Columns:
        np = sys.modules["numpy"]
        mgr = frame._mgr
        blocks, blknos, blklocs = mgr.blocks, mgr.blknos, mgr.blklocs
        tokens, shared, held = [], [], []
        copies_on_write = _copies_on_write()
        for position in range(frame.shape[1]):
            block = blocks[blknos[position]]
            values = block.values
            if isinstance(values, np.ndarray):
                tokens.append((values.__array_interface__["data"][0], values.strides, int(blklocs[position])))
                held.append((None, (_weak(_root(values)),)))
            else:
                owners: list = []
                tokens.append((id(values), _memory_of(values, owners)))
                held.append((_weak(values), tuple(_weak(_root(owner)) for owner in owners)))
            if not copies_on_write:
                # pandas writes a block in place however many objects
                # share it: sharing changes nothing a call does.
                shared.append(False)
                continue
            refs = attribute_or(block, "refs", None)
            shared.append(bool(refs.has_reference()) if refs is not None else None)
        return cls(tuple(tokens), tuple(shared), tuple(held), _held_by_cash(frame))


def _copies_on_write() -> bool:
    """Does pandas copy a block other objects share before writing it?
    Always in pandas 3; before, only with copy-on-write turned on."""
    pd = sys.modules["pandas"]
    try:
        return int(pd.__version__.split(".", 1)[0]) >= 3 or pd.options.mode.copy_on_write is True
    except Exception:  # noqa: BLE001 - unknown: taken as copying, so sharing is compared
        return True


def _memory_of(values: Any, owners: list) -> frozenset:
    """The address ranges an extension array's own arrays and buffers
    span: two array objects over the same memory (``fillna(inplace=True)``
    on an ``Int64`` column wraps the very arrays it filled in a new one)
    are one column written into."""
    from cash.argument_sharing import memory_of

    return frozenset(memory_of(values, owners))


def _column_replaced(before: Any, held: tuple, now: Any, values: Any) -> bool | None:
    """Whether a column whose array was *before* (a `Columns` token, with
    its *held* references) was given a new one, now *now*, held in
    *values*: False when it was written into, None when that cannot be
    told (a new array over part of the old memory)."""
    array, owners = held
    if array is not None and array() is values:
        return False  # the same array object, whatever it holds now
    if any(owner() is None for owner in owners):
        return True  # its memory is gone
    if array is None:
        return before != now  # a NumPy column: its memory and place in its block
    if before[1] and before[1] == now[1]:
        return False  # a new array over the very same memory
    if any(lo < hi2 and lo2 < hi for lo, hi in before[1] for lo2, hi2 in now[1]):
        return None
    return True


class HeldByCash(ValueError):
    """A call gave a column of a frame the frame-hash memo held a new array:
    pandas copies a block cash holds before writing it, so whether plain
    pandas would have written into it cannot be told (`snapshot`)."""


class FrameState(NamedTuple):
    """A frame's state after a call, and the *replaced* positions of the
    columns the call gave a new array (``df["a"] = df["a"] + 1``); it wrote
    into the others (``df.loc[0, "a"] = 5``). A hit does the same, so a
    name for a column's array (``h = df["a"].array``) sees the write as it
    does in a plain run, or keeps the old array as it does there.

    *shared* is whether another pandas object shared each column's block
    before the call (`Columns`): pandas copies a shared block before it
    writes it, so which columns a call gives a new array depends on it, and
    a hit is served only on a frame whose columns are shared the same way."""

    frame: Any
    replaced: tuple[int, ...]
    shared: tuple | None = None


def _same(a: Any, b: Any) -> bool:
    pd = sys.modules.get("pandas")
    if pd is not None and isinstance(a, pd.Index):
        return isinstance(b, pd.Index) and _same_axis(a, b)
    if type(a) is tuple:
        return type(b) is tuple and len(a) == len(b) and all(map(_same, a, b))
    try:
        return bool(type(a) is type(b) and a == b)
    except Exception:  # noqa: BLE001 - not comparable: not the same
        return False


def shape_change(before: Shape, after: Any) -> str | None:
    """Why *after* cannot be written into an object of the shape *before*,
    or None: the computation added a column, reshaped an array, changed a
    dtype or the index."""
    if before.kind is not type(after):
        return f"it is bound to a {before.kind.__name__}, not a {type(after).__name__}"
    try:
        now = shape_of(after)
        if not _same(before.parts, now.parts):
            return "its shape, dtypes, index or columns differ from the stored ones"
    except Exception:  # noqa: BLE001 - not comparable: not written
        return "it could not be compared with the stored state"
    return None


def mismatch(live: Any, stored: Any) -> str | None:
    """Why *stored* cannot be written into *live*, or None when `rewrite`
    can write it without failing halfway, and as the call wrote it."""
    if isinstance(stored, AttrState):
        return stored.mismatch(live)
    if isinstance(stored, FrameState):
        if _pandas_kind(live) == "frame":
            release_frame(live)
            try:
                shared = Columns.of(live).shared
            except Exception:  # noqa: BLE001 - its columns cannot be read: not written
                return "its columns could not be read"
            if stored.shared is None or shared != stored.shared:
                return (
                    "another pandas object shares its columns' memory differently than in the call it was "
                    "stored from, so pandas would copy different columns before writing them"
                )
        stored = stored.frame
    if type(live) is not type(stored):
        return f"it is bound to a {type(live).__name__}, not a {type(stored).__name__}"
    why = refusal(live) or refusal(stored)
    if why is not None:
        return why
    try:
        before = shape_of(live)
    except Exception:  # noqa: BLE001 - not comparable: not written
        return "it could not be compared with the stored state"
    return shape_change(before, stored)


def rewrite(live: Any, stored: Any) -> None:
    """Write the state of *stored* into *live* (`mismatch` said it fits)."""
    if isinstance(stored, AttrState):
        stored.replay(live)
        return
    kind = type(live)
    if kind is list:
        live[:] = stored
    elif kind is dict or kind is set:
        live.clear()
        live.update(stored)
    elif _numpy_array(live):
        sys.modules["numpy"].copyto(live, stored, casting="no")
    elif _pandas_kind(live) == "frame":
        release_frame(live)
        replaced = range(stored.shape[1]) if not isinstance(stored, FrameState) else stored.replaced
        frame = stored.frame if isinstance(stored, FrameState) else stored
        np = sys.modules["numpy"]
        for position in range(frame.shape[1]):
            # The stored column's own array, so its dtype comes with it and
            # nothing is aligned on labels: as a new array where the call
            # gave the column one, else written into the very array that
            # holds the column, as the call wrote into it.
            values = frame.iloc[:, position].array.copy()
            if position in replaced:
                live.isetitem(position, values)
                continue
            mgr = live._mgr
            held = mgr.blocks[mgr.blknos[position]].values
            if isinstance(held, np.ndarray):
                held[mgr.blklocs[position]] = np.asarray(values)
            else:
                held[:] = values
    else:
        live.iloc[:] = stored.to_numpy(copy=True)


def snapshot(obj: Any, arrays: Columns | None = None) -> Any:
    """A copy of *obj*'s state that nothing else holds, for an entry to
    keep. A frame's comes as a `FrameState`, whose replaced columns are
    those whose array differs from *arrays*, its `Columns` before the call
    (all of them when that was not taken)."""
    if type(obj) in _CONTAINERS:
        return type(obj)(obj)
    if _pandas_kind(obj) == "frame":
        now = Columns.of(obj).tokens
        before = arrays.tokens if arrays is not None else ()
        mgr = obj._mgr
        verdicts = [
            True
            if position >= len(before)
            or (type(before[position][0]) is int) != (type(token[0]) is int)
            or len(before[position]) != len(token)
            else _column_replaced(
                before[position], arrays.held[position], token, mgr.blocks[mgr.blknos[position]].values
            )
            for position, token in enumerate(now)
        ]
        if None in verdicts:
            raise ValueError("a column was given a new array over part of its old memory")
        if arrays is not None and arrays.cash_held and any(verdicts):
            raise HeldByCash("pandas may have copied a column only because cash's frame-hash memo held it")
        replaced = tuple(position for position, verdict in enumerate(verdicts) if verdict)
        return FrameState(obj.copy(), replaced, arrays.shared if arrays is not None else None)
    return obj.copy()


class KeyedBefore(NamedTuple):
    """The state an argument was in before the call, as the key held it:
    by all it holds, under the key's argument hash *args_hash*. A hit whose
    key holds the argument the same way is on an argument in that state."""

    args_hash: str


class BoundBefore(NamedTuple):
    """The state the instance a cached callable is bound to was in before
    the call (``c.cache(est.fit)``'s ``est``), as the key held it: by all
    it holds (`PurityChecks.bound_by_content`). A hit's key match says the
    bound instance is in that state. *name* is how the call names it."""

    name: str


class ResultIsArgument(NamedTuple):
    """Stored in place of a result that IS an argument the call changed in
    place (``stft(y, out=buf)`` filling and returning ``buf``): the entry
    keeps that argument's state once, and a hit writes it into the argument
    handed in and returns that very object, so ``result is out`` holds as
    in a plain run. *name* is the parameter it was bound to."""

    name: str


class ArgumentsLeft:
    """A decorated call's result, stored with the state it left the
    arguments it changed in place in: ``{parameter: (before, state
    after)}``, *before* a `KeyedBefore` or the argument's fingerprint. A
    hit writes each state into the argument the caller handed in, while
    that argument is in the state *before* names. *result* is a
    `ResultIsArgument` when the result was one of those arguments."""

    __slots__ = ("reads", "result", "sealed", "states")

    def __init__(
        self, result: Any, states: dict[str, tuple[str, Any]], reads: tuple | None = None, sealed: bytes | None = None
    ) -> None:
        self.result = result
        self.states = states
        #: What the code the call ran reads outside its arguments
        #: (`argument_sharing.reads_of`): a hit checks the arguments it
        #: writes share nothing with it. None: not known, so not served.
        self.reads = reads
        #: *result* and *states* pickled with the call's arguments inside
        #: them as references (`seal`), when an attribute bound holds one:
        #: `opened` unpickles them onto the arguments a hit is handed. Then
        #: *result* and *states* are None until opened.
        self.sealed = sealed

    def __reduce__(self) -> tuple:
        if self.sealed is not None:
            return (ArgumentsLeft, (None, None, self.reads, self.sealed))
        pinned: dict[int, tuple[Any, Relink]] = {}
        for _before, state in self.states.values():
            if isinstance(state, AttrState) and state.pinned:
                pinned.update(state.pinned)
        if pinned:
            sealed, _hits = seal((self.result, self.states), pinned)
            return (ArgumentsLeft, (None, None, self.reads, sealed))
        return (ArgumentsLeft, (self.result, self.states, self.reads))

    def opened(self, find: Callable[[Relink], Any]) -> ArgumentsLeft:
        """This entry with its sealed parts unpickled, each argument inside
        them the object *find* gives for its `Relink` (it raises
        `LookupError` for one it cannot find)."""
        if self.sealed is None:
            return self
        result, states = unseal(self.sealed, find)
        return ArgumentsLeft(result, states, self.reads)


def seal(value: Any, pinned: dict[int, tuple[Any, Relink]]) -> tuple[bytes, set[int]]:
    """``(bytes, ids)``: *value*'s store pickle with each object *pinned*
    names (``{id: (object, relink)}``) written as its `Relink`, and the ids
    of those it met. A hit unpickles it onto the objects it is handed
    (`unseal`), so what held an argument holds the caller's own object, as
    a plain run's does: a tree built over an array handed in is built over
    the array the caller hands in, and sees a later write into it."""
    from . import kept_state
    from .backends.serialization import stored_reducers

    met: set[int] = set()

    def persistent_id(obj: Any) -> tuple | None:
        entry = pinned.get(id(obj))
        if entry is None or entry[0] is not obj:
            return None
        met.add(id(obj))
        return tuple(entry[1])

    return kept_state.dumps(value, 5, extra=stored_reducers(), persistent_id=persistent_id), met


def unseal(sealed: bytes, find: Callable[[Relink], Any]) -> Any:
    """What `seal` pickled, each `Relink` in it the object *find* gives."""

    class _Unpickler(pickle.Unpickler):
        def persistent_load(self, pid: Any) -> Any:
            return find(Relink(*pid))

    return _Unpickler(io.BytesIO(sealed)).load()


# ---------------------------------------------------------------- attributes

#: What a recorded key held before the call when it was not there.
ABSENT: Any = cwatch.ABSENT if cwatch is not None else object()

_HEAPTYPE = 1 << 9
_OWN_SLOTS = frozenset({"__dict__", "__weakref__"})


def attribute_refusal(obj: Any) -> str | None:
    """Why a change to *obj*'s attributes cannot be replayed by a hit, or
    None: *obj* is an instance of a plain class -- classes of Python code
    all the way up to ``object``, no ``__slots__``, no ``__setattr__`` or
    ``__delattr__`` of its own -- whose whole state is its instance
    ``__dict__``, an exact dict."""
    kind = type(obj)
    if isinstance(obj, (type, types.ModuleType, types.FunctionType, types.MethodType)):
        return f"a {kind.__name__} is not an instance whose attributes are replayed"
    if kind.__setattr__ is not object.__setattr__ or kind.__delattr__ is not object.__delattr__:
        return f"{kind.__name__} sets or deletes its attributes through its own code"
    for klass in kind.__mro__[:-1]:
        if not klass.__flags__ & _HEAPTYPE:
            return f"{kind.__name__} keeps state outside its __dict__ (it derives from {klass.__name__})"
        for name, held in vars(klass).items():
            if _setter(held) and name not in _OWN_SLOTS:
                return f"{kind.__name__}.{name} runs code when it is set"
        slots = vars(klass).get("__slots__")
        if slots is not None:
            names = (slots,) if isinstance(slots, str) else tuple(slots)
            if any(name not in _OWN_SLOTS for name in names):
                return f"{kind.__name__} keeps attributes in __slots__"
    if not kind.__dictoffset__:
        return f"{kind.__name__} has no instance __dict__"
    try:
        state = object.__getattribute__(obj, "__dict__")
    except AttributeError:
        return f"the {kind.__name__} has no instance __dict__"
    if type(state) is not dict:
        return f"the {kind.__name__}'s __dict__ is not a plain dict"
    return None


def _setter(held: Any) -> bool:
    """Whether *held*, found on a class, runs code when an attribute of
    its name is set or deleted: a property with a setter or deleter, or any
    other descriptor with ``__set__`` or ``__delete__`` but a read-only
    property."""
    if isinstance(held, property):
        if held.fdel is not None:
            return True
        return held.fset is not None and not (type(held) is property and _stores_only(held.fset))
    if isinstance(held, (types.GetSetDescriptorType, types.MemberDescriptorType)):
        return False
    kind = type(held)
    return hasattr(kind, "__set__") or hasattr(kind, "__delete__")


_PLAIN_LOADS = frozenset({"LOAD_FAST", "LOAD_FAST_BORROW", "LOAD_FAST_CHECK"})
_PAIR_LOADS = frozenset({"LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW_LOAD_FAST_BORROW"})
_SKIPPED = frozenset({"RESUME", "NOP", "CACHE", "PUSH_NULL"})


def _stores_only(fset: Any) -> bool:
    """Whether the property setter *fset* provably does nothing but bind
    its value in the instance's ``__dict__``: its body is only
    ``self.<name> = value`` or ``setattr(self, "<name>", value)``
    statements. Such a setter changes nothing a recording of the instance
    ``__dict__`` misses, so a class that has one still replays."""
    import builtins
    import dis

    code = getattr(fset, "__code__", None)
    if (
        type(fset) is not types.FunctionType
        or code is None
        or code.co_argcount != 2
        or code.co_kwonlyargcount
        or code.co_flags & (0x04 | 0x08 | 0x20 | 0x80 | 0x100 | 0x200)
        or code.co_freevars
        or code.co_cellvars
    ):
        return False
    me, value = code.co_varnames[:2]
    steps: list[tuple[str, Any]] = []
    try:
        for ins in dis.get_instructions(code):
            if ins.opname in _SKIPPED:
                continue
            if ins.opname in _PLAIN_LOADS:
                steps.append(("load", ins.argval))
            elif ins.opname in _PAIR_LOADS:
                steps.extend(("load", name) for name in ins.argval)
            else:
                steps.append((ins.opname, ins.argval))
    except Exception:  # noqa: BLE001 -- unknown bytecode: not provable
        return False
    if steps[-1:] == [("RETURN_CONST", None)]:
        steps = steps[:-1]
    elif steps[-2:] == [("LOAD_CONST", None), ("RETURN_VALUE", None)]:
        steps = steps[:-2]
    else:
        return False
    plain_setattr = fset.__globals__.get("setattr", ABSENT) is ABSENT and (
        fset.__builtins__.get("setattr") is builtins.setattr
    )
    at = 0
    while at < len(steps):
        if steps[at : at + 2] == [("load", value), ("load", me)] and (
            at + 2 < len(steps) and steps[at + 2][0] == "STORE_ATTR"
        ):
            at += 3
            continue
        call = steps[at : at + 6]
        if (
            plain_setattr
            and len(call) == 6
            and call[0] == ("LOAD_GLOBAL", "setattr")
            and call[1] == ("load", me)
            and call[2][0] == "LOAD_CONST"
            and type(call[2][1]) is str
            and call[3] == ("load", value)
            and call[4] in (("CALL", 3), ("CALL_FUNCTION", 3))
            and call[5][0] == "POP_TOP"
        ):
            at += 6
            continue
        return False
    return True


def _data_descriptor(kind: type, name: str) -> bool:
    """Whether *kind* has a data descriptor (a property, a slot) named *name*."""
    for klass in kind.__mro__:
        found = vars(klass).get(name, ABSENT)
        if found is not ABSENT:
            held = type(found)
            return hasattr(held, "__set__") or hasattr(held, "__delete__")
    return False


class AttrChange(NamedTuple):
    """What a call did to an object's attributes (`AttrRecording.finish`):
    the object, its ``__dict__``, that dict as it was before the call
    (*before*, in its order), the attributes it bound to a value not there
    before (*bound*, in the dict's order now) and those it deleted."""

    obj: Any
    state: dict
    before: dict
    bound: dict
    deleted: tuple
    #: The keys of its ``__dict__`` after the call, in order.
    order: tuple


class AttrRecording:
    """Which attributes of *obj* a call binds or deletes, recorded across
    the body: by the C dict watcher, which names the keys that changed and
    what each held before (`cash._cwatch.record`), or, without it, by a
    shallow copy of the ``__dict__`` compared after the call. `finish`
    stops it and says what changed."""

    __slots__ = ("change", "copy", "kind", "obj", "recorder", "state", "why")

    def __init__(self, obj: Any) -> None:
        self.obj = obj
        self.kind = type(obj)
        self.state = object.__getattribute__(obj, "__dict__")
        self.recorder = cwatch.record(self.state) if cwatch is not None else None
        self.copy = None if self.recorder is not None else dict(self.state)
        #: Set by `finish`: the change, None when there is none, and why
        #: it cannot be replayed (*why*) when it cannot.
        self.change: AttrChange | None = None
        self.why: str | None = None

    def finish(self) -> None:
        """Stop recording and keep what changed (`change`, `why`)."""
        recorder, self.recorder = self.recorder, None
        state = self.state
        if recorder is not None:
            keys, held, flags = recorder.take()
            if flags & cwatch.REC_THREAD:
                self.why = "another thread changed its attributes while the call ran"
                return
            if flags:
                self.why = "its __dict__ was cleared or replaced whole, or given a key that is not a str"
                return
            if keys is None:
                return
            before = {key: held[key] if key in held else state[key] for key in keys}
            touched = held
        else:
            before, self.copy = self.copy, None
            touched = {key: before.get(key, ABSENT) for key in (*before, *state)}
        if type(self.obj) is not self.kind or object.__getattribute__(self.obj, "__dict__") is not state:
            self.why = "the call replaced its class or its __dict__"
            return
        deleted: list = []
        for key, was in touched.items():
            now = state.get(key, ABSENT)
            if now is was:
                continue
            if type(key) is not str:
                self.why = "a key of its __dict__ is not a str"
                return
            if _data_descriptor(self.kind, key):
                self.why = f"{self.kind.__name__}.{key} is a property or descriptor, not a plain attribute"
                return
            if now is ABSENT:
                deleted.append(key)
        if not touched:
            return
        bound = {key: value for key, value in state.items() if touched.get(key, value) is not value}
        if not bound and not deleted:
            return
        self.change = AttrChange(self.obj, state, before, bound, tuple(deleted), tuple(state))


class put_back:  # a context manager, named like contextlib's
    """While open, each object of *changes* (`AttrChange`) holds the
    attributes it had before its call, in their order; on the way out, those
    it holds now. For a check that must see the arguments as the key saw
    them (`CallRunner.attribute_changes`)."""

    __slots__ = ("changes", "saved")

    def __init__(self, changes: Iterable[AttrChange]) -> None:
        self.changes = list(changes)
        self.saved: list[tuple[dict, dict]] = []

    def __enter__(self) -> put_back:
        for change in self.changes:
            now = dict(change.state)
            change.state.clear()
            self.saved.append((change.state, now))
            change.state.update(change.before)
        return self

    def __exit__(self, *exc: object) -> None:
        for state, now in self.saved:
            state.clear()
            state.update(now)
        self.saved = []


class Relink(NamedTuple):
    """In an `AttrState`, a value a hit binds by identity instead of a copy:
    ``("argument", parameter)`` -- one of the call's arguments, or the object
    itself --, ``("attribute", parameter, name)`` -- what that argument holds
    as its attribute *name* on the hit, as it held it before the call --,
    or ``("global", module, name)``."""

    kind: str
    where: tuple


class AttrState:
    """The attributes a call bound on an instance it was handed (*values*,
    a stored value or a `Relink` each), those it deleted, the type the
    instance was and the order its ``__dict__`` was left in. A hit binds and
    deletes them on the instance handed in (`replay`), whose ``__dict__``
    then holds what a plain run leaves, in the same order."""

    __slots__ = ("deleted", "kind", "order", "pinned", "values")

    def __init__(self, kind: type, values: dict, deleted: tuple, order: tuple) -> None:
        self.kind = kind
        self.values = values
        self.deleted = deleted
        self.order = order
        #: ``{id: (argument, relink)}`` of the call's arguments an object in
        #: *values* holds, which the entry pickles as references
        #: (`ArgumentsLeft.__reduce__`); not stored itself.
        self.pinned: dict[int, tuple[Any, Relink]] | None = None

    def __reduce__(self) -> tuple:
        return (AttrState, (self.kind, self.values, self.deleted, self.order))

    def relinks(self) -> list[Relink]:
        return [value for value in self.values.values() if type(value) is Relink]

    def resolved(self, find: Callable[[Relink], Any]) -> AttrState:
        """This state with each `Relink` replaced by the object *find*
        gives for it (it raises `LookupError` for one it cannot find)."""
        values = {key: find(value) if type(value) is Relink else value for key, value in self.values.items()}
        return AttrState(self.kind, values, self.deleted, self.order)

    def mismatch(self, live: Any) -> str | None:
        """Why the attributes cannot be replayed on *live*, or None."""
        if type(live) is not self.kind:
            return f"it is a {type(live).__name__}, not a {self.kind.__name__}"
        why = attribute_refusal(live)
        if why is not None:
            return why
        if any(_data_descriptor(self.kind, key) for key in (*self.values, *self.deleted)):
            return "an attribute it binds is a property or descriptor of its class now"
        state = object.__getattribute__(live, "__dict__")
        if any(key not in state for key in self.deleted):
            return "an attribute the call deleted is not there"
        return None

    def replay(self, live: Any) -> None:
        """Bind and delete the attributes on *live* (`mismatch` said it fits)."""
        state = object.__getattribute__(live, "__dict__")
        for key in self.deleted:
            del state[key]
        state.update(self.values)
        if tuple(state) != self.order and set(state) == set(self.order):
            # Rebound in place, or deleted and bound again: put each key
            # where the plain run left it.
            for key in self.order:
                state[key] = state.pop(key)
