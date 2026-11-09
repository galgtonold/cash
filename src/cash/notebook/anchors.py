"""Objects an entry stores as the variable that holds them, not as a copy.

``grid_search.fit(X, y)`` after the cell building ``preprocessing`` was edited
leaves ``grid_search`` holding ``full_pipeline``, which holds the
``cat_pipeline`` the old ``preprocessing`` still holds, and so does every
model built on it earlier in the notebook. Stored with ``grid_search``, a
copy of ``cat_pipeline`` would not be the object those hold, so the fit was
refused, and ran again after every restart. The fit leaves ``full_pipeline``
as it was: the entry stores it as a reference to the variable (an anchor,
`share_group`), and a hit binds the live object there again, which is the
object everything else holds, as in plain Python.

A hit binds an anchor only while the variable's content digest is the one
the entry recorded: in the session that stored it, the very object; after a
restart, the one the notebook built again, when it has the same content.
Anything else is a miss, never a copy.

A numpy view held inside the value whose memory is an anchor's data (a
fitted ``KNeighborsRegressor`` keeps ``y.to_numpy()`` of the labels it was
fitted on) is stored as where it sits there (`MemoryHome`), and a hit builds
it on the live variable's data again, when its bytes there are the ones
stored: a write to the labels in place reaches the restored model, as in
plain Python. Such an anchor is recorded with `MEMORY_HOME` for a digest:
the bytes are checked, not the variable's whole content.

A numpy view bound to one of the statement's inputs and held by the value as
it is (``knn.fit(X_train, y)`` keeps ``X_train = X[:10000]`` itself) is an
anchor too. A view's session hash is its lineage, not its content, so it is
recorded with the digest of its content (`array_digest`), and a hit binds
the live view, which sits on the live ``X`` as the one the run held did,
when its content is the one stored.

Whatever an entry stores beside objects it shares with other variables (its
anchors, or the holders stored with the outputs as one graph) keeps them one
object only through parts pickled by default: ``model = M().fit(cfg)``
keeping ``cfg`` and a ``Snap(cfg)`` whose ``__reduce__`` deep-copies it would
restore a snapshot of a copy. `shared_parts_held_apart` refuses any part cash
cannot look into beside a shared object.
"""

from __future__ import annotations

import abc
import copyreg
import io
import pickle
import sys
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cash import kept_state, value_hash
from cash._user_attrs import attribute_or
from cash.content_hashers import hash_numpy
from cash.notebook.borrowed_memory import MemoryHome, data_arrays, memory_home, rebuild_view
from cash.notebook.shared_objects import check_walk_budget

__all__ = [
    "ARRAY_CONTENT",
    "MEMORY_HOME",
    "AnchorRef",
    "AnchorUnrebuilt",
    "anchor_digests",
    "array_digest",
    "array_memory_held_apart",
    "link_anchors",
    "memory_homes",
    "resolve_anchors",
    "shared_parts_held_apart",
    "with_anchor_refs",
]

#: The digest an entry records for an anchor held only for the memory of
#: the views it builds on it again (`MemoryHome`): a frame's or an array's
#: content has no digest of its own (its lineage stands for it), and the
#: bytes each view sits on are checked instead. Any other cash reads it as
#: a digest that changed: a miss.
MEMORY_HOME = "memory-home"

#: How the digest an entry records for a numpy array anchor checked by its
#: content starts (`array_digest`). Any other cash reads it as a content
#: digest that changed: a miss.
ARRAY_CONTENT = "array-content:"


def array_digest(obj: Any) -> str | None:
    """The digest of the numpy array *obj* as an anchor checked by its
    content: its dtype, its shape and its values (`hash_numpy`), whatever
    its strides. None for anything else (a subclass such as a memmap
    included), and for an array of objects, whose bytes are addresses of
    objects that may change."""
    ndarray = attribute_or(sys.modules.get("numpy"), "ndarray")
    if ndarray is None or type(obj) is not ndarray or obj.dtype.hasobject:
        return None
    digest = hash_numpy(obj)
    return None if digest is None else ARRAY_CONTENT + digest


@dataclass(frozen=True)
class AnchorRef:
    """Where an entry's value held the object of variable *name*."""

    name: str


class AnchorUnrebuilt(Exception):
    """The copy `with_anchor_refs` makes could not be rebuilt around the
    reference to variable *name*, the last one it loaded: the value held
    that object in a form a reference does not stand for (``results.append``
    is rebuilt as ``getattr(ref, 'append')``). The cause is the error the
    rebuild raised."""

    def __init__(self, name: str | None) -> None:
        super().__init__(name)
        self.name = name


#: Marks a persistent id kept as the object itself (`_RefPickler`).
_KEPT = "kept"


class _RefPickler(pickle.Pickler):
    def __init__(self, file: io.BytesIO, name_of: Callable[[Any], Any], buffers: list) -> None:
        super().__init__(file, pickle.HIGHEST_PROTOCOL, buffer_callback=buffers.append)
        self._name_of = name_of
        self.kept: dict[int, Any] = {}

    def persistent_id(self, obj: Any) -> Any:
        name = self._name_of(obj)
        if name is None and type(obj) is types.FunctionType and kept_state.named_now(obj) is not None:
            # A function whose cell ran again since: pickle refuses it by
            # name. The copy holds the one named now when it is the same
            # definition, else the one the value held (`held_now`).
            self.kept[id(obj)] = kept_state.held_now(obj)
            return (_KEPT, id(obj))
        return name


def _round_trip(value: Any, name_of: Callable[[Any], Any], load: Callable[[Any], Any], *, naming: bool = False) -> Any:
    """A copy of *value*, each object *name_of* names replaced by *load* of
    that name: one pickle round trip, so what the parts share stays shared.
    *naming*: a rebuild that fails raises `AnchorUnrebuilt` for the name
    loaded last; the pickling's own errors are the value's."""
    stream = io.BytesIO()
    buffers: list[pickle.PickleBuffer] = []
    pickler = _RefPickler(stream, name_of, buffers)
    pickler.dump(value)
    unpickler = pickle.Unpickler(
        io.BytesIO(stream.getvalue()),
        buffers=[bytes(b) if memoryview(b).readonly else bytearray(b) for b in buffers],
    )
    kept = pickler.kept
    loaded: list[str] = []

    def persistent_load(pid: Any) -> Any:
        if type(pid) is tuple and pid[0] == _KEPT:
            return kept[pid[1]]
        loaded.append(pid.name if type(pid) is MemoryHome else pid)
        return load(pid)

    unpickler.persistent_load = persistent_load  # type: ignore[assignment]
    if not naming:
        return unpickler.load()
    try:
        return unpickler.load()
    except Exception as exc:
        raise AnchorUnrebuilt(loaded[-1] if loaded else None) from exc


def with_anchor_refs(value: Any, anchors: Mapping[str, Any], homes: Iterable[str] = ()) -> Any:
    """A copy of *value* with an `AnchorRef` in each place it holds one of
    the objects of *anchors* (``{name: object}``), and the `MemoryHome` of
    each numpy view it holds that sits in the data of an anchor *homes*
    names: the statement's inputs, whose lineage its key holds. One view
    held in two places is one `MemoryHome`, so it comes back as one."""

    def load(pid: Any) -> Any:
        return pid if type(pid) is MemoryHome else AnchorRef(pid)

    return _round_trip(value, _namer(anchors, homes), load, naming=True)


def _namer(anchors: Mapping[str, Any], homes: Iterable[str]) -> Callable[[Any], Any]:
    """What `with_anchor_refs` puts in the place of an object: the name of
    the anchor it is, the `MemoryHome` of a view in a home's data, or None
    for an object copied as it is."""
    by_id = {id(obj): name for name, obj in anchors.items()}
    homes = set(homes)
    data = {name: obj for name, obj in anchors.items() if name in homes and data_arrays(obj)}
    ndarray = attribute_or(sys.modules.get("numpy"), "ndarray")
    placed: dict[int, Any] = {}

    def name_of(obj: Any) -> Any:
        name = by_id.get(id(obj))
        if name is None and data and type(obj) is ndarray and obj.base is not None:
            if id(obj) not in placed:
                placed[id(obj)] = memory_home(obj, data)
            return placed[id(obj)]
        return name

    return name_of


#: Values that hold no memory of another object's and come back equal:
#: nothing in them to look into.
_PLAIN = (type(None), bool, int, float, complex, str, bytes, range, slice)
#: `_PLAIN` as a set: a container's plain items are not looked at.
_PLAIN_TYPES = frozenset(_PLAIN)
#: Containers pickling saves item by item, each item as it is.
_CONTAINERS = (list, tuple, dict, set, frozenset)
#: Built-in bases besides ``object`` whose subclasses pickle by default
#: (``object.__reduce_ex__``): the items, the attributes and the slots,
#: each as it is.
_DEFAULT_BASES = (object, list, dict)
#: What pickling calls or looks up on an object, or that changes what it
#: finds there, when its class defines it: refused wherever it is defined
#: but on ``object``, ``list`` or ``dict`` (the exception aside,
#: `_sklearn_pickling`), whether or not the built-in base has it.
_PICKLING = frozenset(
    {
        "__reduce__",
        "__reduce_ex__",
        "__getstate__",
        "__setstate__",
        "__getnewargs__",
        "__getnewargs_ex__",
        "__getattr__",
        "__getattribute__",
        "__setattr__",
        "__class__",
        "__new__",
    }
)
#: Names a class may define although its built-in base (``object``,
#: ``list`` or ``dict``) has them: what a class statement or ``dataclass``
#: puts in the class dict (CPython 3.10 to 3.14), and methods pickling
#: never calls. ``__init__``: default pickling rebuilds through
#: ``cls.__new__`` (``copyreg.__newobj__``), never ``__init__``.
#: ``__init_subclass__``: runs when a class is made, not an object.
#: ``__repr__``, ``__dir__``: never called by a pickler or an unpickler.
#: ``__eq__`` and ``__hash__``: called only on a dict's keys and a set's
#: items, as they are put in again; an object reached so is refused
#: (`_memory_apart`).
_CLASS_DICT = frozenset(
    {
        "__init__",
        "__module__",
        "__qualname__",
        "__doc__",
        "__dict__",
        "__weakref__",
        "__slots__",
        "__annotations__",
        "__annotate__",
        "__annotate_func__",
        "__annotations_cache__",
        "__firstlineno__",
        "__static_attributes__",
        "__classcell__",
        "__orig_bases__",
        "__parameters__",
        "__type_params__",
        "__init_subclass__",
        "__repr__",
        "__dir__",
        "__eq__",
        "__hash__",
    }
)
#: Metaclasses whose classes pickling treats as ``type``'s: neither
#: defines how an attribute of the class is found nor how it is called.
_METACLASSES = (type, abc.ABCMeta)
#: The versions whose pickling code the exceptions were read in.
_SKLEARN_READ = "1.9.1"
_NUMPY_READ = "2.5.4"
_HEAP_TYPE = 1 << 9  # Py_TPFLAGS_HEAPTYPE: a class made by a class statement


def array_memory_held_apart(value: Any, anchors: Mapping[str, Any], homes: Iterable[str] = ()) -> str | None:
    """The name of an anchor that is a numpy array which a restore of
    *value* might not give back everywhere *value* holds its memory; None
    when every object *value* reaches pickles in a way cash knows keeps
    each part it shares the very object it was, so a restore rebuilds
    *value* around the anchors exactly.

    Exact by construction, and fail closed. Only these are passed:

    - the anchors themselves and the views a restore builds on their data
      again (`with_anchor_refs`);
    - numpy arrays (``ndarray`` itself, and ``numpy.ma.MaskedArray``,
      `_masked_array`) whose memory does not overlap an anchor's
      (``numpy.may_share_memory``; one that does would come back as a
      copy), with the items of an array of objects walked;
    - numpy scalars and dtypes, plain values, classes and functions;
    - the built-in containers (exact types), item by item;
    - objects of a class written in Python, over ``object``, ``list`` or
      ``dict`` only, that pickle and unpickle by default
      (`_pickles_by_default`): no class of theirs overrides anything its
      built-in base has, but what a class statement puts there and methods
      pickling never calls, nor defines any pickling of its own, and
      nothing in their ``__dict__`` or ``copyreg.dispatch_table`` changes
      it. Default pickling saves the attributes, the slots and the items
      each as the object it is, and rebuilds them so through the built-in
      methods; they are walked the same way. The one exception is
      scikit-learn's ``BaseEstimator``, whose pickling keeps the
      attributes as they are (`_sklearn_pickling`).

    Anything else -- any pickling of its own, a class written in C (a
    ``torch.Tensor``, a ``cKDTree``, a ``functools.partial``), an error on
    the way, the share check's budget running out (`check_walk_budget`)
    -- names the first array anchor."""
    np = sys.modules.get("numpy")
    if np is None:
        return None
    arrays = {name: obj for name, obj in anchors.items() if type(obj) is np.ndarray}
    if not arrays:
        return None
    try:
        return _memory_apart(value, arrays, _namer(anchors, homes), np)
    except Exception:  # noqa: BLE001 - what cannot be looked into refuses
        return next(iter(arrays))


def _memory_apart(value: Any, arrays: Mapping[str, Any], name_of: Callable[[Any], Any], np: Any) -> str | None:
    """`array_memory_held_apart` for *value*."""
    refused = next(iter(arrays))
    exempt = _sklearn_pickling()
    masked = _masked_array()
    #: Every part met, kept alive while ids of them are compared.
    kept: list[Any] = []
    seen: set[int] = set()
    #: Each part with whether pickling puts it in again as a dict's key or
    #: a set's item, which hashes and compares it.
    stack: list[tuple[Any, bool]] = [(value, False)]
    while stack:
        check_walk_budget()
        obj, keyed = stack.pop()
        if id(obj) in seen:
            if keyed:
                seen.discard(id(obj))  # met again as a key: looked at again
            else:
                continue
        seen.add(id(obj))
        kept.append(obj)
        if name_of(obj) is not None:
            continue
        opened = _open(obj, keyed, np, masked, exempt)
        if opened is None:
            return refused
        parts, memory = opened
        for data in memory:
            for name, array in arrays.items():
                if np.may_share_memory(data, array):
                    return name
        stack.extend(parts)
    return None


#: What `_open` hands back for an object with nothing in it to look into.
_NOTHING: tuple[list[tuple[Any, bool]], list[Any]] = ([], [])


def _open(
    obj: Any, keyed: bool, np: Any, masked: type | None, exempt: Mapping[str, Any]
) -> tuple[list[tuple[Any, bool]], list[Any]] | None:
    """``(parts, memory)`` of *obj* when pickling it by default, or by the
    pickling cash read (`_masked_array`, `_sklearn_pickling`), saves each
    part as the object it is and an unpickler rebuilds it so: the parts,
    each with whether it is put in again as a dict's key or a set's item
    (*keyed*), and the arrays over the memory its data is (an ndarray's own,
    a masked array's data and mask, a bytearray's bytes). None for anything
    else: any pickling of its own, a class written in C."""
    cls = type(obj)
    if cls in _PLAIN:
        return _NOTHING
    if np is not None and (cls is np.ndarray or (masked is not None and cls is masked)):
        if cls is np.ndarray:
            memory = [obj]
            parts = []
        else:
            # What `_masked_array` reads it pickles: its data, its mask and
            # ``_fill_value``, read as they are (``fill_value`` would set
            # one on it).
            memory = [obj.data, np.ma.getmaskarray(obj)]
            parts = [(obj.__dict__.get("_fill_value"), False)]
        if obj.dtype.hasobject:
            parts.extend((item, False) for item in np.asarray(obj).ravel(order="K"))
        return parts, memory
    if cls is bytearray:
        return ([], [np.frombuffer(obj, dtype=np.uint8)]) if np is not None else _NOTHING
    if np is not None and (isinstance(obj, np.dtype) or isinstance(obj, np.generic) and not obj.dtype.hasobject):
        return _NOTHING
    if cls in (type, types.FunctionType, types.BuiltinFunctionType) or isinstance(obj, type):
        return _NOTHING  # pickled by name
    if cls in _CONTAINERS:
        return _parts(obj), []
    if not _pickles_by_default(cls, exempt, keyed):
        return None
    parts = []
    attrs = attribute_or(obj, "__dict__", None)
    if attrs is not None:
        if type(attrs) is not dict or _PICKLING.intersection(attrs):
            return None
        parts.extend((part, False) for part in attrs.values())
    parts.extend((attribute_or(obj, slot, None), False) for slot in _slot_names(cls))
    if isinstance(obj, (list, dict)):
        parts.extend(_parts(obj))
    return parts, []


def shared_parts_held_apart(
    values: Iterable[Any],
    anchors: Mapping[str, Any],
    homes: Iterable[str] = (),
    placed: Iterable[Any] = (),
) -> bool:
    """Whether a restore of *values* (the objects an entry stores in one
    pickle, each where one variable or the echo holds it) might not give
    back an object they share as one object everywhere it was held.

    An object is shared when it is held in two places or more: in two
    places of *values*, an anchor (the variable holds it, `share_group`),
    a view a restore builds on an anchor's data again (`with_anchor_refs`),
    or an object of *values* a live holder is given again (*placed*, the
    targets of its `HolderPatch`). Default pickling saves each part as the
    object it is and rebuilds it so (`_open`), so it keeps a shared object
    one object; anything else may hold it, or its memory, where cash cannot
    see, and copy it apart (a ``__reduce__`` that deep-copies, a tensor
    over an array's memory).

    Exact by construction, and fail closed: True when an object cash cannot
    look into (`_open`: any pickling of its own, a class written in C) is
    met beside a shared object other than itself, when an array met
    overlaps the memory of a shared one (``numpy.may_share_memory``), when
    a *placed* object is not met as one looked into, on an error, or when
    the share check's budget runs out (`check_walk_budget`). A shared
    object cash cannot look into is saved once and found again in the
    pickle's memo in every place: one alone, beside objects that pickle by
    default, comes back as one object."""
    try:
        return _held_apart(list(values), anchors, homes, list(placed))
    except Exception:  # noqa: BLE001 - what cannot be looked into refuses
        return True


def _held_apart(values: list[Any], anchors: Mapping[str, Any], homes: Iterable[str], placed: list[Any]) -> bool:
    """`shared_parts_held_apart`."""
    np = sys.modules.get("numpy")
    exempt = _sklearn_pickling()
    masked = _masked_array()
    name_of = _namer(anchors, homes)
    #: Every object met, by id, kept alive while ids are compared.
    met: dict[int, Any] = {}
    #: How many places of the stored values hold each object.
    places: dict[int, int] = {}
    #: The anchors and the views built on their data again: shared always.
    held: set[int] = set()
    #: Objects with no identity a restore must keep (`_no_identity`).
    plain: set[int] = set()
    #: Objects cash cannot look into (`_open`).
    opaque: set[int] = set()
    #: The objects looked into as a dict's key or a set's item.
    keyed_open: set[int] = set()
    memory: dict[int, list[Any]] = {}
    stack: list[tuple[Any, bool]] = [(value, False) for value in values]
    while stack:
        check_walk_budget()
        obj, keyed = stack.pop()
        key = id(obj)
        places[key] = places.get(key, 0) + 1
        if key in met:
            if keyed and key not in keyed_open and key not in held | plain | opaque:
                # Met again as a key: looked at again as one.
                keyed_open.add(key)
                if _open(obj, True, np, masked, exempt) is None:
                    opaque.add(key)
            continue
        met[key] = obj
        if name_of(obj) is not None:
            held.add(key)
            continue
        if _no_identity(obj, np):
            plain.add(key)
            continue
        opened = _open(obj, keyed, np, masked, exempt)
        if keyed:
            keyed_open.add(key)
        if opened is None:
            opaque.add(key)
            continue
        parts, data = opened
        if data:
            memory[key] = data
        stack.extend(parts)
    for obj in placed:
        if _no_identity(obj, np):
            continue
        if id(obj) not in met:
            return True  # held in a part cash did not look into
        places[id(obj)] += 1
    shared = held | {key for key, count in places.items() if count > 1 and key not in plain}
    if any(shared - {key} for key in opaque):
        return True
    if np is None:
        return False
    spans = [(key, key in shared, data) for key, parts in memory.items() for data in parts]
    spans += [(id(obj), True, obj) for obj in anchors.values() if type(obj) is np.ndarray]
    return _overlap(spans, np)


def _overlap(spans: list[tuple[int, bool, Any]], np: Any) -> bool:
    """Whether two arrays of *spans* ``(owner, shared, array)`` of different
    owners, one of them shared, may share memory: whether their memory
    bounds overlap, as ``numpy.may_share_memory`` asks, swept in the order
    they start instead of pair by pair."""
    bounds = []
    for owner, shared, array in spans:
        if array.size:
            low, high = _byte_bounds(np)(array)
            bounds.append((low, high, owner, shared))
    bounds.sort(key=lambda span: span[0])
    #: The spans started and not yet ended, by where they end.
    active: list[tuple[int, int, bool]] = []
    for low, high, owner, shared in bounds:
        check_walk_budget()
        active = [span for span in active if span[0] > low]
        if any(other != owner and (shared or other_shared) for _end, other, other_shared in active):
            return True
        active.append((high, owner, shared))
    return False


def _byte_bounds(np: Any) -> Callable[[Any], tuple[int, int]]:
    """``numpy.lib.array_utils.byte_bounds`` (``numpy.byte_bounds`` before
    numpy 2): the first byte of an array's memory and the one past its last."""
    utils = attribute_or(attribute_or(np, "lib", None), "array_utils", None)
    found = attribute_or(utils, "byte_bounds", None) or attribute_or(np, "byte_bounds", None)
    if found is None:
        raise RuntimeError("numpy has no byte_bounds")
    return found


def _no_identity(obj: Any, np: Any) -> bool:
    """Whether *obj* has no identity a restore must keep: a plain value, a
    numpy scalar or dtype, a class or a function (pickled by name)."""
    cls = type(obj)
    if cls in _PLAIN or cls in (type, types.FunctionType, types.BuiltinFunctionType) or isinstance(obj, type):
        return True
    return np is not None and (isinstance(obj, np.dtype) or isinstance(obj, np.generic) and not obj.dtype.hasobject)


def _parts(obj: Any) -> list[tuple[Any, bool]]:
    """What *obj*, a built-in container or a subclass of ``list`` or
    ``dict``, holds but its plain values (`_PLAIN`), each with whether it
    is a dict's key or a set's item."""
    if isinstance(obj, dict):
        parts = [(key, True) for key in obj if type(key) not in _PLAIN_TYPES]
        parts += [(item, False) for item in obj.values() if type(item) not in _PLAIN_TYPES]
        return parts
    keyed = isinstance(obj, (set, frozenset))
    return [(item, keyed) for item in obj if type(item) not in _PLAIN_TYPES]


def _slot_names(cls: type) -> list[str]:
    """The slots the classes of *cls* declare, as default pickling saves
    them (``copyreg._slotnames``, without ``__dict__`` and ``__weakref__``)."""
    names: list[str] = []
    for klass in cls.__mro__:
        slots = klass.__dict__.get("__slots__", ())
        for slot in (slots,) if isinstance(slots, str) else slots:
            if slot in ("__dict__", "__weakref__"):
                continue
            if slot.startswith("__") and not slot.endswith("__"):
                slot = f"_{klass.__name__.lstrip('_')}{slot}"
            names.append(slot)
    return names


def _pickles_by_default(cls: type, exempt: Mapping[str, Any], keyed: bool) -> bool:
    """Whether an object of class *cls* pickles as ``object.__reduce_ex__``
    does by default, and an unpickler rebuilds it so: ``cls.__new__`` of
    ``object``, ``list`` or ``dict``, then its ``__dict__`` set item by
    item, its slots through their own descriptors, and a list's items or a
    dict's through the built-in ``extend``/``append`` and ``__setitem__``.

    So every class of *cls* but those three is written in Python, made by
    ``type`` or ``abc.ABCMeta``, and defines no name its built-in base has
    (no override of anything pickling may call) but those of `_CLASS_DICT`,
    nothing of `_PICKLING` (but the methods *exempt* names, on the class
    that defines them), and no ``__slotnames__`` other than its slots;
    ``__eq__`` and ``__hash__`` not when *keyed*. And *cls* has no
    ``copyreg.dispatch_table`` entry."""
    if not cls.__flags__ & _HEAP_TYPE or cls in copyreg.dispatch_table:
        return False
    for klass in cls.__mro__:
        if klass in _DEFAULT_BASES:
            continue
        if not klass.__flags__ & _HEAP_TYPE or type(klass) not in _METACLASSES:
            return False  # over a class written in C, or made by another metaclass
        base = next(b for b in klass.__mro__ if b in _DEFAULT_BASES)
        for name, defined in klass.__dict__.items():
            if exempt.get(name) is defined:
                continue
            if name in _PICKLING or (name in ("__eq__", "__hash__") and keyed):
                return False
            if name not in _CLASS_DICT and any(name in b.__dict__ for b in base.__mro__):
                return False
        if "__slotnames__" in klass.__dict__ and list(klass.__dict__["__slotnames__"]) != _slot_names(klass):
            return False
    for slot in _slot_names(cls):
        # Set through ``setattr``: it must reach the slot's own descriptor.
        found = next((klass.__dict__[slot] for klass in cls.__mro__ if slot in klass.__dict__), None)
        if type(found) is not types.MemberDescriptorType:
            return False
    return True


def _masked_array() -> type | None:
    """``numpy.ma.MaskedArray`` when ``numpy.ma`` is loaded, and numpy is
    the version read (`_NUMPY_READ`; any other: None): an array whose
    pickling cash knows (``GridSearchCV.cv_results_`` holds them).

    In numpy 2.5.4, ``MaskedArray.__reduce__`` hands pickling
    ``_mareconstruct`` with the class, ``_baseclass`` and a dtype code, and
    for state ``__getstate__``: ``ndarray``'s own state of the data (its
    bytes, or for an array of objects the objects as they are) plus the
    mask's bytes and the fill value. ``__setstate__`` sets the new array's
    data and mask from them. So it comes back as a copy of its data, mask
    and fill value, holding the objects of an array of objects as they
    are: walked as an ``ndarray`` is, its mask and fill value with it."""
    if attribute_or(sys.modules.get("numpy"), "__version__", None) != _NUMPY_READ:
        return None
    return attribute_or(sys.modules.get("numpy.ma"), "MaskedArray", None)


def _sklearn_pickling() -> dict[str, Any]:
    """``BaseEstimator.__getstate__`` and ``__setstate__`` when scikit-learn
    is loaded, and is the version read (`_SKLEARN_READ`; any other: none):
    pickling an estimator by default but for them is exact.

    In scikit-learn 1.9.1, ``__getstate__`` raises for an estimator with
    ``__slots__``, takes ``object.__getstate__()`` (or a copy of
    ``__dict__``), and for a class whose module is ``sklearn.*`` returns
    that dict with one key added, ``_sklearn_version``; ``__setstate__``
    pops that key (warning when the version differs) and updates
    ``__dict__`` with the rest (``super().__setstate__``, which ``object``
    lacks). The attributes are saved and rebuilt as the objects they are,
    as by default."""
    if attribute_or(sys.modules.get("sklearn"), "__version__", None) != _SKLEARN_READ:
        return {}
    base = attribute_or(sys.modules.get("sklearn.base"), "BaseEstimator")
    if base is None:
        return {}
    return {name: base.__dict__[name] for name in ("__getstate__", "__setstate__") if name in base.__dict__}


def memory_homes(digests: Mapping[str, str] | None) -> set[str]:
    """The anchors among *digests* held only for the views built on their
    data (`MEMORY_HOME`)."""
    return {name for name, digest in (digests or {}).items() if digest == MEMORY_HOME}


def link_anchors(
    derivation_edges: dict[str, set[str]],
    outputs: Sequence[str],
    links: Iterable[str],
    digests: Mapping[str, str] | None,
) -> None:
    """Link *outputs* with the variables *links* names as the entry's
    anchors link them: both ways, but from a `MEMORY_HOME` anchor to the
    outputs only (`link_memory_homes`)."""
    from cash.notebook.statement.derivation_edges import link_memory_homes, record_shared_object_edges

    homes = memory_homes(digests)
    links = list(links)
    record_shared_object_edges(derivation_edges, outputs, [name for name in links if name not in homes])
    link_memory_homes(derivation_edges, [name for name in links if name in homes], outputs)


def anchor_digests(anchors: Mapping[str, Any], compute_hash: Callable[[Any], str] | None) -> dict[str, str] | None:
    """``{name: content digest}`` for *anchors*; None when one has none a
    restore could check (no content digest, or only its identity's)."""
    digests: dict[str, str] = {}
    hash_fn = compute_hash or value_hash.compute_hash
    for name, obj in anchors.items():
        digest = _digest(obj, hash_fn)
        if digest is None:
            return None
        digests[name] = digest
    return digests


def _digest(obj: Any, hash_fn: Callable[[Any], str]) -> str | None:
    try:
        digest = value_hash.budgeted_hash(obj, hash_fn)
    except Exception:  # noqa: BLE001 - a value that cannot be read has no digest
        return None
    if digest is None or value_hash.is_identity_fallback_hash(obj, digest):
        return None
    return digest


def resolve_anchors(
    value: Any,
    digests: Mapping[str, str] | None,
    user_ns: Mapping[str, Any],
    compute_hash: Callable[[Any], str] | None = None,
    holds: Mapping[str, Sequence[str]] | None = None,
) -> Any:
    """*value* with each `AnchorRef` replaced by the live object of its
    variable. KeyError, before anything is bound, when a variable is gone or
    its content is not what *digests* recorded, or when a variable of
    *holds* (``{name: the anchors it held}``) that is bound holds other
    anchors' objects now: the entry is a miss."""
    if not digests:
        return value
    from cash.notebook.shared_objects import anchors_held

    hash_fn = compute_hash or value_hash.compute_hash
    live: dict[str, Any] = {}
    for name, digest in digests.items():
        if name not in user_ns:
            raise KeyError(f"anchor {name!r} is not bound")
        obj = user_ns[name]
        if digest.startswith(ARRAY_CONTENT):
            if array_digest(obj) != digest:
                raise KeyError(f"anchor {name!r} changed since the entry was stored")
        elif digest != MEMORY_HOME and _digest(obj, hash_fn) != digest:
            raise KeyError(f"anchor {name!r} changed since the entry was stored")
        live[name] = obj
    for name, held in (holds or {}).items():
        # Bound again since (after a restart, or to a new object), it shares
        # the anchors' objects as it did, or the outputs would hold one of
        # its objects where the entry binds the anchor's.
        if name in user_ns and anchors_held(user_ns[name], live) != set(held):
            raise KeyError(f"{name!r} no longer holds the objects of {sorted(held)}")

    def name_of(obj: Any) -> Any:
        if type(obj) is MemoryHome:
            return obj
        return obj.name if type(obj) is AnchorRef else None

    built: dict[int, Any] = {}

    def load(pid: Any) -> Any:
        if type(pid) is MemoryHome:
            if pid.name not in live:
                raise pickle.UnpicklingError(f"unknown anchor {pid.name!r}")
            # One view held in two places is one `MemoryHome` object: built once.
            if id(pid) not in built:
                built[id(pid)] = (pid, rebuild_view(pid, live[pid.name]))  # KeyError: laid out otherwise, or changed
            return built[id(pid)][1]
        if pid not in live:
            raise pickle.UnpicklingError(f"unknown anchor {pid!r}")
        return live[pid]

    return _round_trip(value, name_of, load)
