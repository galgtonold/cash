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
"""

from __future__ import annotations

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
#: Containers pickling saves item by item, each item as it is.
_CONTAINERS = (list, tuple, dict, set, frozenset)
#: Built-in bases besides ``object`` whose subclasses pickle by default
#: (``object.__reduce_ex__``): the items, the attributes and the slots,
#: each as it is.
_DEFAULT_BASES = (object, list, dict)
#: What, defined anywhere but on ``object`` (or on ``list``/``dict`` for
#: ``__getattribute__``), changes how pickling saves or rebuilds an object
#: of the class, or what it finds on it.
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
        "__class__",
    }
)
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
      ``dict`` only, that pickle by default: nothing in their classes
      (`_PICKLING`) or in ``copyreg.dispatch_table`` changes how they are
      saved or rebuilt, and nothing in their ``__dict__`` either. Default
      pickling saves the attributes, the slots and the items each as the
      object it is, and rebuilds them so; they are walked the same way.
      The one exception is scikit-learn's ``BaseEstimator``, whose
      pickling keeps the attributes as they are (`_sklearn_pickling`).

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
    stack: list[Any] = [value]
    while stack:
        check_walk_budget()
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        kept.append(obj)
        cls = type(obj)
        if name_of(obj) is not None or cls in _PLAIN:
            continue
        if cls is np.ndarray or cls is bytearray or cls is masked:
            parts = [obj] if cls is np.ndarray else [np.frombuffer(obj, dtype=np.uint8)] if cls is bytearray else []
            if cls is masked:
                parts = [obj.data, np.ma.getmaskarray(obj)]
                stack.append(obj.fill_value)
            for data in parts:
                for name, array in arrays.items():
                    if np.may_share_memory(data, array):
                        return name
            if cls is not bytearray and obj.dtype.hasobject:
                stack.extend(np.asarray(obj).ravel(order="K"))
            continue
        if isinstance(obj, np.dtype) or isinstance(obj, np.generic) and not obj.dtype.hasobject:
            continue
        if cls in (type, types.FunctionType, types.BuiltinFunctionType) or isinstance(obj, type):
            continue  # pickled by name
        if cls in _CONTAINERS:
            stack.extend(_items(obj))
            continue
        if not _pickles_by_default(cls, exempt):
            return refused
        attrs = attribute_or(obj, "__dict__", None)
        if attrs is not None:
            if type(attrs) is not dict or _PICKLING.intersection(attrs):
                return refused
            stack.extend(attrs.values())
        for slot in _slot_names(cls):
            stack.append(attribute_or(obj, slot, None))
        if isinstance(obj, (list, dict)):
            stack.extend(_items(obj))
    return None


def _items(obj: Any) -> list[Any]:
    """What pickling saves of the built-in container *obj*: its items, or
    its keys and values."""
    if isinstance(obj, dict):
        return [part for pair in obj.items() for part in pair]
    return list(obj)


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


def _pickles_by_default(cls: type, exempt: Mapping[str, Any]) -> bool:
    """Whether an object of class *cls* pickles as ``object.__reduce_ex__``
    does by default: a class written in Python over ``object``, ``list``
    or ``dict``, with nothing of `_PICKLING` defined in its classes but on
    those (and the methods *exempt* names on their own class), and no
    ``copyreg.dispatch_table`` entry."""
    if not cls.__flags__ & _HEAP_TYPE or cls in copyreg.dispatch_table:
        return False
    for klass in cls.__mro__:
        if klass in _DEFAULT_BASES:
            continue
        if not klass.__flags__ & _HEAP_TYPE:
            return False  # over a class written in C: its state is not seen
        for name in _PICKLING.intersection(klass.__dict__):
            if exempt.get(name) is not klass.__dict__[name]:
                return False
    return True


def _masked_array() -> type | None:
    """``numpy.ma.MaskedArray`` when ``numpy.ma`` is loaded: an array whose
    pickling cash knows (``GridSearchCV.cv_results_`` holds them).

    In numpy 2.5.4, ``MaskedArray.__reduce__`` hands pickling
    ``_mareconstruct`` with the class, ``_baseclass`` and a dtype code, and
    for state ``__getstate__``: ``ndarray``'s own state of the data (its
    bytes, or for an array of objects the objects as they are) plus the
    mask's bytes and the fill value. ``__setstate__`` sets the new array's
    data and mask from them. So it comes back as a copy of its data, mask
    and fill value, holding the objects of an array of objects as they
    are: walked as an ``ndarray`` is, its mask and fill value with it."""
    ma = sys.modules.get("numpy.ma")
    return attribute_or(ma, "MaskedArray", None)


def _sklearn_pickling() -> dict[str, Any]:
    """``BaseEstimator.__getstate__`` and ``__setstate__`` when scikit-learn
    is loaded: pickling an estimator by default but for them is exact.

    In scikit-learn 1.9.1, ``__getstate__`` raises for an estimator with
    ``__slots__``, takes ``object.__getstate__()`` (or a copy of
    ``__dict__``), and for a class whose module is ``sklearn.*`` returns
    that dict with one key added, ``_sklearn_version``; ``__setstate__``
    pops that key (warning when the version differs) and updates
    ``__dict__`` with the rest (``super().__setstate__``, which ``object``
    lacks). The attributes are saved and rebuilt as the objects they are,
    as by default."""
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
