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
"""

from __future__ import annotations

import io
import pickle
import sys
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cash import kept_state, value_hash
from cash.notebook.borrowed_memory import MemoryHome, data_arrays, memory_home, rebuild_view

__all__ = [
    "MEMORY_HOME",
    "AnchorRef",
    "AnchorUnrebuilt",
    "anchor_digests",
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


def with_anchor_refs(value: Any, anchors: Mapping[str, Any]) -> Any:
    """A copy of *value* with an `AnchorRef` in each place it holds one of
    the objects of *anchors* (``{name: object}``), and the `MemoryHome` of
    each numpy view it holds that sits in an anchor's data."""
    by_id = {id(obj): name for name, obj in anchors.items()}
    homes = {name: obj for name, obj in anchors.items() if data_arrays(obj)}
    ndarray = getattr(sys.modules.get("numpy"), "ndarray", None)

    def name_of(obj: Any) -> Any:
        name = by_id.get(id(obj))
        if name is None and homes and type(obj) is ndarray and obj.base is not None:
            return memory_home(obj, homes)
        return name

    def load(pid: Any) -> Any:
        return pid if type(pid) is MemoryHome else AnchorRef(pid)

    return _round_trip(value, name_of, load, naming=True)


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
        if digest != MEMORY_HOME and _digest(obj, hash_fn) != digest:
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

    def load(pid: Any) -> Any:
        if type(pid) is MemoryHome:
            if pid.name not in live:
                raise pickle.UnpicklingError(f"unknown anchor {pid.name!r}")
            return rebuild_view(pid, live[pid.name])  # KeyError: laid out otherwise, or changed
        if pid not in live:
            raise pickle.UnpicklingError(f"unknown anchor {pid!r}")
        return live[pid]

    return _round_trip(value, name_of, load)
