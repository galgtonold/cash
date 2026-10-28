"""`mutation_fingerprint`: a digest that moves when a value is changed in place.

The notebook takes it around a statement that is executing, before and
after, to see which inputs the statement wrote into. It reads the whole
value, by the same content hashes as `compute_hash`, and knows the parts of
an AnnData-like object that scanpy writes to.
"""

from __future__ import annotations

import contextvars
import hashlib
import itertools
import secrets
from collections.abc import Callable
from typing import Any

from .content_hashers import builtin_family_of, builtin_hash_family
from .value_hash import HASH_ERRORS, compute_hash, identity_hash


def mutation_fingerprint(obj: Any, content: Callable[[Any], str] | None = None) -> str | None:
    """A digest that changes when *obj* is changed IN PLACE, or ``None``.

    Reads the whole value: `compute_hash` (through *content* when given --
    `DigestHandoff`, which hands one digest between the checks around a
    statement and its call), and for an AnnData-like object every part
    scanpy writes to -- the ``obs``/``var`` frames, ``X``, and each value
    of ``uns``/``obsm``/``varm``/``obsp``/``varp``/``layers`` -- each by its
    own content hash. A checksum of ``X`` and the key names alone missed a
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
    if t is contextvars.ContextVar:
        # Nothing changes one in place: ``set`` binds another value to it
        # in the context, which is what it is read by.
        h.update(f"context variable|{id(obj.get(_UNSET))}".encode())
        return h.hexdigest()
    held = _sync_state(obj)
    if held is not None:
        # A queue or an event holds a lock, so it pickles to nothing to
        # compare: what ``put``, ``get``, ``set`` change, read instead.
        h.update(f"sync|{held}".encode())
        return h.hexdigest()
    inner = _held_sync_state(obj)
    if inner:
        # The rest of the value may pickle no better: read with what the
        # lock-holding objects it holds hold.
        h.update(f"held sync|{'|'.join(inner)}".encode())
        digest = (content or compute_hash)(obj)
        if digest != identity_hash(obj):
            h.update(digest.encode("utf-8"))
        return h.hexdigest()
    canon = _canonical_sparse(obj)
    if canon is not obj:
        # A copy in canonical form: hashed on its own, never handed on.
        digest = compute_hash(canon)
    elif builtin_family_of(t) is None and _is_anndata_like(obj):
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
        except (_Unobservable, *HASH_ERRORS):
            return None
        h.update(b"anndata|")
        h.update("|".join(parts).encode("utf-8"))
        return h.hexdigest()
    else:
        digest = (content or compute_hash)(obj)
    if digest == identity_hash(canon):
        return None
    h.update(digest.encode("utf-8"))
    return h.hexdigest()


_UNSET = object()


def _sync_state(obj: Any) -> str | None:
    """What a lock-holding object holds, read without its lock-holding
    parts, which pickle to nothing a change moves; None for anything else.

    A ``queue.Queue`` or ``asyncio.Queue`` (its items, by identity, and the
    tasks not done), a ``queue.SimpleQueue`` or ``multiprocessing.Queue``
    (its size), a ``threading.Event`` (whether it is set), a ``Lock`` or
    ``RLock`` (whether it is held, and by whom), a ``Semaphore`` (its
    count) and a ``Condition`` (its lock and its waiters). Also an
    ``itertools.count``, which from Python 3.14 does not pickle at all: its
    ``repr`` (``count(3)``) is its whole state."""
    import queue
    import threading

    try:
        if isinstance(obj, queue.Queue):
            with obj.mutex:
                items = list(obj.queue)
                return f"queue|{obj.unfinished_tasks}|{[id(item) for item in items]}"
        if type(obj) is itertools.count:
            return "count|" + repr(obj)
        if isinstance(obj, queue.SimpleQueue):
            return f"simple queue|{obj.qsize()}"
        if isinstance(obj, threading.Event):
            return f"event|{obj.is_set()}"
        if isinstance(obj, _LOCK_TYPES):
            return f"lock|{obj.locked()}"
        if isinstance(obj, _RLOCK_TYPES):
            # ``<locked _thread.RLock object owner=1 count=2 at 0x...>``
            return "rlock|" + repr(obj).partition(" at 0x")[0]
        if isinstance(obj, threading.Semaphore):
            return f"semaphore|{obj._value}"
        if isinstance(obj, threading.Condition):
            return f"condition|{_sync_state(obj._lock)}|{len(obj._waiters)}"
        kind = _foreign_queue(obj)
        if kind is not None:
            return kind
    except Exception:  # noqa: BLE001 - a subclass that cannot be read
        return None
    return None


_LOCK_TYPES = (type(__import__("threading").Lock()),)
_RLOCK_TYPES = tuple({type(__import__("threading").RLock()), __import__("threading")._PyRLock})


def _foreign_queue(obj: Any) -> str | None:
    """`_sync_state` of an ``asyncio.Queue`` or a ``multiprocessing.Queue``,
    each only when its module is loaded already."""
    import sys

    aio = sys.modules.get("asyncio")
    if aio is not None and isinstance(obj, aio.Queue):
        return f"asyncio queue|{obj._unfinished_tasks}|{[id(item) for item in obj._queue]}"
    mp_queues = sys.modules.get("multiprocessing.queues")
    if mp_queues is not None and isinstance(obj, mp_queues.Queue):
        try:
            return f"process queue|{obj.qsize()}"
        except NotImplementedError:
            # macOS has no semaphore count: read as changed every time.
            return f"process queue|unreadable|{secrets.token_hex(8)}"
    return None


#: How deep `_held_sync_state` looks into plain containers and objects,
#: and how many items it looks at in all: it runs on every fingerprint.
_SYNC_DEPTH = 3
_SYNC_ITEMS = 256


def _held_sync_state(obj: Any, depth: int = _SYNC_DEPTH, work: list[int] | None = None) -> list[str]:
    """`_sync_state` of each lock-holding object *obj* holds in a plain
    dict, list, tuple or set, or in the ``__dict__`` of an object or class
    (``H = {'e': Event()}``, ``B.q = Queue()``), with where it is held."""
    work = [_SYNC_ITEMS] if work is None else work
    found: list[str] = []
    for path, item in _held_items(obj, work):
        state = _sync_state(item)
        if state is not None:
            found.append(f"{path}={state}")
        elif depth > 1 and work[0] > 0:
            found.extend(f"{path}.{inner}" for inner in _held_sync_state(item, depth - 1, work))
    return found


def _held_items(obj: Any, work: list[int]) -> list[tuple[str, Any]]:
    """``(where, item)`` of what *obj* holds, as many as *work* has left."""
    t = type(obj)
    try:
        if t is dict:
            items = [(repr(k), v) for k, v in itertools.islice(obj.items(), work[0])]
        elif t in (list, tuple):
            items = [(str(i), v) for i, v in enumerate(itertools.islice(obj, work[0]))]
        elif t in (set, frozenset):
            items = [("item", v) for v in itertools.islice(obj, work[0])]
        elif builtin_family_of(t) is not None or isinstance(obj, _MODULE_TYPE):
            return []
        else:
            own = obj.__dict__ if isinstance(obj, type) else vars(obj)
            items = [(str(k), v) for k, v in itertools.islice(own.items(), work[0]) if not str(k).startswith("__")]
    except Exception:  # noqa: BLE001 - no __dict__, or one that cannot be read
        return []
    work[0] -= len(items)
    return items


_MODULE_TYPE = type(contextvars)


def sync_state(value: Any) -> str | None:
    """What *value*, a lock-holding object or a value holding some, holds in
    them (`_sync_state`, `_held_sync_state`): what its content hash, which
    pickles them to nothing, cannot see change. None when it holds none."""
    own = _sync_state(value)
    if own is not None:
        return own
    held = _held_sync_state(value)
    return "|".join(held) if held else None


def _is_anndata_like(obj: Any) -> bool:
    try:
        return all(hasattr(obj, a) for a in ("obs", "var", "uns", "X"))
    except Exception:  # noqa: BLE001 - a property that raises: not AnnData-like
        return False


class _Unobservable(Exception):
    """A part of a value whose only hash is its identity."""


def _part_digest(value: Any) -> str:
    """`compute_hash` of one part of a value, or `_Unobservable`."""
    if value is None:
        return "None"
    value = _canonical_sparse(value)
    digest = compute_hash(value)
    if digest == identity_hash(value):
        raise _Unobservable(type(value).__name__)
    return digest


def _canonical_sparse(value: Any) -> Any:
    """A scipy sparse matrix in canonical form (sorted indices, duplicates
    summed), else *value* unchanged.

    scipy canonicalizes a CSR/CSC/BSR/COO matrix IN PLACE when it is merely
    read: ``m.sum()``, ``m @ v`` and friends sort its indices and set
    ``has_canonical_format``. Its values are unchanged, but `hash_sparse`
    (rightly, for a cache key) tells an unsorted index from a sorted one, so
    ``s = heavy(X)`` with a TF-IDF ``X`` was seen changing ``X`` and ran
    every time. The fingerprint compares the canonical form; a copy is made
    only when the matrix is not canonical already.
    """
    if builtin_hash_family(type(value)) != "scipy.sparse" or not hasattr(value, "sum_duplicates"):
        return value
    try:
        if getattr(value, "has_canonical_format", True):
            return value
        canon = value.copy()
        canon.sum_duplicates()
        return canon
    except (TypeError, ValueError, AttributeError, MemoryError):
        return value
