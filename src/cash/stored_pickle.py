"""One pickle of a value for its digest and for its store.

The notebook digests a value it pickles whole (`value_hash.compute_hash`)
over the bytes a store keeps of it: the protocol 5 pickle the file tier
writes, with each buffer of at least ``OUT_OF_BAND_MIN_BYTES`` out of band
(`pickle_value`). A statement that binds a networkx graph pickled it to
digest it after it ran and again, with nothing run in between, to store it:
two pickles of 0.4-0.7 s each for a graph of 350,000 edges. Digested over the
store's own bytes, the store takes the bytes the digest made (`keeping`,
`kept`).

The digest (`object_digest`) is ``SHA-256`` over a head naming this format
and the bulk-digest scheme (`bulk_digest.TAG`), the ``SHA-256`` of the
stream, and each out-of-band buffer folded in by `bulk_digest.fold_buffer`
(its length, and its tree digest when big). The stream is hashed as the
pickler writes it, so a budgeted digest stops as soon as it runs out of time
(`HashBudgetExceeded`) and an unkept digest never holds the whole stream.

What is digested is always the value's keyed pickle (`kept_state.dumps`
with *keyed*), made the way the store pickles: the same protocol, reducers
and out-of-band buffers. For most values those are the very bytes the store
keeps (`kept_state.dumps_also_keyed` says when), and the store's pickle is
digested. Where they are not -- a record array's padding, an attribute a
user's class leaves out of its own state, a view a store keeps -- the keyed
pickle is made apart, and the store's bytes are never handed on as a
digest's. Either way the digest is of the same bytes, so a value digests
alike whenever it is unchanged, before a statement and after it, whether or
not a view of it was filled meanwhile.
"""

from __future__ import annotations

import hashlib
import pickle
import threading
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from . import kept_state
from .bulk_digest import TAG, fold_buffer

__all__ = [
    "HashBudgetExceeded",
    "Part",
    "keeping",
    "kept",
    "load_part",
    "nest",
    "nested",
    "object_digest",
    "object_digest_and_size",
    "pickle_value",
]


class HashBudgetExceeded(Exception):
    """A hash taken under `value_hash.budgeted_hash` ran past its budget."""


#: The head of a digest: this format and the bulk-digest scheme, so an
#: install hashing with another algorithm never matches.
_HEAD = b"cash-value-pickle-2\x00" + TAG


def _out_of_band_min() -> int:
    from .backends.serialization import OUT_OF_BAND_MIN_BYTES

    return OUT_OF_BAND_MIN_BYTES


def _stored_reducers() -> dict | None:
    from .backends.serialization import stored_reducers

    return stored_reducers()


class Part:
    """A value's store pickle: *stream* and its out-of-band *buffers*
    (cash's own copies when kept: ``bytes`` for a read-only buffer, a
    ``bytearray`` for a writable one, so a pickle holding them marks them
    as the value's were), *readonly* per buffer, whether the bytes are also
    the value's keyed bytes (*also_keyed*), and its *digest* when taken."""

    __slots__ = ("also_keyed", "buffers", "digest", "readonly", "size", "stream")

    def __init__(
        self,
        stream: bytes,
        buffers: list[Any],
        readonly: list[bool],
        also_keyed: bool,
        digest: str | None,
        size: int,
    ) -> None:
        self.stream = stream
        self.buffers = buffers
        self.readonly = readonly
        self.also_keyed = also_keyed
        self.digest = digest
        #: The bytes of the pickle, stream and buffers: kept or not.
        self.size = size


class _Sink:
    """Where the pickler writes: hashed as it comes, kept when asked, and
    stopped once *deadline* has passed (`HashBudgetExceeded`, raised from
    ``write``, which the pickler calls a frame, at most 64 KiB, or one big
    item at a time)."""

    __slots__ = ("_deadline", "_parts", "sha", "size")

    def __init__(self, deadline: float | None, keep: bool, digest: bool) -> None:
        self.sha = hashlib.sha256() if digest else None
        self._parts: list[bytes] | None = [] if keep else None
        self._deadline = deadline
        self.size = 0

    def write(self, data: Any) -> int:
        n = len(data)
        self.size += n
        if self.sha is not None:
            self.sha.update(data)
        if self._parts is not None:
            self._parts.append(bytes(data))
        if self._deadline is not None and perf_counter() > self._deadline:
            raise HashBudgetExceeded
        return n

    def stream(self) -> bytes:
        return b"".join(self._parts or ())


def pickle_value(
    value: Any,
    *,
    keep: bool,
    digest: bool = True,
    deadline: float | None = None,
    as_keyed: bool = True,
    reducer_override: Any = None,
    kept_buffers: Any = None,
) -> Part:
    """*value*'s store pickle (`kept_state.dumps_also_keyed`, protocol 5,
    the reducers a store pickles with, buffers of ``OUT_OF_BAND_MIN_BYTES``
    and more out of band).

    *keep*: the stream is kept, and each out-of-band buffer copied (the
    value's own memory may change once this returns). *digest*: the digest
    is taken (`object_digest`), over the keyed pickle when the store bytes
    are not also the keyed ones (unless *as_keyed* is False: then it is
    None). *deadline*: raise `HashBudgetExceeded` past it.
    *reducer_override* is the pickler's hook of that name, and
    *kept_buffers* says which buffers it hands out are cash's own copies
    already, kept as they are.

    Raises what pickling *value* raises.
    """
    sink = _Sink(deadline, keep, digest)
    folded = hashlib.sha256() if digest else None
    buffers: list[Any] = []
    readonly: list[bool] = []
    smallest = _out_of_band_min()
    apart = 0

    def take(buffer: pickle.PickleBuffer) -> bool:
        nonlocal apart
        raw = buffer.raw()
        if raw.nbytes < smallest:
            return True  # in band, as the store writes it
        apart += raw.nbytes
        if folded is not None:
            fold_buffer(folded, raw)
        if keep:
            if kept_buffers is not None and kept_buffers(raw.obj):
                buffers.append(raw.obj)
            else:
                buffers.append(raw.tobytes() if raw.readonly else bytearray(raw))
            readonly.append(raw.readonly)
        if deadline is not None and perf_counter() > deadline:
            raise HashBudgetExceeded
        return False

    def retry() -> None:
        nonlocal apart
        buffers.clear()
        readonly.clear()
        apart = 0

    stream, also_keyed = kept_state.dumps_also_keyed(
        value,
        protocol=5,
        buffer_callback=take,
        extra=_stored_reducers(),
        on_retry=retry,
        # A store's pickle alone is written whole, and may be pickled flat;
        # a digest's is hashed as it is written (`dumps_also_keyed`).
        sink=sink if digest else None,
        reducer_override=reducer_override,
    )
    found = None
    if digest:
        if also_keyed:
            found = _final(_HEAD, sink.sha.digest(), folded.digest())
        elif as_keyed:
            found = _keyed_digest(value, deadline)[0]
    if digest:
        if keep:
            stream = sink.stream()
        size = sink.size
    else:
        size = len(stream)
    return Part(stream, buffers, readonly, also_keyed, found, size + apart)


def _final(head: bytes, stream_digest: bytes, buffers_digest: bytes) -> str:
    return hashlib.sha256(head + stream_digest + buffers_digest).hexdigest()


def _keyed_digest(value: Any, deadline: float | None) -> tuple[str, int]:
    """The digest and size of *value*'s keyed pickle: what a value whose
    store bytes leave something out is digested over."""
    sink = _Sink(deadline, False, True)
    folded = hashlib.sha256()
    size = 0
    smallest = _out_of_band_min()

    def take(buffer: pickle.PickleBuffer) -> bool:
        nonlocal size
        raw = buffer.raw()
        if raw.nbytes < smallest:
            return True  # in band, as for the store's bytes
        fold_buffer(folded, raw)
        size += raw.nbytes
        if deadline is not None and perf_counter() > deadline:
            raise HashBudgetExceeded
        return False

    kept_state.dumps(value, protocol=5, buffer_callback=take, keyed=True, extra=_stored_reducers(), sink=sink)
    return _final(_HEAD, sink.sha.digest(), folded.digest()), size + sink.size


def object_digest(value: Any, deadline: float | None = None) -> str:
    """The digest of *value* over its store pickle, or over its keyed pickle
    when the store's leaves something out (see the module's docstring).

    While a store is `keeping` the pickles of *value*, the bytes are kept
    for it (`kept`). Raises what pickling raises, and `HashBudgetExceeded`
    past *deadline*."""
    keep = _keeps(value)
    part = pickle_value(value, keep=keep, deadline=deadline)
    if keep and part.also_keyed:
        _remember(value, part)
    return part.digest


def object_digest_and_size(value: Any) -> tuple[str, int]:
    """``(object_digest(value), the bytes of the pickle it is taken over)``."""
    part = pickle_value(value, keep=False, as_keyed=False)
    if part.also_keyed:
        return part.digest, part.size
    return _keyed_digest(value, None)


# -- the handoff to a store ---------------------------------------------------

_local = threading.local()


@contextmanager
def keeping(wanted: Callable[[Any], bool]) -> Iterator[None]:
    """Keep the store pickle each digest takes of a value *wanted* says a
    store that runs before this ends will hold (`kept`).

    Only around code that runs nothing of the user's between a digest and
    the store: the kept bytes are the value as it was when digested. A value
    is kept by a weak reference (an object without one is not kept), so
    nothing here adds to its reference count, which the checks between
    count. Everything kept is let go at the end."""
    outer = getattr(_local, "state", None)
    _local.state = (wanted, {}, {})
    try:
        yield
    finally:
        _local.state = outer


def _keeps(value: Any) -> bool:
    state = getattr(_local, "state", None)
    if state is None:
        return False
    try:
        return bool(state[0](value))
    except Exception:  # noqa: BLE001 - not known to be stored: not kept
        return False


def _remember(value: Any, part: Part) -> None:
    state = getattr(_local, "state", None)
    if state is None:
        return
    try:
        ref = weakref.ref(value)
    except TypeError:
        return  # no weak reference (a list, a dict): not kept
    state[1][id(value)] = (ref, part)


def kept(value: Any) -> Part | None:
    """The store pickle a digest of this very *value* kept (`keeping`), or
    None."""
    state = getattr(_local, "state", None)
    if state is None:
        return None
    found = state[1].get(id(value))
    if found is None or found[0]() is not value:
        return None
    return found[1]


def nest(container: Any, value: Any) -> None:
    """Say that *value*, held in *container* (a statement's entry), may be
    pickled as its own kept pickle inside *container*'s (`nested`): the
    caller has checked that nothing else in *container* is part of
    *value*, so no object is held by both."""
    state = getattr(_local, "state", None)
    if state is None or kept(value) is None:
        return
    state[2][id(container)] = (container, value)


def nested(container: Any) -> tuple[Any, Part] | None:
    """``(value, its kept pickle)`` for the value `nest` named for this
    very *container*, or None."""
    state = getattr(_local, "state", None)
    if state is None:
        return None
    found = state[2].get(id(container))
    if found is None or found[0] is not container:
        return None
    part = kept(found[1])
    return None if part is None else (found[1], part)


def load_part(stream: bytes, *buffers: Any) -> Any:
    """The value a kept pickle nested in another holds (`nested`). Named in
    stored entries: keep its name."""
    return pickle.loads(stream, buffers=buffers)
