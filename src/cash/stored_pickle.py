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
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from . import kept_state
from .bulk_digest import TAG, fold_buffer

__all__ = [
    "HashBudgetExceeded",
    "Part",
    "apart",
    "keeping",
    "kept",
    "load_part",
    "nest",
    "nested",
    "nesting",
    "object_digest",
    "object_digest_and_size",
    "pending",
    "pickle_value",
    "run_on",
    "running_on",
    "sized_object_digest",
]


class HashBudgetExceeded(Exception):
    """A hash taken under `value_hash.budgeted_hash` ran past its budget."""


#: What a digest past its deadline asks on this thread for a later one
#: (`running_on`).
_overrun = threading.local()


@contextmanager
def running_on(ask: Callable[[float, int], float | None]) -> Iterator[None]:
    """While this runs, a digest on this thread that passes its deadline
    asks *ask*, with that deadline and the bytes it has pickled so far, for
    a later deadline to run on to, and stops (`HashBudgetExceeded`) when it
    gives none (`value_hash.budgeted_hash`, which knows what the value came
    to before and how long the check may take, and gives none for a
    deadline that is not its own)."""
    outer = getattr(_overrun, "ask", None)
    _overrun.ask = ask
    try:
        yield
    finally:
        _overrun.ask = outer


def run_on(deadline: float, done: int) -> float:
    """The later deadline a digest past its own, *deadline*, runs on to,
    *done* bytes into its pickle (`running_on`); `HashBudgetExceeded` when
    there is none."""
    ask = getattr(_overrun, "ask", None)
    later = ask(deadline, done) if ask is not None else None
    if later is None or perf_counter() > later:
        raise HashBudgetExceeded
    return later


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
    item at a time, and from `check`, called for each buffer) and no later
    one is given (`run_on`), or once the pickle holds more than *limit*
    bytes, stream and out-of-band buffers (*apart*)."""

    __slots__ = ("_deadline", "_limit", "_parts", "apart", "sha", "size")

    def __init__(self, deadline: float | None, keep: bool, digest: bool, limit: int | None = None) -> None:
        self.sha = hashlib.sha256() if digest else None
        self._parts: list[bytes] | None = [] if keep else None
        self._deadline = deadline
        self._limit = limit
        self.apart = 0
        self.size = 0

    def write(self, data: Any) -> int:
        n = len(data)
        self.size += n
        if self.sha is not None:
            self.sha.update(data)
        if self._parts is not None:
            self._parts.append(bytes(data))
        self.check()
        return n

    def check(self) -> None:
        """Raise `HashBudgetExceeded` past the deadline, unless a later
        one is given (`run_on`), or past the byte limit."""
        if self._deadline is not None and perf_counter() > self._deadline:
            self._deadline = run_on(self._deadline, self.size + self.apart)
        if self._limit is not None and self.size + self.apart > self._limit:
            raise HashBudgetExceeded

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
    after: Any = None,
    limit: int | None = None,
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
    already, kept as they are. *after* is called with the pickler once
    it has pickled *value* (`kept_state.dumps_also_keyed`). *limit*: raise
    `HashBudgetExceeded` once the pickle holds more than that many bytes.
    A store's pickle alone (no *digest*) under a *deadline* or a *limit* is
    written to a sink that checks them, and is then not pickled flat: a
    value too deep for the pickler raises RecursionError.

    Raises what pickling *value* raises.
    """
    sink = _Sink(deadline, keep, digest, limit)
    # A digest's pickle is hashed as it is written; a store's alone is
    # written whole, and may be pickled flat, unless it is checked as it
    # goes (`dumps_also_keyed`).
    sunk = digest or deadline is not None or limit is not None
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
        sink.apart = apart
        if folded is not None:
            fold_buffer(folded, raw)
        if keep:
            if kept_buffers is not None and kept_buffers(raw.obj):
                buffers.append(raw.obj)
            else:
                buffers.append(raw.tobytes() if raw.readonly else bytearray(raw))
            readonly.append(raw.readonly)
        sink.check()
        return False

    def retry() -> None:
        nonlocal apart
        buffers.clear()
        readonly.clear()
        apart = sink.apart = 0

    stream, also_keyed = kept_state.dumps_also_keyed(
        value,
        protocol=5,
        buffer_callback=take,
        extra=_stored_reducers(),
        on_retry=retry,
        sink=sink if sunk else None,
        reducer_override=reducer_override,
        after=after,
    )
    found = None
    if digest:
        if also_keyed:
            found = _final(_HEAD, sink.sha.digest(), folded.digest())
        elif as_keyed:
            found = _keyed_digest(value, deadline)[0]
    if sunk:
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
        sink.apart = size
        sink.check()
        return False

    kept_state.dumps(value, protocol=5, buffer_callback=take, keyed=True, extra=_stored_reducers(), sink=sink)
    return _final(_HEAD, sink.sha.digest(), folded.digest()), size + sink.size


def object_digest(value: Any, deadline: float | None = None) -> str:
    """The digest of *value* over its store pickle, or over its keyed pickle
    when the store's leaves something out (see the module's docstring).

    While a store is `keeping` the pickles of *value*, the bytes are kept
    for it (`kept`). Raises what pickling raises, and `HashBudgetExceeded`
    past *deadline* (and any later one `run_on` gives): then, with
    `keeping`'s *past_budget*, the store that follows makes the pickle
    itself if it keeps *value* as bytes (`nested`)."""
    return sized_object_digest(value, deadline)[0]


def sized_object_digest(value: Any, deadline: float | None = None) -> tuple[str, int]:
    """``(object_digest(value, deadline), the bytes of the pickle it was
    taken over)``."""
    if not _keeps(value):
        # No store takes these bytes, so its pickle buys nothing: the digest
        # is the keyed pickle's either way, and where the two differ (a
        # fitted forest's trees are record arrays with padding) the store's
        # was made only to be thrown away before the keyed one.
        return _keyed_digest(value, deadline)
    # Only a value kept by a weak reference is kept at all (`_remember`).
    others = _others(value) if _weakly_held(value) else ()
    proven: list[frozenset[int]] = []
    after = None
    if others:

        def after(pickler: Any) -> None:
            proven.append(_proven_apart(pickler, others))

    try:
        part = pickle_value(value, keep=True, deadline=deadline, after=after)
    except HashBudgetExceeded:
        _defer(value)
        raise
    if part.also_keyed:
        _remember(value, part, proven[0] if proven else frozenset())
    return part.digest, part.size


def object_digest_and_size(value: Any) -> tuple[str, int]:
    """``(object_digest(value), the bytes of the pickle it is taken over)``."""
    return _keyed_digest(value, None)


# -- the handoff to a store ---------------------------------------------------

_local = threading.local()


class _State:
    """What one `keeping` window holds: its arguments, the pickles kept
    (id -> ``(weak reference, Part, ids proven apart)``), the entries
    nesting one (id -> ``(container, value, others to prove apart)``),
    and the values whose digest ran out of its budget (id -> weak
    reference), whose pickle the store makes if it takes them (`nested`)."""

    __slots__ = ("kept", "nests", "others", "past_bytes", "past_seconds", "pending", "wanted")

    def __init__(
        self,
        wanted: Callable[[Any], bool],
        others: Callable[[Any], Iterable[Any]] | None,
        past_bytes: int | None,
        past_seconds: float | None,
    ) -> None:
        self.wanted = wanted
        self.others = others
        self.past_bytes = past_bytes
        self.past_seconds = past_seconds
        self.kept: dict[int, tuple[weakref.ref, Part, frozenset[int]]] = {}
        self.nests: dict[int, tuple[Any, Any, list[Any]]] = {}
        self.pending: dict[int, weakref.ref] = {}


@contextmanager
def keeping(
    wanted: Callable[[Any], bool],
    others: Callable[[Any], Iterable[Any]] | None = None,
    past_budget: int | None = None,
    past_seconds: float | None = None,
) -> Iterator[None]:
    """Keep the store pickle each digest takes of a value *wanted* says a
    store that runs before this ends will hold (`kept`).

    *others* gives, for such a value, what else the store's entry will
    hold that may share an object with it: the statement's echoed value,
    its other outputs. Each is checked against the value's pickle as it is
    made (`apart`).

    *past_budget*: a digest under a budget (`value_hash.budgeted_hash`)
    that runs out of it while pickling such a value gives up, and the
    pickle is left to the store: when it does keep the value as bytes in
    an entry (`nest`, `nested`), it makes it whole, up to that many bytes
    and *past_seconds* (when given), and nests it as it would a kept one.
    A 915 MB statsmodels result ran out of a change check's 0.5 s with 0.7
    s of pickling to make, and its store took 9.8 s to copy it the other
    way. The pickle is not made on in the digest itself: the store may not
    take the value at all (an openpyxl workbook its own sheets hold is
    refused after it is digested), and 8 s of pickling one was thrown away.

    Only around code that runs nothing of the user's between a digest and
    the store: the kept bytes are the value as it was when digested. A value
    is kept by a weak reference (an object without one is not kept), so
    nothing here adds to its reference count, which the checks between
    count. Everything kept is let go at the end."""
    outer = getattr(_local, "state", None)
    _local.state = _State(wanted, others, past_budget, past_seconds)
    try:
        yield
    finally:
        _local.state = outer


def _keeps(value: Any) -> bool:
    state = getattr(_local, "state", None)
    if state is None:
        return False
    try:
        return bool(state.wanted(value))
    except Exception:  # noqa: BLE001 - not known to be stored: not kept
        return False


def _weakly_held(value: Any) -> bool:
    try:
        weakref.ref(value)
    except TypeError:
        return False
    return True


def _defer(value: Any) -> None:
    """Note that *value*'s digest ran out of its budget, for a store that
    may make its pickle (`keeping`'s *past_budget*)."""
    state = getattr(_local, "state", None)
    if state is None or state.past_bytes is None:
        return
    try:
        state.pending[id(value)] = weakref.ref(value)
    except TypeError:
        return  # no weak reference: not kept


def pending(value: Any) -> bool:
    """Did a digest of this very *value* run out of its budget, leaving its
    pickle to the store (`keeping`'s *past_budget*)?"""
    state = getattr(_local, "state", None)
    if state is None:
        return False
    found = state.pending.get(id(value))
    return found is not None and found() is value


def _others(value: Any) -> list[Any]:
    """What `keeping`'s *others* says the entry holding *value* holds
    besides it, *value* itself left out."""
    state = getattr(_local, "state", None)
    if state is None or state.others is None:
        return []
    try:
        return [other for other in state.others(value) if other is not value and type(other) not in _ATOMS]
    except Exception:  # noqa: BLE001 - nothing known: nothing proven
        return []


#: Values `_others` leaves out: nothing can be part of them.
_ATOMS = frozenset({str, bytes, int, float, complex, bool, type(None)})

#: How much of the stream of each of `keeping`'s *others* `_proven_apart`
#: pickles to look for shared objects, and for how long, before it gives up:
#: an echo is a few kilobytes, and its buffers are not copied.
_APART_MAX_BYTES = 16 << 20
_APART_MAX_SECONDS = 0.25


class _TooMuch(Exception):
    """`_proven_apart` went past its bytes or time."""


class _Counted:
    """A sink that keeps nothing and stops past `_APART_MAX_BYTES` or
    *deadline*."""

    __slots__ = ("_deadline", "size")

    def __init__(self, deadline: float) -> None:
        self._deadline = deadline
        self.size = 0

    def write(self, data: Any) -> int:
        self.size += len(data)
        if self.size > _APART_MAX_BYTES or perf_counter() > self._deadline:
            raise _TooMuch
        return len(data)


def _proven_apart(pickler: Any, others: list[Any]) -> frozenset[int]:
    """The ids of those of *others* that share no object with the value
    *pickler* just pickled, as a store pickles them: no object in both
    pickles' memos -- every object a pickle restores as one wherever it
    appears -- but values that restore alike apart
    (`kept_state.restores_alike_apart`). Both memos are alive together,
    so an id names one object in both.

    Pickled together, an object both hold is restored as one; pickled
    apart, as two. So *value*'s own pickle may stand in an entry beside
    such an other (`nest`) only when none is shared."""
    try:
        inner = pickler.memo.copy()
    except Exception:  # noqa: BLE001 - nothing known: nothing proven
        return frozenset()
    extra = _stored_reducers()
    table = kept_state.reducers(5, extra)
    deadline = perf_counter() + _APART_MAX_SECONDS
    proven = set()
    for other in others:
        try:
            memo = kept_state.store_memo(other, 5, extra=extra, sink=_Counted(deadline))
        except Exception:  # noqa: BLE001 - too big, too slow, or unpicklable: not proven
            continue
        if all(kept_state.restores_alike_apart(memo[key][1], table) for key in memo if key in inner):
            proven.add(id(other))
        del memo
    return frozenset(proven)


def _remember(value: Any, part: Part, apart: frozenset[int]) -> None:
    state = getattr(_local, "state", None)
    if state is None:
        return
    try:
        ref = weakref.ref(value)
    except TypeError:
        return  # no weak reference (a list, a dict): not kept
    state.kept[id(value)] = (ref, part, apart)


def apart(value: Any, other: Any) -> bool:
    """Did the digest that kept *value*'s pickle (`kept`) find that
    *other*, one of `keeping`'s *others* for it, shares no object with it
    (`_proven_apart`)? Both are alive from that digest to the store, so
    the id names the same object."""
    state = getattr(_local, "state", None)
    if state is None:
        return False
    found = state.kept.get(id(value))
    return found is not None and found[0]() is value and id(other) in found[2]


def kept(value: Any) -> Part | None:
    """The store pickle a digest of this very *value* kept (`keeping`), or
    None."""
    state = getattr(_local, "state", None)
    if state is None:
        return None
    found = state.kept.get(id(value))
    if found is None or found[0]() is not value:
        return None
    return found[1]


def nest(container: Any, value: Any, unproven: Iterable[Any] = ()) -> None:
    """Say that *value*, held in *container* (a statement's entry), may be
    pickled as its own kept pickle inside *container*'s (`nested`): the
    caller has checked that nothing else in *container* is part of
    *value*, so no object is held by both (immutable data, or `apart`).

    A value whose pickle was left to the store (`pending`) may be named
    too, with the parts of *container* not yet proven apart from it
    (*unproven*): `nested` proves them as it makes the pickle."""
    state = getattr(_local, "state", None)
    if state is None or (kept(value) is None and not pending(value)):
        return
    state.nests[id(container)] = (container, value, list(unproven))


def nesting(container: Any) -> bool:
    """Was a value named for this very *container* (`nest`)? Nothing is
    made: see `nested`."""
    state = getattr(_local, "state", None)
    if state is None:
        return False
    found = state.nests.get(id(container))
    return found is not None and found[0] is container and (kept(found[1]) is not None or pending(found[1]))


def nested(container: Any) -> tuple[Any, Part] | None:
    """``(value, its kept pickle)`` for the value `nest` named for this
    very *container*, or None.

    A value whose pickle was left to the store (`pending`) has it made
    now, once, whole, for the store that asks: at most `keeping`'s
    *past_budget* bytes and *past_seconds*, past which this raises
    `HashBudgetExceeded` (a value slower to pickle than its statement is
    worth is not kept as bytes; the caller keeps it no other way either).
    None when it does not pickle, or when a part of *container* shares an
    object with it."""
    state = getattr(_local, "state", None)
    if state is None:
        return None
    found = state.nests.get(id(container))
    if found is None or found[0] is not container:
        return None
    value, unproven = found[1], found[2]
    part = kept(value)
    if part is None and pending(value):
        part = _make_pending(state, value, unproven)
    return None if part is None else (value, part)


def _make_pending(state: _State, value: Any, others: list[Any]) -> Part | None:
    """The store pickle of *value*, whose digest ran out of its budget,
    made for the store (`nested`), when each of *others* is proven apart
    from it; `HashBudgetExceeded` past `keeping`'s bytes or seconds."""
    del state.pending[id(value)]  # made once, taken or not
    proven: list[frozenset[int]] = []
    after = None
    if others:

        def after(pickler: Any) -> None:
            proven.append(_proven_apart(pickler, others))

    deadline = None if state.past_seconds is None else perf_counter() + state.past_seconds
    try:
        part = pickle_value(value, keep=True, digest=False, deadline=deadline, limit=state.past_bytes, after=after)
    except HashBudgetExceeded:
        raise
    except Exception:  # noqa: BLE001 - copied the store's usual way, which says why
        return None
    apart_ids = proven[0] if proven else frozenset()
    if any(id(other) not in apart_ids for other in others):
        return None
    _remember(value, part, apart_ids)
    return part


def load_part(stream: bytes, *buffers: Any) -> Any:
    """The value a kept pickle nested in another holds (`nested`). Named in
    stored entries: keep its name."""
    return pickle.loads(stream, buffers=buffers)
