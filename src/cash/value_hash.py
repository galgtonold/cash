"""The notebook's one value hash: `compute_hash`.

A value's whole content, never a sample: a frame, array or table through
its library content hasher (`builtin_hash`), a collection of frames item by
item, anything else pickled -- over the bytes a store keeps of it
(`stored_pickle.object_digest`), so a statement's store takes the pickle its
digest made -- and only when even pickling fails its identity
(`identity_hash`), which `is_identity_fallback_hash` lets a caller recognise.
It is the ``compute_hash_fn`` seam threaded into ``StatementProcessor`` and
``UpstreamChecker``, and
the digest of a loop variable, a call's arguments and the globals a call
writes. A sample would decide "unchanged" for an edit outside it.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import pickle
import threading
import weakref
from collections.abc import Callable, Iterator
from time import perf_counter
from typing import Any

from . import value_watch
from .content_hashers import builtin_hash, builtin_hash_family, is_native_panic
from .sizing import pandas_nbytes
from .stored_pickle import HashBudgetExceeded, run_on, running_on, sized_object_digest

logger = logging.getLogger(__name__)

#: What a hash that cannot read a value raises. RecursionError: a value nested
#: deeper than pickle or a walk follows, such as a linked list of a few
#: hundred objects, has no content hash either. NotImplementedError: a value
#: with no state to pickle, such as ``random.SystemRandom()``, whose
#: ``__reduce__`` asks for a state it does not have.
HASH_ERRORS = (TypeError, ValueError, AttributeError, pickle.PicklingError, RecursionError, NotImplementedError)


_BULKY_TYPE_NAMES = frozenset({"DataFrame", "Series", "ndarray"})


def is_bulky(value: Any) -> bool:
    """A frame, array or table: hashed on its own rather than pickled with
    the collection that holds it."""
    return _is_bulky_type(type(value))


def _is_bulky_type(t: type) -> bool:
    """`is_bulky`, asked of a type."""
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
    # Asked of each TYPE once: asked of each item, a list of 200,000 ints
    # cost 0.9 s here, 30x the pickle that hashes it.
    if any(map(_is_bulky_type, set(map(type, values)))):
        parts = [f"{type(obj).__name__}:{len(obj)}"]
        if items is not None:
            parts.extend(f"{compute_hash(k)}={compute_hash(v)}" for k, v in items)
        else:
            parts.extend(compute_hash(v) for v in values)
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return _pickle_digest(obj)


#: How long a check whether a value changed may spend pickling it before
#: the value counts as too costly to read (`budgeted_hash`). An openpyxl
#: workbook of 400,000 cells pickles in about 3 s, and a formatting cell that
#: binds and changes one was checked after each of its statements: 15-40 s a
#: run next to 11 s of its own work.
CHANGE_CHECK_BUDGET_S = 0.5

#: A check runs on past its budget (`budgeted_hash`) only for a value whose
#: size cash knows (`known_size`) and whose pickle so far projects it done
#: within this many budgets: a value that hashes in about the budget, which a
#: loaded machine pushes past it now and then. A networkx graph of 350,000
#: edges (an 11 MB pickle) hashes in 0.3-0.55 s: cut at 0.5 s one run in a
#: few, it counted as changed after a restart, and 195 s of statements built
#: on it ran again. A value projected past it is a clear overrun, cut at the
#: budget as before.
CHANGE_CHECK_RUN_ON = 2.0

#: How long a check that runs on may take in all, in budgets: room for a
#: projection that comes out low, after which it is cut all the same.
CHANGE_CHECK_STRETCH = 3.0

#: The pickled size from which a value's size is remembered (`note_size`):
#: below it no pickle comes near the budget.
SIZE_MEMO_MIN_BYTES = 1 << 20

_budget = threading.local()

#: ``id -> (weak reference, seconds of the check that gave up on it)`` of
#: the objects a budgeted hash gave up on.
_COSTLY: dict[int, tuple[weakref.ref, float]] = {}

#: ``id -> (weak reference, bytes)``: what each object's pickle came to,
#: from its last full digest or the entry it was restored from.
_SIZES: dict[int, tuple[weakref.ref, int]] = {}


def note_size(obj: Any, nbytes: int) -> None:
    """Remember that *obj* pickles to about *nbytes* (`known_size`): its
    digest took that many, or the entry it was just restored from holds
    that many. Only for an object with a weak reference, of at least
    `SIZE_MEMO_MIN_BYTES`."""
    if nbytes < SIZE_MEMO_MIN_BYTES:
        return
    key = id(obj)
    try:
        ref = weakref.ref(obj, lambda _ref, key=key: _SIZES.pop(key, None))
    except TypeError:
        return
    _SIZES[key] = (ref, nbytes)


def known_size(obj: Any) -> int | None:
    """The bytes *obj* last pickled to, as `note_size` was told, or None."""
    found = _SIZES.get(id(obj))
    if found is None or found[0]() is not obj:
        return None
    return found[1]


def _pickle_digest(obj: Any) -> str:
    """The digest of *obj* over the bytes a store keeps of it
    (`stored_pickle.object_digest`); under a budget (`budgeted_hash`),
    stopped when the budget runs out. The same digest either way. What
    the pickle came to is remembered (`note_size`)."""
    check = getattr(_budget, "check", None)
    if check is None:
        digest, size = sized_object_digest(obj)
    else:
        if perf_counter() > check.deadline:
            run_on(check.deadline, 0)
        digest, size = sized_object_digest(obj, check.deadline)
        check.done += size
    note_size(obj, size)
    return digest


def is_costly(obj: Any) -> bool:
    """Did a budgeted hash give up on *obj* (`budgeted_hash`)? Such a value
    is treated as one whose content no check compares: changed whenever a
    check would have to read it."""
    if not _COSTLY:
        return False
    found = _COSTLY.get(id(obj))
    return found is not None and found[0]() is obj


def costly_after(obj: Any) -> float | None:
    """The seconds of the check that gave up on *obj* (`is_costly`), or
    None when none did: for saying why it counts as changed."""
    if not _COSTLY:
        return None
    found = _COSTLY.get(id(obj))
    return found[1] if found is not None and found[0]() is obj else None


def _remember_costly(obj: Any, seconds: float) -> None:
    key = id(obj)
    try:
        ref = weakref.ref(obj, lambda _ref, key=key: _COSTLY.pop(key, None))
    except TypeError:
        return  # no weak reference (a list, a dict): asked again next time
    _COSTLY[key] = (ref, seconds)


class _Check:
    """One budgeted hash (`budgeted_hash`): when it started, its budget
    (*seconds*) and deadline, the size its value is known to pickle to
    (*size*), and the bytes its finished pickles came to (*done*)."""

    __slots__ = ("budget", "deadline", "done", "ran_on", "seconds", "size", "start")

    def __init__(self, seconds: float, size: int | None) -> None:
        self.start = perf_counter()
        self.seconds = seconds
        self.budget = self.deadline = self.start + seconds
        self.size = size
        self.done = 0
        self.ran_on = False

    def ask(self, passed: float, pickled: int) -> float | None:
        """The later deadline a pickle past this check's (*passed*),
        *pickled* bytes in, may run on to (`stored_pickle.run_on`), or
        None; None for a pickle with a deadline of its own (a store's).

        Only once, only for a value whose size is known, and only when the
        rate so far says it is done within `CHANGE_CHECK_RUN_ON` budgets;
        then up to `CHANGE_CHECK_STRETCH` budgets in all. A value nothing
        is known of, or a clear overrun, stops at the budget, as it always
        did, and costs no more."""
        if passed != self.budget and passed != self.deadline:
            return None
        now = perf_counter()
        if now <= self.deadline:
            return self.deadline  # a later deadline given already
        if self.ran_on or not self.size:
            return None
        done = self.done + pickled
        if done <= 0 or done >= self.size:
            return None  # nothing read yet, or more than it ever was
        projected = (now - self.start) * self.size / done
        if projected > self.seconds * CHANGE_CHECK_RUN_ON:
            return None
        self.ran_on = True
        self.deadline = self.start + self.seconds * CHANGE_CHECK_STRETCH
        return self.deadline


def budgeted_hash(obj: Any, hash_fn: Callable[[Any], str] | None = None, seconds: float | None = None) -> str | None:
    """``hash_fn(obj)`` (`compute_hash` by default) for a check whether
    *obj* changed, or None when pickling it takes longer than *seconds*
    (`CHANGE_CHECK_BUDGET_S`); *obj* is then remembered as costly
    (`is_costly`) and later budgeted hashes of it return None at once.

    A value whose size is known (`known_size`: from its last digest, or
    from the entry it was restored from) runs on past the budget when what
    it pickled by then projects it done within `CHANGE_CHECK_RUN_ON`
    budgets, for up to `CHANGE_CHECK_STRETCH` budgets in all (`_Check.ask`):
    a value that hashes in about the budget is not cut short one run in a
    few by a loaded machine, and does not count as changed for it. A value
    nothing is known of, or a clear overrun, stops at the budget.

    None means "cannot say": every caller treats the value as changed, or
    as one whose content it does not compare -- never as unchanged. Only
    the checks around a statement take this; a key never does, so no key
    loses part of a value.
    """
    if is_costly(obj):
        return None
    outer = getattr(_budget, "check", None)
    if outer is not None:
        # Within another check: its budget holds.
        try:
            return (hash_fn or compute_hash)(obj)
        except HashBudgetExceeded:
            _remember_costly(obj, outer.deadline - outer.start)
            return None
    check = _Check(CHANGE_CHECK_BUDGET_S if seconds is None else seconds, known_size(obj))
    _budget.check = check
    try:
        with running_on(check.ask):
            return (hash_fn or compute_hash)(obj)
    except HashBudgetExceeded:
        _remember_costly(obj, check.deadline - check.start)
        return None
    finally:
        _budget.check = None


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

_PANDAS_TYPE_NAMES = frozenset({"DataFrame", "Series"})

#: The size from which a pandas frame's digest is memoised
#: (`_memoised_frame_digest`): below it a full read costs about what the check
#: does, as for the call checks' memo (`call_effects.ArgFingerprints`).
FRAME_MEMO_MIN_BYTES = 1 << 20


def _memoised_frame_digest(obj: Any) -> str | None:
    """The content digest of a big pandas frame under copy-on-write, read
    once and then checked, not read again, while it is provably unchanged
    (`cash.decorator.arg_hashing.cow_frame_digest`): its blocks, axes and
    manager are the same objects, which copy-on-write makes an exact "not
    written" signal, and no handle that could write past it was given out.
    The same digest `builtin_hash` gives, from the memo the decorator keys
    its arguments through. None for a small frame or any other value.

    A notebook loop that hands a 50 MB frame to a cached function on every
    iteration took that frame's digest several times per iteration (the
    loop's mutation watch before and after, the value checks around the
    call) and spent most of its time doing so.
    """
    if (pandas_nbytes(obj) or 0) < FRAME_MEMO_MIN_BYTES:
        return None
    from .decorator.arg_hashing import cow_frame_digest

    return cow_frame_digest(obj)


#: Values `compute_hash` never asks the watchers about: they are hashed
#: in less time than a lookup in the watch table takes.
_ATOMS = frozenset((int, str, float, bool, type(None), bytes))


def digest_via_pickle(obj: Any) -> bool:
    """Does `compute_hash` take *obj*'s digest over its whole pickle
    (`stored_pickle.object_digest`), as a call's store digests its result
    (`call_refs.digest_and_size`)? Then that digest is *obj*'s
    `compute_hash`, when it pickles at all."""
    kind = type(obj)
    if kind.__name__ in _PANDAS_TYPE_NAMES or _is_bulky_type(kind):
        return False
    if isinstance(obj, tuple) and isinstance(getattr(kind, "_fields", None), tuple):
        return False
    if kind is dict or kind is list or kind is tuple:
        values = obj.values() if kind is dict else obj
        return not any(map(_is_bulky_type, set(map(type, values))))
    return True


def compute_hash(obj: Any) -> str:
    """Hash *obj* over its whole content, with explicit fallbacks.

    With the C extension, a value hashed before whose content the watchers
    prove unchanged since gets that earlier digest back without being read
    (`value_watch.digest`): the same digest, so nothing downstream can tell.

    Strategy order:
    1. A frame, array or table through its built-in content hasher
       (`builtin_hash`), every byte and its schema; a collection item by item
       (`_hash_collection`)
    2. Generic pickle hash, over the store's own pickle
       (`stored_pickle.object_digest`)
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
    window = _one_read.memo
    if window is not None and type(obj) not in _ATOMS:
        found = window.get(id(obj))
        if found is not None and found[0]() is obj:
            return found[1]
    if value_watch.WATCHING and type(obj) not in _ATOMS:
        digest = value_watch.digest(obj, _compute_hash, value_watch.KIND_VALUE, _reusable)
    else:
        digest = _compute_hash(obj)
    if window is not None and type(obj) not in _ATOMS and _reusable(obj, digest):
        try:
            window[id(obj)] = (weakref.ref(obj), digest)
        except TypeError:
            pass  # a list, dict or tuple: see `one_read`
    return digest


class _OneRead(threading.local):
    """`one_read`'s digests, on the thread that opened it only."""

    memo: dict[int, tuple[weakref.ref, str]] | None = None


_one_read = _OneRead()


@contextlib.contextmanager
def one_read() -> Iterator[None]:
    """Within it, `compute_hash` reads a value once on this thread: every
    later digest of the same object is the first one's.

    Only where nothing can change a value between two checks that read it:
    after a statement ran and before the next one starts, its in-place
    checks, its outputs' lineage and their session hashes all hashed a model
    again, 7 reads of a 500-tree ensemble a statement where 2 do. Every one
    of them is `compute_hash` of the same object, with no option of its
    own (`budgeted_hash` only gives up past its budget, which a digest
    already taken spares it), so the bytes are the same. The caller vouches
    that no code of the user's runs inside, on any thread.

    Kept by weak reference, and only for a value that takes one: a
    reference held here would count as a holder to the checks that tell a
    value shared (`_plain_data`), and a list's digest follows those. A
    plain list, dict or tuple is read each time, as before. The identity
    tier's digest is never kept. Nested, the outer one holds.
    """
    if _one_read.memo is not None:
        yield
        return
    _one_read.memo = {}
    try:
        yield
    finally:
        _one_read.memo = None


def _reusable(obj: Any, digest: str) -> bool:
    """May the watchers hand *digest* back for *obj*? Not the identity
    tier's: it says nothing of the content, and callers treat it apart."""
    return digest != identity_hash(obj)


def _compute_hash(obj: Any) -> str:
    """`compute_hash`, read from the value itself."""
    type_name = type(obj).__name__

    try:
        digest = _memoised_frame_digest(obj) if type_name in _PANDAS_TYPE_NAMES else None
        if digest is not None:
            return digest
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
        return _pickle_digest(obj)
    except HashBudgetExceeded:
        raise
    except HASH_ERRORS as exc:
        logger.debug("Primary hash failed for %s: %s", type_name, exc)
    except Exception as exc:  # noqa: BLE001 - see below
        # Whatever a user's `__reduce__` or `__getstate__` raises: the value
        # has no content hash, so it gets the identity tier, which callers
        # recognise as content-blind. Raised to the user, it failed a cell
        # that plain Python runs.
        logger.debug("Hash failed for %s: %r", type_name, exc)
        return identity_hash(obj)
    except BaseException as exc:
        if not is_native_panic(exc):
            raise
        return identity_hash(obj)

    try:
        return _pickle_digest(obj)
    except HashBudgetExceeded:
        raise
    except Exception:  # noqa: BLE001 - see below
        # Python 3.13 raises AttributeError for an instance of a class
        # defined inside a function ("Can't get local object"); a user's
        # `__reduce__` may raise anything.
        pass
    except BaseException as exc:
        if not is_native_panic(exc):
            raise

    return identity_hash(obj)
