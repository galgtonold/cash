"""The notebook's one value hash: `compute_hash`.

A value's whole content, never a sample: a frame, array or table through
its library content hasher (`builtin_hash`), a collection of frames item by
item, anything else pickled, and only when even pickling fails its identity
(`identity_hash`), which `is_identity_fallback_hash` lets a caller recognise.
It is the ``compute_hash_fn`` seam threaded into ``StatementProcessor`` and
``UpstreamChecker``, what ``Restorer`` checks a restored object against, and
the digest of a loop variable, a call's arguments and the globals a call
writes. A sample would decide "unchanged" for an edit outside it.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from typing import Any

from .content_hashers import builtin_hash, builtin_hash_family, is_native_panic

logger = logging.getLogger(__name__)

#: What a hash that cannot read a value raises. RecursionError: a value nested
#: deeper than pickle or a walk follows, such as a linked list of a few
#: hundred objects, has no content hash either.
HASH_ERRORS = (TypeError, ValueError, AttributeError, pickle.PicklingError, RecursionError)


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
    except HASH_ERRORS as exc:
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
