"""`mutation_fingerprint`: a digest that moves when a value is changed in place.

The notebook takes it around a statement that is executing, before and
after, to see which inputs the statement wrote into. It reads the whole
value, by the same content hashes as `compute_hash`, and knows the parts of
an AnnData-like object that scanpy writes to.
"""

from __future__ import annotations

import hashlib
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
