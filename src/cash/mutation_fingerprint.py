"""`mutation_fingerprint`: a digest that moves when a value is changed in place.

The notebook takes it around a statement that is executing, before and
after, to see which inputs the statement wrote into. It reads the whole
value, by the same content hashes as `compute_hash`, and knows the parts of
an AnnData-like object that scanpy writes to.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .object_hashing import builtin_hash
from .value_hash import HASH_ERRORS, compute_hash, identity_hash


def mutation_fingerprint(obj: Any) -> str | None:
    """A digest that changes when *obj* is changed IN PLACE, or ``None``.

    Reads the whole value: `builtin_hash` for the library types (pandas,
    numpy, polars, ...), and for an AnnData-like object every part scanpy
    writes to -- the ``obs``/``var`` frames, ``X``, and each value of
    ``uns``/``obsm``/``varm``/``obsp``/``varp``/``layers`` -- each by its own
    content hash. A checksum of ``X`` and the key names alone missed a
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
    digest = builtin_hash(obj)
    if digest is not None:
        h.update(digest.encode("utf-8"))
        return h.hexdigest()
    try:
        is_anndata = all(hasattr(obj, a) for a in ("obs", "var", "uns", "X"))
    except Exception:  # noqa: BLE001 - a property that raises: not AnnData-like
        is_anndata = False
    if is_anndata:
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
        h.update("|".join(parts).encode("utf-8"))
        return h.hexdigest()
    digest = compute_hash(obj)
    if digest == identity_hash(obj):
        return None
    return digest


class _Unobservable(Exception):
    """A part of a value whose only hash is its identity."""


def _part_digest(value: Any) -> str:
    """`compute_hash` of one part of a value, or `_Unobservable`."""
    if value is None:
        return "None"
    digest = compute_hash(value)
    if digest == identity_hash(value):
        raise _Unobservable(type(value).__name__)
    return digest
