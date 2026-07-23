"""Pure functions for hashing and sizing arbitrary Python objects.

Hash helpers (``compute_hash``) are used as the ``compute_hash_fn``
callable seam threaded into ``StatementProcessor`` and ``UpstreamChecker``
and by ``Restorer`` to verify restored objects against their stored
lineage.

Size helpers (``estimate_object_size``) are used by
``StatementProcessor`` to enforce per-statement cache-budget decisions
(skip caching when an object would exceed the configured RAM/disk
budgets).  Order-of-magnitude correct, not precise; bounded in runtime
by a depth cap and sampling.

**Anti-god-class rule (load-bearing):** this module is *pure functions*.
No state, no class, no IPython, no ``Cash`` dependency. If a caller
needs context (e.g. "hash relative to lineage X"), the context stays
in the caller — do not grow this module into a class with options.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import pickle
import sys
from typing import Any

logger = logging.getLogger(__name__)

_HASH_ERRORS = (TypeError, ValueError, AttributeError, pickle.PicklingError)


def _hash_dataframe_or_series(obj: Any, type_name: str) -> str:
    """Hash a pandas DataFrame or Series using shape + dtypes + data sample."""
    shape_str = f"{obj.shape}"
    dtypes_str = str(obj.dtypes.to_dict()) if type_name == 'DataFrame' else str(obj.dtype)
    try:
        sample = str(obj.head(5).values.tobytes() if len(obj) > 0 else b'')
    except (TypeError, ValueError, AttributeError):
        sample = str(obj.head(5))
    combined = f"{shape_str}:{dtypes_str}:{sample}"
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()


def _hash_collection(obj: Any) -> str:
    """Hash a list/tuple/dict/set/frozenset — sampling large ones to avoid O(n) pickle."""
    n = len(obj)
    if n <= 200:
        return hashlib.sha256(pickle.dumps(obj)).hexdigest()
    if isinstance(obj, (list, tuple)):
        combined = f"list:{n}:{repr(obj[:5])}:{repr(obj[-5:])}"
    elif isinstance(obj, dict):
        combined = f"dict:{n}:{repr(sorted(obj.keys())[:10])}"
    else:
        combined = f"set:{n}:{repr(sorted(obj)[:10])}"
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()


def compute_hash(obj: Any) -> str:
    """Compute a hash for an object using type-specific methods with explicit fallbacks.

    Strategy order:
    1. Type-specific fast hash (DataFrame/ndarray/collections)
    2. Generic pickle hash
    3. Identity hash (always succeeds)
    """
    type_name = type(obj).__name__

    try:
        if type_name in ('DataFrame', 'Series'):
            return _hash_dataframe_or_series(obj, type_name)
        if type_name == 'ndarray':
            shape_str = str(obj.shape)
            dtype_str = str(obj.dtype)
            sample = str(obj.flat[:100].tobytes() if obj.size > 0 else b'')
            combined = f"{shape_str}:{dtype_str}:{sample}"
            return hashlib.sha256(combined.encode('utf-8')).hexdigest()
        if isinstance(obj, (list, tuple, dict, set, frozenset)):
            return _hash_collection(obj)
        return hashlib.sha256(pickle.dumps(obj)).hexdigest()
    except _HASH_ERRORS as exc:
        logger.debug("Primary hash failed for %s: %s", type_name, exc)

    try:
        return hashlib.sha256(pickle.dumps(obj)).hexdigest()
    except (TypeError, pickle.PicklingError):
        pass

    return hashlib.sha256(str(id(obj)).encode('utf-8')).hexdigest()


def compute_hash_full(obj: Any) -> str:
    """Full-content hash for cache-KEY discrimination.

    ``compute_hash`` SAMPLES large objects (ndarray: first 100 elements,
    DataFrame: first 5 rows, collections >200: head/tail). That is fine for
    cheap freshness heuristics, but unsound wherever the hash *is* the key
    discriminator — per-iteration loop caching keyed two iterations over
    arrays that agreed in the sample onto one entry and produced a wrong
    result on the very first run. This variant hashes every byte.

    Fallback ladder mirrors ``compute_hash`` (generic pickle, then the
    sampled/identity ladder as a last resort).
    """
    type_name = type(obj).__name__
    try:
        if type_name in ('DataFrame', 'Series'):
            import pandas as pd
            if type_name == 'DataFrame':
                schema = f"{list(obj.columns)!r}:{list(obj.index.names)!r}:"
            else:
                schema = f"{obj.name!r}:{list(obj.index.names)!r}:"
            h = hashlib.sha256(schema.encode('utf-8'))
            h.update(pd.util.hash_pandas_object(obj).values.tobytes())
            return h.hexdigest()
        if type_name == 'ndarray':
            if getattr(obj.dtype, 'hasobject', False):
                # Object arrays' buffer bytes are raw pointers, not content.
                return hashlib.sha256(pickle.dumps(obj)).hexdigest()
            h = hashlib.sha256(f"{obj.shape}:{obj.dtype}:".encode('utf-8'))
            try:
                h.update(memoryview(obj).cast('B'))   # no copy if contiguous
            except (TypeError, ValueError):
                h.update(obj.tobytes())
            return h.hexdigest()
        return hashlib.sha256(pickle.dumps(obj)).hexdigest()
    except _HASH_ERRORS as exc:
        logger.debug("Full hash failed for %s: %s", type_name, exc)
    return compute_hash(obj)


# ---------------------------------------------------------------------------
# Object size estimation
# ---------------------------------------------------------------------------

# Recursion-depth cap for ``estimate_object_size`` walks; bounds total cost.
_MAX_ESTIMATE_DEPTH = 4

# scipy.sparse type names. Dispatched via type-name string to avoid an
# import-time dependency on scipy.
_SPARSE_CSR_CSC_TYPES = frozenset({
    'csr_matrix', 'csc_matrix', 'csr_array', 'csc_array',
})
_SPARSE_COO_TYPES = frozenset({'coo_matrix', 'coo_array'})


def _est_sparse_csr_csc(m: Any) -> int:
    return int(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes)


def _est_sparse_coo(m: Any) -> int:
    return int(m.data.nbytes + m.row.nbytes + m.col.nbytes)


def estimate_object_size(obj: Any, _depth: int = 0) -> int:
    """Estimate the memory size of an object in bytes.

    Uses ``sys.getsizeof`` for basic types and specialised estimators for
    common data-science types (DataFrame, Series, ndarray, scipy sparse).
    Recursion is bounded by ``_MAX_ESTIMATE_DEPTH``.

    Must remain **order-of-magnitude correct** (not precise) and bounded
    in runtime — see the K=3-outer / K=1-inner sampling rule below.
    Circular references are naturally bounded by the depth cap, so no
    ``seen`` set is required.
    """
    if _depth >= _MAX_ESTIMATE_DEPTH:
        return sys.getsizeof(obj)
    try:
        type_name = type(obj).__name__
        if type_name == 'DataFrame':
            return int(obj.memory_usage(deep=True).sum())
        if type_name == 'Series':
            return int(obj.memory_usage(deep=True))
        if type_name == 'ndarray':
            return int(obj.nbytes)
        if type_name in _SPARSE_CSR_CSC_TYPES:
            return _est_sparse_csr_csc(obj)
        if type_name in _SPARSE_COO_TYPES:
            return _est_sparse_coo(obj)
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            base = sys.getsizeof(obj)
            return base + sum(
                estimate_object_size(getattr(obj, f.name), _depth + 1)
                for f in dataclasses.fields(obj)
            )
        if isinstance(obj, tuple) and hasattr(obj, '_fields'):  # namedtuple
            base = sys.getsizeof(obj)
            return base + sum(
                estimate_object_size(getattr(obj, name), _depth + 1)
                for name in obj._fields
            )
        if type_name in ('bytes', 'bytearray'):
            return len(obj)
        if isinstance(obj, (list, tuple)):
            return _estimate_indexable(obj, _depth)
        if isinstance(obj, dict):
            return _estimate_dict(obj, _depth)
        if isinstance(obj, (set, frozenset)):
            return _estimate_set(obj, _depth)
        return sys.getsizeof(obj)
    except (TypeError, AttributeError, IndexError, KeyError, StopIteration):
        return sys.getsizeof(obj)


def _estimate_indexable(obj: Any, _depth: int) -> int:
    """Estimate size of a list or tuple.

    K=3 (first / middle / last) at depth 0 to catch position-correlated
    element sizes from monotonic-build patterns; K=1 (first element) at
    depth >= 1 to bound total recursive cost.
    """
    n = len(obj)
    base = sys.getsizeof(obj)
    if n == 0:
        return base
    if n <= 2 or _depth >= 1:
        return base + n * estimate_object_size(obj[0], _depth + 1)
    samples = (obj[0], obj[n // 2], obj[-1])
    avg = sum(estimate_object_size(s, _depth + 1) for s in samples) // 3
    return base + n * avg


def _estimate_dict(obj: dict, _depth: int) -> int:
    """Estimate size of a dict.

    K=2 (first value + last value) at depth 0; K=1 (first value) at depth
    >= 1.  CPython doesn't support O(1) middle-by-index access, so we
    accept the weaker position-bias detection for dicts.
    """
    n = len(obj)
    base = sys.getsizeof(obj)
    if n == 0:
        return base
    first_val = next(iter(obj.values()))
    if n == 1:
        return base + estimate_object_size(first_val, _depth + 1)
    if _depth >= 1:
        return base + n * estimate_object_size(first_val, _depth + 1)
    last_val = next(reversed(obj.values()))
    avg = (
        estimate_object_size(first_val, _depth + 1) +
        estimate_object_size(last_val, _depth + 1)
    ) // 2
    return base + n * avg


def _estimate_set(obj: Any, _depth: int) -> int:
    """Estimate size of a set or frozenset (unordered; K=1 always)."""
    n = len(obj)
    base = sys.getsizeof(obj)
    if n == 0:
        return base
    sample = next(iter(obj))
    return base + n * estimate_object_size(sample, _depth + 1)
