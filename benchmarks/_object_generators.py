"""Object generators per type family for the ser/deser cost-model bench.

Each family has a generator that produces a representative object of
*approximately* the requested in-memory size. Exact sizing is impossible
in general (pandas/numpy memory layout is not linear in element count
for all dtypes), but we get within a small constant factor — good enough
to span the 1 KB → 100 MB range.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Callable

FAMILIES = [
    "dataframe_numeric",
    "series_numeric",
    "ndarray_dense",
    "sparse",
    "dict_shallow",
    "list_flat",
    "bytes",
]


def _dataframe_numeric(target_bytes: int) -> Any:
    import numpy as np
    import pandas as pd
    # 8 float64 cols × 8 bytes per element = 64 B/row. Add a small header.
    rows = max(1, target_bytes // 64)
    return pd.DataFrame(np.random.rand(rows, 8))


def _series_numeric(target_bytes: int) -> Any:
    import numpy as np
    import pandas as pd
    n = max(1, target_bytes // 8)
    return pd.Series(np.random.rand(n))


def _ndarray_dense(target_bytes: int) -> Any:
    import numpy as np
    n = max(1, target_bytes // 8)
    return np.random.rand(n)


def _sparse(target_bytes: int) -> Any:
    try:
        from scipy.sparse import random as sparse_random
    except ImportError as e:
        raise ImportError("scipy required for sparse family") from e
    # Density 1% — a 1000x1000 1%-dense matrix has ~10000 nnz entries,
    # ~80 bytes each (float64 + indices) → ~800 KB. Scale by sqrt of target.
    nnz_target = max(10, target_bytes // 80)
    n = max(10, int(nnz_target / 0.01) ** 0.5)
    n = int(n)
    return sparse_random(n, n, density=0.01, format="csr")


def _dict_shallow(target_bytes: int) -> Any:
    # CPython dict overhead is ~232 bytes empty + ~96 B/entry (PyObject + hash).
    n = max(1, target_bytes // 100)
    return {f"k{i}": float(i) for i in range(n)}


def _list_flat(target_bytes: int) -> Any:
    # ~28 B/element for float in list (PyObject overhead).
    n = max(1, target_bytes // 28)
    return [float(i) for i in range(n)]


def _bytes(target_bytes: int) -> Any:
    return os.urandom(max(1, target_bytes))


_GENERATORS: dict[str, Callable[[int], Any]] = {
    "dataframe_numeric": _dataframe_numeric,
    "series_numeric": _series_numeric,
    "ndarray_dense": _ndarray_dense,
    "sparse": _sparse,
    "dict_shallow": _dict_shallow,
    "list_flat": _list_flat,
    "bytes": _bytes,
}


def make_object(family: str, target_bytes: int) -> Any:
    """Generate a representative object of ``family`` near ``target_bytes``
    in size. May raise ``ImportError`` for families that need optional deps
    not installed (e.g. ``sparse`` needs scipy)."""
    if family not in _GENERATORS:
        raise KeyError(f"unknown family: {family!r}")
    return _GENERATORS[family](target_bytes)


def estimate_in_memory_size(obj: Any) -> int:
    """Best-effort in-memory size estimate.

    Mirrors the kind of estimate the cash policy uses at runtime: prefer
    framework-native size accessors when available, fall back to
    sys.getsizeof for primitives.
    """
    # pandas
    if hasattr(obj, "memory_usage"):
        try:
            mu = obj.memory_usage(deep=True)
            if hasattr(mu, "sum"):
                return int(mu.sum())
            return int(mu)
        except (TypeError, ValueError):
            pass
    # numpy
    if hasattr(obj, "nbytes"):
        return int(obj.nbytes)
    # scipy.sparse
    if hasattr(obj, "data") and hasattr(obj, "indices") and hasattr(obj, "indptr"):
        return int(obj.data.nbytes + obj.indices.nbytes + obj.indptr.nbytes)
    # generic
    if isinstance(obj, bytes):
        return len(obj)
    if isinstance(obj, (list, tuple)):
        return sys.getsizeof(obj) + sum(sys.getsizeof(x) for x in obj)
    if isinstance(obj, dict):
        return sys.getsizeof(obj) + sum(
            sys.getsizeof(k) + sys.getsizeof(v) for k, v in obj.items()
        )
    return sys.getsizeof(obj)
