"""Pure functions for hashing and sizing arbitrary Python objects.

Used as the `compute_hash_fn` and `calculate_memory_fn` callable seam
threaded into `StatementProcessor` and `UpstreamChecker` (see
``CashMagics._init_processing_components``), and by the upcoming
``Restorer`` to verify restored objects against their stored lineage.

**Anti-god-class rule (load-bearing):** this module is *pure functions*.
No state, no class, no IPython, no ``Cash`` dependency. If a caller
needs context (e.g. "hash relative to lineage X"), the context stays
in the caller — do not grow this module into a class with options.
"""

from __future__ import annotations

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


def _recursive_getsizeof(obj: Any, seen: set[int] | None = None) -> int:
    """Recursively calculate size of an object including its contents.

    More accurate than plain ``sys.getsizeof()`` for containers.
    """
    size = sys.getsizeof(obj)

    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum(_recursive_getsizeof(k, seen) + _recursive_getsizeof(v, seen)
                    for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(_recursive_getsizeof(item, seen) for item in obj)

    return size


def calculate_memory_size(variables_dict: dict[str, Any]) -> int:
    """Calculate the total memory size of output variables using type-specific methods.

    Much faster than ``pickle.dumps()`` on the entire payload.

    Args:
        variables_dict: Dictionary of variable names to their values.

    Returns:
        Total memory size in bytes.
    """
    total_size = 0

    for _var_name, value in variables_dict.items():
        try:
            type_name = type(value).__name__

            if type_name == 'DataFrame':
                try:
                    total_size += value.memory_usage(deep=True).sum()
                    continue
                except (TypeError, AttributeError):
                    pass

            elif type_name == 'ndarray':
                try:
                    total_size += value.nbytes
                    continue
                except (TypeError, AttributeError):
                    pass

            elif type_name == 'Series':
                try:
                    total_size += value.memory_usage(deep=True)
                    continue
                except (TypeError, AttributeError):
                    pass

            total_size += _recursive_getsizeof(value)

        except (TypeError, ValueError, RecursionError):
            try:
                total_size += len(pickle.dumps(value))
            except (TypeError, pickle.PicklingError):
                total_size += sys.getsizeof(value)

    return total_size
