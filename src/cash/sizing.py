"""How big a value is, from one set of rules for what a frame, an array or a
sparse matrix holds.

* ``estimate_object_size`` -- order of magnitude, bounded by a depth cap and
  sampling: statement budgets, chunk sizes, restore-cost estimates.
* ``memory_footprint`` -- every item of a container counted once, for the RAM
  tier's byte cap, where one big value missed by a sample would overrun it.
* ``pandas_nbytes`` -- a frame's ``memory_usage(deep=True)`` without paying
  for it; ``pickled_size_estimate`` -- about what a value pickles to.

Functions only; nothing here hashes or stores a value.
"""

from __future__ import annotations

import dataclasses
import logging
import pickle
import random
import sys
from typing import Any

from . import _plain_data
from .value_types import BUILTIN_CONTAINERS, CODELESS_PRIMS, PLAIN_SEQS

logger = logging.getLogger(__name__)


#: The arrays a scipy sparse matrix is, by format: everything its values
#: and their positions are stored in.
SPARSE_PARTS = ("data", "indices", "indptr", "offsets")


# The in-memory size of a pandas frame or series, without ``memory_usage``.
#
# ``DataFrame.memory_usage()`` builds a result Series -- one per column, then
# concatenated with the index's -- and that is nearly all it costs: ~0.23 ms
# for a 200-row, five-column frame, deep or not, against ~0.05 ms to sum the
# column arrays' ``nbytes`` to the same number. Cash sized every stored frame
# twice, once for the RAM tier's cap and once for the restore-cost estimate;
# in a loop over a thousand small files that was 18% of the cell.
#
# ``deep=True`` is also unbounded where it differs at all: a column of Python
# objects is walked value by value, seconds for millions of strings. Those are
# sampled here instead -- the callers want the order of magnitude, not the
# byte. Arrow-backed strings (pandas 3's default) report their real size as
# ``nbytes``.

#: Values looked at per column of Python objects.
_OBJECT_SAMPLE = 64


def _holds_python_objects(dtype: Any) -> bool:
    """Object dtype, or strings stored as Python objects rather than in Arrow.

    By name: categorical and string dtypes report ``kind == 'O'`` too, and
    their arrays' ``nbytes`` is already the real size."""
    return str(dtype) == "object" or getattr(dtype, "storage", None) == "python"


def _sampled_bytes(values: Any) -> int:
    """Pointers plus the sampled mean size of what they point at.

    A random sample, seeded by the length so one value always gets one size.
    Not every k-th value: data with a period -- rows cycling through a few
    shapes -- lands a fixed stride on one of them (measured: 22% low on a
    column repeating every 5 rows, stride 3,125).
    """
    n = len(values)
    if not n:
        return 0
    if n <= _OBJECT_SAMPLE:
        sample = values
    else:
        sample = values[sorted(random.Random(n).sample(range(n), _OBJECT_SAMPLE))]
    return 8 * n + int(sum(map(sys.getsizeof, sample)) / len(sample) * n)


def _column_bytes(col: Any) -> int:
    """One column's data, as ``memory_usage(deep=True)`` counts it."""
    if str(col.dtype) == "category":
        # Codes, plus the categories counted deep: they are strings held as
        # Python objects on pandas < 3.
        categorical = col.array
        return int(categorical.codes.nbytes) + _index_bytes(categorical.categories)
    if _holds_python_objects(col.dtype):
        return _sampled_bytes(col.to_numpy(dtype=object))
    return int(col.array.nbytes)


def _index_bytes(index: Any) -> int:
    """The index's own ``nbytes``: a RangeIndex is a few numbers, not n of them."""
    if type(index).__name__ != "MultiIndex" and _holds_python_objects(index.dtype):
        return _sampled_bytes(index.to_numpy(dtype=object))
    return int(index.nbytes)


def _arrays_bytes(obj: Any) -> int | None:
    """A frame's column arrays summed straight from its block manager, or None
    when a column needs a closer look (objects, categories) or the manager's
    shape is not the one known. ``items()`` boxes every column as a Series:
    ~1.5 ms a call for a 20-column group frame, sized once per element of a
    comprehension over 360 of them."""
    try:
        arrays = obj._mgr.arrays
    except Exception:  # noqa: BLE001 - a private attribute: any surprise means "no"
        return None
    total = 0
    for arr in arrays:
        dtype = getattr(arr, "dtype", None)
        if dtype is None or _holds_python_objects(dtype) or str(dtype) == "category":
            return None
        nbytes = getattr(arr, "nbytes", None)
        if not isinstance(nbytes, int):
            return None
        total += nbytes
    return total


def pandas_nbytes(obj: Any) -> int | None:
    """The size of a DataFrame or Series, index included; None for anything else."""
    kind = type(obj).__name__
    try:
        if kind == "DataFrame":
            fast = _arrays_bytes(obj)
            if fast is not None:
                return _index_bytes(obj.index) + fast
            return _index_bytes(obj.index) + sum(_column_bytes(col) for _, col in obj.items())
        if kind == "Series":
            return _index_bytes(obj.index) + _column_bytes(obj)
    except (AttributeError, TypeError, ValueError):
        return None
    return None


#: How deep into tuples, lists and dicts `pickled_size_estimate` looks.
_ESTIMATE_DEPTH = 2


def _pickled_item_cost(item: Any) -> int:
    """What one Python object adds to a pickle, the first time it is written."""
    if type(item) is str:
        n = len(item.encode("utf-8", "surrogatepass"))
        return n + (2 if n < 256 else 5) + 1  # opcode, length, memo
    if type(item) in (int, float, bool) or item is None:
        return 9
    try:
        return len(pickle.dumps(item, protocol=5))
    except Exception:  # noqa: BLE001 - an estimate: count what cannot be seen as small
        return 8


def _objects_estimate(values: Any, seen: set[int]) -> int:
    """Python objects, sampled. Pickle writes an object once and each repeat
    of the SAME object as a memo reference -- two bytes while the memo is
    small, five past 256 entries -- while equal strings that are distinct
    objects are each written whole. An object sampled from an earlier array
    counts as repeated: on pandas 2 a column taken from a frame is a new
    array of the same strings."""
    n = len(values)
    if not n:
        return 0
    if n <= _OBJECT_SAMPLE:
        sample = list(values)
    else:
        sample = [values[i] for i in sorted(random.Random(n).sample(range(n), _OBJECT_SAMPLE))]
    distinct = {id(v): v for v in sample}
    new = [v for k, v in distinct.items() if k not in seen]
    seen.update(distinct)
    share = len(new) / len(sample)
    # A sample that keeps meeting the same objects has seen about all there
    # are; otherwise their count scales with the column.
    objects = len(distinct) if len(distinct) < len(sample) // 2 else share * n
    repeat = 2 if objects < 256 else 5
    if not new:
        return repeat * n
    mean = sum(map(_pickled_item_cost, new)) / len(new)
    return int(n * (share * mean + (1 - share) * repeat))


def _array_estimate(arr: Any, seen: set[int]) -> int:
    """One array's share of a pickle, once per array object (pickle writes a
    shared one once; a pandas 3 frame shares string columns with its series)."""
    if id(arr) in seen:
        return 0
    seen.add(id(arr))
    if str(getattr(arr, "dtype", "")) == "category":
        return _array_estimate(arr.codes, seen) + _array_estimate(arr.categories.to_numpy(dtype=object), seen)
    if type(arr).__name__ == "ndarray":
        if arr.dtype.kind != "O":
            return int(arr.nbytes)
        # pandas 2 hands out its 2-D blocks: a column of strings is one row
        # of one, which sampled as ONE item pickled it whole.
        return _objects_estimate(arr.reshape(-1) if arr.ndim != 1 else arr, seen)
    if _holds_python_objects(getattr(arr, "dtype", None)):
        return _objects_estimate(arr.to_numpy(dtype=object), seen)
    nbytes = getattr(arr, "nbytes", None)  # Arrow and other extension arrays
    return int(nbytes) if isinstance(nbytes, int) else 0


def _index_estimate(index: Any, seen: set[int]) -> int:
    """A RangeIndex pickles as its three numbers, not as the values."""
    if type(index).__name__ == "RangeIndex":
        return 0
    return _array_estimate(index._values, seen)


def pickled_size_estimate(value: Any, _depth: int = 0, _seen: set[int] | None = None) -> int:
    """About what ``pickle.dumps(value)`` comes to, read off the arrays.

    For frames, series, numpy arrays and tuples, lists and dicts of them;
    0 for anything else. Fixed-width data counts exactly; Python objects are
    sampled. Enough to see that a value is far too big to be worth storing
    without pickling it to find out: a 1.7 GiB result took 2.7 s to pickle,
    after 2.8 s of compute.
    """
    seen = set() if _seen is None else _seen
    kind = type(value).__name__
    try:
        if kind == "DataFrame":
            return sum(_array_estimate(arr, seen) for arr in value._mgr.arrays) + _index_estimate(value.index, seen)
        if kind == "Series":
            # ``_values``: the ndarray itself for a numpy dtype, else the extension array.
            return _array_estimate(value._values, seen) + _index_estimate(value.index, seen)
        if kind == "ndarray":
            return _array_estimate(value, seen)
    except Exception:  # noqa: BLE001 - an estimate is optional: none is 0
        return 0
    if _depth >= _ESTIMATE_DEPTH:
        return 0
    if type(value) in (tuple, list):
        return sum(pickled_size_estimate(v, _depth + 1, seen) for v in value)
    if type(value) is dict:
        return sum(pickled_size_estimate(v, _depth + 1, seen) for v in value.values())
    return 0


# Recursion-depth cap for ``estimate_object_size`` walks; bounds total cost.
_MAX_ESTIMATE_DEPTH = 4

# scipy.sparse type names. Dispatched via type-name string to avoid an
# import-time dependency on scipy.
_SPARSE_CSR_CSC_TYPES = frozenset(
    {
        "csr_matrix",
        "csc_matrix",
        "csr_array",
        "csc_array",
    }
)
_SPARSE_COO_TYPES = frozenset({"coo_matrix", "coo_array"})


def _est_sparse_csr_csc(m: Any) -> int:
    return int(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes)


def _est_sparse_coo(m: Any) -> int:
    return int(m.data.nbytes + m.row.nbytes + m.col.nbytes)


def _data_size(obj: Any) -> int | None:
    """The bytes a frame, array or sparse matrix holds, or None for anything
    else -- a builtin, or an object that does not say.

    A frame is summed from its column arrays (`pandas_nbytes`); an array, an
    Arrow table or a tensor reports ``nbytes``. Always a Python ``int``: the
    RAM tier writes this into entry metadata, and a ``numpy.int64`` there made
    the metadata unreadable without numpy (``%cash_on`` failed with a
    ModuleNotFoundError).
    """
    kind = type(obj)
    if kind in CODELESS_PRIMS or kind in BUILTIN_CONTAINERS:
        return None
    type_name = kind.__name__
    if type_name in ("DataFrame", "Series"):
        size = pandas_nbytes(obj)
        if size is not None:
            return size
        try:
            usage = obj.memory_usage(deep=True)
            return int(usage.sum() if type_name == "DataFrame" else usage)
        except (TypeError, AttributeError, ValueError):
            return None
    if type_name in _SPARSE_CSR_CSC_TYPES:
        return _est_sparse_csr_csc(obj)
    if type_name in _SPARSE_COO_TYPES:
        return _est_sparse_coo(obj)
    try:
        nbytes = getattr(obj, "nbytes", None)
    except Exception:  # noqa: BLE001 - a property of someone else's class: any failure means "does not say"
        return None
    if isinstance(nbytes, int) or type(nbytes).__name__.startswith(("int", "uint")):
        try:
            return int(nbytes)
        except (TypeError, ValueError):
            return None
    return None


def estimate_object_size(obj: Any, _depth: int = 0) -> int:
    """Estimate the memory size of an object in bytes.

    Uses ``sys.getsizeof`` for basic types and `_data_size` for frames,
    arrays and sparse matrices.
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
        size = _data_size(obj)
        if size is not None:
            return size
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            base = sys.getsizeof(obj)
            return base + sum(estimate_object_size(getattr(obj, f.name), _depth + 1) for f in dataclasses.fields(obj))
        if isinstance(obj, tuple) and hasattr(obj, "_fields"):  # namedtuple
            base = sys.getsizeof(obj)
            return base + sum(estimate_object_size(getattr(obj, name), _depth + 1) for name in obj._fields)
        if type_name in ("bytes", "bytearray"):
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
    avg = (estimate_object_size(first_val, _depth + 1) + estimate_object_size(last_val, _depth + 1)) // 2
    return base + n * avg


def _estimate_set(obj: Any, _depth: int) -> int:
    """Estimate size of a set or frozenset (unordered; K=1 always)."""
    n = len(obj)
    base = sys.getsizeof(obj)
    if n == 0:
        return base
    sample = next(iter(obj))
    return base + n * estimate_object_size(sample, _depth + 1)


def memory_footprint(obj: Any, _seen: set[int] | None = None) -> int:
    """What *obj* takes in memory, for the RAM tier's byte cap: every item of
    a dict, list, tuple or set counted, each object once however often it is
    referenced.

    `estimate_object_size` samples a dict's first and last values; a cap sized
    that way would miss the one big frame in the middle of a statement's
    variables. Plain data -- lists and tuples of primitives -- is summed a level
    at a time (`cash._plain_data.size_of`): the per-item walk took 3.5 s for two
    million parsed rows, and every notebook entry holds the RNG
    state, a tuple of 625 ints one dict down (1.7M calls in one cell).
    """
    seen = set() if _seen is None else _seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if type(obj) in PLAIN_SEQS:
        plain = _plain_data.size_of(obj)
        if plain is not None:
            return plain
    try:
        size = _data_size(obj)
        if size is not None:
            return size
        size = sys.getsizeof(obj)
        if isinstance(obj, dict):
            size += sum(memory_footprint(v, seen) for v in obj.values())
            size += sum(memory_footprint(k, seen) for k in obj)
        elif isinstance(obj, (list, tuple, set, frozenset)):
            size += sum(memory_footprint(i, seen) for i in obj)
        return size
    except (TypeError, RecursionError, ValueError):
        logger.debug("Could not estimate size of %s object", type(obj).__name__, exc_info=True)
        return 0
