"""Content hashers, one per library cash knows: pandas, numpy, polars,
PyArrow, modin, dask and scipy.sparse.

`builtin_hash` reads every byte of a value together with its schema --
column names, dtypes, an array's memory layout -- so the same values under
two dtypes or two layouts hash apart. The decorator keys arguments on these
hashes (directly, and inside containers through `BUILTIN_CONTENT` or its own
memo around it), and the notebook's value hash (`cash.value_hash`) uses
them for every frame, array and table it meets.

Python objects held inside one (an object column, ``attrs``) are hashed in
their canonical key form (`cash.canonical_form.canonical_bytes`).
`builtin_family_of` remembers per type which hasher claims it.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from typing import Any

from . import _plain_data
from .canonical_form import ContentHashing, canonical_bytes
from .sizing import SPARSE_PARTS
from .value_types import LEAF_TYPES

logger = logging.getLogger(__name__)


def builtin_hash_family(type_: type) -> str | None:
    """Name the built-in content hasher that claims *type_*, or ``None``.

    Asked of a bare TYPE as well as of a value: ``register_hasher`` needs it
    to tell a user that the hasher they are registering would never run.

    Matched on module PREFIX, so a user's own subclass defined in their own
    module is deliberately not claimed -- registering a hasher for it has
    always worked and still does.
    """
    type_name = getattr(type_, "__name__", "")
    module = getattr(type_, "__module__", "") or ""
    if module.startswith("pandas") and type_name in ("DataFrame", "Series"):
        return "pandas"
    if type_name == "ndarray" and module.startswith("numpy"):
        return "numpy"
    if module.startswith("polars"):
        return "polars"
    if module.startswith("pyarrow"):
        return "pyarrow"
    if module.startswith("modin"):
        return "modin"
    if module.startswith("dask"):
        return "dask"
    if module.startswith("scipy.sparse") and hasattr(type_, "tocsr") and hasattr(type_, "format"):
        return "scipy.sparse"
    return None


def builtin_family_of(type_: type) -> str | None:
    """`builtin_hash_family`, remembered per type: asked of every value a
    key walks."""
    try:
        return _FAMILIES[type_]
    except KeyError:
        family = _FAMILIES[type_] = builtin_hash_family(type_)
        return family
    except TypeError:  # a class whose metaclass makes it unhashable
        return builtin_hash_family(type_)


_FAMILIES: dict[type, str | None] = {}


def builtin_hash(value: Any) -> str | None:
    """Every byte of *value*, with its schema, for the library types cash knows.

    A hex digest for pandas, numpy, polars, PyArrow, modin and dask values, or
    ``None`` when no built-in hasher claims *value*'s type or hashing it failed
    -- a normal answer: the caller falls through to its own next step.

    Every hasher folds in what the values alone do not say: column and index
    names, dtypes, the memory layout of an array. The same values under two
    dtypes are two different objects to the code reading them.
    """
    family = builtin_hash_family(type(value))
    if family == "pandas":
        return hash_pandas(value)
    if family == "numpy":
        return hash_numpy(value)
    if family == "polars":
        return hash_polars(value)
    if family == "pyarrow":
        return hash_pyarrow(value)
    if family == "modin":
        return hash_modin(value)
    if family == "dask":
        return hash_dask(value)
    if family == "scipy.sparse":
        return hash_sparse(value)
    return None


#: How a key walk reads a frame, array or table with no memo in front:
#: every byte, through `builtin_hash`.
BUILTIN_CONTENT = ContentHashing(builtin_family_of, builtin_hash)


def hash_pandas(value: Any) -> str | None:
    """Hash a pandas DataFrame or Series over values AND schema.

    ``hash_pandas_object`` covers row values + index values but NOT the
    schema labels: column names, ``Series.name``, and index name(s) are
    invisible to it, so ``df.rename(columns=...)`` (or an empty frame of any
    shape) collided with the original and returned its cached result. Fold
    the labels in as a digest prefix.

    The dtypes go in for the same reason, and it is the sharper one: the same
    values under two dtypes are two different objects to the body. A tz-naive
    and a tz-aware series collided, and the tz-aware call was served the naive
    one's ``TypeError: Cannot convert tz-naive timestamps``; so did
    ``int64``/``Int64`` (pd.NA semantics), ``int64``/``int32`` and a
    categorical against an object column.

    The values are hashed column by column and index level by index level
    (`_fold_pandas_values`), each in the form that keeps it exact.
    """
    try:
        import pandas as pd

        index = value.index
        index_dtypes = [_pandas_dtype_key(dt) for dt in getattr(index, "dtypes", [index.dtype])]
        # What the labels and dtypes leave out, and code reads: the column
        # axis's own name (``melt``, ``stack`` and ``reset_index`` name
        # columns after it), an index's ``freq`` (``shift(freq=...)``,
        # ``asfreq``) and ``attrs``.
        axes = f"{list(index.names)!r}:{index_dtypes!r}:{getattr(index, 'freq', None)!r}:"
        if type(value).__name__ == "DataFrame":
            dtypes = [_pandas_dtype_key(dt) for dt in value.dtypes]
            schema = f"{list(value.columns)!r}:{list(value.columns.names)!r}:{dtypes!r}:{axes}"
        else:  # Series
            schema = f"{value.name!r}:{_pandas_dtype_key(value.dtype)!r}:{axes}"
        h = hashlib.sha256(schema.encode("utf-8"))
        if value.attrs:
            h.update(canonical_bytes(value.attrs, BUILTIN_CONTENT))
        _fold_pandas_values(h, value, pd)
        return h.hexdigest()
    except (ImportError, TypeError, ValueError, AttributeError, pickle.PicklingError):
        logger.debug("Failed to hash pandas %s via hash_pandas_object", type(value).__name__)
        return None


def _fold_pandas_values(h: Any, value: Any, pd: Any) -> None:
    """Fold a frame's or series' values, then its index's, into *h*.

    ``hash_pandas_object`` keys an array of Python objects by ``str()`` of
    each: when one value is not a string it stringifies the whole column, so
    ``1`` and ``'1'``, ``True`` and ``'True'``, ``b'a'`` and ``'a'``, or a date
    and its ISO string hashed alike, and ``s * 2`` was served ``2`` for
    ``'11'``. The same held for an index of them. Such an array -- object
    dtype, and pandas' string dtypes, whose values are Python strings -- is
    keyed by its items' pickled form instead (`_object_items_bytes`), which is
    also several times faster than ``hash_pandas_object`` on strings. A
    categorical is keyed by its codes, the categories being in the schema.
    Everything else keeps ``hash_pandas_object``'s per-value hash, which reads
    the raw bits of numbers and dates.
    """
    if type(value).__name__ == "DataFrame":
        rest = []
        for pos, dtype in enumerate(value.dtypes):
            if _pandas_value_route(dtype, pd) is None:
                rest.append(pos)
            else:
                h.update(f"|{pos}|".encode())
                _fold_pandas_array(h, value.iloc[:, pos], pd)
        if rest:
            others = value if len(rest) == value.shape[1] else value.iloc[:, rest]
            h.update(_np_bytes(pd.util.hash_pandas_object(others, index=False)))
    else:
        _fold_pandas_array(h, value, pd)
    index = value.index
    if isinstance(index, pd.MultiIndex):
        # Each level's distinct values, then which one each row holds.
        for level, codes in zip(index.levels, index.codes):
            h.update(b"|level|")
            _fold_pandas_array(h, level, pd)
            h.update(codes.tobytes())
    elif isinstance(index, pd.RangeIndex):
        h.update(f"|range({index.start}, {index.stop}, {index.step})".encode())
    else:
        h.update(b"|index|")
        _fold_pandas_array(h, index, pd)


def _pandas_value_route(dtype: Any, pd: Any) -> str | None:
    """How `_fold_pandas_array` keys values of *dtype*: ``"objects"``,
    ``"codes"``, or None for ``hash_pandas_object``."""
    if isinstance(dtype, pd.CategoricalDtype):
        return "codes"
    if str(dtype) == "object" or isinstance(dtype, pd.StringDtype):
        return "objects"
    return None


def _fold_pandas_array(h: Any, values: Any, pd: Any) -> None:
    """Fold one Series' or Index's values into *h* (`_fold_pandas_values`)."""
    route = _pandas_value_route(values.dtype, pd)
    if route == "codes":
        h.update(_np_bytes(values.codes if isinstance(values, pd.Index) else values.cat.codes))
    elif route == "objects":
        h.update(_object_items_bytes(_np_array(values, object).tolist()))
    else:
        h.update(_np_bytes(pd.util.hash_pandas_object(values, index=False)))


def _np_array(values: Any, dtype: Any = None) -> Any:
    """``np.asarray(values)``: a pandas value's array through ``__array__``,
    not ``to_numpy()``, which the decorator's frame memo watches for handles
    a caller could write through (`arg_hashing.watch_writable_handles`)."""
    import numpy as np

    return np.asarray(values, dtype=dtype)


def _np_bytes(values: Any) -> bytes:
    return _np_array(values).tobytes()


def _object_items_bytes(items: list) -> bytes:
    """The bytes a list of Python objects keys on: plain data pickled as it
    is (`_plain_data`, C speed), anything else in its canonical form
    (`stable_key_repr`), so sets and dicts inside are in a stable order."""
    if _plain_data.is_plain(items):
        return b"P" + _plain_data.pickle_unshared(items)
    # The array or column holds each item besides *items*.
    tree = _plain_data.sharing(items, tree=True, held_twice_at=0)
    if tree is not None:
        return b"T" + _plain_data.pickle_unshared((items, tree[0]))
    return b"S" + canonical_bytes(items, BUILTIN_CONTENT)


def _pandas_dtype_key(dtype: Any) -> str:
    """A pandas dtype as a key reads it: ``repr``, and all of a categorical.

    ``str`` is ``'category'`` for every categorical, and ``repr`` elides
    a long list of categories, so the categories (every one, by content) and
    the ``ordered`` flag are spelled out: ``get_dummies``, ``value_counts`` and ``cat.codes``
    read them, and two series of the same values over different categories
    were served each other's columns.
    """
    categories = getattr(dtype, "categories", None)
    if categories is None or getattr(dtype, "name", None) != "category":
        return repr(dtype)
    listed = hashlib.sha256(_object_items_bytes(categories.tolist())).hexdigest()
    return f"category:{dtype.ordered!r}:{_pandas_dtype_key(categories.dtype)}:{listed}"


def array_layout(value: Any) -> str:
    """The order *value*'s axes are laid out in memory: ``C``, ``F`` or ``K…``.

    The key used to fold in the raw strides (0fd2cb5), which separated C-
    from F-ordered arrays -- the point -- but also a strided VIEW from its
    contiguous copy. Those hold the same values in the same memory order,
    so no order-reading callee (``ravel(order='A'/'K')``, ``reshape``) tells
    them apart -- only ``.flags`` does, and a result computed FROM
    contiguity now shares an entry between the two, knowingly. What did
    tell them apart, on every run, was the cache itself. A function that
    returned ``arr[:, 0]`` handed its caller a view on the computing run and
    a contiguous copy on every restored one, so the caller's key changed
    between the two and its expensive step ran twice after every upstream
    edit.

    So: the axes of length > 1, ordered by |stride| from outermost in, with
    a broadcast (zero-stride) axis outermost. Identity is ``C``, reversed is
    ``F``; anything else spells the permutation. Stride MAGNITUDE and sign
    do not change what a memory-order read returns, so they stay out.

    Except for one flag. ``order='A'`` (``ravel``, ``reshape``, ``tobytes``,
    ``copy``) reads in Fortran order only when the array is F-CONTIGUOUS,
    and C order otherwise -- so an F-like strided view (``a.T[::2]``) and
    its F-contiguous copy read differently, and sharing ``F`` handed one the
    other's result. ``Fs`` is the F-like array that is not
    F-contiguous. Nothing else needs the flag: with two or more axes longer
    than 1, only an F-like layout can be F-contiguous, and ``order='A'``
    reads everything else in C order, as ``C`` and ``K…`` already imply.

    One case still re-keys once: an F-like but non-contiguous view is stored
    by pickle as a C-ordered copy, and it genuinely ravels differently from
    one, so the restored value must key apart. Safe direction.
    """
    axes = [(axis, stride) for axis, (n, stride) in enumerate(zip(value.shape, value.strides)) if n > 1]
    if len(axes) <= 1:
        return "C"
    outer_first = sorted(axes, key=lambda a: (-abs(a[1]) if a[1] else float("-inf"), a[0]))
    perm = tuple(axis for axis, _ in outer_first)
    natural = tuple(axis for axis, _ in axes)
    if perm == natural:
        return "C"
    if perm == natural[::-1]:
        return "F" if value.flags.f_contiguous else "Fs"
    return "K" + ",".join(map(str, perm))


def hash_numpy(value: Any) -> str | None:
    """Hash a numpy ndarray over its FULL contents.

    Correctness requires hashing every byte, not a sample: two large arrays
    that differ only outside a sampled window would otherwise collide and
    return a wrong cached result (a silent data-corruption bug, especially for
    the large ML/data arrays caching targets). Shape and dtype are folded in
    so a reshape or retype of the same bytes does not collide. Uses a
    zero-copy ``memoryview`` for contiguous arrays and falls back to
    ``tobytes()`` (C-order copy) otherwise.

    The LAYOUT is folded in too -- the order the axes sit in memory, see
    `array_layout` -- because the C-order fallback above erases it. Without
    it a C-ordered and an F-ordered array holding equal values hash
    identically, and a layout-sensitive callee is served the other one's
    result: measured, ``np.ravel(x, order='A')`` returned ``[0, 1, 2, …]``
    for an F-ordered input whose true answer is ``[0, 4, 8, 1, …]``.
    Normalising to C-order is right for value EQUALITY and wrong for a KEY.
    """
    try:
        h = hashlib.sha256(f"{value.shape}:{value.dtype}:{array_layout(value)}:".encode())
        if getattr(value.dtype, "hasobject", False):
            # object-dtype arrays: the buffer holds raw PyObject *pointers*,
            # not content, so tobytes() hashes memory addresses - identical
            # content in fresh objects never collides (permanent misses,
            # cross-process-unstable) and address reuse could alias distinct
            # content onto one key. Hash the elements' stable representation
            # instead (canonicalising nested sets/dicts so the key is order-
            # and PYTHONHASHSEED-independent).
            h.update(_object_items_bytes(value.tolist()))
            return h.hexdigest()
        try:
            h.update(memoryview(value).cast("B"))  # no copy if C-contiguous
        except (TypeError, ValueError):
            h.update(value.tobytes())  # non-contiguous / odd layout
        return h.hexdigest()
    except (TypeError, ValueError, AttributeError, MemoryError, pickle.PicklingError):
        logger.debug("Failed to hash numpy ndarray")
        return None


def hash_polars(value: Any) -> str | None:
    """Hash a polars DataFrame, Series, or LazyFrame, schema included.

    ``hash_rows()`` and ``hash()`` see the values only: an ``Int32`` and an
    ``Int64`` column holding the same numbers, or a renamed column, would
    collide. The schema -- names and dtypes -- is folded in ahead of them.

    An ``Object`` column is keyed by its items' content
    (`_object_items_bytes`). Polars hashes Object values with Python's
    ``hash()``, which panics on an unhashable one (an ndarray, a dict) and is
    no content key: ``hash(-1) == hash(-2)``, ``1``, ``1.0`` and ``True``
    share a hash, and a string's changes with ``PYTHONHASHSEED``.

    A ``LazyFrame`` is identified by ``serialize()``, never by ``explain()``.
    ``explain()`` renders the human-readable QUERY PLAN, and two frames over
    different in-memory data print identically -- both
    ``pl.DataFrame({"x": [1, 2, 3]}).lazy()`` and the same over
    ``[10, 20, 30]`` are ``DF ["x"]; PROJECT */1 COLUMNS``, so the second
    call was served the first's result. ``serialize()`` carries the plan
    *and* the data the plan closes over, and is byte-identical across
    processes, so persisted entries still hit after a restart. A plan
    ``serialize()`` refuses gets no built-in hash at all.

    KNOWN GAP: a plan that reads from an external source
    (``scan_csv``/``scan_parquet``/...) serializes the PATH, not the file's
    contents, so editing that file in place does not move the key. Closing
    that would mean collecting the frame to build a cache key, which defeats
    the point of a LazyFrame and can be arbitrarily expensive. Collect before
    passing, or name the file with ``file_depends_on=``.
    """
    try:
        import polars as pl

        if isinstance(value, pl.DataFrame):
            h = hashlib.sha256(f"{value.schema}:{value.height}:".encode("utf-8"))
            objects = [name for name, dtype in value.schema.items() if dtype == pl.Object]
            rest = value.drop(objects) if objects else value
            if rest.width:
                h.update(rest.hash_rows().to_numpy().tobytes())
            for name in objects:
                h.update(f"|{name!r}|".encode("utf-8"))
                h.update(_object_items_bytes(value.get_column(name).to_list()))
            return h.hexdigest()
        if isinstance(value, pl.Series):
            h = hashlib.sha256(f"{value.name!r}:{value.dtype}:{len(value)}:".encode("utf-8"))
            if value.dtype == pl.Object:
                h.update(_object_items_bytes(value.to_list()))
            else:
                h.update(value.hash().to_numpy().tobytes())
            return h.hexdigest()
        if isinstance(value, pl.LazyFrame):
            try:
                return hashlib.sha256(value.serialize()).hexdigest()
            except Exception:  # noqa: BLE001 - polars raises its own types
                logger.debug("polars LazyFrame serialize() failed; no built-in hash for it")
                return None
    except (ImportError, TypeError, ValueError, AttributeError):
        logger.debug("Failed to hash polars %s", type(value).__name__)
    except BaseException as exc:
        if not is_native_panic(exc):
            raise
        logger.debug("polars panicked hashing a %s: %s", type(value).__name__, exc)
    return None


def held_objects(value: Any) -> list | None:
    """The Python objects a library value keeps in object storage, other
    than plain leaves; None when it keeps none.

    A numpy ``object`` array, a pandas ``object`` column and a polars
    ``Object`` column hold arbitrary objects -- a model, a function -- where
    no attribute walk reaches them: in the array buffer, in the blocks. Their
    content hashers pickle such an object by reference, so its code is in no
    key unless the code search is handed them here. Strings, numbers and
    dates, which fill most object columns, are left out at C speed.
    """
    family = builtin_family_of(type(value))
    columns: list = []
    try:
        if family == "numpy":
            if getattr(value.dtype, "hasobject", False):
                columns.append(value.ravel(order="K"))
        elif family == "pandas":
            frame = type(value).__name__ == "DataFrame"
            for pos, dtype in enumerate(value.dtypes if frame else [value.dtype]):
                if str(dtype) == "object":
                    columns.append(_np_array(value.iloc[:, pos] if frame else value, object))
        elif family == "polars":
            import polars as pl

            if isinstance(value, pl.DataFrame):
                columns.extend(value.get_column(n).to_list() for n, dt in value.schema.items() if dt == pl.Object)
            elif isinstance(value, pl.Series) and value.dtype == pl.Object:
                columns.append(value.to_list())
    except Exception:  # a library's internals changed: nothing found
        logger.debug("Could not list the objects a %s holds", type(value).__name__, exc_info=True)
        return None
    found: list = []
    for column in columns:
        if not all(k in LEAF_TYPES for k in set(map(type, column))):
            found.extend(v for v in column if type(v) not in LEAF_TYPES)
    return found or None


def is_native_panic(exc: BaseException) -> bool:
    """Is *exc* a panic raised out of a Rust extension (``pyo3``), such as
    polars'? It derives from ``BaseException``, so no ``except Exception``
    stops it, and a panic while building a key ended the user's call."""
    return type(exc).__name__ == "PanicException"


class _HashSink:
    """A write-only file that feeds what is written to it into a hash."""

    closed = False

    def __init__(self, h: Any) -> None:
        self._h = h

    def write(self, data: Any) -> int:
        self._h.update(data)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def hash_pyarrow(value: Any) -> str | None:
    """Hash a PyArrow Table or RecordBatch by what it holds, not its buffers.

    The table is written as an Arrow IPC stream into the hash, never into
    memory. The raw buffers are not the content: a slice (``t.slice(2, 2)``,
    each batch of ``to_batches()``) shares its parent's buffers and differs
    only in an offset they do not show, so every equal-length slice of one
    table hashed alike and was served the first one's result. A dictionary
    column's buffers hold only the indices, so ``["red", "blue"]`` and
    ``["cat", "dog"]`` hashed alike too. The IPC stream carries the schema,
    each batch's rows from its offset, and every dictionary.
    """
    try:
        import pyarrow as pa

        if isinstance(value, (pa.Table, pa.RecordBatch)):
            h = hashlib.sha256(f"{type(value).__name__}:{value.num_rows}:".encode())
            sink = pa.PythonFile(_HashSink(h), mode="w")
            with pa.ipc.new_stream(sink, value.schema) as writer:
                writer.write(value)
            return h.hexdigest()
    except (ImportError, TypeError, ValueError, AttributeError, MemoryError, NotImplementedError):
        logger.debug("Failed to hash PyArrow %s", type(value).__name__)
    return None


def hash_modin(value: Any) -> str | None:
    """Hash a modin DataFrame or Series as the pandas one it converts to --
    schema included (`hash_pandas`)."""
    try:
        return hash_pandas(value._to_pandas() if hasattr(value, "_to_pandas") else value)
    except (TypeError, ValueError, AttributeError):
        logger.debug("Failed to hash modin %s", type(value).__name__)
        return None


def hash_dask(value: Any) -> str | None:
    """Hash a dask collection by its task-graph keys, plus its schema.

    The keys carry a data-derived token. The schema is the collection's
    ``_meta``, the empty pandas frame or numpy array that stands for its
    columns and dtypes, hashed as that type is.
    """
    try:
        h = hashlib.sha256(str(value.__dask_keys__()).encode("utf-8"))
        meta = getattr(value, "_meta", None)
        if meta is not None:
            h.update(f":{builtin_hash(meta)}".encode("utf-8"))
        return h.hexdigest()
    except (TypeError, ValueError, AttributeError):
        logger.debug("Failed to hash dask object via __dask_keys__")
        return None


def hash_sparse(value: Any) -> str | None:
    """Hash a scipy sparse matrix or array by its content, format included.

    The type, shape and dtype, then the arrays its format keeps (`SPARSE_PARTS`,
    and a COO's coordinates), each hashed as an array (`hash_numpy`), so an
    explicit zero, an unsorted index or an ``int32`` against an ``int64``
    index keys apart: code reading ``.data`` or ``.indices`` sees them. A
    DOK or LIL matrix, whose entries live in Python dicts and lists, is keyed
    as the CSR matrix it converts to. Without this every sparse argument --
    a TF-IDF matrix, a one-hot encoding -- ran uncached.
    """
    try:
        t = type(value)
        h = hashlib.sha256(f"{t.__module__}.{t.__qualname__}:{value.shape}:{value.dtype}:{value.format}:".encode())
        if value.format in ("dok", "lil"):
            canon = value.tocsr()
            canon.sort_indices()
            arrays = [(name, getattr(canon, name)) for name in ("data", "indices", "indptr")]
        else:
            arrays = [(name, getattr(value, name)) for name in SPARSE_PARTS if hasattr(value, name)]
            coords = getattr(value, "coords", None)
            if coords is None and value.format == "coo":
                coords = (value.row, value.col)
            if coords is not None:
                arrays.extend((f"coords{i}", c) for i, c in enumerate(coords))
        for name, array in arrays:
            digest = hash_numpy(array)
            if digest is None:
                return None
            h.update(f"|{name}:{digest}".encode())
        return h.hexdigest()
    except (TypeError, ValueError, AttributeError, MemoryError):
        logger.debug("Failed to hash scipy sparse %s", type(value).__name__)
        return None
