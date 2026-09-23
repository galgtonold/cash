"""Pure functions for hashing and sizing arbitrary Python values.

One module for every value hash cash takes, so the decorator and the notebook
cannot drift apart on what makes two values the same:

* ``builtin_hash`` and the per-library hashers under it (pandas, numpy,
  polars, PyArrow, modin, dask) read every byte of a value together with its
  schema -- column names, dtypes, an array's memory layout. The decorator keys
  arguments on them; ``compute_hash_full`` uses them for the notebook's
  per-iteration loop keys and call keys.
* ``stable_key_repr`` is the canonical form a key pickles a value in: sets and
  dicts in a stable order, every container tagged with its type.
* ``compute_hash`` SAMPLES large values. It is the ``compute_hash_fn`` seam
  threaded into ``StatementProcessor`` and ``UpstreamChecker``, and what
  ``Restorer`` checks a restored object against: a cheap freshness signal,
  never a key discriminator.

And every size cash estimates, from one set of rules for what a frame, an
array or a sparse matrix holds:

* ``estimate_object_size`` -- order of magnitude, bounded by a depth cap and
  sampling: statement budgets, chunk sizes, restore-cost estimates.
* ``memory_footprint`` -- every item of a container counted once, for the RAM
  tier's byte cap, where one big value missed by a sample would overrun it.
* ``pandas_nbytes`` -- a frame's ``memory_usage(deep=True)`` without paying
  for it; ``pickled_size_estimate`` -- about what a value pickles to.

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
import random
import sys
from typing import Any

from . import _plain_data
from .value_types import BUILTIN_CONTAINERS, CODELESS_PRIMS, PLAIN_SEQS

logger = logging.getLogger(__name__)

_HASH_ERRORS = (TypeError, ValueError, AttributeError, pickle.PicklingError)


# ---------------------------------------------------------------------------
# The canonical form a key pickles a value in
# ---------------------------------------------------------------------------


class CyclicValueError(TypeError):
    """A value whose object graph loops back on itself and holds a set."""


def object_state(value: Any) -> dict:
    """Return an object's instance state as a name -> value dict, covering both
    ``__dict__`` and ``__slots__`` (collected across the MRO so slots declared
    on base classes are included). Builtins and leaf values yield ``{}``. Used
    so set-canonicalisation reaches a set buried inside a ``__slots__`` object,
    not just a ``__dict__``-backed one.
    """
    state: dict = {}
    obj_dict = getattr(value, "__dict__", None)
    if isinstance(obj_dict, dict):
        state.update(obj_dict)
    for klass in type(value).__mro__:
        slots = getattr(klass, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name in ("__dict__", "__weakref__") or name in state:
                continue
            try:
                state[name] = getattr(value, name)
            except AttributeError:
                pass  # slot declared but never assigned
    return state


#: The tag a builtin container is keyed under: one string object per type, so a
#: key's pickle stores it once however many containers it holds.
_BUILTIN_CONTAINER_TAGS = {t: t.__qualname__ for t in (dict, list, tuple, set, frozenset)}


def _typed(value: Any, canon: Any) -> tuple:
    """*canon*, a container's canonical items, tagged with the container's type.

    Every container carries its type, so containers holding equal items key
    apart when their types differ: a list and a tuple, a set and a frozenset,
    or ``P(1, 2)`` and ``Q(1, 2)`` from two namedtuple types, which otherwise
    shared one entry and were served each other's results.

    A subclass also brings the state it holds beside its items, which its
    items drop: ``defaultdict(list)`` and ``defaultdict(set)`` shared one
    entry, and a ``dict`` subclass holding ``self.source`` served the first
    caller's answer for every source. Nothing is caught here: a part that
    cannot be read is not left out of the key, it makes the call unkeyable
    (run uncached, with a warning).
    """
    t = type(value)
    tag = _BUILTIN_CONTAINER_TAGS.get(t)
    if tag is not None:
        return ("__cash_type__", tag, canon)
    state: Any = ()
    factory = getattr(value, "default_factory", None)
    own = {k: v for k, v in (getattr(value, "__dict__", None) or {}).items() if not k.startswith("__")}
    if factory is not None:
        state += (("default_factory", getattr(factory, "__qualname__", repr(factory))),)
    if own:
        state += tuple(sorted((k, stable_key_repr(v, 45)) for k, v in own.items()))
    tag = f"{t.__module__}.{t.__qualname__}"
    return ("__cash_type__", tag, canon, state) if state else ("__cash_type__", tag, canon)


def stable_key_repr(value: Any, _depth: int = 0, _stack: set | None = None) -> Any:
    """The form a cache key hashes *value* in: equal values pickle to equal
    bytes, in any process.

    * Every dict, list, tuple, set and frozenset becomes a tuple tagged with its
      type (`_typed`), so containers of different types never key alike.
    * The items of a set, and of a plain dict, are sorted by their pickled
      bytes: a set of strings iterates in an order PYTHONHASHSEED picks, and a
      dict equals its reordering. A dict subclass keeps its order, which may be
      what it means (``OrderedDict``).
    * An object with a set somewhere inside becomes its type and its
      canonicalised instance state (`object_state`), so that set is sorted
      too. Any other object is left to pickle, which stores it as it asks to
      be stored (its ``__reduce__``) and keeps the loops in its graph.

    A container graph that loops back on itself raises `CyclicValueError` (a
    TypeError, so the value is reported as unhashable and the call runs
    uncached). Expanding it path by path to the depth limit never returned,
    and a form that stood in for the loop could make two different graphs key
    alike, which would be a wrong answer.
    """
    if _depth > 50:
        return value
    if type(value) in CODELESS_PRIMS:
        return value
    if _stack is None:
        _stack = set()
    if id(value) in _stack:
        raise CyclicValueError(f"a {type(value).__qualname__} that contains itself has no stable form to key on")
    _stack.add(id(value))
    try:
        return _stable_key_repr_of(value, _depth, _stack)
    finally:
        _stack.discard(id(value))


def _stable_key_repr_of(value: Any, _depth: int, _stack: set) -> Any:
    """`stable_key_repr` of one object, with the path walked so far."""

    def sub(v: Any) -> Any:
        return stable_key_repr(v, _depth + 1, _stack)

    if isinstance(value, (set, frozenset)):
        items = [sub(v) for v in value]
        items.sort(key=_plain_data.key_dumps)
        return _typed(value, tuple(items))
    if isinstance(value, dict):
        items = [(sub(k), sub(v)) for k, v in value.items()]
        if type(value) is dict:
            items.sort(key=lambda kv: _plain_data.key_dumps(kv[0]))
        return _typed(value, tuple(items))
    if isinstance(value, (list, tuple)):
        return _typed(value, tuple(sub(v) for v in value))
    if not contains_set(value):
        return value
    t = type(value)
    return ("__cash_obj__", f"{t.__module__}.{t.__qualname__}", sub(object_state(value)))


def contains_set(value: Any, _depth: int = 0, _seen: set[int] | None = None) -> bool:
    """True if *value* contains a set/frozenset anywhere (recursively, including
    inside objects). `stable_key_repr` opens an object up only when it holds
    one; any other object is left to pickle.

    Each container or object is looked at once per walk. Without that, a
    cyclic graph was walked once per PATH to the depth limit: a module-level
    ``logger = logging.getLogger(...)`` read in a cached function reaches the
    logging manager, whose dict of every logger reaches the manager again, and
    the first call never returned -- in every release up to 0.10.0. A node
    seen before is either still being walked (its other branches answer for
    it) or was walked and held no set, or the walk would have stopped there.
    """
    if _depth > 50:
        return False
    if _seen is None:
        _seen = set()
    # An exact builtin primitive cannot contain anything, so it cannot contain
    # a set. Without this the fall-through below called ``object_state`` on
    # EVERY element -- which walks ``type(value).__mro__`` looking for
    # ``__slots__`` -- so hashing a 10k-element list of ints made 10k such
    # walks per cache hit. Exact-type test, matching ``CODELESS_PRIMS``'s own
    # contract: a str/int SUBCLASS can carry a ``__dict__`` holding a set and
    # must still be walked.
    if type(value) in CODELESS_PRIMS:
        return False
    if isinstance(value, (set, frozenset)):
        return True
    if id(value) in _seen:
        return False
    if isinstance(value, logging.Logger):
        # Pickled by NAME (`Logger.__reduce__`), so nothing inside it reaches
        # the key -- and walking it means walking every logger in the process,
        # 270 us on each call of any function that reads a module `logger`.
        return False
    _seen.add(id(value))
    if isinstance(value, dict):
        return any(contains_set(k, _depth + 1, _seen) or contains_set(v, _depth + 1, _seen) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(contains_set(v, _depth + 1, _seen) for v in value)
    obj_state = object_state(value)
    if obj_state:
        return any(contains_set(v, _depth + 1, _seen) for v in obj_state.values())
    return False


# ---------------------------------------------------------------------------
# Content hashers: one per library, shared by the decorator's argument keys and
# the notebook's full-content hash (`compute_hash_full`)
# ---------------------------------------------------------------------------


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
    return None


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
    return None


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
    categorical against an object column (found attacking the decorator
    before round 26).
    """
    try:
        import pandas as pd

        index_dtypes = [str(dt) for dt in getattr(value.index, "dtypes", [value.index.dtype])]
        if type(value).__name__ == "DataFrame":
            schema = (
                f"{list(value.columns)!r}:{list(value.index.names)!r}:"
                f"{[str(dt) for dt in value.dtypes]!r}:{index_dtypes!r}:"
            )
        else:  # Series
            schema = f"{value.name!r}:{list(value.index.names)!r}:{str(value.dtype)!r}:{index_dtypes!r}:"
        h = hashlib.sha256(schema.encode("utf-8"))
        h.update(pd.util.hash_pandas_object(value).values.tobytes())
        return h.hexdigest()
    except (ImportError, TypeError, ValueError, AttributeError):
        logger.debug("Failed to hash pandas %s via hash_pandas_object", type(value).__name__)
        return None


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
    edit (round 17, measured 1 then 1 then 0 executions).

    So: the axes of length > 1, ordered by |stride| from outermost in, with
    a broadcast (zero-stride) axis outermost. Identity is ``C``, reversed is
    ``F``; anything else spells the permutation. Stride MAGNITUDE and sign
    do not change what a memory-order read returns, so they stay out.

    Except for one flag. ``order='A'`` (``ravel``, ``reshape``, ``tobytes``,
    ``copy``) reads in Fortran order only when the array is F-CONTIGUOUS,
    and C order otherwise -- so an F-like strided view (``a.T[::2]``) and
    its F-contiguous copy read differently, and sharing ``F`` handed one the
    other's result (round 18, 8/8). ``Fs`` is the F-like array that is not
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
            h.update(pickle.dumps(stable_key_repr(value.tolist()), protocol=4))
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
            h.update(repr(value.hash_rows().to_list()).encode("utf-8"))
            return h.hexdigest()
        if isinstance(value, pl.Series):
            h = hashlib.sha256(f"{value.name!r}:{value.dtype}:{len(value)}:".encode("utf-8"))
            h.update(repr(value.hash().to_list()).encode("utf-8"))
            return h.hexdigest()
        if isinstance(value, pl.LazyFrame):
            try:
                return hashlib.sha256(value.serialize()).hexdigest()
            except Exception:  # noqa: BLE001 - polars raises its own types
                logger.debug("polars LazyFrame serialize() failed; no built-in hash for it")
                return None
    except (ImportError, TypeError, ValueError, AttributeError):
        logger.debug("Failed to hash polars %s", type(value).__name__)
    return None


def hash_pyarrow(value: Any) -> str | None:
    """Hash a PyArrow Table or RecordBatch: schema, row count and every buffer.

    The previous size-gated path hashed ONLY schema+row-count for tables
    >=10 MB, so any two same-shape tables collided into a wrong cache hit.
    Buffer hashing is zero-copy and total.
    """
    try:
        import pyarrow as pa

        if isinstance(value, (pa.Table, pa.RecordBatch)):
            h = hashlib.sha256(f"{value.schema}:{value.num_rows}:".encode())
            for col in value.columns:
                chunks = col.chunks if hasattr(col, "chunks") else [col]
                for chunk in chunks:
                    for buf in chunk.buffers():
                        if buf is not None:
                            h.update(memoryview(buf))
            return h.hexdigest()
    except (ImportError, TypeError, ValueError, AttributeError, MemoryError):
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


# ---------------------------------------------------------------------------
# Sampled and full hashes
# ---------------------------------------------------------------------------


def _content_bytes(values: Any) -> bytes:
    """The bytes of an array's content -- never of its pointers.

    An object array's buffer holds PyObject pointers: memory addresses, which
    differ in every process and between a value and its copy. A frame with a
    text column (whose ``.values`` is an object array) hashed differently after
    every restart, so an ``if`` or ``for`` body that might reassign it gave it
    a new lineage each time, and nothing downstream restored (round 21). Its
    elements are pickled instead, which is content. Numeric arrays keep the raw
    bytes, so their hashes -- and the keys built on them -- do not move.
    """
    if getattr(getattr(values, "dtype", None), "hasobject", False):
        return pickle.dumps(values.tolist(), protocol=4)
    return values.tobytes()


def _frame_dtypes_signature(obj: Any) -> str:
    """``str(obj.dtypes.to_dict())``, byte for byte, without its per-column cost.

    That expression iterated the column index element by element (slow for
    pandas' Arrow-backed string index) and called ``repr`` on every column's
    dtype object. A loop mutating a 3130x800 frame re-hashed it after every
    iteration, and building that string was 42% of a loop that ran 70x slower
    under cash than without it (round 28, r28s3). The OUTPUT must not change:
    it is part of every frame's hash, and keys already on disk must not move
    (``test_a_numeric_frame_hash_is_unchanged``). So: the columns come out in
    one ``tolist()``, the dict keeps its semantics for duplicate names, and a
    dtype is repr'd once however many columns share it.
    """
    mapping = dict(zip(obj.columns.tolist(), obj.dtypes.tolist()))
    reprs: dict[int, str] = {}
    parts = []
    for name, dtype in mapping.items():
        text = reprs.get(id(dtype))
        if text is None:
            text = reprs[id(dtype)] = repr(dtype)
        parts.append(f"{name!r}: {text}")
    return "{" + ", ".join(parts) + "}"


def _hash_dataframe_or_series(obj: Any, type_name: str) -> str:
    """Hash a pandas DataFrame or Series using shape + dtypes + data sample."""
    shape_str = f"{obj.shape}"
    if type_name == "DataFrame":
        try:
            dtypes_str = _frame_dtypes_signature(obj)
        except _HASH_ERRORS:
            dtypes_str = str(obj.dtypes.to_dict())
    else:
        dtypes_str = str(obj.dtype)
    try:
        sample = str(_content_bytes(obj.head(5).values) if len(obj) > 0 else b"")
    except (TypeError, ValueError, AttributeError, pickle.PicklingError):
        sample = str(obj.head(5))
    combined = f"{shape_str}:{dtypes_str}:{sample}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


_BULKY_TYPE_NAMES = frozenset({"DataFrame", "Series", "ndarray"})


def _hash_collection(obj: Any) -> str:
    """Hash a list/tuple/dict/set/frozenset — sampling large ones to avoid O(n) pickle."""
    n = len(obj)
    if n <= 200:
        # A few frames in a dict (`blocks = {w: net_returns(orders, w) ...}`)
        # were pickled WHOLE -- every byte of every frame -- after each restore
        # and each loop iteration that changed the dict, while a frame on its
        # own is hashed by sampling: seconds per hit at 400 MiB a frame (round
        # 28, r28s5). Such a collection is hashed element by element, each
        # element as ``compute_hash`` would hash it alone. Only then: a plain
        # collection keeps the hash it always had, so its keys do not move.
        items = list(obj.items()) if isinstance(obj, dict) else None
        values = [v for _, v in items] if items is not None else (list(obj) if isinstance(obj, (list, tuple)) else [])
        if any(type(v).__name__ in _BULKY_TYPE_NAMES for v in values):
            parts = [f"{type(obj).__name__}:{n}"]
            if items is not None:
                parts.extend(f"{k!r}={compute_hash(v)}" for k, v in items)
            else:
                parts.extend(compute_hash(v) for v in values)
            return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
        return hashlib.sha256(pickle.dumps(obj)).hexdigest()
    if isinstance(obj, (list, tuple)):
        combined = f"list:{n}:{repr(obj[:5])}:{repr(obj[-5:])}"
    elif isinstance(obj, dict):
        combined = f"dict:{n}:{repr(sorted(obj.keys())[:10])}"
    else:
        combined = f"set:{n}:{repr(sorted(obj)[:10])}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


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
    before/after mutation detection (``CallUnit._hash_args``). Content-hashed
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


def compute_hash(obj: Any) -> str:
    """Compute a hash for an object using type-specific methods with explicit fallbacks.

    Strategy order:
    1. Type-specific fast hash (DataFrame/ndarray/collections)
    2. Generic pickle hash
    3. Identity hash (always succeeds) -- see ``identity_hash`` /
       ``is_identity_fallback_hash`` for why this tier is content-BLIND, not
       merely a cruder content hash.
    """
    type_name = type(obj).__name__

    try:
        if type_name in ("DataFrame", "Series"):
            return _hash_dataframe_or_series(obj, type_name)
        if type_name == "ndarray":
            shape_str = str(obj.shape)
            dtype_str = str(obj.dtype)
            sample = str(_content_bytes(obj.flat[:100]) if obj.size > 0 else b"")
            combined = f"{shape_str}:{dtype_str}:{sample}"
            return hashlib.sha256(combined.encode("utf-8")).hexdigest()
        if isinstance(obj, tuple) and isinstance(getattr(type(obj), "_fields", None), tuple):
            # A namedtuple is its name, its fields and its values. Pickling it
            # pickles its CLASS by reference, which fails for a class made on
            # the spot -- as `df.itertuples()` makes one per call -- and the
            # identity tier below then keyed every row on `id(row)`, new on
            # every run: a loop over itertuples() never restored (r28s1, r28s3).
            fields = type(obj)._fields
            return hashlib.sha256(
                f"namedtuple:{type(obj).__name__}:{fields!r}:{_hash_collection(tuple(obj))}".encode("utf-8")
            ).hexdigest()
        if isinstance(obj, (list, tuple, dict, set, frozenset)):
            return _hash_collection(obj)
        return hashlib.sha256(pickle.dumps(obj)).hexdigest()
    except _HASH_ERRORS as exc:
        logger.debug("Primary hash failed for %s: %s", type_name, exc)

    try:
        return hashlib.sha256(pickle.dumps(obj)).hexdigest()
    except (TypeError, pickle.PicklingError):
        pass

    return identity_hash(obj)


def compute_hash_full(obj: Any) -> str:
    """Full-content hash for cache-KEY discrimination.

    ``compute_hash`` SAMPLES large objects (ndarray: first 100 elements,
    DataFrame: first 5 rows, collections >200: head/tail). That is fine for
    cheap freshness heuristics, but unsound wherever the hash *is* the key
    discriminator — per-iteration loop caching keyed two iterations over
    arrays that agreed in the sample onto one entry and produced a wrong
    result on the very first run. This variant hashes every byte.

    A library value goes through ``builtin_hash``, the hasher the decorator
    keys arguments on, so it carries the value's schema as well: an ``int64``
    and an ``Int64`` column, a tz-naive and a tz-aware one, or a C- and an
    F-ordered array holding equal values key apart here too. Anything else is
    pickled whole; what cannot be pickled falls back to ``compute_hash``.
    """
    digest = builtin_hash(obj)
    if digest is not None:
        return digest
    try:
        return hashlib.sha256(pickle.dumps(obj)).hexdigest()
    except _HASH_ERRORS as exc:
        logger.debug("Full hash failed for %s: %s", type(obj).__name__, exc)
    return compute_hash(obj)


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

# The in-memory size of a pandas frame or series, without ``memory_usage``.
#
# ``DataFrame.memory_usage()`` builds a result Series -- one per column, then
# concatenated with the index's -- and that is nearly all it costs: ~0.23 ms
# for a 200-row, five-column frame, deep or not, against ~0.05 ms to sum the
# column arrays' ``nbytes`` to the same number. Cash sized every stored frame
# twice, once for the RAM tier's cap and once for the restore-cost estimate;
# in a loop over a thousand small files that was 18% of the cell (round 23).
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
    comprehension over 360 of them (round 25, r25s5)."""
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
    without pickling it to find out: r28s5's 1.7 GiB result took 2.7 s to
    pickle, after 2.8 s of compute.
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
    million parsed rows (round 19), and every notebook entry holds the RNG
    state, a tuple of 625 ints one dict down (round 23: 1.7M calls in one cell).
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


def mutation_fingerprint(obj: Any) -> str | None:
    """A digest that changes when *obj* is changed IN PLACE, or ``None``.

    ``compute_hash`` samples a large frame or array, which is right for a
    cache key and wrong for "did this call change its argument": a function
    that adds a column or rescales values in place can leave the sample alone.
    This reads the whole value -- `builtin_hash` for the library types
    (pandas, numpy, polars, ...), and for an AnnData-like object the key sets
    scanpy adds to (``obs``/``var`` columns, ``uns``/``obsm``/``varm``/``obsp``/``layers``
    keys) plus a checksum of ``X``. Taken only around a statement that is
    actually executing, twice, so its O(n) cost is paid next to real work.

    ``None`` when the value cannot be observed (it cannot be pickled, so its
    only hash would be its ``id``, which no in-place change moves).
    """
    h = hashlib.sha256()
    t = type(obj)
    h.update(f"{t.__module__}.{t.__qualname__}".encode("utf-8"))
    digest = builtin_hash(obj)
    if digest is not None:
        h.update(digest.encode("utf-8"))
        return h.hexdigest()
    try:
        if all(hasattr(obj, a) for a in ("obs", "var", "uns", "X")):
            parts = [getattr(obj, "shape", None), [str(c) for c in obj.obs.columns], [str(c) for c in obj.var.columns]]
            for slot in ("uns", "obsm", "varm", "obsp", "varp", "layers"):
                mapping = getattr(obj, slot, None)
                parts.append(sorted(map(str, mapping.keys())) if mapping is not None else None)
            h.update(repr(parts).encode("utf-8"))
            x = obj.X
            data = getattr(x, "data", x)
            try:
                h.update(repr((getattr(x, "nnz", None), float(data.sum()))).encode("utf-8"))
            except (TypeError, ValueError, AttributeError):
                pass
            return h.hexdigest()
    except _HASH_ERRORS:
        pass
    digest = compute_hash(obj)
    if digest == identity_hash(obj):
        return None
    return digest
