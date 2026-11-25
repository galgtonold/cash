"""Everything cash knows about how pandas keeps its data, in one place.

pandas is not a dependency of cash, so a user's ``pip install -U pandas``
decides which pandas cash meets. cash reads pandas past its public
interface in three ways, and each of them is spelled only here:

* **Private attributes** (``_mgr``, the block manager; ``_ndarray``, the
  array inside a date or categorical array; ``_pa_array``, the Arrow data
  of an Arrow-backed array; ``_references``, who shares an index's memory;
  ...): the accessor functions below.
* **Private modules** (``pandas.core.*``, ``pandas._libs.*``): `internal`
  names each class cash takes from one.
* **The accessors cash wraps on the pandas classes** so that a handle to a
  frame's memory is recorded (`HANDLE_ACCESSORS`, `CALLBACK_METHODS`).

`tests/test_tooling/test_pandas_internals_have_one_home.py` holds the rest
of the package to it.

The frame memo
--------------
The one use that can serve a wrong answer when pandas changes is the frame
memo (`cash.decorator.arg_hashing.FrameDigests`): it does not hash a frame
again while it can prove that nothing wrote to it, and the proof is a list
of the routes by which a frame's memory can be written. A route that is
not on the list is a stale key. So before the memo is used at all, the
first time a pandas frame reaches cash, `frame_memo_usable` asks two
questions:

1. Is this pandas one the memo was tested with (`NEWEST_TESTED`)?
2. Does the memo notice a write through each route of `WRITE_ROUTES`,
   on a small frame, right now (`self_test`)?

A "no" to either switches the memo off for the process: every frame is
hashed in full every time, which is always right, and
``CACHE-PANDAS-UNTESTED`` says so once. Nothing here imports pandas: a
frame reaching cash means the program has.
"""

from __future__ import annotations

import importlib
import logging
import threading
import warnings
from typing import Any, Callable

from .loaded_modules import loaded_module

__all__ = [
    "AXIS_PROPERTY",
    "column_block",
    "column_row",
    "ARROW_ARRAY_MODULE",
    "reset_frame_memo_state",
    "probe_frames",
    "compile_routes",
    "loaded_arrow_array_module",
    "owner_class",
    "held_values",
    "index_cache",
    "multi_index_codes",
    "multi_index_levels",
    "cow_write_hooks",
    "RESULT_OBJECT_MODULES",
    "TIMEDELTA_PANDAS_2",
    "TIMESTAMP_BASE",
    "CACHED_PROPERTY",
    "INDEX_ENGINE_MODULE",
    "DATA_SLOT",
    "ARRAY_SLOTS",
    "ARROW_SLOT",
    "CALLBACK_METHODS",
    "HANDLE_ACCESSORS",
    "ITEM_CACHE_SLOT",
    "MANAGER_SLOT",
    "NDARRAY_SLOT",
    "NEWEST_TESTED",
    "NOT_HELD_SLOTS",
    "WRITE_ROUTES",
    "arrays",
    "arrow_data",
    "block_refs",
    "blocks",
    "column_layout",
    "column_values",
    "frame_from_manager",
    "frame_memo_state",
    "frame_memo_usable",
    "from_backing_data",
    "index_data",
    "index_refs",
    "inner_ndarray",
    "internal",
    "is_shared",
    "manager",
    "manager_or_none",
    "multi_index_arrays",
    "pandas_version",
    "self_test",
    "set_arrow_data",
    "sharers",
    "structure_problems",
]

logger = logging.getLogger("cash")

# ---------------------------------------------------------------------------
# Private attribute names
# ---------------------------------------------------------------------------

#: A frame's or series' block manager.
MANAGER_SLOT = "_mgr"
#: The ndarray inside a date, duration, categorical or numpy-wrapping array.
NDARRAY_SLOT = "_ndarray"
#: The Arrow data of an Arrow-backed extension array.
ARROW_SLOT = "_pa_array"
#: pandas 2 without copy-on-write: the columns a frame has handed out.
ITEM_CACHE_SLOT = "_item_cache"
#: The array of an index or of a nullable array.
DATA_SLOT = "_data"
#: The attributes an extension array keeps its memory in: a date's or a
#: categorical's ndarray, a nullable array's values and mask, and (pandas 2)
#: a categorical's codes.
ARRAY_SLOTS = ("_ndarray", "_data", "_mask", "_codes")
#: What a frame's ``__dict__`` holds that is pandas' own bookkeeping, not
#: something the user put there.
NOT_HELD_SLOTS = frozenset({"_mgr", "_item_cache", "_flags", "_cache", "_references", "_id"})

_REQUIRED: Any = object()


def manager(obj: Any) -> Any:
    """The block manager of a frame or series."""
    return obj._mgr


def manager_or_none(obj: Any) -> Any:
    """`manager`, or None for an object that has none."""
    return getattr(obj, "_mgr", None)


def blocks(obj: Any) -> Any:
    """The blocks of a frame or series: each holds the columns of one dtype
    in one array (``block.values``) and who shares it (`block_refs`)."""
    return obj._mgr.blocks


def arrays(obj: Any) -> Any:
    """The arrays of a frame's or series' blocks."""
    return obj._mgr.arrays


def column_layout(mgr: Any) -> tuple[Any, Any, Any]:
    """``(blocks, blknos, blklocs)`` of a frame's manager: column *i* is row
    ``blklocs[i]`` of ``blocks[blknos[i]].values``."""
    return mgr.blocks, mgr.blknos, mgr.blklocs


def column_values(mgr: Any, position: int) -> Any:
    """The array of the block that holds column *position*."""
    return mgr.blocks[mgr.blknos[position]].values


def column_block(mgr: Any, position: int) -> Any:
    """The block that holds column *position*."""
    return mgr.blocks[mgr.blknos[position]]


def column_row(mgr: Any, position: int) -> Any:
    """Which row of its block's 2-d array column *position* is."""
    return mgr.blklocs[position]


def block_refs(block: Any) -> Any:
    """The copy-on-write record of who shares a block's memory."""
    return block.refs


def index_refs(index: Any) -> Any:
    """The copy-on-write record of who shares an index's memory, or None."""
    return index._references


def sharers(refs: Any) -> Any:
    """The weak references a copy-on-write record (`block_refs`,
    `index_refs`) holds to the blocks and indexes sharing the memory."""
    return refs.referenced_blocks


def is_shared(refs: Any) -> bool:
    """Does anything else share the memory of this copy-on-write record?"""
    return refs.has_reference()


def inner_ndarray(values: Any, default: Any = _REQUIRED) -> Any:
    """The ndarray inside a date, duration, categorical or numpy-wrapping
    array; *default* for a value that has none."""
    if default is _REQUIRED:
        return values._ndarray
    return getattr(values, "_ndarray", default)


def arrow_data(values: Any, default: Any = None) -> Any:
    """The Arrow data of an Arrow-backed array; *default* for any other."""
    return getattr(values, "_pa_array", default)


def set_arrow_data(values: Any, data: Any) -> None:
    """Give an Arrow-backed array other Arrow data to hold."""
    values._pa_array = data


def index_data(index: Any, default: Any = _REQUIRED) -> Any:
    """The array an index keeps its labels in (a ``RangeIndex`` builds one
    when asked), or a nullable array its values in; *default* for a value
    that has none."""
    if default is _REQUIRED:
        return index._data
    return getattr(index, "_data", default)


def multi_index_levels(axis: Any) -> Any:
    """The level indexes a ``MultiIndex`` keeps (not the copies ``.levels``
    hands out)."""
    return axis._levels


def multi_index_codes(axis: Any) -> Any:
    """The list of code arrays a ``MultiIndex`` keeps."""
    return axis._codes


def index_cache(index: Any) -> dict:
    """What an index has computed and kept (``cache_readonly``): the array
    ``.array`` handed out, its lookup engine, a MultiIndex's ``levels``."""
    return index._cache


def held_values(obj: Any) -> Any:
    """The array a series or an index keeps, as it is (``.values`` may
    convert it)."""
    return obj._values


def multi_index_arrays(axis: Any) -> list:
    """The arrays a ``MultiIndex`` keeps: each level's labels, then the codes."""
    return [level._data for level in axis._levels] + list(axis._codes)


def frame_from_manager(like: Any, mgr: Any) -> Any:
    """A frame or series of *like*'s class over *mgr*."""
    return like._constructor_from_mgr(mgr, axes=mgr.axes)


def from_backing_data(array: Any, data: Any) -> Any:
    """An extension array like *array* over the ndarray *data*."""
    return array._from_backing_data(data)


# ---------------------------------------------------------------------------
# Private modules
# ---------------------------------------------------------------------------

#: ``name -> module`` of every class cash takes from a private pandas module.
_INTERNAL = {
    "ArrowExtensionArray": "pandas.core.arrays.arrow",
    "BaseMaskedArray": "pandas.core.arrays.masked",
    "Block": "pandas.core.internals.blocks",
    "BaseBlockManager": "pandas.core.internals.managers",
    "SingleBlockManager": "pandas.core.internals.managers",
    "FrozenList": "pandas.core.indexes.frozen",
    "IndexOpsMixin": "pandas.core.base",
    "NDFrame": "pandas.core.generic",
    "DatetimeIndexOpsMixin": "pandas.core.indexes.datetimelike",
    "DatetimeTimedeltaMixin": "pandas.core.indexes.datetimelike",
    "GroupBy": "pandas.core.groupby.groupby",
    "DataFrameGroupBy": "pandas.core.groupby.generic",
    "SeriesGroupBy": "pandas.core.groupby.generic",
    "Rolling": "pandas.core.window.rolling",
    "Expanding": "pandas.core.window.expanding",
    "ExponentialMovingWindow": "pandas.core.window.ewm",
}

#: The module that defines the Arrow-backed extension array, for a caller
#: that only wants to know whether it is loaded.
ARROW_ARRAY_MODULE = "pandas.core.arrays.arrow.array"


#: The module of an index's lookup engines.
INDEX_ENGINE_MODULE = "pandas._libs.index"
#: ``(module, class)`` of the descriptor behind ``cache_readonly``.
CACHED_PROPERTY = ("pandas._libs.properties", "CachedProperty")
#: ``(module, class)`` of the descriptor behind ``df.index`` / ``df.columns``.
AXIS_PROPERTY = ("pandas._libs.properties", "AxisProperty")
#: ``(module, class)`` of the C base of ``Timestamp``, and of ``Timedelta``
#: as pandas 2 names it.
TIMESTAMP_BASE = ("pandas._libs.tslibs.timestamps", "_Timestamp")
TIMEDELTA_PANDAS_2 = ("pandas._libs.tslibs.timedeltas", "Timedelta")
#: Module prefixes of the objects pandas builds over a frame (a groupby,
#: a window, an indexer).
RESULT_OBJECT_MODULES = ("pandas.core.", "pandas.api.typing")

#: ``method -> class`` of the private methods pandas runs before it writes
#: a block under copy-on-write; `cash.backends.frame_sharing` wraps them to
#: learn of a write to memory it handed out.
_COW_WRITE_HOOKS = {
    "_has_no_reference_block": "BaseBlockManager",
    "_has_no_reference": "SingleBlockManager",
    "_maybe_copy": "Block",
    "_get_refs_and_copy": "Block",
}


def cow_write_hooks() -> dict[str, tuple[type, Any]]:
    """``method name -> (class, the method as pandas defines it)`` for each
    of the methods pandas runs before writing a block under copy-on-write."""
    found = {}
    for name, owner in _COW_WRITE_HOOKS.items():
        cls = internal(owner)
        found[name] = (cls, getattr(cls, name))
    return found


def internal(name: str) -> Any:
    """The pandas class *name* (a key of `_INTERNAL`). ImportError when
    this pandas has no such module, AttributeError when the module has no
    such class: the caller's handling of a pandas it cannot read applies."""
    return getattr(importlib.import_module(_INTERNAL[name]), name)


def loaded_arrow_array_module() -> Any:
    """The module of pandas' Arrow-backed array when the program has loaded
    it, else None."""
    return loaded_module(ARROW_ARRAY_MODULE)


# ---------------------------------------------------------------------------
# The accessors the frame memo wraps
# ---------------------------------------------------------------------------

#: ``(owner, attribute, whose reads are recorded)`` for each accessor that
#: hands out memory a frame keeps. *owner* is a public name of ``pandas``
#: or a name `internal` knows. The third item: "every" read, reads from
#: "outside" pandas only, or only a read-only "view" asked for from outside
#: pandas and cash.
HANDLE_ACCESSORS = (
    ("Series", "array", "every"),
    ("Index", "array", "every"),
    ("DatetimeIndexOpsMixin", "asi8", "outside"),
    ("Series", "values", "view"),
    ("DataFrame", "values", "view"),
    ("Index", "values", "view"),
    ("DatetimeTimedeltaMixin", "values", "view"),
    ("IndexOpsMixin", "to_numpy", "view"),
    ("DataFrame", "to_numpy", "view"),
    ("Series", "__array__", "view"),
    ("NDFrame", "__array__", "view"),
    ("Index", "__array__", "view"),
    ("Categorical", "codes", "view"),
    ("CategoricalIndex", "codes", "view"),
)

#: ``(owner, method, position of raw)`` for each pandas method that hands
#: a frame's data to the user's callable; the position is where ``raw``
#: sits among the positional arguments after ``self``, None for a method
#: that has no such argument. The last two are missing from some builds.
CALLBACK_METHODS = (
    ("DataFrame", "apply", 2),
    ("NDFrame", "pipe", None),
    *((owner, name, None) for owner in ("GroupBy", "SeriesGroupBy", "DataFrameGroupBy")
      for name in ("apply", "agg", "aggregate", "transform", "pipe")),
    ("Rolling", "apply", 1),
    ("Expanding", "apply", 1),
)  # fmt: skip


def owner_class(pd: Any, owner: str) -> Any:
    """The class an entry of `HANDLE_ACCESSORS` or `CALLBACK_METHODS` names."""
    return internal(owner) if owner in _INTERNAL else getattr(pd, owner)


# ---------------------------------------------------------------------------
# Which pandas the frame memo may be used with
# ---------------------------------------------------------------------------

#: The newest pandas ``(major, minor)`` the frame memo's list of write
#: routes was checked against (`WRITE_ROUTES`, and the wider probe in
#: ``tests/test_core/arguments/test_a_frame_written_by_any_route_is_hashed_again.py``,
#: which the nightly workflow also runs against pandas pre-releases). Raise
#: it when that probe passes on a newer pandas.
NEWEST_TESTED = (3, 1)


def pandas_version() -> tuple[int, int] | None:
    """``(major, minor)`` of the loaded pandas; None when pandas is not
    loaded or its version cannot be read."""
    pd = loaded_module("pandas")
    if pd is None:
        return None
    try:
        major, minor = str(pd.__version__).split(".")[:2]
        return int(major), int("".join(ch for ch in minor if ch.isdigit()) or 0)
    except Exception:  # noqa: BLE001 - fails closed: an unreadable version is an untested pandas
        return None


# ---------------------------------------------------------------------------
# The start-up self-test
# ---------------------------------------------------------------------------

#: ``(name, frame kind, statements)``: each way a program can write into a
#: frame ``df`` that the frame memo knows of. The statements run with
#: ``pd``, ``np`` and ``df`` in a namespace that is not cash's or pandas'
#: (the memo ignores their own reads), inside a function, so every handle
#: is dropped before the memo is asked: that is the case in which only the
#: record of the handle is left. Frame kinds: ``num`` (two float columns
#: under an integer index), ``cat`` (a categorical column ``c`` too) and
#: ``dates`` (a date index).
_SET_WRITABLE = "v.flags.writeable = True"
WRITE_ROUTES: tuple[tuple[str, str, str], ...] = (
    # Through pandas: copy-on-write gives the frame new arrays.
    ("iloc", "num", "df.iloc[0, 0] = 9.0"),
    ("loc", "num", "df.loc[df.index[0], 'a'] = 9.0"),
    ("at", "num", "df.at[df.index[0], 'a'] = 9.0"),
    ("column assignment", "num", "df['a'] = df['a'] + 1.0"),
    ("augmented assignment", "num", "df += 1.0"),
    ("insert", "num", "df.insert(0, 'z', 9.0)"),
    ("pop", "num", "df.pop('a')"),
    # On the same objects: compared by value.
    ("index name", "num", "df.index.name = 'k'"),
    ("attrs", "num", "df.attrs['k'] = 1"),
    ("duplicate-labels flag", "num", "df.flags.allows_duplicate_labels = False"),
    # A read-only view made writable again.
    ("Series.values", "num", f"v = df['a'].values; {_SET_WRITABLE}; v[0] = 9.0"),
    ("DataFrame.values", "num", f"v = df.values; {_SET_WRITABLE}; v[0, 0] = 9.0"),
    ("Series.to_numpy", "num", f"v = df['a'].to_numpy(); {_SET_WRITABLE}; v[0] = 9.0"),
    ("DataFrame.to_numpy", "num", f"v = df.to_numpy(); {_SET_WRITABLE}; v[0, 0] = 9.0"),
    ("asarray of a column", "num", f"v = np.asarray(df['a']); {_SET_WRITABLE}; v[0] = 9.0"),
    ("asarray of a frame", "num", f"v = np.asarray(df); {_SET_WRITABLE}; v[0, 0] = 9.0"),
    ("Index.values", "num", f"v = df.index.values; {_SET_WRITABLE}; v[0] = 9"),
    ("asarray of an index", "num", f"v = np.asarray(df.index); {_SET_WRITABLE}; v[0] = 9"),
    # A handle to the array pandas keeps.
    ("Series.array", "num", "df['a'].array[0] = 9.0"),
    ("Index.array", "num", "df.index.array[0] = 9"),
    ("pd.array without a copy", "num", "pd.array(df['a'], copy=False)[0] = 9.0"),
    ("asi8", "dates", f"v = df.index.asi8; {_SET_WRITABLE}; v[0] += 1"),
    # A categorical: the object itself, and its codes.
    ("values of a categorical", "cat", "df['c'].values[0] = 'y'"),
    ("array of a categorical", "cat", "df['c'].array[0] = 'y'"),
    ("codes of a categorical", "cat", f"v = df['c'].values.codes; {_SET_WRITABLE}; v[0] = 1"),
    ("Categorical without a copy", "cat", "pd.Categorical(df['c'], copy=False)[0] = 'y'"),
    # Memory pandas hands to the program's own function.
    (
        "raw apply",
        "num",
        "def f(v):\n        v.flags.writeable = True\n        v[0] = 9.0\n        return 0.0\n    df.apply(f, raw=True)",
    ),
    (
        "raw rolling apply",
        "num",
        "def f(v):\n        v.flags.writeable = True\n        v[0] = 9.0\n        return 0.0\n"
        "    df['a'].rolling(2).apply(f, raw=True)",
    ),
)

#: The module name the routes run under: neither cash's nor pandas'.
_PROBE_MODULE = "pandas_write_probe"


def compile_routes(routes: tuple[tuple[str, str, str], ...] | None = None) -> dict[str, Callable]:
    """``name -> function(pd, np, df)`` for each of *routes* (`WRITE_ROUTES`)."""
    namespace: dict[str, Any] = {"__name__": _PROBE_MODULE}
    names = {}
    for position, (name, _kind, statements) in enumerate(WRITE_ROUTES if routes is None else routes):
        exec(f"def route_{position}(pd, np, df):\n    {statements}\n", namespace)
        names[name] = namespace[f"route_{position}"]
    return names


def probe_frames(pd: Any, np: Any) -> dict[str, Any]:
    """One small frame of each kind `WRITE_ROUTES` names. Copy one
    (``.copy()``) before writing to it."""
    data = {"a": [1.0, 2.0, 3.0, 4.0], "b": [5.0, 6.0, 7.0, 8.0]}
    return {
        "num": pd.DataFrame(data, index=pd.Index([10, 11, 12, 13])),
        "cat": pd.DataFrame({**data, "c": pd.Categorical(["x", "y", "x", "y"])}, index=pd.Index([10, 11, 12, 13])),
        "dates": pd.DataFrame(data, index=pd.date_range("2020-01-01", periods=4)),
    }


def structure_problems(pd: Any) -> list[str]:
    """What this pandas does not keep where the accessors above look, as
    one line each; empty when everything is where it is expected."""
    problems = []

    def check(what: str, test: Callable[[], Any]) -> None:
        try:
            if not test():
                problems.append(what)
        except Exception as exc:  # noqa: BLE001 - fails closed: a failed check is a problem found
            problems.append(f"{what} ({type(exc).__name__})")

    frame = pd.DataFrame({"a": [1.0, 2.0], "c": pd.Categorical(["x", "y"])}, index=pd.Index([3, 4]))
    check("a frame's blocks", lambda: len(blocks(frame)) == 2 and len(arrays(frame)) == 2)
    check("a frame's column layout", lambda: len(column_layout(manager(frame))[1]) == 2)
    check("a column's block", lambda: column_values(manager(frame), 0) is not None)
    check("a block's sharing record", lambda: all(isinstance(sharers(block_refs(b)), list) for b in blocks(frame)))
    check("a block's sharing count", lambda: is_shared(block_refs(blocks(frame)[0])) is False)
    check("an index's sharing record", lambda: index_refs(frame.index) is None or sharers(index_refs(frame.index)))
    check("an index's array", lambda: len(index_data(frame.index)) == 2)
    check("a categorical's codes", lambda: len(inner_ndarray(frame["c"].values)) == 2)
    check("a frame from a manager", lambda: frame_from_manager(frame, manager(frame)).shape == (2, 2))
    for name in ("FrozenList", "IndexOpsMixin", "NDFrame", "DatetimeIndexOpsMixin", "DatetimeTimedeltaMixin"):
        check(f"pandas' {name}", lambda name=name: isinstance(internal(name), type))
    for owner, name, _whose in HANDLE_ACCESSORS:
        check(f"{owner}.{name}", lambda owner=owner, name=name: _defined_on(owner_class(pd, owner), name))
    return problems


def _defined_on(cls: type, name: str) -> bool:
    """Does *cls* itself define *name* as a function, a property or another
    descriptor? One inherited, or implemented in C on the class, is not an
    accessor a Python wrapper put on *cls* stands in front of."""
    found = vars(cls).get(name)
    return found is not None and (callable(found) or hasattr(found, "__get__"))


def self_test(pd: Any, np: Any, remember: Callable[[Any], bool], vouched: Callable[[Any], bool]) -> list[str]:
    """Run the frame memo over every route of `WRITE_ROUTES` and name what
    it missed; empty when the memo may be used.

    *remember(frame)* stores *frame* in a memo and says whether it took it;
    *vouched(frame)* asks that memo whether *frame* is still as stored.
    For each route a fresh small frame is remembered, written through the
    route and asked for again: the memo must no longer vouch for it. A
    route this pandas refuses (it raises) cannot be used by a program
    either and passes. The memo must also take and vouch for an untouched
    frame, before the routes and after them: a memo that never engages
    proves nothing.
    """
    problems = structure_problems(pd)
    if problems:
        return ["pandas does not keep " + what for what in problems]
    frames = probe_frames(pd, np)
    # Its own index: a copy shares the index of what it was copied from,
    # and a route that reaches that index's memory reaches every sharer.
    control = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.Index([3, 4]))
    if not (remember(control) and vouched(control)):
        return ["the memo does not take an untouched frame"]
    for (name, kind, _statements), route in zip(WRITE_ROUTES, compile_routes().values()):
        frame = frames[kind].copy()
        if not remember(frame):
            continue  # never vouched for, so never served stale
        try:
            route(pd, np, frame)
        except Exception:  # noqa: BLE001 - fails closed: a route pandas refuses writes nothing
            continue
        if vouched(frame) and not frame.equals(frames[kind]):
            problems.append(f"a write through {name} goes unnoticed")
    if not problems and not vouched(control):
        problems.append("the memo no longer vouches for an untouched frame")
    return problems


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

#: None until the first frame is met; then ``(usable, reason)``. *reason*
#: is "" when usable or when copy-on-write is simply off (pandas 2), else
#: what the user is told.
_STATE: tuple[bool, str] | None = None
_DECIDING = threading.RLock()
_TESTING = threading.local()


def frame_memo_state() -> tuple[bool, str] | None:
    """``(usable, reason)`` once decided (`frame_memo_usable`), else None."""
    return _STATE


def frame_memo_usable(remember: Callable[[Any], bool], vouched: Callable[[Any], bool]) -> bool:
    """May the frame memo be used with the loaded pandas? Decided once per
    process, when the first pandas frame reaches cash (never before pandas
    is loaded), and told once when the answer is no for a pandas that has
    copy-on-write (``CACHE-PANDAS-UNTESTED``).

    While the self-test runs, this thread is answered True (the test uses
    the memo) and every other thread waits for the verdict.
    """
    state = _STATE
    if state is not None:
        return state[0]
    if getattr(_TESTING, "active", False):
        return True
    with _DECIDING:
        if _STATE is None:
            _decide(remember, vouched)
        return _STATE[0]  # type: ignore[index]


def _decide(remember: Callable[[Any], bool], vouched: Callable[[Any], bool]) -> None:
    global _STATE
    pd = loaded_module("pandas")
    np = loaded_module("numpy")
    version = pandas_version()
    if pd is None or np is None or version is None:
        _STATE = (False, "")  # no pandas whose frames could reach the memo
        return
    try:
        cow = version[0] >= 3 or pd.options.mode.copy_on_write is True
    except Exception:  # noqa: BLE001 - fails closed: unknown mode, no memo
        cow = False
    if not cow:
        _STATE = (False, "")
        return
    shown = ".".join(map(str, version))
    if version > NEWEST_TESTED:
        tested = ".".join(map(str, NEWEST_TESTED))
        reason = f"pandas {shown} is newer than the newest pandas this cash was tested with ({tested})"
    else:
        _TESTING.active = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # nothing of the check's reaches the user
                problems = self_test(pd, np, remember, vouched)
        except Exception as exc:  # noqa: BLE001 - fails closed: a self-test that cannot run switches the memo off
            problems = [f"the check raised {type(exc).__name__}"]
        finally:
            _TESTING.active = False
        reason = f"cash's start-up check of pandas {shown} failed: {'; '.join(problems[:3])}" if problems else ""
    _STATE = (not reason, reason)
    if reason:
        _tell(reason)


def _tell(reason: str) -> None:
    from .diagnostics import warn_diagnostic  # imports what imports this module
    from .exceptions import CashCacheIneffectiveWarning

    try:
        warn_diagnostic(
            CashCacheIneffectiveWarning,
            "CACHE-PANDAS-UNTESTED",
            f"{reason}, so cash reads every pandas DataFrame and Series in full each time it keys one, "
            f"instead of checking that it is unchanged. Results are correct; keying a large frame is slower.",
            f"upgrade cash-lib, or install a pandas it was tested with (up to {'.'.join(map(str, NEWEST_TESTED))}.x).",
        )
    except Exception:  # noqa: BLE001 - fails closed: the memo is already off; a filter that raises must not fail a key build
        logger.warning("CACHE-PANDAS-UNTESTED: %s; pandas frames are hashed in full on every call.", reason)


def reset_frame_memo_state() -> None:
    """Forget the verdict, so that the next frame decides again (tests)."""
    global _STATE
    with _DECIDING:
        _STATE = None
