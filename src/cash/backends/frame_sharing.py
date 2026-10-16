"""Pandas tables the RAM tier shares instead of copying.

Under pandas copy-on-write a table can be handed out as a shallow copy of
the one the RAM tier keeps: pandas copies a block before writing it as long
as another table shares it. Copy-on-write covers only writes made through
pandas. ``s.array[0] = ...``, ``s.values`` of a nullable or categorical
column, and a read-only view made writable again write into the block
itself, past it -- into the stored entry and every other hit.

So a table the tier shares is *frozen* first (`freeze`):

* every array its blocks keep their data in is made read-only, and a block
  never holds the array that owns the memory, only a view of it: numpy will
  not make a view of a read-only array writable again, so no handle pandas
  gives out can write it;
* each block's copy-on-write reference group gets a reference that never
  dies (`_PIN`). pandas then always sees the block as shared and copies it
  before writing, even once the stored table and every other hit are gone:
  a frozen block is never written in place, by pandas or anyone else.

Only columns kept in plain numpy arrays (numbers, dates and times, object
columns whose cells cannot change) and Arrow-backed ones are frozen. A
nullable, categorical or period column hands out ``s.values`` as the
extension array itself, writable by design, so a table holding one is not
shared: the tier copies it.

The data a table reads stays where it is; pandas writes to a frozen table
behave as on any other, they just copy first. A table frozen this way pickles
as a writable one (`_install_pickling`): a disk entry, or a table the user
pickles themselves, loads as an ordinary table.

Only a table cash owns is frozen: its own copy made when storing, or a
table it has just unpickled from disk. A table the caller holds is never
frozen -- cash does not change what an object of the caller's allows -- so
the store copies it once, and every hit shares that copy. That copy holds
nothing of the caller's (`own_parts`): its labels are copied, and so is
any Arrow data that may be the caller's numpy memory under Arrow's name;
Arrow data in memory Arrow allocated itself is shared (`arrow_owned`).

A big table of numbers and dates is not copied at all (`share_checked`):
the tier keeps a shallow copy, which pandas copies before writing either
side, with the digest of its data, and checks that digest before it serves
the table. Once nothing but the tier reaches that data -- the caller has
rebound the name (``df = df[...]``) -- the tier takes it over and freezes
it as its own (`adopt`).

Used only where pandas has copy-on-write (pandas 3, or opted in before) and
these internals behave as described (`enabled`, checked once). Otherwise the
tier copies as it always has.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
import weakref
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "arrow_data_owned",
    "arrow_owned",
    "copy_keeping_blocks",
    "enabled",
    "freeze",
    "frozen_already",
    "hand_out",
    "is_frozen",
    "is_pin",
    "own_parts",
    "owned_arrow_data",
]


class _Pin:
    """What the reference that never dies points to (`_PIN`)."""

    __slots__ = ("__weakref__",)

    def __repr__(self) -> str:
        return "<cash: data shared with the RAM tier>"


_PIN_TARGET = _Pin()
#: The reference a frozen block's copy-on-write group keeps: alive for good,
#: so pandas always copies the block before writing it.
_PIN = weakref.ref(_PIN_TARGET)


def is_pin(obj: Any) -> bool:
    """Is *obj* what `_PIN` points to? (not a block: walks of a reference
    group skip it)."""
    return obj is _PIN_TARGET


#: ``id(array) -> weak reference`` for each array whose memory `freeze` made
#: read-only: the arrays that pickle as writable (`_pickle_form`).
_FROZEN: dict[int, Any] = {}
_FROZEN_LOCK = threading.Lock()


def _remember_frozen(array: Any) -> None:
    key = id(array)
    with _FROZEN_LOCK:
        if key not in _FROZEN:
            _FROZEN[key] = weakref.ref(array, lambda _ref, key=key: _FROZEN.pop(key, None))


def _root(array: Any) -> Any:
    """The array at the end of *array*'s chain of views: the one that owns the
    memory, or whose base is not an array."""
    import numpy as np

    while isinstance(array.base, np.ndarray):
        array = array.base
    return array


def is_frozen(array: Any) -> bool:
    """Did `freeze` make the memory *array* views read-only?"""
    ref = _FROZEN.get(id(_root(array)))
    return ref is not None and ref() is not None


# ---------------------------------------------------------------- enabled


_ENABLED: list[bool] = []


def enabled() -> bool:
    """Can tables be shared? pandas with copy-on-write, its internals as this
    module expects them (a self-test, run once)."""
    if _ENABLED:
        return _ENABLED[0]
    pd = sys.modules.get("pandas")
    if pd is None:
        return False  # not decided: pandas may be imported later
    ok = False
    try:
        major = int(pd.__version__.split(".", 1)[0])
        cow = major >= 3 or pd.options.mode.copy_on_write is True
        ok = cow and _self_test(pd)
    except Exception:
        logger.debug("cash: pandas tables are copied, not shared", exc_info=True)
        ok = False
    _ENABLED.append(ok)
    return ok


def _self_test(pd: Any) -> bool:
    """Freeze a small table of each kind of column, write a hit of it
    through pandas with the stored table gone, and check nothing leaked."""
    import numpy as np

    df = pd.DataFrame(
        {
            "f": np.arange(4.0),
            "i": np.arange(4),
            "d": pd.date_range("2020-01-01", periods=4),
            "s": ["a", "b", "c", "d"],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )
    by_date = pd.Series(np.arange(4.0), index=pd.timedelta_range("1D", periods=4, freq="h"))
    stored = df.copy(deep=True)
    stored_series = by_date.copy(deep=True)
    if not _freeze(stored) or not _freeze(stored_series):
        return False
    hit = hand_out(stored)
    other = hand_out(stored)
    if not all(blk.refs.has_reference() for blk in stored._mgr.blocks):
        return False
    # A dated index keeps its freq through the freeze, a hit and a pickle.
    import pickle

    for kept, original in ((hit, df), (hand_out(stored_series), by_date)):
        if kept.index.freq != original.index.freq:
            return False
        if pickle.loads(pickle.dumps(kept)).index.freq != original.index.freq:
            return False
    del stored
    hit.iloc[0, 0] = 100.0
    hit.iloc[1, 2] = pd.Timestamp("2021-01-01")
    hit.iloc[2, 3] = "z"
    hit.iloc[3, 1] = 9
    if other["f"].iloc[0] != 0.0 or other["i"].iloc[3] != 3 or other["s"].iloc[2] != "c":
        return False
    try:
        other["f"].array[1] = 5.0
    except ValueError:
        pass
    else:
        return False
    return bool(other["f"].iloc[1] == 1.0)


# ---------------------------------------------------------------- freezing


def _ndarray_parts(values: Any) -> list[tuple[str, Any]] | None:
    """``(attribute, array)`` for each numpy array a block's *values* keep
    their data in (``""``: *values* is one); None for values this module
    does not freeze.

    Only values no public pandas handle writes but ``.array``: numbers,
    dates and durations, and Arrow-backed data. A nullable, categorical,
    period or numpy-backed text column hands out ``.values`` as the array
    itself, which is written in place as an ordinary pandas operation; the
    tier keeps copying those tables, so such a write still works on a hit.
    """
    import numpy as np

    if type(values) is np.ndarray:
        return [("", values)]
    import pandas as pd

    if type(values) in (pd.arrays.DatetimeArray, pd.arrays.TimedeltaArray):
        return [("_ndarray", values._ndarray)]
    from pandas.core.arrays.arrow import ArrowExtensionArray

    if isinstance(values, ArrowExtensionArray):
        return []  # Arrow memory is immutable; a write swaps the Arrow array
    return None


def _cells_unchangeable(array: Any) -> bool:
    """Is every cell of an object array a value nothing can change in place?"""
    from .memory_backend import immutable_cells

    return immutable_cells(array.ravel())


def _freezable_values(values: Any, cells_known: bool = False) -> bool:
    parts = _ndarray_parts(values)
    if parts is None:
        return False
    for _name, array in parts:
        if array.dtype.hasobject and not cells_known and not _cells_unchangeable(array):
            return False
    return True


def _freezable_index(index: Any) -> bool:
    import pandas as pd

    if isinstance(index, pd.RangeIndex):
        return True
    if isinstance(index, pd.MultiIndex):
        return False
    return _freezable_values(index._data)


def freezable(frame: Any, *, cells_known: bool = False) -> bool:
    """Is *frame* a plain pandas DataFrame or Series made only of columns and
    labels `freeze` knows (`_ndarray_parts`; Python objects only when nothing
    can change them in place)? *cells_known*: the caller has just found that
    no object column holds a changeable cell (``_holds_mutable_cells``), so
    the columns are not scanned again."""
    import pandas as pd

    if type(frame) not in (pd.DataFrame, pd.Series):
        return False  # a subclass may keep state of its own
    if not all(_freezable_values(blk.values, cells_known) for blk in frame._mgr.blocks):
        return False
    axes = (frame.index,) if frame.ndim == 1 else (frame.index, frame.columns)
    return all(_freezable_index(axis) for axis in axes)


def _read_only(array: Any) -> Any:
    """*array* and every array it views made read-only; a view of it when it
    owns its memory, so no one holding the result can make it writable."""
    import numpy as np

    link = array
    while isinstance(link, np.ndarray):
        if link.flags.writeable:
            link.flags.writeable = False
        if not isinstance(link.base, np.ndarray):
            break
        link = link.base
    _remember_frozen(link)
    if array.base is None:
        return array.view()
    return array


def _freeze_values(values: Any) -> Any:
    """*values* with their arrays read-only (`_read_only`); new values when an
    array had to be replaced by a view of it."""
    parts = _ndarray_parts(values)
    if not parts:
        return values
    if parts[0][0] == "":
        return _read_only(values)
    frozen = _read_only(values._ndarray)
    return values if frozen is values._ndarray else _same_dates(values, frozen)


def _same_dates(values: Any, ndarray: Any) -> Any:
    """A DatetimeArray or TimedeltaArray like *values* over *ndarray*, its
    ``freq`` kept: ``_from_backing_data`` drops it, and a table with a
    ``date_range`` index would come back from a hit, or from disk, without
    it (``df.shift(1, freq=df.index.freq)`` then shifts the data)."""
    return type(values)._simple_new(ndarray, dtype=values.dtype, freq=values.freq)


def _frozen_index(index: Any) -> Any:
    """A new index with *index*'s labels, copied and made read-only; *index*
    itself when it holds no array (a range). An index is never written
    through pandas, only through a handle (``index.array``). Copied, because
    a deep pandas copy of a table may keep the caller's label array, and the
    caller's index must stay as writable as it was."""
    import pandas as pd

    if isinstance(index, pd.RangeIndex):
        return index
    data = index._data
    if not _ndarray_parts(data):
        return index  # Arrow labels: immutable memory (`_private_axes`)
    return type(index)._simple_new(_freeze_values(data.copy()), name=index.name)


def frozen_already(frame: Any) -> bool:
    """Is all of *frame*'s data frozen by cash already (a hit, a table cash
    froze after unpickling it)? Then it can be kept as it is, by a shallow
    copy (`hand_out`): no holder can write that data, and storing it changes
    nothing any holder may do."""
    try:
        for blk in frame._mgr.blocks:
            if not any(ref is _PIN for ref in blk.refs.referenced_blocks):
                return False
            parts = _ndarray_parts(blk.values)
            if parts is None or not all(not array.flags.writeable and is_frozen(array) for _name, array in parts):
                return False
        axes = (frame.index,) if frame.ndim == 1 else (frame.index, frame.columns)
        for axis in axes:
            parts = _ndarray_parts(getattr(axis, "_data", None)) if not _is_range(axis) else []
            if parts is None or not all(not array.flags.writeable and is_frozen(array) for _name, array in parts):
                return False
        return type(frame).__module__.startswith("pandas") and freezable(frame, cells_known=True)
    except Exception:  # noqa: BLE001 - unknown internals: not frozen
        return False


def _is_range(index: Any) -> bool:
    import pandas as pd

    return isinstance(index, pd.RangeIndex)


def freeze(frame: Any, *, cells_known: bool = False) -> bool:
    """Make *frame*'s data immutable in place (see the module docstring), and
    every pandas object's that shares its blocks. *frame* must be cash's
    own: a copy no caller holds, or a table just unpickled. False, with
    nothing done, when *frame* is not `freezable`.
    """
    return enabled() and _freeze(frame, cells_known=cells_known)


def _freeze(frame: Any, *, cells_known: bool = False) -> bool:
    try:
        if not freezable(frame, cells_known=cells_known):
            return False
        _install_pickling()
        # Not consolidated: the blocks stay as the caller's table had them.
        # One block for every float column made ``data[cols].to_numpy()`` a
        # read-only view of it, where the caller's own table (one block per
        # column, from ``read_csv``) gives a writable copy.
        for blk in frame._mgr.blocks:
            refs = blk.refs
            sharing = [ref() for ref in refs.referenced_blocks]
            for other in [blk, *sharing]:
                if other is None or is_pin(other):
                    continue
                values = other.values
                frozen = _freeze_values(values)
                if frozen is not values:
                    other.values = frozen
            if not any(ref is _PIN for ref in refs.referenced_blocks):
                refs.referenced_blocks.append(_PIN)
        for name in ("index",) if frame.ndim == 1 else ("index", "columns"):
            axis = getattr(frame, name)
            frozen = _frozen_index(axis)
            if frozen is not axis:
                setattr(frame, name, frozen)
        _private_axes(frame)
        return True
    except Exception:
        logger.debug("cash: could not freeze a %s", type(frame).__name__, exc_info=True)
        return False


# ------------------------------------------------- shared with the caller, checked

#: The blocks of the tables `share_checked` made: the RAM tier's own block
#: objects over data a caller may still hold.
_CHECKED_BLOCKS: weakref.WeakSet = weakref.WeakSet()

#: The smallest table `share_checked` keeps uncopied: its numbers and dates
#: in bytes. A smaller one is copied, which costs about what its digest does.
CHECKED_MIN_BYTES = 16 << 20

_CHECKED_TAG = b"cash: a table's data, checked\0"


def share_checked(frame: Any) -> tuple[Any, bytes] | None:
    """A table for the RAM tier to keep in place of a copy of the caller's
    *frame*, and the digest of its data (`checked_digest`); None when
    *frame* is not one to keep so, and the tier copies it as before.

    Copying a table into fresh memory is most of what storing a big one
    costs: on a virtual machine that hands memory back to the host, 1.4 s
    for 380 MB, where its digest takes 0.05 s. And the copy doubles the
    memory the table takes, though the caller usually drops its own a
    statement later (``df = df[df.fare > 0]``).

    The table kept is a shallow copy: its blocks are the tier's own
    objects over the caller's arrays, in the caller's blocks' copy-on-write
    groups, so pandas copies a block before writing it on either side. A
    write past pandas (``s.array[0] = ...``, a numpy array the table was
    built over) is not copied first: the digest, taken now, is checked
    before the table is served or written elsewhere, and a table whose data
    changed is not served. Its labels are copied (`_owned_axis`), and so is
    its Arrow data where it may be someone else's memory (`_copy_values`).

    Only a plain DataFrame `freezable` with *cells_known* (the caller has
    found no object column holding a changeable cell), holding at least
    `CHECKED_MIN_BYTES` in numpy arrays of numbers, dates and durations,
    each in one piece of memory. An object column is copied: its digest is
    of pointers, and a cell written twice can come back to an address."""
    if not _checked_works():
        return None
    try:
        return _share_checked(frame)
    except Exception:  # unknown internals: copied as before
        logger.debug("cash: could not share a %s", type(frame).__name__, exc_info=True)
        return None


def _share_checked(frame: Any) -> tuple[Any, bytes] | None:
    import pandas as pd

    if type(frame) is not pd.DataFrame or not freezable(frame, cells_known=True):
        return None
    mgr = frame._mgr
    blocks = []
    checked = []
    nbytes = 0
    for block in mgr.blocks:
        parts = _ndarray_parts(block.values)
        if parts is None:
            return None
        if not parts:  # Arrow-backed: copied where it may be another's memory
            values = _copy_values(block.values, True)
            blocks.append(type(block)(values, placement=block._mgr_locs, ndim=block.ndim, refs=None))
            continue
        for _name, array in parts:
            if array.dtype.hasobject or _flat_bytes(array) is None:
                return None
            nbytes += array.nbytes
        shallow = block.copy(deep=False)
        blocks.append(shallow)
        if not any(ref is _PIN for ref in block.refs.referenced_blocks):
            checked.append(shallow)  # frozen data (a hit's) is no one's to write
    if nbytes < CHECKED_MIN_BYTES:
        return None
    axes = [_owned_axis(axis, True) for axis in mgr.axes]
    shared_mgr = type(mgr).from_blocks(blocks, axes)
    shared = frame._constructor_from_mgr(shared_mgr, axes=shared_mgr.axes).__finalize__(frame, method="copy")
    if any(kept is not block for kept, block in zip(shared._mgr.blocks, blocks)):
        return None  # pandas rebuilt the blocks: not the ones made here
    for block in checked:
        _CHECKED_BLOCKS.add(block)
    digest = checked_digest(shared)
    if digest is None:
        return None
    return shared, digest


def _group(block: Any) -> list[Any]:
    """The live blocks in *block*'s copy-on-write group, the pin left out."""
    found = []
    for ref in block.refs.referenced_blocks:
        other = ref()
        if other is not None and not is_pin(other):
            found.append(other)
    return found


def _flat_bytes(array: Any) -> Any:
    """*array*'s memory as one flat byte array, in its own order, without a
    copy; None when it is not in one piece."""
    import numpy as np

    if array.flags.c_contiguous:
        flat = array.reshape(-1)
    elif array.flags.f_contiguous:
        flat = array.T.reshape(-1)
    else:
        return None
    return flat.view(np.uint8)  # a contiguous array reshapes to a view


def checked_digest(frame: Any) -> bytes | None:
    """The digest of the data a table `share_checked` made keeps in the
    caller's arrays (its Arrow data and labels are its own); None when an
    array is no longer in one piece."""
    import hashlib

    from cash import bulk_digest

    digest = hashlib.sha256(_CHECKED_TAG)
    for block in frame._mgr.blocks:
        if block not in _CHECKED_BLOCKS:
            continue
        for _name, array in _ndarray_parts(block.values) or ():
            flat = _flat_bytes(array)
            if flat is None:
                return None
            bulk_digest.fold_buffer(digest, flat)
    return digest.digest()


def adopt(frame: Any) -> bool:
    """Freeze a table `share_checked` made, in place, when nothing but the
    RAM tier reaches its data any more (`_only_tier_reaches`): the caller's
    table, and every handle to its arrays, are gone. Its data is then the
    tier's own, as a copy's is, and hits share it. False, with nothing
    done, while anything else may still write it. Check its digest first:
    what is frozen is what is served from then on."""
    try:
        return _adoptable(frame) and _freeze(frame, cells_known=True)
    except Exception:  # unknown internals: left as it is
        logger.debug("cash: could not take over a %s", type(frame).__name__, exc_info=True)
        return False


def _only_tier_reaches(blocks: list[Any]) -> bool:
    """Does nothing but the tier's own blocks reach the arrays of *blocks*?

    Every live block in their copy-on-write groups must be one of the
    tier's (`_CHECKED_BLOCKS`): a caller's table, or any table pandas made
    over that data (a column, a view), is another block in the group. And
    nothing outside a block may hold the arrays: a numpy view a caller
    took (``a = df["x"].array``) holds its memory's array, which a
    reference count shows. Each object is counted against the holders the
    tier's blocks account for: the block holding the values, the wrapper
    holding its numpy array (a date column), the views whose base is it."""
    seen: dict[int, Any] = {}
    for block in blocks:
        for other in _group(block):
            if other not in _CHECKED_BLOCKS:
                return False
            seen[id(other)] = other
    counted = _holders(list(seen.values()))
    if counted is None:
        return False
    objects, holders = counted
    # Each object: the references counted, one from `objects`, one for the call.
    return all(sys.getrefcount(objects[key]) == count + 2 for key, count in holders.items())


def _holders(blocks: list[Any]) -> tuple[dict[int, Any], dict[int, int]] | None:
    """For `_only_tier_reaches`: each object *blocks* keep their data in
    (``id -> object``), and how many references to it they account for;
    None for data held another way (memory an Arrow buffer or another
    object owns)."""
    import numpy as np

    objects: dict[int, Any] = {}
    holders: dict[int, int] = {}

    def held(obj: Any) -> None:
        objects[id(obj)] = obj
        holders[id(obj)] = holders.get(id(obj), 0) + 1

    for block in blocks:
        parts = _ndarray_parts(block.values)
        if not parts:
            return None
        held(block.values)
        for name, array in parts:
            if name:
                held(array)  # the wrapper's array
            base = array.base
            if base is None:
                continue
            if not isinstance(base, np.ndarray) or base.base is not None:
                return None
            held(base)  # a view's reference to the array owning its memory
    return objects, holders


def _adoptable(frame: Any) -> bool:
    """Would `adopt` take *frame* over now?"""
    blocks = [block for block in frame._mgr.blocks if block in _CHECKED_BLOCKS]
    return not blocks or _only_tier_reaches(blocks)  # none: frozen data and its own copies only


def _checked_self_test() -> bool:
    """Do the checks behave as `share_checked` and `adopt` need here? A
    table shared with its caller is not adopted while the caller holds it,
    nor while a numpy handle to its data lives; it is once both are gone.
    A write past pandas changes its digest."""
    import numpy as np
    import pandas as pd

    n = CHECKED_MIN_BYTES // 16 + 1
    caller = pd.DataFrame({"f": np.arange(float(n)), "d": pd.date_range("2020-01-01", periods=n, freq="s")})
    caller = caller[caller["f"] >= 0]  # blocks over views of fresh arrays, as most tables hold
    kept = _share_checked(caller)
    if kept is None:
        return False
    shared, digest = kept
    if _adoptable(shared):
        return False  # the caller's table still holds the data
    handle = caller["f"].array
    dates = caller["d"].array
    del caller
    if _adoptable(shared):
        return False  # numpy handles still do
    del dates
    if _adoptable(shared):
        return False
    handle[0] = -1.0  # a write past pandas
    if checked_digest(shared) == digest:
        return False
    del handle
    return _adoptable(shared)


_CHECKED_OK: list[bool] = []


def _checked_works() -> bool:
    """Is checked sharing on here: pandas tables shared (`enabled`), and the
    checks it relies on behaving as expected (`_checked_self_test`, once)?"""
    if _CHECKED_OK:
        return _CHECKED_OK[0]
    if not enabled():
        return False
    ok = False
    try:
        ok = _checked_self_test()
    except Exception:  # off: tables are copied as before
        logger.debug("cash: tables are copied, not shared with their caller", exc_info=True)
    _CHECKED_OK.append(ok)
    return ok


def copy_keeping_blocks(frame: Any, *, arrow: bool = True, reuse: Any = None) -> Any:
    """``frame.copy(deep=True)`` with *frame*'s blocks as they are, and
    nothing of *frame*'s left in it (`own_parts`).

    pandas joins the blocks of one dtype into one on a deep copy. What a
    table hands out then changes: ``data[["f1", "f2"]].to_numpy()`` on one
    block is a view, read-only under copy-on-write, where the same call on
    one block per column (a fresh ``read_csv``) builds a writable array, so
    the user's ``X -= X.mean(axis=0)`` raised on a cached table only.
    Copied block by block instead, which also skips the join's copy.

    Each block keeps its memory layout, too (`_copy_values`): pandas copies
    a block's array in C order, and a frame built by ``pivot`` holds its
    block in F order -- its ``.values`` row-major. Served such a frame,
    ``held @ close.values[i]`` summed in another order than the computed
    frame, and a backtest moved in the last bit (1.4e-17).

    *arrow* False: *frame* is the RAM tier's own, so its Arrow data, which
    no pandas handle writes, is shared rather than copied (a hit).

    *reuse*, when storing: a `shared_blocks.BlockReuse` for *frame*, whose
    ``find`` gives the values of a block of the same content the RAM tier
    already keeps; such a block is not copied again.
    """
    try:
        mgr = frame._mgr
        if frame.ndim != 2:
            copied = frame.copy(deep=True)
            own_parts(copied, arrow=arrow)
            return copied
        blocks = []
        for block in mgr.blocks:
            values = reuse.find(block) if reuse is not None else None
            if values is None:
                values = _copy_values(block.values, arrow)
            blocks.append(type(block)(values, placement=block._mgr_locs, ndim=block.ndim, refs=None))
        axes = [_owned_axis(axis, arrow) for axis in mgr.axes]
        copied_mgr = type(mgr).from_blocks(blocks, axes)
        return frame._constructor_from_mgr(copied_mgr, axes=copied_mgr.axes).__finalize__(frame, method="copy")
    except Exception:
        logger.debug("cash: copying a %s the pandas way", type(frame).__name__, exc_info=True)
        copied = frame.copy(deep=True)
        own_parts(copied, arrow=arrow)
        return copied


def own_parts(copied: Any, *, arrow: bool = True) -> None:
    """Make a deep pandas copy *copied* keep nothing of the table it was
    copied from, in place.

    A deep pandas copy keeps three things of the original. Its labels: the
    axes are views of the original's (``Index.view``), and ``df.index.array``
    or ``df.columns.values`` write a label array in place. A categorical
    column's categories, which ``s.cat.categories.array`` writes the same
    way (`_owned_categories`). And its Arrow
    data: an Arrow-backed column's copy is a new wrapper over the same
    Arrow arrays, which may be the caller's numpy memory itself --
    ``pa.array(a)`` and ``pa.table(...).to_pandas(types_mapper=pd.ArrowDtype)``
    copy nothing, so ``a[0] = ...`` after the store changed what was stored.
    Every chunk with a buffer that is not memory Arrow allocated itself is
    copied (`_owned_arrow`); the rest, which nothing but Arrow holds, is
    shared. *arrow* False: the RAM tier's own data (a hit), all shared.
    """
    from pandas.core.arrays.arrow import ArrowExtensionArray

    if arrow:
        for block in copied._mgr.blocks:
            values = block.values
            if isinstance(values, ArrowExtensionArray):
                values._pa_array = _owned_arrow(values._pa_array)  # a new wrapper: pandas' copy made it
    for block in copied._mgr.blocks:
        if isinstance(block.values, _categorical()):
            block.values = _owned_categories(block.values, arrow)
    for name in ("index",) if copied.ndim == 1 else ("index", "columns"):
        axis = getattr(copied, name)
        owned = _owned_axis(axis, arrow)
        if owned is not axis:
            setattr(copied, name, owned)


def _owned_axis(axis: Any, arrow: bool = True) -> Any:
    """A copy of *axis* holding no label array of *axis*'s (`own_parts`); a
    range itself, which holds none."""
    import pandas as pd

    if _is_range(axis):
        return axis
    if isinstance(axis, pd.MultiIndex):
        copied = axis.copy(deep=True)
        levels = [_owned_axis(level, arrow) for level in copied.levels]
        if any(new is not old for new, old in zip(levels, copied.levels)):
            copied = copied.set_levels(levels, verify_integrity=False)
        return copied
    from pandas.core.arrays.arrow import ArrowExtensionArray

    data = getattr(axis, "_data", None)
    if isinstance(data, (ArrowExtensionArray, _categorical())):
        return type(axis)._simple_new(_copy_values(data, arrow), name=axis.name)
    return axis.copy(deep=True)


def _copy_values(values: Any, arrow: bool = True) -> Any:
    """A copy of a block's array in its own memory order: a numpy array, or
    a datetime or timedelta array over one, take ``order="K"``; other
    extension arrays have no order to keep. Arrow data that may be someone
    else's memory is copied too (`_owned_arrow`) unless *arrow* is False."""
    import numpy as np

    if isinstance(values, _categorical()):  # over a numpy array too: its codes
        return _owned_categories(values, arrow)
    if isinstance(values, np.ndarray) or isinstance(getattr(values, "_ndarray", None), np.ndarray):
        try:
            return values.copy(order="K")
        except TypeError:  # an array type whose copy takes no order
            pass
    copied = values.copy()
    if arrow:
        from pandas.core.arrays.arrow import ArrowExtensionArray

        if isinstance(copied, ArrowExtensionArray):
            copied._pa_array = _owned_arrow(copied._pa_array)  # a new wrapper: ``copy`` made it
    return copied


def _categorical() -> type:
    import pandas as pd

    return pd.Categorical


def _owned_categories(values: Any, arrow: bool = True) -> Any:
    """A copy of the Categorical *values* over a copy of its categories:
    ``Categorical.copy`` keeps the categories, an index whose array
    ``s.cat.categories.array[0] = ...`` writes in place."""
    import pandas as pd

    dtype = pd.CategoricalDtype(_owned_axis(values.categories, arrow), ordered=values.ordered)
    return pd.Categorical.from_codes(values.codes.copy(), dtype=dtype, validate=False)


def _owned_arrow(chunked: Any) -> Any:
    """The Arrow ChunkedArray *chunked* with every chunk that may hold
    memory someone else writes copied into memory Arrow allocated for it;
    *chunked* itself when no chunk does.

    Arrow data may wrap memory someone else owns and writes: ``pa.array(a)``
    over a numpy array copies nothing, and neither does
    ``pa.table(...).to_pandas(types_mapper=pd.ArrowDtype)``. A chunk whose
    buffers are all Arrow's own (`arrow_owned`: allocated from a memory pool,
    or a slice of such a buffer) is kept as it is: Arrow never writes a
    buffer an array holds, and no Python object views that memory writably
    but the buffer's own ``memoryview``. Every other chunk is copied, and
    only it: chunks are never joined. A copied slice comes out as its rows.
    """
    import pyarrow as pa

    chunks = chunked.chunks
    owned = [chunk if _chunk_owned(chunk) else _owned_chunk(chunk) for chunk in chunks]
    if all(new is old for new, old in zip(owned, chunks)):
        return chunked
    return pa.chunked_array(owned, type=chunked.type)


def owned_arrow_data(data: Any) -> Any:
    """An Arrow Array, ChunkedArray or Table holding no memory but Arrow's
    own (`_owned_arrow`): *data* itself when it holds none else, or when it
    is none of them."""
    import pyarrow as pa

    if isinstance(data, pa.ChunkedArray):
        return _owned_arrow(data)
    if isinstance(data, pa.Array):
        return data if _chunk_owned(data) else _owned_chunk(data)
    if isinstance(data, pa.Table):
        columns = [_owned_arrow(column) for column in data.columns]
        if all(new is old for new, old in zip(columns, data.columns)):
            return data
        return pa.Table.from_arrays(columns, schema=data.schema)
    return data


def _owned_chunk(chunk: Any) -> Any:
    """A copy of the Arrow Array *chunk*, every buffer written anew."""
    import pyarrow as pa

    if pa.types.is_dictionary(chunk.type):
        indices, dictionary = _owned_chunk(chunk.indices), _owned_chunk(chunk.dictionary)
        owned = pa.DictionaryArray.from_arrays(indices, dictionary, ordered=chunk.type.ordered)
        if owned.type == chunk.type:
            return owned
    elif not _holds_dictionary(chunk.type):
        # ``concat_arrays`` writes every buffer anew, but a type it may
        # append as it is (views over data buffers) is checked, not trusted.
        owned = pa.concat_arrays([chunk])
        if not _overlapping(owned.buffers(), chunk.buffers()):
            return owned
    return _ipc_copy(chunk)


def _holds_dictionary(arrow_type: Any) -> bool:
    import pyarrow as pa

    if pa.types.is_dictionary(arrow_type):
        return True
    storage = getattr(arrow_type, "storage_type", None)
    if storage is not None:
        return _holds_dictionary(storage)
    return any(_holds_dictionary(arrow_type.field(i).type) for i in range(arrow_type.num_fields))


def _overlapping(buffers: list[Any], others: list[Any]) -> bool:
    """Does a buffer of *buffers* share memory with one of *others*?"""
    spans = [(b.address, b.address + b.size) for b in others if b is not None and b.size]
    for buffer in buffers:
        if buffer is None or not buffer.size:
            continue
        start, end = buffer.address, buffer.address + buffer.size
        if any(start < other_end and other_start < end for other_start, other_end in spans):
            return True
    return False


def _ipc_copy(chunk: Any) -> Any:
    """*chunk* written to Arrow's IPC format in a buffer of Arrow's own and
    read back from it: a copy of every buffer, of any type (slower than
    ``concat_arrays``, so only for nested dictionaries and the like)."""
    import pyarrow as pa

    batch = pa.RecordBatch.from_arrays([chunk], names=["c"])
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return pa.ipc.open_stream(sink.getvalue()).read_next_batch().column(0)


# ------------------------------------------------- whose memory an Arrow buffer is
#
# pyarrow says nothing of where a buffer's memory came from: ``is_mutable``
# is True for one its pool allocated and for one over a writable numpy
# array alike, and ``parent`` is only set on a slice. The C++ object behind
# a ``pa.Buffer`` does say it, by its class: a ``PoolBuffer`` is memory a
# memory pool allocated for Arrow, a ``NumPyBuffer`` or ``PyBuffer`` wraps a
# Python object's, an imported one (the C data interface: polars,
# DuckDB, ...) someone else's. Its class is read off its vtable pointer, the
# first word of the C++ object, which the ``pa.Buffer`` holds by a
# ``shared_ptr``. Where to find both is learnt once from buffers of known
# classes (`_find_layout`), and checked on every buffer read (the data
# address and size must be the ones pyarrow reports); when anything does
# not match, nothing is taken as Arrow's own and every chunk is copied.


class _Layout:
    """Where a ``pa.Buffer``'s C++ buffer is, and the vtables that tell its
    class (`_find_layout`)."""

    __slots__ = ("cpp", "data", "pool", "slice")

    def __init__(self, cpp: int, data: int, pool: int, slice_: int) -> None:
        self.cpp = cpp  # offset of the C++ Buffer* in the pa.Buffer object
        self.data = data  # offset of data_ (then size_) in the C++ Buffer
        self.pool = pool  # PoolBuffer's vtable: memory a pool allocated
        self.slice = slice_  # plain Buffer's vtable: a slice of its parent


_LAYOUT: list[_Layout | None] = []
_LAYOUT_LOCK = threading.Lock()


def _word(address: int) -> int:
    return ctypes.c_size_t.from_address(address).value


def _arrow_layout() -> _Layout | None:
    if _LAYOUT:
        return _LAYOUT[0]
    with _LAYOUT_LOCK:
        if not _LAYOUT:
            try:
                layout = _find_layout()
            except Exception:  # unknown internals: copy everything
                logger.debug("cash: Arrow buffers' owners unknown, Arrow data is copied", exc_info=True)
                layout = None
            _LAYOUT.append(layout)
    return _LAYOUT[0]


def _find_layout() -> _Layout | None:
    """Learn `_Layout` from buffers of known classes; None when they do not
    tell pool memory from numpy's, Python's and a slice's apart."""
    import numpy as np
    import pyarrow as pa

    if sys.implementation.name != "cpython" or ctypes.sizeof(ctypes.c_size_t) != 8 or pa.Buffer.__basicsize__ > 256:
        return None  # id() is an object's address in CPython only
    small, other = pa.allocate_buffer(1024), pa.allocate_buffer(2048)
    again = pa.Array.from_buffers(pa.uint8(), 1024, [None, small]).buffers()[1]  # a second pa.Buffer of `small`
    if type(small) is not pa.Buffer or type(again) is not pa.Buffer or again.address != small.address:
        return None
    # The shared_ptr's two words are the same in both objects over `small`
    # and differ for `other`. The object's other words that do so are small
    # numbers (its shape: 1024), never read through.
    candidates = []
    for offset in range(16, pa.Buffer.__basicsize__ - 7, 8):
        word = _word(id(small) + offset)
        if word == _word(id(again) + offset) and word != _word(id(other) + offset) and word > 1 << 16 and not word % 8:
            candidates.append(offset)
    for cpp in candidates:
        pointer = _word(id(small) + cpp)
        data = next(
            (at for at in range(8, 64, 8) if _word(pointer + at) == small.address and _word(pointer + at + 8) == 1024),
            None,
        )
        if data is None:
            continue
        layout = _Layout(cpp, data, _vtable(small, cpp, data), 0)
        sliced = small.slice(16, 64)
        layout.slice = _vtable(sliced, cpp, data)
        numbers = np.arange(128, dtype=np.int64)
        foreign = [pa.py_buffer(bytearray(64)), pa.array(numbers).buffers()[1], pa.py_buffer(numbers)]
        pooled = [other, pa.concat_arrays([pa.array(numbers)]).buffers()[1]]
        if (
            layout.pool
            and layout.slice
            and layout.slice != layout.pool
            and sliced.parent is not None
            and all(_vtable(buffer, cpp, data) == layout.pool for buffer in pooled)
            and arrow_owned(sliced, layout)
            and not any(arrow_owned(buffer, layout) for buffer in foreign)
        ):
            return layout
    return None


def _vtable(buffer: Any, cpp: int, data: int) -> int:
    """The vtable of *buffer*'s C++ object; 0 when its data address and size
    are not where `_Layout` has them."""
    pointer = _word(id(buffer) + cpp)
    if not pointer or _word(pointer + data) != buffer.address or _word(pointer + data + 8) != buffer.size:
        return 0
    return _word(pointer)


def arrow_owned(buffer: Any, layout: _Layout | None = None) -> bool:
    """Is the ``pa.Buffer`` *buffer* memory an Arrow memory pool allocated,
    or a slice of such, so nothing but Arrow holds it? False when that
    cannot be told (`_find_layout`).

    Pool memory is written only through a ``pa.Buffer`` of it, by its
    buffer protocol (``memoryview(arr.buffers()[1])``, ``np.frombuffer`` of
    it, a buffer from ``pa.allocate_buffer`` filled in place): that writes
    into an array Arrow treats as immutable. A hit, which shares the
    stored copy's Arrow memory, has always relied on the same."""
    import pyarrow as pa

    layout = layout or _arrow_layout()
    if layout is None:
        return False
    for _depth in range(64):
        if type(buffer) is not pa.Buffer:
            return False
        vtable = _vtable(buffer, layout.cpp, layout.data)
        if vtable == layout.pool:
            return True
        if vtable != layout.slice:
            return False  # numpy's, Python's, imported, mapped from a file, or unknown
        buffer = buffer.parent
    return False


def _chunk_owned(chunk: Any) -> bool:
    """Is every buffer of the Arrow Array *chunk* (its children's and its
    dictionary's too) Arrow's own (`arrow_owned`)?"""
    import pyarrow as pa

    if _arrow_layout() is None:
        return False
    if pa.types.is_dictionary(chunk.type):
        return _buffers_owned(chunk.indices.buffers()) and _chunk_owned(chunk.dictionary)
    if _holds_dictionary(chunk.type):
        return False  # a dictionary under a list or struct: not walked, copied
    return _buffers_owned(chunk.buffers())


def _buffers_owned(buffers: list[Any]) -> bool:
    return all(buffer is None or arrow_owned(buffer) for buffer in buffers)


def arrow_data_owned(data: Any) -> bool:
    """Is the Arrow Array, ChunkedArray, Table or RecordBatch *data* held
    in memory Arrow allocated alone (`arrow_owned`, every chunk of every
    column)? Then nothing but a write through a ``pa.Buffer``'s own buffer
    protocol can change it. False for anything else, and when that cannot
    be told."""
    import pyarrow as pa

    if isinstance(data, pa.Array):
        return _chunk_owned(data)
    if isinstance(data, pa.ChunkedArray):
        return all(_chunk_owned(chunk) for chunk in data.chunks)
    if isinstance(data, (pa.Table, pa.RecordBatch)):
        return all(arrow_data_owned(column) for column in data.columns)
    return False


def hand_out(stored: Any) -> Any:
    """A table to give a caller for the frozen *stored* one: a shallow copy,
    which pandas copies before any write."""
    copied = stored.copy(deep=False)
    _private_axes(copied)
    return copied


def _private_axes(frame: Any) -> None:
    """Give *frame* axes of its own where a shallow copy would share a
    writable label array: Arrow-backed labels (pandas 3 text) are written by
    swapping the Arrow array inside the object both copies hold."""
    from pandas.core.arrays.arrow import ArrowExtensionArray

    for name in ("index",) if frame.ndim == 1 else ("index", "columns"):
        axis = getattr(frame, name)
        if _is_range(axis):
            # No labels to share, and reading its ``_data`` builds them: an
            # arange of every row, 0.8 s for 21M rows, on each store and hit.
            continue
        data = getattr(axis, "_data", None)
        if isinstance(data, ArrowExtensionArray):
            fresh = type(axis)._simple_new(data.copy(), name=axis.name)  # a new wrapper; Arrow data is not copied
            setattr(frame, name, fresh)


# ---------------------------------------------------------------- pickling


def _writable_alias(array: Any) -> Any:
    """A writable array over a frozen *array*'s memory, for pickle to read
    alone: pickle writes a read-only array as read-only, and a table loaded
    from it could then not be written by pandas (it would hold no `_PIN`).
    Keeps *array* alive. Never handed to anything but a pickler."""
    import numpy as np

    if (
        array.flags.writeable
        or array.dtype.hasobject
        or not (array.flags.c_contiguous or array.flags.f_contiguous)
        or array.nbytes == 0
        or not is_frozen(array)
    ):
        return array  # numpy pickles these writable, or it was read-only before cash
    memory = (ctypes.c_char * array.nbytes).from_address(array.ctypes.data)
    memory._cash_keeps = array  # ctypes arrays take attributes: the memory lives as long as the alias
    order = "C" if array.flags.c_contiguous else "F"
    return np.ndarray(array.shape, dtype=array.dtype, buffer=memory, order=order)


def _pickle_form(values: Any) -> Any:
    """*values* as pickle should write them: with frozen arrays as writable
    aliases (`_writable_alias`)."""
    try:
        parts = _ndarray_parts(values)
        if not parts:
            return values
        if parts[0][0] == "":
            return _writable_alias(values)
        alias = _writable_alias(values._ndarray)
        return values if alias is values._ndarray else _same_dates(values, alias)
    except Exception:
        logger.debug("cash: pickling a frozen block as it is", exc_info=True)
        return values


_PICKLING_INSTALLED: list[bool] = []
_INSTALL_LOCK = threading.Lock()


def _install_pickling() -> None:
    """Make pandas pickle a frozen block's data as writable (`_pickle_form`).
    Once per process, before the first table is frozen."""
    if _PICKLING_INSTALLED:
        return
    with _INSTALL_LOCK:
        if _PICKLING_INSTALLED:
            return
        from pandas.core.internals.blocks import Block
        from pandas.core.internals.managers import SingleBlockManager

        block_reduce = Block.__reduce__

        def __reduce__(self: Any) -> Any:
            reduced = block_reduce(self)
            func, args = reduced[0], reduced[1]
            form = _pickle_form(args[0])
            if form is args[0]:
                return reduced
            return (func, (form, *args[1:]), *reduced[2:])

        single_getstate = SingleBlockManager.__getstate__

        def __getstate__(self: Any) -> Any:
            axes, block_values, block_items, extra = single_getstate(self)
            forms = {id(values): _pickle_form(values) for values in block_values}
            block_values = [forms[id(values)] for values in block_values]
            for block in extra["0.14.1"]["blocks"]:
                block["values"] = forms.get(id(block["values"]), block["values"])
            return axes, block_values, block_items, extra

        Block.__reduce__ = __reduce__
        SingleBlockManager.__getstate__ = __getstate__
        _PICKLING_INSTALLED.append(True)
