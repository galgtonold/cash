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
the store copies it once, and every hit shares that copy.

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

__all__ = ["enabled", "freeze", "frozen_already", "hand_out", "is_frozen", "is_pin"]


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
    except Exception:  # noqa: BLE001 - unknown pandas: copy as before
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
        frame._consolidate_inplace()
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
    except Exception:  # noqa: BLE001 - a pandas internals change: copy instead
        logger.debug("cash: could not freeze a %s", type(frame).__name__, exc_info=True)
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
    except Exception:  # noqa: BLE001 - pickled as it is: read-only, never wrong
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
