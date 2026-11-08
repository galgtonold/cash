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
  dies (`_PinRef`). pandas then always sees the block as shared and copies it
  before writing, even once the stored table and every other hit are gone:
  a frozen block is never written in place, by pandas or anyone else.

A hit therefore differs from a plain run in one way: a numpy array taken
from it (``v = s.to_numpy()``, ``s.values``) keeps the old values when
pandas then writes the column, where a plain run writes the column in place
and the array sees the write. Such writes are watched
(`_install_write_watch`), and `views_left_behind` names the memory such an
array still reads, for the notebook to warn about. A handle that writes past
pandas (``s.array[0] = ...``) raises on a hit; `read_only_note` says why.

Only columns kept in plain numpy arrays (numbers, dates and times, object
columns whose cells cannot change) and Arrow-backed ones are frozen. A
nullable, categorical or period column hands out ``s.values`` as the
extension array itself, writable by design, so a table holding one is not
shared: the tier copies it.

The data a table reads stays where it is; pandas writes to a frozen table
behave as on any other, they just copy first. A table frozen this way pickles
as a writable one (`_install_pickling`): a disk entry, or a table the user
pickles themselves, loads as an ordinary table.

Only a table cash owns is frozen: its own copy made when storing. A table
read from disk goes to the caller unfrozen, as its own (`InMemoryBackend.promote`).
A table the caller holds is never
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

from . import heap_blocks

logger = logging.getLogger(__name__)

__all__ = [
    "arrow_data_owned",
    "arrow_owned",
    "copy_keeping_blocks",
    "enabled",
    "forget_asked",
    "freeze",
    "frozen_already",
    "hand_out",
    "has_pin",
    "is_frozen",
    "is_pin",
    "own_parts",
    "owned_arrow_data",
    "read_only_note",
    "reads_memory_of",
    "views_left_behind",
]


class _Pin:
    """What the reference that never dies points to (`_PinRef`)."""

    __slots__ = ("__weakref__",)

    def __repr__(self) -> str:
        return "<cash: data shared with the RAM tier>"


_PIN_TARGET = _Pin()


class _PinRef(weakref.ref):
    """The reference a frozen block's copy-on-write group keeps: alive for
    good, so pandas always copies the block before writing it. One per
    group, holding the arrays that own the group's frozen memory (*roots*)
    and the blocks cash knows to be in the group (*blocks*, weakly), so
    that a write can count who still reads that memory (`_note_shared`).
    *writes* and *left*: how many writes pandas checked the group before,
    and whether the last left a reader behind (`_watch_self_test`)."""

    __slots__ = ("__weakref__", "blocks", "left", "roots", "writes")

    def __new__(cls, roots: list[Any]) -> _PinRef:
        return super().__new__(cls, _PIN_TARGET)

    def __init__(self, roots: list[Any]) -> None:
        super().__init__(_PIN_TARGET)
        self.roots = [weakref.ref(root) for root in roots]
        self.blocks: list[Any] = []
        self.writes = 0
        self.left = False

    def know(self, block: Any) -> None:
        """Remember *block*, one of the group's, to find the group by."""
        self.blocks = [ref for ref in self.blocks if ref() is not None]
        self.blocks.append(weakref.ref(block))


#: ``id(root) -> {id(pin): weak reference to pin}`` for the groups whose
#: frozen memory *root* owns; a column the RAM tier keeps for two entries is
#: in two groups.
_PINS_BY_ROOT: dict[int, Any] = {}


def is_pin(obj: Any) -> bool:
    """Is *obj* what a `_PinRef` points to? (not a block: walks of a
    reference group skip it)."""
    return obj is _PIN_TARGET


def _pin_of(refs: Any) -> _PinRef | None:
    """The `_PinRef` in the copy-on-write group *refs*; None in a group
    cash has not frozen."""
    for ref in refs.referenced_blocks:
        if type(ref) is _PinRef:
            return ref
    return None


def has_pin(block: Any) -> bool:
    """Is *block* in a group cash froze (its data the RAM tier's)?"""
    return _pin_of(block.refs) is not None


# ------------------------------------------------- writes a view does not see
#
# A served table shares its memory with the RAM tier's copy, so pandas
# copies a column before writing it, and a numpy array taken from the
# column before (``v = s.to_numpy()``, ``s.values``, ``s.array``) keeps the
# old values, where a plain run writes the column in place and the array
# sees the write. That cannot be matched without copying every hit, so the
# write is counted as pandas checks the column (`_install_write_watch`),
# and warned about: at the end of the statement in a notebook cell cash
# runs (`defer_reports`, `views_left_behind`), at the write itself
# anywhere else (`_warn_at_write`).

#: What writes left behind while reports are deferred, oldest first: for
#: each, weak references to the roots still read.
_LEFT: list[list[Any]] = []
_LEFT_LOCK = threading.Lock()
#: How many callers defer the reports (`defer_reports`).
_DEFERRING: list[int] = [0]
#: This thread's `_watch_self_test` writes: counted, never reported.
_QUIET = threading.local()
#: The places `_warn_at_write` warned about, ``(file, line)``.
_WARNED_AT: set[tuple[str, int]] = set()
_UNCHECKED_WARNED: list[bool] = []


def _note_shared(refs: Any) -> None:
    """pandas found the group *refs* shared before writing one of its
    blocks: when cash froze it, count who still reads its memory besides
    pandas' blocks (`_read_elsewhere`), and report it (`_left_behind`)."""
    pin = _pin_of(refs)
    if pin is None:
        return
    pin.writes += 1
    if not _counts_exact():
        _warn_unchecked()
        return
    try:
        left = _read_elsewhere(pin)
    except Exception:
        logger.debug("cash: could not count the readers of a written table", exc_info=True)
        return
    pin.left = bool(left)
    if left and not getattr(_QUIET, "on", False):
        _left_behind(left)


def _left_behind(left: list[Any]) -> None:
    if _DEFERRING[0] > 0:
        with _LEFT_LOCK:
            _LEFT.append(left)
    else:
        _warn_at_write()


#: ``id(root) -> references`` the views `for_storing` made hold to *root*:
#: cash's own, credited when a write counts who reads that memory.
_STORE_HELD: dict[int, int] = {}
_STORE_LOCK = threading.Lock()
#: How many `for_storing` calls are running, on whatever thread: their
#: locals hold the blocks' values for a moment (`_held_by_pandas_and_cash`).
_DETACHING = [0]


def for_storing(value: Any) -> Any:
    """*value* for a disk write to serialize off the caller's thread: each
    frozen pandas table in it (as deep as the RAM tier looks for tables)
    replaced by an equal one whose frozen columns are fresh views of the
    same memory, outside its copy-on-write group.

    The pickler and the payload's buffers then hold only those views, which
    a write into a served table does not count as readers, while each view
    itself is credited for as long as it lives (`_STORE_HELD`): the count
    stays exact while the disk write runs. *value* itself when it holds no
    frozen table, or when a table cannot be rebuilt so; the disk write's
    references then count as readers (a write warns, never misses)."""
    with _STORE_LOCK:
        _DETACHING[0] += 1
    try:
        return _for_storing(value, {}, 0)
    except Exception:
        logger.debug("cash: a table stored to disk as it is", exc_info=True)
        return value
    finally:
        with _STORE_LOCK:
            _DETACHING[0] -= 1


def _for_storing(value: Any, done: dict[int, Any], depth: int) -> Any:
    if id(value) in done:
        return done[id(value)]
    kind = type(value)
    if kind in (dict, list, tuple):
        if depth >= 2:
            return value
        if kind is dict:
            items = {key: _for_storing(item, done, depth + 1) for key, item in value.items()}
            changed = any(items[key] is not value[key] for key in value)
            result = items if changed else value
        else:
            parts = [_for_storing(item, done, depth + 1) for item in value]
            changed = any(new is not old for new, old in zip(parts, value, strict=True))
            result = kind(parts) if changed else value
    elif getattr(value, "_mgr", None) is not None and kind.__module__.startswith("pandas"):
        result = _detached(value)
    else:
        return value
    done[id(value)] = result
    return result


def _detached(table: Any) -> Any:
    """*table* rebuilt on fresh views of its frozen columns (`for_storing`);
    *table* itself when none of its blocks is frozen."""
    mgr = table._mgr
    if not any(_pin_of(block.refs) is not None for block in mgr.blocks):
        return table
    blocks = []
    for block in mgr.blocks:
        values = block.values
        if _pin_of(block.refs) is not None:
            values = _fresh_view(values)
        blocks.append(block.make_block_same_class(values))
    if table.ndim == 2:
        fresh = type(mgr)(tuple(blocks), list(mgr.axes), verify_integrity=False)
    else:
        fresh = type(mgr)(blocks[0], mgr.axes[0], verify_integrity=False)
    return table._constructor_from_mgr(fresh, axes=fresh.axes).__finalize__(table)


def _fresh_view(values: Any) -> Any:
    """A new view of the frozen *values*, credited to its root while alive."""
    import numpy as np

    if type(values) is np.ndarray:
        view = values.view()
        fresh = view
    else:
        parts = _ndarray_parts(values)
        if not parts:
            return values  # Arrow: no numpy memory to count
        view = values._ndarray.view()
        fresh = values._from_backing_data(view)
        if getattr(values, "_freq", None) is not None:
            fresh._freq = values._freq
    key = id(_root(view))
    # Credited only once it exists: in between, it counts as a reader.
    with _STORE_LOCK:
        _STORE_HELD[key] = _STORE_HELD.get(key, 0) + 1
    weakref.finalize(view, _store_view_gone, key).atexit = False
    return fresh


def _store_view_gone(key: int) -> None:
    with _STORE_LOCK:
        left = _STORE_HELD.pop(key, 1) - 1
        if left > 0:
            _STORE_HELD[key] = left


def defer_reports(on: bool) -> None:
    """While on, a write that leaves an array behind is kept for
    `views_left_behind` rather than warned about where it happens: the
    notebook warns at the end of the statement, naming the variables.
    Calls nest; each ``True`` is matched by a ``False``."""
    with _LEFT_LOCK:
        _DEFERRING[0] = max(0, _DEFERRING[0] + (1 if on else -1))


def forget_asked() -> None:
    """Drop what `views_left_behind` would report."""
    with _LEFT_LOCK:
        _LEFT.clear()


def views_left_behind() -> tuple[int, list[Any]]:
    """The writes kept since the last call (`defer_reports`) that left a
    numpy array reading a served table's old memory, and the arrays owning
    that memory still alive (an array taken inside a function is gone by
    the end of the statement; its write is still counted).

    In a plain run the write reaches the array when no other table shares
    the column; that is not told apart, so a write pandas would have
    copied anyway is reported too."""
    with _LEFT_LOCK:
        taken = list(_LEFT)
        _LEFT.clear()
    roots: list[Any] = []
    for refs in taken:
        for ref in refs:
            root = ref()
            if root is not None and not any(root is seen for seen in roots):
                roots.append(root)
    return len(taken), roots


def _counts_exact() -> bool:
    """Are reference counts exact here? Not on a free-threaded CPython
    with the GIL off, where another thread's references are counted
    apart."""
    gil = getattr(sys, "_is_gil_enabled", None)
    return gil is None or bool(gil())


def _warn_unchecked() -> None:
    """Said once: writes into served tables cannot be checked here."""
    if _UNCHECKED_WARNED or getattr(_QUIET, "on", False):
        return
    _UNCHECKED_WARNED.append(True)
    from ..diagnostics import warn_diagnostic
    from ..exceptions import CashWarning

    what = (
        "a pandas table served from the cache was written, and this free-threaded Python counts "
        "references inexactly, so cash cannot tell whether a numpy array taken from the table before "
        "the write still holds the old values (in a plain run it would see the write)."
    )
    fix = "take arrays from a served table after writing it, or copy the table first (df = df.copy())."
    warn_diagnostic(CashWarning, "CACHE-SERVED-VIEW", what, fix, location=_write_site())


def _write_site() -> tuple[str, int]:
    """The line of the caller's own code that wrote the table: the first
    frame outside pandas and cash."""
    import os

    import pandas as pd

    skip = (os.path.dirname(pd.__file__) + os.sep, os.path.dirname(os.path.dirname(__file__)) + os.sep)
    frame = sys._getframe(2)
    while frame is not None and frame.f_code.co_filename.startswith(skip):
        frame = frame.f_back
    if frame is None:
        return "<cash>", 0
    return frame.f_code.co_filename, frame.f_lineno


def _warn_at_write() -> None:
    """CACHE-SERVED-VIEW, once per line of the caller's that writes: a
    table ``@cash.cache`` served is about to be written while a numpy
    array taken from it still reads its old memory (`_note_shared`)."""
    from ..diagnostics import warn_diagnostic
    from ..exceptions import CashWarning

    site = _write_site()
    if site in _WARNED_AT:
        return
    _WARNED_AT.add(site)
    what = (
        "this line writes into a pandas table a cached function returned from the cache, while a numpy "
        "array taken from the table before (with .to_numpy(), .values or .array) is still held. The "
        "served table shares its memory with the cache's copy, so pandas copies the column before "
        "writing it, and the array keeps the old values. In a plain run the write reaches the array "
        "whenever no other table shares the column."
    )
    fix = (
        "take the array after the write, or copy the returned table (df = df.copy()) and take the "
        "array from the copy, so that it sees later writes as in a plain run."
    )
    warn_diagnostic(CashWarning, "CACHE-SERVED-VIEW", what, fix, location=site)


#: ``id(array) -> weak reference`` for each array whose memory `freeze` made
#: read-only: the arrays that pickle as writable (`_pickle_form`).
_FROZEN: dict[int, Any] = {}
_FROZEN_LOCK = threading.Lock()


def _remember_frozen(array: Any) -> None:
    key = id(array)
    with _FROZEN_LOCK:
        if key not in _FROZEN:
            _FROZEN[key] = weakref.ref(array, lambda _ref, key=key: _forget_root(key))


def _forget_root(key: int) -> None:
    _FROZEN.pop(key, None)
    _PINS_BY_ROOT.pop(key, None)


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
    if not ok:
        _uninstall_write_watch()  # the self-test may have put it in
    _ENABLED.append(ok)
    return ok


def _self_test(pd: Any) -> bool:
    """Freeze a small table of each kind of column, write a hit of it
    through pandas with the stored table gone, and check nothing leaked."""
    import numpy as np

    columns = {"f": np.arange(4.0), "i": np.arange(4), "d": pd.date_range("2020-01-01", periods=4)}
    text = pd.array(["a", "b", "c", "d"])
    if _ndarray_parts(text) is not None:
        # Text is frozen where pandas keeps it in Arrow. Without pyarrow,
        # pandas 3 keeps it in a numpy array of str a table hands out
        # writable (``.array``): such a table is copied, as before, and
        # the test is of the kinds frozen, so tables of numbers and dates
        # are still shared.
        columns["s"] = text
    df = pd.DataFrame(columns, index=pd.date_range("2024-01-01", periods=4, freq="D"))
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
    if not _watch_self_test(pd, hit, stored_series):
        return False
    hit.iloc[1, 2] = pd.Timestamp("2021-01-01")
    hit.iloc[3, 1] = 9
    if other["f"].iloc[0] != 0.0 or other["i"].iloc[3] != 3:
        return False
    if "s" in columns:
        hit.iloc[2, 3] = "z"
        if other["s"].iloc[2] != "c":
            return False
    try:
        other["f"].array[1] = 5.0
    except ValueError:
        pass
    else:
        return False
    return bool(other["f"].iloc[1] == 1.0)


def _watch_self_test(pd: Any, hit: Any, stored_series: Any) -> bool:
    """Does the write watch see writes into a hit, through each place it
    hooks, count a write with nothing else holding the data as clean, and
    one with a numpy array taken before it as leaving it behind? When not,
    tables are not shared (`enabled` takes the watch out again): copied,
    they behave as in a plain run."""
    import numpy as np

    if not _counts_exact():
        return True  # nothing is counted: a write says once it cannot be (`_warn_unchecked`)
    _QUIET.on = True
    try:
        ok = _watch_checks(pd, np, hit, stored_series)
    except Exception:
        logger.debug("cash: the write watch failed its self-test", exc_info=True)
        ok = False
    finally:
        _QUIET.on = False
    return ok


def _watch_checks(pd: Any, np: Any, hit: Any, stored_series: Any) -> bool:
    def written(table: Any, write: Any) -> tuple[bool, bool]:
        """Did writing *table* count as a write (`_note_shared`), and did
        it leave a reader behind?"""
        mgr = table._mgr
        pin = _pin_of(mgr.blocks[mgr.blknos[0] if table.ndim == 2 else 0].refs)
        if pin is None:
            return False, False
        before = pin.writes
        write(table)
        return pin.writes > before, pin.left

    clean = (True, False)
    # A frame's block, through the manager (`_has_no_reference_block`).
    if written(hand_out(hit), lambda t: t.iloc.__setitem__((0, 0), 100.0)) != clean:
        return False
    # A Series (`SingleBlockManager._has_no_reference`).
    if written(hand_out(stored_series), lambda t: t.iloc.__setitem__(0, 100.0)) != clean:
        return False
    # Block methods writing in place: `_maybe_copy` (mask, where) and
    # `_get_refs_and_copy` (ffill, fillna, interpolate).
    if written(hand_out(stored_series), lambda t: t.mask(t > 1, 0.0, inplace=True)) != clean:
        return False
    if written(hand_out(stored_series), lambda t: t.ffill(inplace=True)) != clean:
        return False
    # An array taken before the write still reads the old memory. Kept in
    # a list, as a caller's would be: the locals of cash's own frames are
    # counted as the write's (`_held_by_pandas_and_cash`).
    frame = hand_out(hit)
    kept = [frame["f"].to_numpy()]
    seen = written(frame, lambda t: t.iloc.__setitem__((0, 0), 100.0))
    return seen == (True, True) and kept[0][0] != 100.0


# ---------------------------------------------------------------- freezing


def block_arrays(frame: Any) -> list[Any]:
    """The numpy arrays *frame*'s blocks keep their data in (`_ndarray_parts`)."""
    try:
        blocks = frame._mgr.blocks
    except AttributeError:
        return []
    return [array for block in blocks for _name, array in (_ndarray_parts(block.values) or ())]


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


def _label_parts(data: Any) -> list[tuple[str, Any]] | None:
    """`_ndarray_parts` of an index's labels, and the numpy array of str
    pandas 3 keeps text labels in without pyarrow: a column kind tables are
    copied for, as its ``.array`` is written in place, but an index is never
    written through pandas, and its cells are str or missing. Without it,
    every table with named columns was copied, and so was each hit."""
    parts = _ndarray_parts(data)
    if parts is None and _python_text(data):
        return [("_ndarray", data._ndarray)]
    return parts


def _python_text(data: Any) -> bool:
    import pandas as pd

    return isinstance(data, pd.arrays.StringArray)


def _freezable_index(index: Any) -> bool:
    import pandas as pd

    if isinstance(index, pd.RangeIndex):
        return True
    if isinstance(index, pd.MultiIndex):
        return False
    return _python_text(index._data) or _freezable_values(index._data)


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


#: What the `_freeze` running on this thread has changed so far, for
#: `_undo` to put back when it fails part way (`_Journal`).
_JOURNAL = threading.local()


class _Journal:
    """The changes one `_freeze` makes, in order: arrays made read-only,
    block values replaced, pins added, axes set. A freeze that fails part
    way puts them all back (`_undo`): a table it did not finish is left as
    it was, never with read-only arrays pandas does not know are shared
    (it would write them in place, and the write would raise)."""

    __slots__ = ("arrays", "axes", "pins", "values")

    def __init__(self) -> None:
        self.arrays: list[Any] = []
        self.values: list[tuple[Any, Any]] = []
        self.pins: list[list[Any]] = []
        self.axes: list[tuple[str, Any]] = []


def _undo(frame: Any, journal: _Journal) -> None:
    """Put back what a failed `_freeze` of *frame* changed (`_Journal`)."""
    for name, axis in reversed(journal.axes):
        setattr(frame, name, axis)
    for referenced, pin in journal.pins:
        while any(ref is pin for ref in referenced):
            referenced.remove(pin)
    for block, values in reversed(journal.values):
        block.values = values
    # Bases were made read-only after the views over them: made writable
    # again first, as numpy requires of a view.
    for array in reversed(journal.arrays):
        array.flags.writeable = True
        with _FROZEN_LOCK:
            _FROZEN.pop(id(array), None)


def _read_only(array: Any) -> Any:
    """*array* and every array it views made read-only; a view of it when it
    owns its memory, so no one holding the result can make it writable."""
    import numpy as np

    journal = getattr(_JOURNAL, "current", None)
    link = array
    while isinstance(link, np.ndarray):
        if link.flags.writeable:
            link.flags.writeable = False
            if journal is not None:
                journal.arrays.append(link)
        if not isinstance(link.base, np.ndarray):
            break
        link = link.base
    frozen = heap_blocks.sealed(array, link)
    if frozen is not None:
        _remember_frozen(_root(frozen))
        return frozen
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


def array_freq(values: Any) -> Any:
    """The ``freq`` the DatetimeArray or TimedeltaArray *values* keeps;
    None where the array type keeps none.

    Up to pandas 3.0 the array keeps it and its index reads it from there;
    pandas 3.1 removed ``freq`` from the arrays (GH 24566), and keeps it
    on the DatetimeIndex or TimedeltaIndex alone (`_same_index`)."""
    return values.freq if _arrays_keep_freq(type(values)) else None


def _arrays_keep_freq(kind: type) -> bool:
    return isinstance(getattr(kind, "freq", None), property)


def _same_dates(values: Any, ndarray: Any) -> Any:
    """A DatetimeArray or TimedeltaArray like *values* over *ndarray*, its
    ``freq`` kept where the array keeps one (`array_freq`):
    ``_from_backing_data`` drops it, and a table with a ``date_range``
    index would come back from a hit, or from disk, without it
    (``df.shift(1, freq=df.index.freq)`` then shifts the data)."""
    if _arrays_keep_freq(type(values)):
        return type(values)._simple_new(ndarray, dtype=values.dtype, freq=values.freq)
    return type(values)._simple_new(ndarray, dtype=values.dtype)


def _same_index(index: Any, data: Any) -> Any:
    """An index of *index*'s type and name over *data*, its ``freq`` kept.

    Where the arrays keep ``freq`` (pandas up to 3.0), *data* carries it
    (`_same_dates`). pandas 3.1 keeps it on the index alone, in ``_freq``,
    which its own ``copy`` and ``_view`` carry over the same way; a new
    index made from the data has none. Set without the check the public
    setter makes (a pass over every label): the labels are *index*'s own.
    Raises when the freq still differs, so the table is not frozen."""
    fresh = type(index)._simple_new(data, name=index.name)
    freq = getattr(index, "freq", None)
    if freq is not None and fresh.freq != freq and hasattr(type(fresh), "_freq"):
        fresh._freq = freq
    if getattr(fresh, "freq", None) != freq:
        raise ValueError("an index's freq would be lost")
    return fresh


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
    if _python_text(data):
        copied = data.copy()
        return _same_index(index, copied._from_backing_data(_read_only(copied._ndarray)))
    if not _ndarray_parts(data):
        return index  # Arrow labels: immutable memory (`_private_axes`)
    return _same_index(index, _freeze_values(data.copy()))


def frozen_already(frame: Any) -> bool:
    """Is all of *frame*'s data frozen by cash already (a hit)? Then it can be kept as it is, by a shallow
    copy (`hand_out`): no holder can write that data, and storing it changes
    nothing any holder may do."""
    try:
        for blk in frame._mgr.blocks:
            if _pin_of(blk.refs) is None:
                return False
            parts = _ndarray_parts(blk.values)
            if parts is None or not all(not array.flags.writeable and is_frozen(array) for _name, array in parts):
                return False
        axes = (frame.index,) if frame.ndim == 1 else (frame.index, frame.columns)
        for axis in axes:
            parts = _label_parts(getattr(axis, "_data", None)) if not _is_range(axis) else []
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
    own: a copy no caller holds. False, with
    nothing done, when *frame* is not `freezable`.
    """
    return enabled() and _freeze(frame, cells_known=cells_known)


def _freeze(frame: Any, *, cells_known: bool = False) -> bool:
    journal = _Journal()
    try:
        if not freezable(frame, cells_known=cells_known):
            return False
        _install_pickling()
        _install_write_watch()
        _JOURNAL.current = journal
        _freeze_in_place(frame)
        return True
    except Exception:
        logger.debug("cash: could not freeze a %s", type(frame).__name__, exc_info=True)
        try:
            _undo(frame, journal)
        except Exception:
            logger.debug("cash: could not undo a part freeze", exc_info=True)
        return False
    finally:
        _JOURNAL.current = None


def _freeze_in_place(frame: Any) -> None:
    """`_freeze`'s changes, each recorded in the thread's `_Journal`."""
    journal = _JOURNAL.current
    # Not consolidated: the blocks stay as the caller's table had them.
    # One block for every float column made ``data[cols].to_numpy()`` a
    # read-only view of it, where the caller's own table (one block per
    # column, from ``read_csv``) gives a writable copy.
    for blk in frame._mgr.blocks:
        refs = blk.refs
        sharing = [ref() for ref in refs.referenced_blocks]
        group = [other for other in [blk, *sharing] if other is not None and not is_pin(other)]
        for other in group:
            values = other.values
            frozen = _freeze_values(values)
            if frozen is not values:
                journal.values.append((other, values))
                other.values = frozen
        pin = _pin_of(refs)
        if pin is None:
            pin = _PinRef([_root(array) for _name, array in _ndarray_parts(blk.values) or ()])
            journal.pins.append((refs.referenced_blocks, pin))
            refs.referenced_blocks.append(pin)
            for root in pin.roots:
                # By id: pins are weak references to one target, so all compare equal.
                _PINS_BY_ROOT.setdefault(id(root()), {})[id(pin)] = weakref.ref(pin)
        for other in group:
            pin.know(other)
    for name in ("index",) if frame.ndim == 1 else ("index", "columns"):
        axis = getattr(frame, name)
        frozen = _frozen_index(axis)
        if frozen is not axis:
            journal.axes.append((name, axis))
            setattr(frame, name, frozen)
    _private_axes(frame)


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
    if _total_bytes(frame) < CHECKED_MIN_BYTES:
        # Too small to keep uncopied whatever it holds: no need to know yet
        # whether the checks work here. Finding out builds a 16 MB table
        # (`_checked_self_test`), which a process storing only small tables
        # paid at its first store.
        return None
    if not _checked_works():
        return None
    try:
        return _share_checked(frame)
    except Exception:  # unknown internals: copied as before
        logger.debug("cash: could not share a %s", type(frame).__name__, exc_info=True)
        return None


def keeps_uncopied(frame: Any) -> bool:
    """Would the RAM tier keep *frame* without copying its data, frozen
    already (`frozen_already`) or shared and checked (`share_checked`)?
    Looked at from its blocks alone: nothing is copied or hashed."""
    try:
        if not enabled():
            return False
        if frozen_already(frame):
            return True
        import pandas as pd

        if type(frame) is not pd.DataFrame or _total_bytes(frame) < CHECKED_MIN_BYTES:
            return False  # before `_checked_works`, which builds a 16 MB table
        if not _checked_works() or not freezable(frame, cells_known=True):
            return False
        nbytes = 0
        for block in frame._mgr.blocks:
            parts = _ndarray_parts(block.values)
            if parts is None:
                return False
            for _name, array in parts:
                if array.dtype.hasobject or _flat_bytes(array) is None:
                    return False
                nbytes += array.nbytes
        return nbytes >= CHECKED_MIN_BYTES
    except Exception:  # noqa: BLE001 - unknown internals: copied
        return False


def _total_bytes(frame: Any) -> int:
    """All of *frame*'s block data in bytes, numbers or not; at least what
    `_share_checked` counts. A frame of unknown shape counts as big, so it
    goes through the checks as before."""
    try:
        return sum(block.values.nbytes for block in frame._mgr.blocks)
    except Exception:  # noqa: BLE001 - not a pandas frame as known here
        return CHECKED_MIN_BYTES


def _share_checked(frame: Any) -> tuple[Any, bytes] | None:
    import pandas as pd

    if type(frame) is not pd.DataFrame or not freezable(frame, cells_known=True):
        return None
    keys = _known_contents(frame)
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
        if _pin_of(block.refs) is None:
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
    digest = checked_digest(shared, keys)
    if digest is None:
        return None
    return shared, digest


def _known_contents(frame: Any) -> dict[tuple, tuple]:
    """The content key of each block of *frame* the frame memo vouches for
    (`shared_blocks.content_keys`), asked before `_share_checked`'s shallow
    copies reference its arrays: a loop storing its table again with the
    same columns does not read them again (`checked_digest`).

    A table the memo knows some columns of but not all -- a loop's
    ``h = matches.assign(p=...)``, ``matches`` hashed before -- is read into
    the memo first (`read_into_memo`), its known columns not read: its new
    columns are then read once, where the checked digest, the disk entry's
    chunks and the next call taking ``h`` read them each."""
    from ..content_hashers import read_into_memo
    from .shared_blocks import content_keys

    keys = content_keys(frame)
    if not keys:
        return keys
    numpy_blocks = sum(1 for block in frame._mgr.blocks if _ndarray_parts(block.values))
    if len(keys) < numpy_blocks and read_into_memo(frame):
        keys = content_keys(frame)
    return keys


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


def checked_digest(frame: Any, keys: dict[tuple, tuple] | None = None) -> bytes | None:
    """The digest of the data a table `share_checked` made keeps in the
    caller's arrays (its Arrow data and labels are its own); None when an
    array is no longer in one piece.

    *keys*: the content key of each block, by its columns' positions
    (`shared_blocks.content_keys`), asked of the frame memo just now; a
    block whose tree digest they give (`block_digests.known`) is not read,
    and one read is remembered under its key. Without them, every block is
    read: what checks the data before it is served."""
    import hashlib

    from cash import bulk_digest

    from . import block_digests
    from .shared_blocks import block_layout

    digest = hashlib.sha256(_CHECKED_TAG)
    for block in frame._mgr.blocks:
        if block not in _CHECKED_BLOCKS:
            continue
        key = keys.get(tuple(int(pos) for pos in block.mgr_locs.as_array)) if keys else None
        if key is not None and block_layout(block.values) != key[0]:
            key = None
        for _name, array in _ndarray_parts(block.values) or ():
            flat = _flat_bytes(array)
            if flat is None:
                return None
            if key is None or flat.nbytes <= bulk_digest.DIRECT_MAX_BYTES:
                bulk_digest.fold_buffer(digest, flat)
                continue
            known = block_digests.known(key, flat.nbytes)
            if known is None:
                known = bulk_digest.bulk_digest(flat)
                block_digests.remember(key, known)
            bulk_digest.fold_digest(digest, flat.nbytes, known)
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
        fresh = heap_blocks.fresh_copy(values)  # a store's big block: out of glibc's heap
        if fresh is not None:
            return fresh
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
    todo = [i for i, chunk in enumerate(chunks) if not _chunk_owned(chunk)]
    if not todo:
        return chunked
    owned = list(chunks)
    copied = _copied_together(chunked.type, [chunks[i] for i in todo]) if len(todo) > 1 else None
    for i, copy in zip(todo, copied or [_owned_chunk(chunks[i]) for i in todo]):
        owned[i] = copy
    return pa.chunked_array(owned, type=chunked.type)


def _copied_together(arrow_type: Any, chunks: list[Any]) -> list[Any] | None:
    """Copies of the Arrow *chunks* of one column, as `_owned_chunk` copies
    each, made by the C extension for all of them in one call into buffers
    Arrow allocated, one per kind of buffer, each chunk over its slice of
    them; None when it cannot (no C extension, another type).

    p08's 30 parsed log files come in 384 chunks each: copied one by one,
    a text chunk cast to ``large_string`` and back, that was 3.3 s of
    which a third was the calls themselves."""
    import pyarrow as pa

    from .._native import kernels

    if kernels is None or not hasattr(kernels, "copy_view_chunks"):
        return None
    t = pa.types
    if t.is_string_view(arrow_type) or t.is_binary_view(arrow_type):
        return _views_copied_together(arrow_type, chunks, kernels)
    if t.is_boolean(arrow_type) or t.is_dictionary(arrow_type) or isinstance(arrow_type, pa.ExtensionType):
        return None
    if (
        not (
            t.is_integer(arrow_type)
            or t.is_floating(arrow_type)
            or t.is_timestamp(arrow_type)
            or t.is_date(arrow_type)
            or t.is_time(arrow_type)
            or t.is_duration(arrow_type)
        )
        or arrow_type.bit_width % 8
    ):
        return None
    width = arrow_type.bit_width // 8
    pieces, rows, nulls = [], 0, 0
    for chunk in chunks:
        n, buffers = len(chunk), chunk.buffers()
        validity = buffers[0] if chunk.null_count else None
        pieces.append(
            (
                buffers[1],
                width,
                validity,
                chunk.offset,
                chunk.offset + n,
                rows * width,
                None if validity is None else nulls,
            )
        )
        rows += n
        nulls += (n + 7) // 8 if validity is not None else 0
    values_out, validity_out = pa.allocate_buffer(rows * width), pa.allocate_buffer(nulls) if nulls else None
    kernels.copy_fixed_chunks(pieces, values_out, validity_out)
    out = []
    for chunk, piece in zip(chunks, pieces):
        n = len(chunk)
        validity = None if piece[6] is None else validity_out.slice(piece[6], (n + 7) // 8)
        values = values_out.slice(piece[5], n * width)
        out.append(pa.Array.from_buffers(arrow_type, n, [validity, values], null_count=chunk.null_count))
    return out


def _views_copied_together(arrow_type: Any, chunks: list[Any], kernels: Any) -> list[Any] | None:
    """`_copied_together` for a string or binary view column: each chunk's
    views and the bytes its own rows use (``copy_view_chunks``)."""
    import pyarrow as pa

    pieces = []
    for chunk in chunks:
        buffers = chunk.buffers()
        validity = buffers[0] if chunk.null_count else None
        pieces.append((buffers[1], tuple(buffers[2:]), validity, chunk.offset, chunk.offset + len(chunk)))
    sizes = kernels.view_data_sizes(pieces)
    if any(size > 2**31 - 1 for size in sizes):
        return None  # a view's offset is 32 bits
    placed, rows, data, nulls = [], 0, 0, 0
    for piece, size in zip(pieces, sizes):
        n = piece[4] - piece[3]
        placed.append((*piece, rows * 16, data, None if piece[2] is None else nulls))
        rows += n
        data += size
        nulls += (n + 7) // 8 if piece[2] is not None else 0
    views_out, data_out = pa.allocate_buffer(rows * 16), pa.allocate_buffer(data)
    validity_out = pa.allocate_buffer(nulls) if nulls else None
    kernels.copy_view_chunks(placed, views_out, data_out, validity_out)
    out = []
    for chunk, piece, size in zip(chunks, placed, sizes):
        n = len(chunk)
        validity = None if piece[7] is None else validity_out.slice(piece[7], (n + 7) // 8)
        buffers = [validity, views_out.slice(piece[5], n * 16)]
        if size:
            buffers.append(data_out.slice(piece[6], size))
        out.append(pa.Array.from_buffers(arrow_type, n, buffers, null_count=chunk.null_count))
    return out


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
    elif pa.types.is_string_view(chunk.type) or pa.types.is_binary_view(chunk.type):
        buffers = chunk.buffers()
        if not any(b is not None and b.size for b in buffers[2:]):
            # Every value is inline in its view (12 bytes or fewer): the
            # validity bitmap and the views are all it holds, copied as
            # they are. Writing the rows out and back was 3.3 s of p21's
            # 10M-row events frame, whose text columns are all short.
            return pa.Array.from_buffers(
                chunk.type,
                len(chunk),
                [None if b is None else _copied_buffer(b) for b in buffers[:2]],
                null_count=chunk.null_count,
                offset=chunk.offset,
            )
        # Its rows written out and back: a polars text chunk lists every data
        # buffer of its column, and ``concat_arrays`` keeps them while the IPC
        # stream writes them all, so a 23-chunk frame of 5,760 rows sliced
        # from a big one took 7.6 s to copy, each chunk ~1 GB of IPC. The
        # cast may keep the validity bitmap: ``concat_arrays`` writes it anew.
        large = pa.large_string() if pa.types.is_string_view(chunk.type) else pa.large_binary()
        try:
            owned = pa.concat_arrays([chunk.cast(large)]).cast(chunk.type)
        except (pa.ArrowNotImplementedError, pa.ArrowInvalid):
            owned = None
        if owned is not None and not _overlapping(owned.buffers(), chunk.buffers()):
            return owned
    elif not _holds_dictionary(chunk.type):
        # ``concat_arrays`` writes every buffer anew, but a type it may
        # append as it is (views over data buffers) is checked, not trusted.
        owned = pa.concat_arrays([chunk])
        if not _overlapping(owned.buffers(), chunk.buffers()):
            return owned
    return _ipc_copy(chunk)


def _copied_buffer(buffer: Any) -> Any:
    """A copy of the Arrow *buffer* in memory Arrow's pool allocated."""
    import pyarrow as pa

    copy = pa.allocate_buffer(buffer.size)
    memoryview(copy)[:] = memoryview(buffer)
    return copy


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
    for block in copied._mgr.blocks:
        pin = _pin_of(block.refs)
        if pin is not None:
            pin.know(block)
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


# ------------------------------------------------- readers of a written table


def reads_memory_of(value: Any, roots: list[Any]) -> bool:
    """Is *value* a numpy array, or an array wrapping one (``s.array``),
    over the memory one of *roots* owns (`views_left_behind`)?"""
    import numpy as np

    array = value if isinstance(value, np.ndarray) else getattr(value, "_ndarray", None)
    if not isinstance(array, np.ndarray):
        return False
    root = _root(array)
    return any(root is other for other in roots)


def read_only_note(exc: BaseException) -> str | None:
    """A note for *exc* when it is numpy's "read-only" error from a write
    into a table served from the RAM tier (``s.array[0] = ...``); None for
    any other error.

    A plain run writes such a handle in place. A served table's memory is
    the tier's, shared with every other hit, and read-only (`freeze`): the
    write cannot be let through without changing them."""
    import numpy as np

    if not isinstance(exc, ValueError) or "read-only" not in str(exc):
        return None
    tb = exc.__traceback__
    while tb is not None:
        for value in list(tb.tb_frame.f_locals.values()):
            array = value if isinstance(value, np.ndarray) else getattr(value, "_ndarray", None)
            if isinstance(array, np.ndarray) and is_frozen(array):
                return (
                    "cash: this column belongs to a table served from the cache. Its memory is shared "
                    "with the cache's copy and is read-only, so a write past pandas (such as "
                    "s.array[0] = ...) cannot change it; a plain run writes it. Write through pandas "
                    "(s.iloc[0] = ...), or copy the table first (df = df.copy())."
                )
        tb = tb.tb_next
    return None


def _read_elsewhere(pin: _PinRef) -> list[Any]:
    """`views_left_behind` for one group: weak references to the roots read
    elsewhere. The roots are held weakly here, so that the references
    counted are the holders' alone."""
    found = []
    for ref in pin.roots:
        root = ref()
        if root is None:
            continue
        known = [ref() for ref in list(_PINS_BY_ROOT.get(id(root), {}).values())]
        pins = [other for other in known if other is not None] or [pin]
        del root
        blocks: dict[int, Any] = {}
        countable = True
        for other in pins:
            group = _known_group(other)
            if group is None:
                countable = False
                break
            for block in group:
                blocks[id(block)] = block
        if not countable or _unaccounted(list(blocks.values()), ref):
            found.append(ref)
    return found


def _known_group(pin: _PinRef) -> list[Any] | None:
    """The live blocks of *pin*'s group, through a block cash knows in it;
    None when none of those is alive."""
    for ref in pin.blocks:
        block = ref()
        if block is not None:
            return _group(block)
    return None if pin.blocks else []


def _unaccounted(blocks: list[Any], root_ref: Any) -> bool:
    """Is the root *root_ref* refers to, or an array or wrapper the
    *blocks* keep over it, held by anything the blocks do not account for?"""
    objects, holders = _block_holders(blocks)
    del blocks
    root = root_ref()
    if root is None:
        return False
    objects[id(root)] = root
    # The views a disk write serializes from are cash's own (`for_storing`).
    holders[id(root)] = holders.get(id(root), 0) + _STORE_HELD.get(id(root), 0)
    del root
    held_here = _held_by_pandas_and_cash(objects)
    for key in list(objects):
        # Besides those counted: `objects`, the call's argument, and the
        # locals of the pandas and cash code writing (`values = self.values`).
        if sys.getrefcount(objects[key]) > holders[key] + held_here.get(key, 0) + 2:
            return True
    return False


def _credit_locals(
    frame: Any, objects: dict[int, Any], counted: dict[int, int], views: dict[int, list[Any]] | None
) -> None:
    """Credit the references *frame*'s locals hold to *objects*: one per
    slot holding one of them. With *views*, also note each slot holding a
    numpy view of one of them (`_credit_views`)."""
    import numpy as np

    for value in list(frame.f_locals.values()):
        if objects.get(id(value)) is value:
            counted[id(value)] = counted.get(id(value), 0) + 1
        elif (
            views is not None and isinstance(value, np.ndarray) and value.base is not None and id(value.base) in objects
        ):
            noted = views.setdefault(id(value), [value, 0])
            noted[1] += 1


def _credit_views(views: dict[int, list[Any]], counted: dict[int, int]) -> None:
    """Credit the base of each view the stack's locals hold, once per view
    (one view holds its base once, however many slots hold the view), and
    only when those slots are all that hold the view: a view the caller
    also holds, such as the array a write is given (``df.iloc[:, 0] = v``),
    is the caller's reader, not the write's."""
    for key in list(views):
        view, slots = views.pop(key)
        # Besides the slots: `view`, getrefcount's argument.
        if sys.getrefcount(view) - 2 <= slots:
            counted[id(view.base)] = counted.get(id(view.base), 0) + 1
        del view


def _held_by_pandas_and_cash(objects: dict[int, Any]) -> dict[int, int]:
    """How many references to each of *objects* the locals of the pandas
    and cash frames on this thread's stack hold, directly or through a
    numpy view whose base is one of them: the write under way, not a
    reader the caller keeps."""
    import os

    import pandas as pd

    inside = (os.path.dirname(pd.__file__) + os.sep, os.path.dirname(os.path.dirname(__file__)) + os.sep)
    counted: dict[int, int] = {}
    views: dict[int, list[Any]] = {}
    frame = sys._getframe(2)  # past `_unaccounted`, which holds `objects` alone
    while frame is not None:
        if frame.f_code.co_filename.startswith(inside):
            _credit_locals(frame, objects, counted, views)
        frame = frame.f_back
    _credit_views(views, counted)
    if _DETACHING[0]:
        here = threading.get_ident()
        for thread, frame in list(sys._current_frames().items()):
            while frame is not None and thread != here:
                if frame.f_code.co_filename == __file__:
                    # Direct references only: a fresh view's base is
                    # credited once registered (`_fresh_view`), and before
                    # that it counts, so the count errs towards a warning.
                    _credit_locals(frame, objects, counted, None)
                frame = frame.f_back
    return counted


def _block_holders(blocks: list[Any]) -> tuple[dict[int, Any], dict[int, int]]:
    """Each object *blocks* keep their data in (``id -> object``: the
    values, a wrapper's array, the array owning the memory), and how many
    references to it they account for. A values object several blocks
    share (a shallow copy's) is counted once per block, the arrays below it
    once."""
    objects: dict[int, Any] = {}
    holders: dict[int, int] = {}

    def held(obj: Any) -> None:
        objects[id(obj)] = obj
        holders[id(obj)] = holders.get(id(obj), 0) + 1

    values_seen: set[int] = set()
    arrays_seen: set[int] = set()
    for block in blocks:
        values = block.values
        held(values)
        if id(values) in values_seen:
            continue
        values_seen.add(id(values))
        for name, array in _ndarray_parts(values) or ():
            if name:
                held(array)
            if id(array) in arrays_seen:
                continue
            arrays_seen.add(id(array))
            if array.base is not None:
                held(array.base)
    return objects, holders


# ---------------------------------------------------------------- pickling


def _writable_alias(array: Any) -> Any:
    """A writable array over a frozen *array*'s memory, for pickle to read
    alone: pickle writes a read-only array as read-only, and a table loaded
    from it could then not be written by pandas (it would hold no `_PinRef`).
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


#: The pandas methods `_install_write_watch` replaced: ``(owner, name, original)``.
_WATCH_INSTALLED: list[tuple[Any, str, Any]] = []
_WATCH_DONE: list[bool] = []


def _install_write_watch() -> None:
    """Note each frozen group pandas finds shared before a write
    (`_note_shared`): the places pandas' copy-on-write asks whether a block
    it is about to write in place is shared. Once per process, before the
    first table is frozen; `_watch_self_test` checks it."""
    if _WATCH_DONE:
        return
    with _INSTALL_LOCK:
        if _WATCH_DONE:
            return
        from pandas.core.internals.blocks import Block
        from pandas.core.internals.managers import BaseBlockManager, SingleBlockManager

        no_reference_block = BaseBlockManager._has_no_reference_block
        single_no_reference = SingleBlockManager._has_no_reference
        maybe_copy = Block._maybe_copy
        refs_and_copy = Block._get_refs_and_copy

        def _has_no_reference_block(self: Any, blkno: int) -> bool:
            alone = no_reference_block(self, blkno)
            if not alone:
                _note_shared(self.blocks[blkno].refs)
            return alone

        def _has_no_reference(self: Any, i: int = 0) -> bool:
            alone = single_no_reference(self, i)
            if not alone:
                _note_shared(self.blocks[0].refs)
            return alone

        def _maybe_copy(self: Any, inplace: bool, deep: bool = True) -> Any:
            if inplace and deep:  # a shallow copy keeps the frozen memory: no write follows
                _note_shared(self.refs)
            return maybe_copy(self, inplace, deep)

        def _get_refs_and_copy(self: Any, inplace: bool) -> Any:
            if inplace:
                _note_shared(self.refs)
            return refs_and_copy(self, inplace)

        for owner, name, original, wrapper in (
            (BaseBlockManager, "_has_no_reference_block", no_reference_block, _has_no_reference_block),
            (SingleBlockManager, "_has_no_reference", single_no_reference, _has_no_reference),
            (Block, "_maybe_copy", maybe_copy, _maybe_copy),
            (Block, "_get_refs_and_copy", refs_and_copy, _get_refs_and_copy),
        ):
            _WATCH_INSTALLED.append((owner, name, original))
            setattr(owner, name, wrapper)
        _WATCH_DONE.append(True)


def _uninstall_write_watch() -> None:
    """Put back the pandas methods `_install_write_watch` replaced."""
    with _INSTALL_LOCK:
        while _WATCH_INSTALLED:
            owner, name, original = _WATCH_INSTALLED.pop()
            setattr(owner, name, original)
