"""What writing a big value of many Python objects to disk costs, measured on a sample.

The fitted cost model prices a value by its bytes, which suits a frame of
numbers or an array: one buffer, copied. A value of millions of small
Python objects -- a dict of parsed sessions, a column of ``datetime.date`` --
is pickled one object at a time, and costs ten times what its bytes say:
a dict of 87,000 users' sessions, 310 MB, took 3.7 s to pickle and 1.25 s to
load against a 3.0 s statement. The disk write runs in the background, but
pickling holds the interpreter lock, so the notebook's next cell waits for it.

`sampled_cost` pickles and loads an evenly spread sample of such a value
(every k-th item of a big list, dict, or object column) and scales by the
item count: about 1% of the work, which the write costs anyway.

A list of many small pandas tables is priced the same way, by a sample of
its tables (`many_frames_cost`): each costs a fixed fraction of a
millisecond to copy into the RAM tier and to pickle, whatever its bytes,
so 6,000 three-row frames built in 1.2 s took 1.1 s to copy and 1.15 s to
pickle, against a byte size the model reads as free.
"""

from __future__ import annotations

import gc
import pickle
import time
from collections.abc import Callable
from typing import Any, NamedTuple

from .. import kept_state
from .._user_attrs import from_family

__all__ = ["SampledCost", "many_frames_cost", "sampled_cost"]

#: Containers with fewer items than this are not sampled: small enough to be
#: priced by their bytes, or looked into when they are a payload's dicts.
_MIN_ITEMS = 20_000
#: Items in each sample.
_SAMPLE_ITEMS = 2_000
#: How deep the payload's small dicts are looked into for big values.
_MAX_DEPTH = 3
_SMALL_DICT = 64
#: A list, tuple or dict of this many pandas tables or more is priced by a
#: sample of them (`_frames_cost`), when they are small (`_SMALL_FRAME_CELLS`).
_MIN_FRAMES = 500
#: Tables in each sample.
_SAMPLE_FRAMES = 64
#: Cells a sampled table may hold for the list to be priced by its tables'
#: count: a list of big tables is priced by its bytes, and sampling it
#: would cost a share of the write.
_SMALL_FRAME_CELLS = 10_000


class SampledCost(NamedTuple):
    """Predicted seconds to pickle (``dump``) and to load (``load``) a value;
    for `many_frames_cost`, also to copy it into the RAM tier (``copy``),
    and the tables it holds (``items``)."""

    dump: float
    load: float
    copy: float = 0.0
    items: int = 0


def sampled_cost(value: Any) -> SampledCost | None:
    """The predicted cost of pickling and loading *value*'s big object-heavy
    parts, or None when it holds none (the byte model prices it then)."""
    # The collector paused: listing a big dict's items and loading a sample
    # make thousands of containers, and each collection they set off walks
    # the notebook's whole heap -- 0.5 s of a 0.07 s measurement next to two
    # million parsed objects.
    return _paused_gc(value, frames_only=False)


def many_frames_cost(value: Any) -> SampledCost | None:
    """The predicted cost of storing *value*'s lists of many small pandas
    tables (`_frames_cost`): pickling, loading and copying them into the RAM
    tier, measured on a sample. None when it holds no such list."""
    return _paused_gc(value, frames_only=True)


def _paused_gc(value: Any, *, frames_only: bool) -> SampledCost | None:
    enabled = gc.isenabled()
    gc.disable()
    try:
        total = _cost(value, 0, frames_only)
    finally:
        if enabled:
            gc.enable()
    if total is None or total.dump <= 0:
        return None
    return total


def _cost(value: Any, depth: int, frames_only: bool = False) -> SampledCost | None:
    kind = type(value)
    if kind is dict and len(value) <= _SMALL_DICT and depth < _MAX_DEPTH:
        found = [part for item in value.values() if (part := _cost(item, depth + 1, frames_only)) is not None]
        if not found:
            return None
        return SampledCost(
            sum(part.dump for part in found),
            sum(part.load for part in found),
            sum(part.copy for part in found),
            sum(part.items for part in found),
        )
    if kind in (list, tuple, dict) and len(value) >= _MIN_FRAMES:
        frames = _frames_cost(list(value.values()) if kind is dict else value, copy=frames_only)
        if frames is not None or frames_only:
            return frames
    elif frames_only:
        return None
    if kind in (list, tuple, dict) and len(value) >= _MIN_ITEMS:
        items = list(value.items()) if kind is dict else value
        step = max(1, len(items) // _SAMPLE_ITEMS)
        return _measure(items[::step], len(items) / len(items[::step]))
    return _frame_cost(value)


def _frame_cost(value: Any) -> SampledCost | None:
    """A pandas frame's or series' object columns, by a sample of rows."""
    if not from_family(type(value), "pandas") or not hasattr(value, "dtypes") and not hasattr(value, "dtype"):
        return None
    try:
        rows = len(value)
        if rows < _MIN_ITEMS:
            return None
        if getattr(value, "ndim", 2) == 1:
            if str(value.dtype) != "object":
                return None
            objects = value
        else:
            positions = [i for i, dtype in enumerate(value.dtypes) if str(dtype) == "object"]
            if not positions:
                return None
            objects = value.iloc[:, positions]
        step = max(1, rows // _SAMPLE_ITEMS)
        sample = objects.iloc[::step]
        return _measure(sample, rows / len(sample))
    except Exception:  # noqa: BLE001 - an estimate it cannot make: the byte model's
        return None


def _small_frame(item: Any) -> bool:
    kind = type(item)
    if kind.__name__ not in ("DataFrame", "Series") or not from_family(kind, "pandas"):
        return False
    try:
        return int(item.size) <= _SMALL_FRAME_CELLS
    except Exception:  # noqa: BLE001 - not one to price by its count
        return False


def _frames_cost(items: list | tuple, *, copy: bool) -> SampledCost | None:
    """What storing *items*, a list of many small pandas tables, costs, by an
    evenly spread sample of them (None unless each sampled item is a small table):
    a table's fixed costs, not its bytes, decide it. *copy*: the copy into
    the RAM tier is timed too, as that tier's store makes it."""
    step = max(1, len(items) // _SAMPLE_FRAMES)
    sample = list(items[::step])
    if not all(_small_frame(item) for item in sample):
        return None
    scale = len(items) / len(sample)
    # Pickled as the disk tier writes them: an Arrow column by its own
    # reducer (`stored_reducers`), which is most of a small table's pickle.
    from .serialization import PickleSerializer

    serializer = PickleSerializer()
    measured = _measure(sample, scale, serializer.serialize, serializer.deserialize)
    if measured is None:
        return None
    copied = 0.0
    if copy:
        from .memory_backend import InMemoryBackend

        try:
            start = time.perf_counter()
            for item in sample:
                # As a store copies it (``record_cells`` given); thrown away.
                InMemoryBackend._copy_frame(item, None, {})
            copied = (time.perf_counter() - start) * scale
        except Exception:  # noqa: BLE001 - not priced: the store finds out
            return None
    return measured._replace(copy=copied, items=len(items))


def _measure(
    sample: Any,
    scale: float,
    dumps: Callable[[Any], bytes] | None = None,
    loads: Callable[[bytes], Any] = pickle.loads,
) -> SampledCost | None:
    try:
        start = time.perf_counter()
        data = dumps(sample) if dumps is not None else kept_state.dumps(sample, protocol=5)
        dumped = time.perf_counter()
        loads(data)
        loaded = time.perf_counter()
    except Exception:  # noqa: BLE001 - what pickle refuses is priced by its bytes
        return None
    return SampledCost((dumped - start) * scale, (loaded - dumped) * scale)
