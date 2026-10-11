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
"""

from __future__ import annotations

import gc
import pickle
import time
from typing import Any, NamedTuple

from .. import kept_state

__all__ = ["SampledCost", "sampled_cost"]

#: Containers with fewer items than this are not sampled: small enough to be
#: priced by their bytes, or looked into when they are a payload's dicts.
_MIN_ITEMS = 20_000
#: Items in each sample.
_SAMPLE_ITEMS = 2_000
#: How deep the payload's small dicts are looked into for big values.
_MAX_DEPTH = 3
_SMALL_DICT = 64


class SampledCost(NamedTuple):
    """Predicted seconds to pickle (``dump``) and to load (``load``) a value."""

    dump: float
    load: float


def sampled_cost(value: Any) -> SampledCost | None:
    """The predicted cost of pickling and loading *value*'s big object-heavy
    parts, or None when it holds none (the byte model prices it then)."""
    # The collector paused: listing a big dict's items and loading a sample
    # make thousands of containers, and each collection they set off walks
    # the notebook's whole heap -- 0.5 s of a 0.07 s measurement next to two
    # million parsed objects.
    enabled = gc.isenabled()
    gc.disable()
    try:
        total = _cost(value, 0)
    finally:
        if enabled:
            gc.enable()
    if total is None or total.dump <= 0:
        return None
    return total


def _cost(value: Any, depth: int) -> SampledCost | None:
    kind = type(value)
    if kind is dict and len(value) <= _SMALL_DICT and depth < _MAX_DEPTH:
        found = [part for item in value.values() if (part := _cost(item, depth + 1)) is not None]
        if not found:
            return None
        return SampledCost(sum(part.dump for part in found), sum(part.load for part in found))
    if kind in (list, tuple, dict) and len(value) >= _MIN_ITEMS:
        items = list(value.items()) if kind is dict else value
        step = max(1, len(items) // _SAMPLE_ITEMS)
        return _measure(items[::step], len(items) / len(items[::step]))
    return _frame_cost(value)


def _frame_cost(value: Any) -> SampledCost | None:
    """A pandas frame's or series' object columns, by a sample of rows."""
    module = type(value).__module__ or ""
    if not module.startswith("pandas") or not hasattr(value, "dtypes") and not hasattr(value, "dtype"):
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


def _measure(sample: Any, scale: float) -> SampledCost | None:
    try:
        start = time.perf_counter()
        data = kept_state.dumps(sample, protocol=5)
        dumped = time.perf_counter()
        pickle.loads(data)
        loaded = time.perf_counter()
    except Exception:  # noqa: BLE001 - what pickle refuses is priced by its bytes
        return None
    return SampledCost((dumped - start) * scale, (loaded - dumped) * scale)
