"""What a pool task returned is released with the pool, under a tracker too.

To bring a worker's reads back, a pool's result object is handed a ``_set``
of cash's own while a tracker is active (``multiprocessing.pool``, and the
future of a process pool's ``submit``). That wrapper closed over the result
object's own bound method, so the two made a reference cycle holding the
task's values until the garbage collector next ran.

The notebook tells an object nothing else holds by its reference count, so a
model fitted through joblib's threading backend (scikit-learn's histogram
binning, inside every ``HistGradientBoostingRegressor.fit``) looked held by
something else: ``m = HistGradientBoostingRegressor().fit(X, y)`` was refused
as shared and ran on every run. The collector is switched off here, so only a
plain reference keeps a value alive.
"""

from __future__ import annotations

import gc
import weakref
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.pool import ThreadPool

import pytest

from cash.tracking.file_tracker import FileAccessTracker


class Box:
    """A task's value, weakly referenceable (a plain list is not)."""


def make_box(_i):
    return Box()


@pytest.fixture
def no_collector():
    was = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        if was:
            gc.enable()


@pytest.mark.parametrize("how", ["map", "apply_async"])
def test_a_thread_pool_value_dies_with_its_last_reference(no_collector, how):
    with FileAccessTracker(), ThreadPool(2) as pool:
        if how == "map":
            boxes = pool.map(make_box, range(4))
        else:
            boxes = [pool.apply_async(make_box, (i,)).get() for i in range(4)]
    refs = [weakref.ref(box) for box in boxes]
    assert all(ref() is not None for ref in refs)  # positive control

    del boxes

    assert [ref() for ref in refs] == [None] * 4, "a reference cycle of the tracking keeps the task's values alive"


def test_a_process_pool_value_dies_with_its_last_reference(no_collector):
    with FileAccessTracker(), ProcessPoolExecutor(1) as ex:
        future = ex.submit(make_box, 0)
        box = future.result()
    ref = weakref.ref(box)

    del future, box

    assert ref() is None, "a reference cycle of the tracking keeps the task's value alive"
