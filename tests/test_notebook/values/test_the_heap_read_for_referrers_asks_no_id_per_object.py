"""The heap read for referrers raises no audit event per object it reads.

Python raises an ``builtins.id`` audit event for every ``id()``, and every
audit hook written in Python runs for it. Read in Python (no C kernels:
Python 3.10 and 3.11), `_referrers_of` took the id of each referent of each
object on the heap: with cash's own hook and the four the test suite adds,
one cell of ``test_a_module_function_held_in_a_value_is_keyed`` ran past
pytest-timeout's 30 s in a Windows 3.10 xdist worker late in its run, and
the worker was ended ("node down: Not properly terminated"). The read now
names each referent by its address as ``object``'s hash gives it: the same
referrers, no event per object.
"""

from __future__ import annotations

import gc
import sys

import pytest

import cash.notebook.shared_objects as shared_objects

#: ``[count]`` while a read is being counted, else empty.
COUNTING: list[int] = []


def _hook(event, args):
    if event == "builtins.id" and COUNTING:
        COUNTING[0] += 1


sys.addaudithook(_hook)


@pytest.fixture(params=["python", "kernel"])
def read(request, monkeypatch):
    if request.param == "python":
        monkeypatch.setattr(shared_objects, "_REFERRERS_IN", None)
    elif shared_objects._REFERRERS_IN is None:
        pytest.skip("no C extension")
    return shared_objects._referrers_of


def test_reading_the_heap_asks_no_id_per_object(read):
    targets = tuple([] for _ in range(3))
    holders = [[t] for t in targets]
    filler = [[i] for i in range(20_000)]  # 20,000 heap objects, one referent each

    COUNTING[:] = [0]
    try:
        found = read(targets, set())
        ids_asked = COUNTING[0]
    finally:
        COUNTING.clear()

    assert ids_asked <= 10, f"{ids_asked} id() calls for a heap of {len(gc.get_objects())} objects"
    found_ids = {id(obj) for obj in found}
    assert all(id(holder) in found_ids for holder in holders)
    assert len(filler) == 20_000
