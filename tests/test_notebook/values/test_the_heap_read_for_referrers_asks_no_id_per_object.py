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

#: A read that asked an id per object would ask at least one per filler
#: object (20,000). What else runs during the read asks a few: weakref
#: callbacks and finalizers a collection runs (34 on a 3.8M-object macOS
#: 3.14 worker in CI).
FEW = 1_000


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

    assert ids_asked < FEW, f"{ids_asked} id() calls for a heap of {len(gc.get_objects())} objects"
    found_ids = {id(obj) for obj in found}
    assert all(id(holder) in found_ids for holder in holders)
    assert len(filler) == 20_000


def _ids_asked(fn, *args):
    COUNTING[:] = [0]
    try:
        return fn(*args), COUNTING[0]
    finally:
        COUNTING.clear()


def test_the_other_reads_of_the_heap_or_a_namespace_ask_no_id_per_object():
    """The same for the other reads the sampled 3.10 workers spent their time
    in: the classes with an object alive (every object on the heap), the
    module references of a value (every value of each namespace) and a
    namespace's memo holders (every value of a module's namespace)."""
    from cash.notebook import helper_memos
    from cash.notebook.statement import unseen_writes

    class Alive:
        pass

    class Gone:
        pass

    kept = [Alive() for _ in range(3)]
    namespace = {f"v{i}": [i] for i in range(20_000)}
    namespace["target"] = []
    namespace.update({f"f{i}": (lambda i=i: i) for i in range(2_000)})

    alive, asked = _ids_asked(unseen_writes._with_objects_alive, [Alive, Gone])
    assert asked < FEW
    assert alive == [Alive]

    counts, asked = _ids_asked(shared_objects._module_references, [namespace["target"]], [namespace])
    assert asked < FEW
    assert counts == [1]

    holders, asked = _ids_asked(helper_memos._holders, namespace)
    assert asked < FEW
    assert len(holders) == 2_000
    assert len(kept) == 3
