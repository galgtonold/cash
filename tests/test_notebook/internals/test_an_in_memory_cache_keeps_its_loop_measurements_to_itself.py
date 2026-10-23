"""Two Cash instances over in-memory backends keep their loop measurements
apart. They shared one store: a heavy ``for k in range(3)`` measured by one
made every later ``for k in range(1500)`` of another run pass by pass, the
same header with numbers blanked, and a test after it took 50 s, not 0.5 s."""

from __future__ import annotations

from cash.backends.memory_backend import InMemoryBackend
from cash.notebook import heavy_loops, loop_split


def test_each_in_memory_backend_has_a_store_of_its_own():
    first, second = InMemoryBackend(), InMemoryBackend()
    for module in (heavy_loops, loop_split):
        a, b = module.store_for_backend(first), module.store_for_backend(second)
        assert a is not None and b is not None
        assert a is not b
        assert module.store_for_backend(first) is a


def test_a_measurement_one_records_is_not_read_by_the_other():
    first, second = InMemoryBackend(), InMemoryBackend()
    heavy_loops.store_for_backend(first).record("k in range(_)", 0.3, "abc")
    assert heavy_loops.store_for_backend(second).get("k in range(_)") is None
