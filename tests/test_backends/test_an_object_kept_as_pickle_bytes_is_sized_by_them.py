"""An object the RAM tier keeps as pickle bytes is sized by those bytes.

The tier sized every value by walking it (`sizing.memory_footprint`), then
kept an object of a class of its own as its pickle bytes. For a graph of
350,000 edges the walk took 1.35 s and the pickle 0.35 s, on every store,
for the size of something the tier does not hold. The bytes are now made
first and their length is the entry's size; the walk is left to values
the tier holds as objects.
"""

from __future__ import annotations

import numpy as np

from cash.backends import FileBackend, InMemoryBackend, TieredBackend, memory_backend

ROWS = 1 << 18


class Graph:
    """Dicts of dicts, as networkx keeps a graph, and an array."""

    def __init__(self) -> None:
        self._adj = {n: {n + 1: {"w": float(n)}} for n in range(20_000)}
        self.weights = np.ones(ROWS)


def test_its_size_is_the_bytes_kept_and_it_is_not_walked(monkeypatch):
    walked = []
    real = memory_backend.memory_footprint
    monkeypatch.setattr(memory_backend, "memory_footprint", lambda v, *a: walked.append(v) or real(v, *a))
    ram = InMemoryBackend()
    graph = Graph()
    ram.set("g", graph, {"execution_time": 60.0})
    kept = ram._store["g"][1]
    assert type(kept) is memory_backend._Pickled
    assert ram.get_metadata("g")["size"] == kept.nbytes
    assert not walked


def test_a_small_object_is_copied_as_before():
    ram = InMemoryBackend()
    small = Graph.__new__(Graph)
    small._adj = {0: {1: {}}}
    ram.set("s", small, {"execution_time": 60.0})
    assert type(ram._store["s"][1]) is Graph


def test_one_over_the_cap_is_refused():
    ram = InMemoryBackend(max_size_bytes=1 << 20)
    assert ram.set("g", Graph(), {"execution_time": 60.0}) is False
    assert ram.get("g") == (None, None)


def test_it_reads_back_after_a_restart(tmp_path):
    graph = Graph()
    first = TieredBackend([InMemoryBackend(), FileBackend(cache_dir=str(tmp_path))])
    first.set("g", graph, {"execution_time": 60.0, "force_persist": True})
    graph._adj[0][1]["w"] = -1.0  # after the store: in neither copy
    first.backends[1].shutdown()
    again = TieredBackend([InMemoryBackend(), FileBackend(cache_dir=str(tmp_path))])
    _meta, served = again.get("g")
    assert served._adj[0][1]["w"] == 0.0 and served._adj[19_999] == {20_000: {"w": 19_999.0}}
    assert np.array_equal(served.weights, np.ones(ROWS))
