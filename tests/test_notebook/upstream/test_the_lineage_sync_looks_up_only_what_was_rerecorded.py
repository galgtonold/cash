"""After an upstream pass, the simulation's snapshots take the runtime
lineage of what the pass re-ran -- looked up in each snapshot, not found by
walking all of its names.

Each cell's snapshot holds every name bound above it, so walking them all
was quadratic in a long notebook: 160,000 names for every cell at cell 400,
the largest single piece of a trivial cell's cost there.
"""

from __future__ import annotations

from types import SimpleNamespace

from cash.notebook.upstream.simulator import NotebookSimulator


class _Snapshots(dict):
    """A snapshot that counts how often its names are walked."""

    walked = 0

    def keys(self):
        _Snapshots.walked += 1
        return super().keys()

    def __iter__(self):
        _Snapshots.walked += 1
        return super().__iter__()


def _simulator(n: int, runtime: dict) -> tuple[NotebookSimulator, list]:
    entries = [
        SimpleNamespace(trace_segment=[], virtual_lineage=_Snapshots({f"x{j}": f"old{j}" for j in range(i + 1)}))
        for i in range(n)
    ]
    sim = object.__new__(NotebookSimulator)
    sim.cache = type("C", (), {"__len__": lambda self: n, "entry": lambda self, i: entries[i]})()
    sim.tracking_state = SimpleNamespace(variable_lineage=runtime)
    sim._should_sync_cache_var = lambda *a: True
    return sim, entries


def test_only_the_rerecorded_names_are_synced():
    sim, entries = _simulator(5, {"x2": "new2"})
    _Snapshots.walked = 0
    sim._sync_simulation_cache_lineages({"x2"})
    assert _Snapshots.walked == 0, "a snapshot's names were walked"
    assert [e.virtual_lineage.get("x2") for e in entries] == [None, None, "new2", "new2", "new2"]
    assert entries[4].virtual_lineage["x3"] == "old3"


def test_nothing_rerecorded_changes_nothing():
    sim, entries = _simulator(3, {"x1": "new1"})
    sim._sync_simulation_cache_lineages(set())
    assert entries[2].virtual_lineage["x1"] == "old1"
