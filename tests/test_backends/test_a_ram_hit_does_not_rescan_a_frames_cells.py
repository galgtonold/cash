"""A RAM hit of a pandas frame reuses the store's verdict on its object cells.

Whether an object column holds a list, dict or array (and so needs a deep copy
on every hit) is decided by scanning the column: 22 ms per hit at a million
rows, 105 ms at five million, where a hit is otherwise a shallow copy of a few
microseconds. The stored frame is private to its entry and never written, so
the store decides once and every hit reuses that.
"""

import pytest

pd = pytest.importorskip("pandas")

from cash.backends import memory_backend
from cash.backends.memory_backend import InMemoryBackend


@pytest.fixture
def scans(monkeypatch):
    calls = []
    real = memory_backend._holds_mutable_cells

    def counting(frame):
        calls.append(frame)
        return real(frame)

    monkeypatch.setattr(memory_backend, "_holds_mutable_cells", counting)
    return calls


def _strings():
    return pd.DataFrame({"s": pd.array(["a", "b", "c"], dtype=object), "n": [1, 2, 3]})


def _lists():
    return pd.DataFrame({"tags": [["a"], ["b"], ["c"]]})


def test_hits_of_a_frame_do_not_scan_its_cells_again(scans):
    backend = InMemoryBackend()
    backend.set("k", _strings(), {"execution_time": 1.0, "copy_required": True})
    assert len(scans) == 1, "the store decides once"
    for _ in range(3):
        _meta, hit = backend.get("k")
        assert hit.equals(_strings())
    assert len(scans) == 1, f"{len(scans) - 1} hit(s) scanned the cells again"


def test_hits_of_a_frame_inside_an_entry_do_not_scan_again(scans):
    backend = InMemoryBackend()
    backend.set("k", {"variables": {"a": _strings(), "b": _lists()}}, {"execution_time": 1.0})
    assert len(scans) == 2
    for _ in range(3):
        backend.get("k")
    assert len(scans) == 2


def test_a_frame_with_mutable_cells_is_still_isolated_on_every_hit(scans):
    """The remembered verdict is "deep copy", so a positive control that the
    verdict is used, not dropped."""
    backend = InMemoryBackend()
    backend.set("k", _lists(), {"execution_time": 1.0, "copy_required": True})
    for _ in range(2):
        _meta, hit = backend.get("k")
        assert hit["tags"].iloc[0] == ["a"], "an edit to an earlier hit's cell reached the cache"
        hit["tags"].iloc[0].append("x")
    _meta, entry_hit = backend.get("k")
    assert entry_hit["tags"].iloc[0] == ["a"]
    assert len(scans) == 1


def test_replacing_an_entry_decides_again(scans):
    backend = InMemoryBackend()
    backend.set("k", _strings(), {"execution_time": 1.0, "copy_required": True})
    backend.set("k", _lists(), {"execution_time": 1.0, "copy_required": True})
    _meta, hit = backend.get("k")
    hit["tags"].iloc[0].append("x")
    _meta, again = backend.get("k")
    assert again["tags"].iloc[0] == ["a"]
    assert len(scans) == 2
