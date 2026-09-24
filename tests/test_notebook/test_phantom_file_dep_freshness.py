"""A file dependency on a path that NEVER existed must not invalidate.

``tracking_state.executed_file_deps`` is a strict superset of the producer's
persisted ``file_dependencies`` snapshot:

* ``FileAccessTracker.track_path`` records every access *attempt* — the
  ``open()`` wrapper records the path before it calls through, so a read that
  raises ``FileNotFoundError`` still lands in the set.
* ``snapshot_file_deps`` silently drops any path it cannot ``stat``, so a
  phantom never reaches the persisted snapshot.

Importing sklearn makes ``importlib.metadata`` probe for optional metadata
that legitimately does not exist (``direct_url.json``, ``entry_points.txt``,
``pythonXY.zip`` on ``sys.path``). Those probes are how a phantom gets in.

The bug: ``_input_file_changed`` judged the path missing BEFORE consulting the
producer's snapshot, so every consumer of such a variable missed on every run,
forever (seen on a ``make_classification`` chain). A path that never existed
cannot have *changed*.
"""

import types

import pytest

from cash.backends import CacheBackend
from cash.notebook.statement.freshness import CacheFreshnessChecker
from cash.tracking.file_dep_snapshot import file_dep_is_fresh, snapshot_file_deps

PHANTOM = "C:/nonexistent-dir-phantom/numpy-1.0.dist-info/direct_url.json"


class _StubBackend(CacheBackend):
    """Returns one producer metadata dict for any key."""

    def __init__(self, metadata):
        self._metadata = metadata

    def get(self, key):
        return self._metadata, "cached-payload"

    def set(self, key, value, metadata=None, serializer=None):
        raise NotImplementedError

    def delete(self, key):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def list_entries(self):
        return []


@pytest.fixture
def real_file(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    return str(p)


def _state(deps):
    return types.SimpleNamespace(
        executed_file_deps={"X": set(deps)},
        variable_sources={"X": "stmt:producer"},
    )


def test_snapshot_drops_phantom_but_tracking_keeps_it(real_file):
    """The asymmetry that makes the bug possible."""
    snapshot = snapshot_file_deps({real_file, PHANTOM})
    assert real_file in snapshot
    assert PHANTOM not in snapshot, "a phantom must never reach the persisted snapshot"


def test_phantom_freshness_verdict_is_stable(real_file):
    """The phantom resolves to the SAME fact every run — it is not churn."""
    verdicts = [file_dep_is_fresh(PHANTOM, {"mtime": 0.0, "size": 0}) for _ in range(3)]
    assert verdicts == [(False, "unreadable")] * 3


def test_phantom_input_dep_does_not_invalidate(real_file):
    """A never-existed path must not make the consumer stale."""
    producer_meta = {
        "key": "stmt:producer",
        "file_dependencies": snapshot_file_deps({real_file}),
    }
    checker = CacheFreshnessChecker(_StubBackend(producer_meta))
    state = _state({real_file, PHANTOM})

    # Repeated across runs: the verdict must be a stable "not changed".
    for _ in range(3):
        assert checker._input_file_changed(state, "X", PHANTOM) is False
        assert checker._input_file_changed(state, "X", real_file) is False

    assert checker._invalidate_if_input_file_changed(state, {"X"}, "payload") == "payload"


def test_snapshotted_file_that_disappears_still_invalidates(real_file, tmp_path):
    """The guard must not swallow a REAL deletion: a dep that was snapshotted
    (so it did exist) and has since been deleted is still stale."""
    producer_meta = {
        "key": "stmt:producer",
        "file_dependencies": snapshot_file_deps({real_file}),
    }
    checker = CacheFreshnessChecker(_StubBackend(producer_meta))
    state = _state({real_file})

    import os

    os.remove(real_file)

    assert checker._input_file_changed(state, "X", real_file) is True
    assert "missing" in (checker.last_miss_reason or "")
    assert checker._invalidate_if_input_file_changed(state, {"X"}, "payload") is None


def test_snapshotted_file_that_changes_still_invalidates(real_file):
    """The guard must not swallow a real content change."""
    producer_meta = {
        "key": "stmt:producer",
        "file_dependencies": snapshot_file_deps({real_file}),
    }
    checker = CacheFreshnessChecker(_StubBackend(producer_meta))
    state = _state({real_file, PHANTOM})

    with open(real_file, "w", encoding="utf-8") as fh:
        fh.write("a,b\n9,9\n")  # same size, different content

    assert checker._input_file_changed(state, "X", real_file) is True
    assert checker._invalidate_if_input_file_changed(state, {"X"}, "payload") is None
