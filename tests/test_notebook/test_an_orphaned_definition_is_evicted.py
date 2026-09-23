"""A variable whose definition was removed is evicted, with its lineage.

When no cell produces a variable any more, the upstream checker evicts it
and everything computed from it, so a consumer re-runs and raises the
``NameError`` a fresh kernel would. The eviction also forgets the variable's
lineage -- and once lineage became writable only through the lineage store,
the eviction still popped it from the read-only view, failed on the first
variable, and left the stale value in place.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from cash.notebook._protocols import TrackingState
from cash.notebook.upstream.checker import UpstreamChecker


def test_a_removed_definition_and_its_consumer_are_evicted():
    shell = MagicMock()
    shell.user_ns = {"y": 5, "z": 6}
    state = TrackingState()
    state.lineage.record("y", "lineage-y")
    state.lineage.record("z", "lineage-z")
    state.executed_input_lineages["z"] = {"y": "lineage-y"}
    checker = UpstreamChecker(shell, tracking_state=state)

    # Cell 1 (`y = 5`) became `pass`; cell 2 still computes z from y.
    checker._evict_orphaned_definitions(["pass", "z = y + 1"], "print(z)")

    assert "y" not in shell.user_ns
    assert "z" not in shell.user_ns
    assert "y" not in state.variable_lineage
    assert "z" not in state.variable_lineage
