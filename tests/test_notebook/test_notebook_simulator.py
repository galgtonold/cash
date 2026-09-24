"""Tests demonstrating the new test surface unlocked by extracting
NotebookSimulator from UpstreamChecker.

Before the extraction, simulator behavior could only be tested through
``UpstreamChecker``, which required constructing a ``Cash`` instance with
a backend, an IPython shell, and the whole tracking-state graph. These
tests construct a ``NotebookSimulator`` directly with a minimal fake shell
and verify simulator-owned state without touching the orchestrator.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from cash.analysis.ast_util import parse_cached
from cash.notebook.tracking_state import TrackingState
from cash.notebook.upstream import NotebookSimulator
from cash.notebook.upstream.virtual_lineage import VirtualLineage


def _make_simulator() -> NotebookSimulator:
    """Construct a NotebookSimulator with a minimal fake shell.

    This is the new test surface — no Cash, no backend, no UpstreamChecker.
    """
    shell = SimpleNamespace(user_ns={})
    return NotebookSimulator(
        shell=shell,
        cash_instance=None,
        tracking_state=TrackingState(),
        compute_hash_fn=None,
    )


class TestConstructionWithoutOrchestrator:
    """The simulator can be constructed in isolation — that's the win."""

    def test_constructs_with_minimal_dependencies(self):
        sim = _make_simulator()
        assert sim is not None
        assert sim.cash_instance is None

    def test_owns_its_caches(self):
        sim = _make_simulator()
        assert sim.cache.entries == []
        assert sim.cache.cell_hashes == {}
        assert sim.cache.last_index_by_cell_id == {}

    def test_shares_tracking_state_refs(self):
        """Mutating the simulator's view must be visible through the original
        TrackingState — this is the shared-dict invariant."""
        ts = TrackingState()
        shell = SimpleNamespace(user_ns={})
        sim = NotebookSimulator(
            shell=shell,
            cash_instance=None,
            tracking_state=ts,
            compute_hash_fn=None,
        )
        sim.tracking_state.lineage.record("x", "abc")
        assert ts.variable_lineage["x"] == "abc"


class TestStaticHelpers:
    """The pure helpers can be exercised without ever constructing an instance.

    Before extraction, these were on ``UpstreamChecker`` and tests reached
    into them via the class — fine, but the class was 3700 lines so the test
    surface was tangled with orchestration concerns. The simulator class is
    just simulator code.
    """

    def test_validate_file_freshness_empty(self):
        assert VirtualLineage._validate_file_freshness({}) is True

    def test_validate_file_freshness_missing_is_stale(self, tmp_path):
        missing = str(tmp_path / "absent.csv")
        assert VirtualLineage._validate_file_freshness({missing: 0.0}) is False

    def test_stat_file_deps_empty(self):
        assert VirtualLineage._stat_file_deps({}) == {}

    def test_stat_file_deps_excludes_missing(self, tmp_path):
        missing = str(tmp_path / "absent.csv")
        result = VirtualLineage._stat_file_deps({missing: 0.0})
        assert missing not in result


class TestParseCached:
    """Cell and statement text is parsed once, through one bounded memo."""

    def test_returns_none_for_syntax_error(self):
        assert parse_cached("def (:") is None

    def test_caches_parsed_tree(self):
        assert parse_cached("x = 1") is parse_cached("x = 1")

    def test_is_bounded(self):
        assert parse_cached.cache_info().maxsize is not None


class TestResetCaches:
    def test_clears_all_simulator_caches(self):
        sim = _make_simulator()
        sim.cache.entries.append(MagicMock())
        sim.cache.cell_hashes[0] = "h"
        sim.reset_caches()
        assert sim.cache.entries == []
        assert sim.cache.cell_hashes == {}
