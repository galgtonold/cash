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

from cash.notebook._protocols import TrackingState
from cash.notebook.notebook_simulator import NotebookSimulator


def _make_simulator(*, debug: bool = False) -> NotebookSimulator:
    """Construct a NotebookSimulator with a minimal fake shell.

    This is the new test surface — no Cash, no backend, no UpstreamChecker.
    """
    shell = SimpleNamespace(user_ns={})
    return NotebookSimulator(
        shell=shell,
        cash_instance=None,
        tracking_state=TrackingState(),
        compute_hash_fn=None,
        debug=debug,
    )


class TestConstructionWithoutOrchestrator:
    """The simulator can be constructed in isolation — that's the win."""

    def test_constructs_with_minimal_dependencies(self):
        sim = _make_simulator()
        assert sim is not None
        assert sim.cash_instance is None

    def test_owns_its_caches(self):
        sim = _make_simulator()
        assert sim._simulation_cache == []
        assert sim._ast_cache == {}
        assert sim._simulation_cell_hashes == {}
        assert sim._cell_id_to_last_index == {}

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
            debug=False,
        )
        sim.variable_lineage["x"] = "abc"
        assert ts.variable_lineage["x"] == "abc"


class TestStaticHelpers:
    """The pure helpers can be exercised without ever constructing an instance.

    Before extraction, these were on ``UpstreamChecker`` and tests reached
    into them via the class — fine, but the class was 3700 lines so the test
    surface was tangled with orchestration concerns. The simulator class is
    just simulator code.
    """

    def test_validate_file_freshness_empty(self):
        assert NotebookSimulator._validate_file_freshness({}) is True

    def test_validate_file_freshness_missing_is_stale(self, tmp_path):
        missing = str(tmp_path / "absent.csv")
        assert NotebookSimulator._validate_file_freshness({missing: 0.0}) is False

    def test_stat_file_deps_empty(self):
        assert NotebookSimulator._stat_file_deps({}) == {}

    def test_stat_file_deps_excludes_missing(self, tmp_path):
        missing = str(tmp_path / "absent.csv")
        result = NotebookSimulator._stat_file_deps({missing: 0.0})
        assert missing not in result


class TestAstCacheEviction:
    """The simulator caches AST parses with a size-limited LRU-ish eviction.

    A regression test on its own surface, not through UpstreamChecker.
    """

    def test_returns_none_for_syntax_error(self):
        sim = _make_simulator()
        assert sim._get_cached_ast("def (:") is None

    def test_caches_parsed_tree(self):
        sim = _make_simulator()
        tree1 = sim._get_cached_ast("x = 1")
        tree2 = sim._get_cached_ast("x = 1")
        assert tree1 is tree2

    def test_evicts_when_over_capacity(self):
        sim = _make_simulator()
        sim._ast_cache_max_size = 4
        for i in range(8):
            sim._get_cached_ast(f"a{i} = {i}")
        # Eviction removes the first quarter when at capacity, so a few
        # entries survive but not all 8.
        assert 0 < len(sim._ast_cache) < 8


class TestResetCaches:
    def test_clears_all_simulator_caches(self):
        sim = _make_simulator()
        sim._get_cached_ast("x = 1")
        sim._simulation_cache.append(MagicMock())
        sim._simulation_cell_hashes[0] = "h"
        sim.reset_caches()
        assert sim._ast_cache == {}
        assert sim._simulation_cache == []
        assert sim._simulation_cell_hashes == {}
