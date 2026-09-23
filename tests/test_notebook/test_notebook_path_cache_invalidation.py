"""Switching notebooks forgets the cached notebook path and the simulation built
from it, so an upstream check never reads another notebook's cells.
"""

import json
from unittest.mock import patch

from cash.notebook.upstream._types import SimulationCacheEntry


class TestNotebookPathCacheInvalidation:
    """Test that invalidate_notebook_path_cache() properly clears the cache."""

    def test_invalidate_clears_cache(self):
        """invalidate_notebook_path_cache should reset cached path and time."""
        import cash.notebook.server_discovery as discovery_mod
        from cash.notebook.server_discovery import (
            invalidate_notebook_path_cache,
        )

        # Set a fake cached path (state lives in server_discovery)
        discovery_mod._cached_notebook_path = "/fake/path/notebook.ipynb"
        discovery_mod._cached_notebook_path_time = 999999.0

        # Invalidate
        invalidate_notebook_path_cache()

        assert discovery_mod._cached_notebook_path is None
        assert discovery_mod._cached_notebook_path_time == 0.0

    def test_get_notebook_path_after_invalidation(self):
        """After invalidation, get_notebook_path should re-discover (not use stale cache)."""
        import cash.notebook.server_discovery as discovery_mod
        from cash.notebook.server_discovery import get_notebook_path, invalidate_notebook_path_cache

        # Set a fake cached path (state lives in server_discovery)
        discovery_mod._cached_notebook_path = "/old/notebook.ipynb"
        discovery_mod._cached_notebook_path_time = 999999.0

        # Invalidate
        invalidate_notebook_path_cache()

        # Now get_notebook_path should NOT return the old path
        # (In test environment without IPython/Jupyter, it returns None)
        result = get_notebook_path()
        assert result != "/old/notebook.ipynb", "Should not return stale cached path"

    def test_cash_on_clears_simulation_cache(self, cash_magics):
        """%cash_on should clear the upstream checker's simulation and AST caches."""
        magics = cash_magics

        # Populate the simulation cache with fake data. Caches now live on
        # the simulator (extracted from UpstreamChecker).
        simulator = magics._upstream_checker.simulator
        simulator.cache.entries = [
            SimulationCacheEntry("fake_hash", {"x": "lineage1"}, set(), [], set(), set(), {}),
            SimulationCacheEntry("fake_hash2", {"y": "lineage2"}, set(), [], set(), set(), {}),
        ]

        # Enable auto-caching (this should clear the caches)
        magics.cash_on("")

        assert simulator.cache.entries == []

    def test_upstream_checker_reset_caches(self):
        """UpstreamChecker.reset_caches() should clear simulation and AST caches."""
        from unittest.mock import MagicMock

        from cash.notebook.upstream import UpstreamChecker

        shell = MagicMock()
        checker = UpstreamChecker(shell)

        # Add some data to caches
        checker.simulator.cache.entries.append(
            SimulationCacheEntry("hash1", {"var": "lin"}, set(), [], set(), set(), {})
        )

        checker.reset_caches()

        assert checker.simulator.cache.entries == []

    def test_no_glob_fallback_for_notebook_discovery(self, tmp_path):
        """get_notebook_cells should NOT use glob fallback.

        The glob fallback can pick the wrong notebook when multiple .ipynb
        files exist in the working directory, leading to wrong upstream cells.
        """
        import os

        from cash.notebook.server_discovery import get_notebook_cells

        # Create a notebook in tmp_path
        nb = {"cells": [{"cell_type": "code", "source": ["wrong = True"]}]}
        nb_path = tmp_path / "wrong_notebook.ipynb"
        nb_path.write_text(json.dumps(nb), encoding="utf-8")

        # Change to tmp_path so glob would find the notebook
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with patch("cash.notebook.server_discovery.get_notebook_path", return_value=None):
                cells = get_notebook_cells(None)
                # Should return empty, NOT the cells from wrong_notebook.ipynb
                assert cells == [], f"Should not pick up notebook via glob, got: {cells}"
        finally:
            os.chdir(old_cwd)
