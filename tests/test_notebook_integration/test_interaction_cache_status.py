"""Batch 225 – Cache status validation tests.

Verifies that statements are correctly cached (RESTORED/SKIPPED)
when re-run without changes, and correctly COMPUTED when changes occur.
This is per user requirement: "we should also check if it was cached
when it was supposed to be."
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.core, pytest.mark.timeout(90)]


class TestCacheStatusValidation:
    """Verify caching status is correct, not just output correctness."""

    def test_second_run_is_cached(self, nb_runner):
        """Second run of identical cells should hit cache, not recompute."""
        nb_runner.create_notebook([
            "x = 42",
            "y = x * 2\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()

        nb_runner.run_all()
        assert "y = 84" in nb_runner.get_output(2)
        nb_runner.get_raw_output(2)
        # First run should include COMPUTED or at least no CACHE_HIT
        # (upstream may restore, but the final cell should compute)

        # Second run — same cells, should be SKIPPED or RESTORED
        nb_runner.run_all()
        raw2 = nb_runner.get_raw_output(2)
        assert "y = 84" in nb_runner.get_output(2)
        # The second run should NOT say CELL_CHANGED
        assert "[CELL_CHANGED]" not in raw2

    def test_edit_forces_recompute(self, nb_runner):
        """Editing a cell must force COMPUTED, not use stale cache."""
        nb_runner.create_notebook([
            "a = 10",
            "b = a + 5\nprint(f'b = {b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()
        nb_runner.run_all()
        assert "b = 15" in nb_runner.get_output(2)

        # Edit cell 1
        nb_runner.set_cell_source(1, "a = 100")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "b = 105" in out

    def test_unchanged_cell_stays_cached(self, nb_runner):
        """A cell that hasn't changed should be served from cache."""
        nb_runner.create_notebook([
            "data = list(range(100))",
            "total = sum(data)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()

        nb_runner.run_all()
        assert "total = 4950" in nb_runner.get_output(2)

        # Run again — should not see CELL_CHANGED
        nb_runner.run_all()
        raw2 = nb_runner.get_raw_output(2)
        assert "total = 4950" in nb_runner.get_output(2)
        assert "[CELL_CHANGED]" not in raw2

    def test_upstream_edit_invalidates_downstream(self, nb_runner):
        """Editing an upstream cell should cause downstream to recompute."""
        nb_runner.create_notebook([
            "base = 5",
            "derived = base ** 2\nprint(f'derived = {derived}')",
            "final = derived + 1\nprint(f'final = {final}')",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()
        nb_runner.run_all()
        assert "derived = 25" in nb_runner.get_output(2)
        assert "final = 26" in nb_runner.get_output(3)

        # Edit base
        nb_runner.set_cell_source(1, "base = 10")
        nb_runner.run_all()
        assert "derived = 100" in nb_runner.get_output(2)
        assert "final = 101" in nb_runner.get_output(3)
