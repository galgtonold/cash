"""Batch 134 – Simulation cache edge case interaction tests (advanced).

Tests that specifically stress the simulation cache and upstream
detection logic with tricky patterns that could cause divergence.
"""

import pytest

pytestmark = [pytest.mark.upstream, pytest.mark.stress, pytest.mark.timeout(30)]


class TestSimCacheAfterEdits:
    """Verify simulation cache remains coherent after edits."""

    def test_edit_root_long_chain_no_restart(self, nb_runner):
        """Edit root of long chain without restart.
        Upstream simulation should propagate correctly."""
        nb_runner.create_notebook([
            "x = 1",
            "y = x + 1",
            "z = y + 1",
            "w = z + 1",
            "result = w\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 4" in nb_runner.get_output(5)

        # Edit root, run only last cell (upstream must detect and re-exec)
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_cell(5)
        assert "result = 103" in nb_runner.get_output(5)

    def test_edit_two_independent_roots(self, nb_runner):
        """Edit two independent roots, verify both paths update."""
        nb_runner.create_notebook([
            "a = 1",
            "b = 2",
            "c = a * 10",
            "d = b * 10",
            "result = c + d\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(5)

        # Edit both roots
        nb_runner.set_cell_source(1, "a = 5")
        nb_runner.set_cell_source(2, "b = 7")
        nb_runner.run_cell(5)
        assert "result = 120" in nb_runner.get_output(5)

    def test_repeated_edits_same_cell_five_times(self, nb_runner):
        """Edit the same cell 5 times, run last each time."""
        nb_runner.create_notebook([
            "x = 0",
            "result = x * 10\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 0" in nb_runner.get_output(2)

        for val in [1, 2, 3, 4, 5]:
            nb_runner.set_cell_source(1, f"x = {val}")
            nb_runner.run_cell(2)
            assert f"result = {val * 10}" in nb_runner.get_output(2)


class TestSimCacheWithRestart:
    """Simulation cache coherence after kernel restart."""

    def test_edit_restart_edit_again(self, nb_runner):
        """Edit → restart → edit again → verify coherence."""
        nb_runner.create_notebook([
            "a = 1",
            "b = a + 1",
            "c = b + 1\nprint(f'c = {c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 3" in nb_runner.get_output(3)

        # First edit + restart
        nb_runner.set_cell_source(1, "a = 10")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 12" in nb_runner.get_output(3)

        # Second edit + restart
        nb_runner.set_cell_source(1, "a = 100")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 102" in nb_runner.get_output(3)

    def test_restart_without_edit_restores(self, nb_runner):
        """Restart without any edits — should restore from cache."""
        nb_runner.create_notebook([
            "x = 42",
            "y = x * 2\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 84" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 84" in nb_runner.get_output(2)


class TestUpstreamPropagationEdges:
    """Edge cases in upstream propagation detection."""

    def test_edit_does_not_change_output_value(self, nb_runner):
        """Edit code but the output value doesn't change."""
        nb_runner.create_notebook([
            "x = 5 + 5",
            "y = x * 2\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)

        # Change code but same result
        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)

    def test_edit_comment_only_change(self, nb_runner):
        """Edit only a comment — code is different but effect is same."""
        nb_runner.create_notebook([
            "x = 10  # initial value",
            "y = x * 2\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)

        # Only change comment
        nb_runner.set_cell_source(1, "x = 10  # updated comment")
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)

    def test_whitespace_only_change_still_correct(self, nb_runner):
        """Change only whitespace — should still produce correct results."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x + 1\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 11" in nb_runner.get_output(2)

        # Add trailing whitespace
        nb_runner.set_cell_source(1, "x = 10  ")
        nb_runner.run_all()
        assert "y = 11" in nb_runner.get_output(2)

    def test_reorder_independent_cells_swap(self, nb_runner):
        """Reorder two independent cells (swap order in notebook)."""
        nb_runner.create_notebook([
            "a = 10",
            "b = 20",
            "result = a + b\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        # Swap cells 1 and 2
        nb_runner.set_cell_source(1, "b = 20")
        nb_runner.set_cell_source(2, "a = 10")
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)
