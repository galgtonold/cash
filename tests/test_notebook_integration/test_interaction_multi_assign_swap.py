"""Batch 267 – Multiple variable assignment and swap patterns.

Tests tuple assignment, swap, augmented assignment with edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestMultiAssignSwap:
    """Multiple variable assignment edit patterns."""

    def test_tuple_swap_edit(self, nb_runner):
        """Edit initial values, swap and downstream reflect."""
        nb_runner.create_notebook([
            "x, y = 10, 20",
            "x, y = y, x\nprint(f'x={x} y={y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=20 y=10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "x, y = 100, 200")
        nb_runner.run_all()
        assert "x=200 y=100" in nb_runner.get_output(2)

    def test_augmented_assignment_edit(self, nb_runner):
        """Edit initial value, augmented assignments chain correctly."""
        nb_runner.create_notebook([
            "val = 10",
            "val = val + 5\nval = val * 2\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "val = 100")
        nb_runner.run_all()
        assert "val = 210" in nb_runner.get_output(2)

    def test_chained_assignment_edit(self, nb_runner):
        """Edit one var in chained assignment."""
        nb_runner.create_notebook([
            "a = 1\nb = 2\nc = 3",
            "total = a + b + c\navg = total / 3\nprint(f'total={total} avg={avg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=6 avg=2.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "a = 10\nb = 20\nc = 30")
        nb_runner.run_all()
        assert "total=60 avg=20.0" in nb_runner.get_output(2)

    def test_multi_target_from_function(self, nb_runner):
        """Edit multi-target assignment from function return."""
        nb_runner.create_notebook([
            "def get_bounds(data):\n    return min(data), max(data)",
            "nums = [5, 2, 8, 1, 9]",
            "lo, hi = get_bounds(nums)\nspan = hi - lo\nprint(f'lo={lo} hi={hi} span={span}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lo=1 hi=9 span=8" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "nums = [100, 200, 300]")
        nb_runner.run_all()
        assert "lo=100 hi=300 span=200" in nb_runner.get_output(3)
