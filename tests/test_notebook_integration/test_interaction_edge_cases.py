"""Batch 164 – Edge case interaction tests.

Tests empty cells, whitespace-only changes, very large output,
cell reordering scenarios, and other boundary conditions.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestWhitespaceEdits:
    """Whitespace-only cell edits."""

    def test_add_trailing_newline(self, nb_runner):
        """Adding trailing newline should not invalidate cache."""
        nb_runner.create_notebook([
            "x = 42",
            "print(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 42" in nb_runner.get_output(2)

        # Add trailing newlines — should still work
        nb_runner.set_cell_source(1, "x = 42\n\n")
        nb_runner.run_all()
        assert "x = 42" in nb_runner.get_output(2)

    def test_add_comment_only(self, nb_runner):
        """Adding a comment changes the code hash → recomputes."""
        nb_runner.create_notebook([
            "val = 10",
            "print(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 10" in nb_runner.get_output(2)

        # Add a comment — semantically identical but different hash
        nb_runner.set_cell_source(1, "# Important value\nval = 10")
        nb_runner.run_all()
        assert "val = 10" in nb_runner.get_output(2)


class TestLargeOutput:
    """Large output scenarios."""

    def test_large_list_output(self, nb_runner):
        """Generate a large list, edit the size."""
        nb_runner.create_notebook([
            "n = 100  # list size",
            "data = list(range(n))\nprint(f'len = {len(data)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "len = 100" in nb_runner.get_output(2)

        # Make it bigger
        nb_runner.set_cell_source(1, "n = 1000  # list size bigger")
        nb_runner.run_all()
        assert "len = 1000" in nb_runner.get_output(2)

    def test_large_string_output(self, nb_runner):
        """Generate a large string, then edit pattern."""
        nb_runner.create_notebook([
            "pattern = 'ab'  # string pattern",
            "big = pattern * 500\nprint(f'length = {len(big)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "length = 1000" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "pattern = 'xyz'  # string pattern changed")
        nb_runner.run_all()
        assert "length = 1500" in nb_runner.get_output(2)


class TestMultipleOutputsPerCell:
    """Cells producing multiple variables."""

    def test_multi_output_edit_one(self, nb_runner):
        """Cell producing multiple vars, edit to change one."""
        nb_runner.create_notebook([
            "a = 1\nb = 2\nc = 3",
            "total = a + b + c\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(2)

        # Change one variable
        nb_runner.set_cell_source(1, "a = 100\nb = 2\nc = 3")
        nb_runner.run_all()
        assert "total = 105" in nb_runner.get_output(2)

    def test_swap_variable_assignments(self, nb_runner):
        """Swap which variables get which values."""
        nb_runner.create_notebook([
            "x = 10\ny = 20",
            "diff = x - y\nprint(f'diff = {diff}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "diff = -10" in nb_runner.get_output(2)

        # Swap values
        nb_runner.set_cell_source(1, "x = 20\ny = 10")
        nb_runner.run_all()
        assert "diff = 10" in nb_runner.get_output(2)


class TestRerunPatterns:
    """Patterns of re-running cells."""

    def test_run_same_cell_twice(self, nb_runner):
        """Run a cell twice without edits — idempotent."""
        nb_runner.create_notebook([
            "x = 5  # initial",
            "y = x * 2\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 10" in nb_runner.get_output(2)

        # Re-run without edits
        nb_runner.run_all()
        assert "y = 10" in nb_runner.get_output(2)

    def test_edit_then_revert(self, nb_runner):
        """Edit a cell, run, then revert and run again."""
        nb_runner.create_notebook([
            "val = 'original'  # version 1",
            "print(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = original" in nb_runner.get_output(2)

        # Edit
        nb_runner.set_cell_source(1, "val = 'modified'  # version 2")
        nb_runner.run_all()
        assert "val = modified" in nb_runner.get_output(2)

        # Revert to original (note: different comment to avoid identical cell ambiguity)
        nb_runner.set_cell_source(1, "val = 'original'  # version 3 reverted")
        nb_runner.run_all()
        assert "val = original" in nb_runner.get_output(2)
