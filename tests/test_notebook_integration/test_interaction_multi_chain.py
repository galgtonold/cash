"""Batch 231 – Multiple cell chain edit interaction tests.

Tests editing a cell in the middle of a multi-cell pipeline to 
verify both upstream restoration and downstream propagation work.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestMultiCellChainEdits:
    """Editing cells in multi-step pipelines."""

    def test_edit_middle_of_3_cell_chain(self, nb_runner):
        """Edit the middle cell in a 3-cell chain."""
        nb_runner.create_notebook([
            "raw = [1, 2, 3, 4, 5]",
            "processed = [x ** 2 for x in raw]",
            "total = sum(processed)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 55" in nb_runner.get_output(3)

        # Edit middle cell to cube instead of square
        nb_runner.set_cell_source(2, "processed = [x ** 3 for x in raw]")
        nb_runner.run_all()
        assert "total = 225" in nb_runner.get_output(3)

    def test_edit_first_of_4_cell_chain(self, nb_runner):
        """Edit the first cell in a 4-cell chain — all downstream recompute."""
        nb_runner.create_notebook([
            "a = 2",
            "b = a * 3",
            "c = b + 1",
            "print(f'c = {c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 7" in nb_runner.get_output(4)

        # Edit first cell
        nb_runner.set_cell_source(1, "a = 10")
        nb_runner.run_all()
        assert "c = 31" in nb_runner.get_output(4)

    def test_edit_adds_intermediate_step(self, nb_runner):
        """Edit middle cell to add an extra processing step in same cell."""
        nb_runner.create_notebook([
            "data = [3, 1, 4, 1, 5, 9]",
            "result = sorted(data)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [1, 1, 3, 4, 5, 9]" in nb_runner.get_output(2)

        # Add dedup step after sort
        nb_runner.set_cell_source(2, "result = sorted(set(data))\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = [1, 3, 4, 5, 9]" in nb_runner.get_output(2)

    def test_edit_preserves_unrelated_cells(self, nb_runner):
        """Editing one branch shouldn't affect an unrelated branch."""
        nb_runner.create_notebook([
            "x = 10",
            "y = 20",
            "sum_xy = x + y\nprint(f'sum = {sum_xy}')",
            "prod_xy = x * y\nprint(f'prod = {prod_xy}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sum = 30" in nb_runner.get_output(3)
        assert "prod = 200" in nb_runner.get_output(4)

        # Edit y — both downstream cells should update
        nb_runner.set_cell_source(2, "y = 30")
        nb_runner.run_all()
        assert "sum = 40" in nb_runner.get_output(3)
        assert "prod = 300" in nb_runner.get_output(4)
