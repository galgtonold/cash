"""Batch 255 – Matrix/nested list computation patterns.

Tests 2D list operations with edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestMatrixComputations:
    """2D list/matrix operation edit patterns."""

    def test_matrix_transpose_edit(self, nb_runner):
        """Edit matrix, transpose updates."""
        nb_runner.create_notebook([
            "matrix = [[1, 2, 3], [4, 5, 6]]",
            "transposed = list(map(list, zip(*matrix)))\nprint(f'transposed = {transposed}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "transposed = [[1, 4], [2, 5], [3, 6]]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "matrix = [[10, 20], [30, 40], [50, 60]]")
        nb_runner.run_all()
        assert "transposed = [[10, 30, 50], [20, 40, 60]]" in nb_runner.get_output(2)

    def test_matrix_row_sums_edit(self, nb_runner):
        """Edit matrix data, row sum computation updates."""
        nb_runner.create_notebook([
            "grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]",
            "row_sums = [sum(row) for row in grid]\nprint(f'row_sums = {row_sums}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "row_sums = [6, 15, 24]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "grid = [[10, 10], [20, 20]]")
        nb_runner.run_all()
        assert "row_sums = [20, 40]" in nb_runner.get_output(2)

    def test_flatten_and_aggregate(self, nb_runner):
        """Edit nested list, flatten + aggregate updates."""
        nb_runner.create_notebook([
            "nested = [[1, 2], [3, 4], [5, 6]]",
            "flat = [x for row in nested for x in row]\ntotal = sum(flat)\nprint(f'flat={flat} total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "flat=[1, 2, 3, 4, 5, 6]" in nb_runner.get_output(2)
        assert "total=21" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "nested = [[10, 20, 30], [40, 50]]")
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "flat=[10, 20, 30, 40, 50]" in out2
        assert "total=150" in out2
