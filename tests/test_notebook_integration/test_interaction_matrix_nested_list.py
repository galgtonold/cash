"""Batch 379: matrix operations with nested lists and edits."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestMatrixNestedList:
    def test_matrix_transpose(self, nb_runner):
        nb_runner.create_notebook([
            "matrix = [[1, 2, 3], [4, 5, 6]]",
            "transposed = list(map(list, zip(*matrix)))\nprint(f'transposed={transposed}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "transposed=[[1, 4], [2, 5], [3, 6]]" in nb_runner.get_output(2)

    def test_matrix_multiply_edit(self, nb_runner):
        nb_runner.create_notebook([
            "a = [[1, 2], [3, 4]]\nb = [[5, 6], [7, 8]]",
            "result = [[sum(a[i][k]*b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[[19, 22], [43, 50]]" in nb_runner.get_output(2)
        # Edit matrix a
        nb_runner.set_cell_source(1, "a = [[1, 0], [0, 1]]\nb = [[5, 6], [7, 8]]")
        nb_runner.run_all()
        assert "result=[[5, 6], [7, 8]]" in nb_runner.get_output(2)

    def test_matrix_flatten(self, nb_runner):
        nb_runner.create_notebook([
            "grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]",
            "flat = [x for row in grid for x in row]\ndiag = [grid[i][i] for i in range(len(grid))]\nprint(f'flat={flat}')\nprint(f'diag={diag}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "flat=[1, 2, 3, 4, 5, 6, 7, 8, 9]" in nb_runner.get_output(2)
        assert "diag=[1, 5, 9]" in nb_runner.get_output(2)
