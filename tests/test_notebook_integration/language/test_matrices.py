"""Matrices as nested lists across cells."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress]


# Matrix/nested list computation patterns.
#
# Tests 2D list operations with edits.
@pytest.mark.timeout(90)
class TestMatrixComputations:
    """2D list/matrix operation edit patterns."""

    def test_matrix_transpose_edit(self, nb_runner):
        """Edit matrix, transpose updates."""
        nb_runner.create_notebook(
            [
                "matrix = [[1, 2, 3], [4, 5, 6]]",
                "transposed = list(map(list, zip(*matrix)))\nprint(f'transposed = {transposed}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "transposed = [[1, 4], [2, 5], [3, 6]]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "matrix = [[10, 20], [30, 40], [50, 60]]")
        nb_runner.run_all()
        assert "transposed = [[10, 30, 50], [20, 40, 60]]" in nb_runner.get_output(2)

    def test_matrix_row_sums_edit(self, nb_runner):
        """Edit matrix data, row sum computation updates."""
        nb_runner.create_notebook(
            [
                "grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]",
                "row_sums = [sum(row) for row in grid]\nprint(f'row_sums = {row_sums}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "row_sums = [6, 15, 24]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "grid = [[10, 10], [20, 20]]")
        nb_runner.run_all()
        assert "row_sums = [20, 40]" in nb_runner.get_output(2)

    def test_flatten_and_aggregate(self, nb_runner):
        """Edit nested list, flatten + aggregate updates."""
        nb_runner.create_notebook(
            [
                "nested = [[1, 2], [3, 4], [5, 6]]",
                "flat = [x for row in nested for x in row]\ntotal = sum(flat)\nprint(f'flat={flat} total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "flat=[1, 2, 3, 4, 5, 6]" in nb_runner.get_output(2)
        assert "total=21" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "nested = [[10, 20, 30], [40, 50]]")
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "flat=[10, 20, 30, 40, 50]" in out2
        assert "total=150" in out2


@pytest.mark.timeout(90)
class TestMatrixNestedList:
    """matrix operations with nested lists and edits."""

    def test_matrix_transpose(self, nb_runner):
        nb_runner.create_notebook(
            [
                "matrix = [[1, 2, 3], [4, 5, 6]]",
                "transposed = list(map(list, zip(*matrix)))\nprint(f'transposed={transposed}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "transposed=[[1, 4], [2, 5], [3, 6]]" in nb_runner.get_output(2)

    def test_matrix_multiply_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "a = [[1, 2], [3, 4]]\nb = [[5, 6], [7, 8]]",
                "result = [[sum(a[i][k]*b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[[19, 22], [43, 50]]" in nb_runner.get_output(2)
        # Edit matrix a
        nb_runner.set_cell_source(1, "a = [[1, 0], [0, 1]]\nb = [[5, 6], [7, 8]]")
        nb_runner.run_all()
        assert "result=[[5, 6], [7, 8]]" in nb_runner.get_output(2)

    def test_matrix_flatten(self, nb_runner):
        nb_runner.create_notebook(
            [
                "grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]",
                "flat = [x for row in grid for x in row]\ndiag = [grid[i][i] for i in range(len(grid))]\nprint(f'flat={flat}')\nprint(f'diag={diag}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "flat=[1, 2, 3, 4, 5, 6, 7, 8, 9]" in nb_runner.get_output(2)
        assert "diag=[1, 5, 9]" in nb_runner.get_output(2)


# Matrix operations interaction tests.
#
# Tests editing cells with nested list matrix operations
# (transpose, multiply, etc.) and verifying propagation.
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestMatrixOpsEdits:
    """Editing matrix operation patterns."""

    def test_edit_matrix_sum(self, nb_runner):
        """Edit two matrices and check element-wise sum."""
        nb_runner.create_notebook(
            [
                "a = [[1, 2], [3, 4]]\nb = [[5, 6], [7, 8]]",
                "s = [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]\nprint(f'sum = {s}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sum = [[6, 8], [10, 12]]" in nb_runner.get_output(2)

        # Change matrix b
        nb_runner.set_cell_source(1, "a = [[1, 2], [3, 4]]\nb = [[10, 20], [30, 40]]")
        nb_runner.run_all()
        assert "sum = [[11, 22], [33, 44]]" in nb_runner.get_output(2)


# matrix/linear algebra with pure Python lists.
@pytest.mark.integration
class TestMatrixOps:
    """Matrix operations implemented with lists."""

    def test_matrix_propagation(self, nb_runner):
        """Matrix computation with upstream change propagation."""
        nb_runner.create_notebook(
            [
                "scale = 2",
                textwrap.dedent("""\
                def scale_matrix(M, s):
                    return [[x * s for x in row] for row in M]
                M = [[1, 2], [3, 4]]
                scaled = scale_matrix(M, scale)
            """),
                "print(f'scaled={scaled}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[[2, 4], [6, 8]]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "scale = 10")
        nb_runner.run_cells([1, 2, 3])
        assert "[[10, 20], [30, 40]]" in nb_runner.get_output(3)
