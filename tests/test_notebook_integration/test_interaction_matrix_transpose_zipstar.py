"""Batch 451: matrix transpose and zip star pattern."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestMatrixTransposeZipStar:
    def test_transpose(self, nb_runner):
        nb_runner.create_notebook([
            "matrix = [[1, 2, 3], [4, 5, 6]]",
            "transposed = list(map(list, zip(*matrix)))\nprint(f'transposed={transposed}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "transposed=[[1, 4], [2, 5], [3, 6]]" in nb_runner.get_output(2)

    def test_unzip(self, nb_runner):
        nb_runner.create_notebook([
            "pairs = [('a', 1), ('b', 2), ('c', 3)]",
            "keys, vals = zip(*pairs)\nprint(f'keys={list(keys)} vals={list(vals)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys=['a', 'b', 'c']" in nb_runner.get_output(2)
        assert "vals=[1, 2, 3]" in nb_runner.get_output(2)

    def test_transpose_edit(self, nb_runner):
        nb_runner.create_notebook([
            "m = [[1, 2], [3, 4], [5, 6]]",
            "t = [list(row) for row in zip(*m)]\nrows = len(t)\ncols = len(t[0])\nprint(f'rows={rows} cols={cols}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "rows=2" in nb_runner.get_output(2)
        assert "cols=3" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "m = [[1, 2, 3, 4], [5, 6, 7, 8]]")
        nb_runner.run_all()
        assert "rows=4" in nb_runner.get_output(2)
        assert "cols=2" in nb_runner.get_output(2)
