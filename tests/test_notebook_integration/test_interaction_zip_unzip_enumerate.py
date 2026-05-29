"""Batch 511: zip unzip and enumerate patterns."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestZipUnzipEnumerate:
    def test_zip_and_unzip(self, nb_runner):
        nb_runner.create_notebook([
            "names = ['Alice', 'Bob', 'Carol']\nages = [30, 25, 35]",
            "paired = list(zip(names, ages))\nun_names, un_ages = zip(*paired)\nprint(f'paired={paired}')\nprint(f'names={list(un_names)} ages={list(un_ages)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "('Alice', 30)" in out
        assert "names=['Alice', 'Bob', 'Carol']" in out


    def test_zip_edit(self, nb_runner):
        nb_runner.create_notebook([
            "a = [1, 2]\nb = ['x', 'y']",
            "result = list(zip(a, b))\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[(1, 'x'), (2, 'y')]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "a = [10, 20, 30]\nb = ['p', 'q', 'r']")
        nb_runner.run_all()
        assert "result=[(10, 'p'), (20, 'q'), (30, 'r')]" in nb_runner.get_output(2)
