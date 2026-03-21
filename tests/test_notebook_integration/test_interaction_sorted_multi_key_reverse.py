"""Batch 526: sorted with multiple keys and reverse."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestSortedMultiKeyReverse:
    def test_multi_key_sort(self, nb_runner):
        nb_runner.create_notebook([
            "data = [('Alice', 30), ('Bob', 25), ('Carol', 30), ('Dave', 25)]",
            "by_age_name = sorted(data, key=lambda x: (x[1], x[0]))\nnames = [n for n, a in by_age_name]\nprint(f'names={names}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "names=['Bob', 'Dave', 'Alice', 'Carol']" in nb_runner.get_output(2)

    def test_reverse_sort(self, nb_runner):
        nb_runner.create_notebook([
            "nums = [5, 2, 8, 1, 9, 3]",
            "asc = sorted(nums)\ndesc = sorted(nums, reverse=True)\nprint(f'asc={asc} desc={desc}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "asc=[1, 2, 3, 5, 8, 9]" in out
        assert "desc=[9, 8, 5, 3, 2, 1]" in out

    def test_sort_edit(self, nb_runner):
        nb_runner.create_notebook([
            "items = ['banana', 'apple', 'cherry']",
            "result = sorted(items)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=['apple', 'banana', 'cherry']" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "items = ['zebra', 'mango', 'fig']")
        nb_runner.run_all()
        assert "result=['fig', 'mango', 'zebra']" in nb_runner.get_output(2)
