"""Batch 444: built-in sorted with key functions."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestSortedKeyFunctions:
    def test_sorted_len_key(self, nb_runner):
        nb_runner.create_notebook([
            "words = ['banana', 'pie', 'strawberry', 'kiwi']",
            "by_len = sorted(words, key=len)\nby_last = sorted(words, key=lambda w: w[-1])\nprint(f'by_len={by_len} by_last={by_last}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "by_len=['pie', 'kiwi', 'banana', 'strawberry']" in out

    def test_sorted_multi_key(self, nb_runner):
        nb_runner.create_notebook([
            "data = [('Bob', 85), ('Alice', 90), ('Charlie', 85), ('Alice', 80)]",
            "ordered = sorted(data, key=lambda x: (x[1], x[0]))\nprint(f'ordered={ordered}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "('Alice', 80)" in out

    def test_sorted_edit(self, nb_runner):
        nb_runner.create_notebook([
            "items = [3, 1, 4, 1, 5, 9, 2, 6]",
            "asc = sorted(items)\ndesc = sorted(items, reverse=True)\nprint(f'asc={asc} desc={desc}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "asc=[1, 1, 2, 3, 4, 5, 6, 9]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "items = [10, 30, 20, 50, 40]")
        nb_runner.run_all()
        assert "asc=[10, 20, 30, 40, 50]" in nb_runner.get_output(2)
