"""Batch 368: any/all/filter/map builtins with lambdas and edits."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestBuiltinHigherOrder:
    def test_any_all(self, nb_runner):
        nb_runner.create_notebook([
            "data = [2, 4, 6, 8, 10]",
            "all_even = all(x % 2 == 0 for x in data)\nany_gt5 = any(x > 5 for x in data)\nprint(f'all_even={all_even} any_gt5={any_gt5}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "all_even=True any_gt5=True" in nb_runner.get_output(2)

    def test_filter_map_edit(self, nb_runner):
        nb_runner.create_notebook([
            "numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
            "evens = list(filter(lambda x: x % 2 == 0, numbers))\nsquared = list(map(lambda x: x ** 2, evens))\nprint(f'squared={squared}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "squared=[4, 16, 36, 64, 100]" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "numbers = [10, 15, 20, 25, 30]")
        nb_runner.run_all()
        assert "squared=[100, 400, 900]" in nb_runner.get_output(2)

    def test_sorted_key_lambda(self, nb_runner):
        nb_runner.create_notebook([
            "items = [('b', 2), ('a', 3), ('c', 1)]",
            "by_val = sorted(items, key=lambda x: x[1])\nby_name = sorted(items, key=lambda x: x[0])\nprint(f'by_val={by_val}')\nprint(f'by_name={by_name}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "by_val=[('c', 1), ('b', 2), ('a', 3)]" in out
        assert "by_name=[('a', 3), ('b', 2), ('c', 1)]" in out
