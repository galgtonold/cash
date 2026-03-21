"""Batch 393: map with multiple iterables and starmap patterns."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestMapMultiIter:
    def test_map_two_lists(self, nb_runner):
        nb_runner.create_notebook([
            "a = [1, 2, 3]\nb = [10, 20, 30]",
            "sums = list(map(lambda x, y: x + y, a, b))\nprint(f'sums={sums}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sums=[11, 22, 33]" in nb_runner.get_output(2)

    def test_map_edit_lists(self, nb_runner):
        nb_runner.create_notebook([
            "xs = [1, 2, 3]\nys = [4, 5, 6]",
            "products = list(map(lambda x, y: x * y, xs, ys))\ntotal = sum(products)\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=32" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "xs = [10, 20]\nys = [3, 4]")
        nb_runner.run_all()
        assert "total=110" in nb_runner.get_output(2)

    def test_map_type_convert(self, nb_runner):
        nb_runner.create_notebook([
            "strings = ['1', '2', '3', '4', '5']",
            "ints = list(map(int, strings))\ntotal = sum(ints)\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=15" in nb_runner.get_output(2)
