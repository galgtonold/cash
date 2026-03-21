"""Batch 485: zip_longest and starmap from itertools."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestZipLongestStarmap:
    def test_zip_longest_fill(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import zip_longest",
            "a = [1, 2, 3]\nb = ['x', 'y']\npaired = list(zip_longest(a, b, fillvalue='?'))\nprint(f'paired={paired}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "paired=[(1, 'x'), (2, 'y'), (3, '?')]" in nb_runner.get_output(2)

    def test_starmap_operations(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import starmap",
            "pairs = [(2, 3), (4, 5), (6, 7)]\nprods = list(starmap(lambda a, b: a * b, pairs))\nprint(f'prods={prods} sum={sum(prods)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "prods=[6, 20, 42]" in out
        assert "sum=68" in out

    def test_zip_longest_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import zip_longest",
            "r = list(zip_longest([1], [2, 3], fillvalue=0))\nprint(f'r={r}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r=[(1, 2), (0, 3)]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "r = list(zip_longest([1, 2, 3], [10], fillvalue=-1))\nprint(f'r={r}')")
        nb_runner.run_all()
        assert "r=[(1, 10), (2, -1), (3, -1)]" in nb_runner.get_output(2)
