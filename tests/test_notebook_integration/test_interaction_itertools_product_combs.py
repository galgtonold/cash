"""Batch 418: itertools.product and combinations_with_replacement."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestItertoolsProductCombs:
    def test_product(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import product\na = [1, 2]\nb = ['x', 'y']",
            "result = list(product(a, b))\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[(1, 'x'), (1, 'y'), (2, 'x'), (2, 'y')]" in nb_runner.get_output(2)

    def test_combinations_with_replacement(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import combinations_with_replacement\nitems = ['a', 'b', 'c']",
            "result = list(combinations_with_replacement(items, 2))\ncount = len(result)\nprint(f'count={count} first={result[0]} last={result[-1]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "count=6" in out
        assert "first=('a', 'a')" in out
        assert "last=('c', 'c')" in out

    def test_product_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import product\ncolors = ['R', 'G']\nsizes = ['S', 'L']",
            "combos = list(product(colors, sizes))\ncount = len(combos)\nprint(f'count={count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=4" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from itertools import product\ncolors = ['R', 'G', 'B']\nsizes = ['S', 'M', 'L']")
        nb_runner.run_all()
        assert "count=9" in nb_runner.get_output(2)
