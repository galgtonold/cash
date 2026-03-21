"""Batch 353: itertools.chain, product, starmap combinations."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestItertoolsChainProduct:
    def test_chain_flat(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import chain\na = [1, 2]\nb = [3, 4]\nc = [5]",
            "result = list(chain(a, b, c))\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[1, 2, 3, 4, 5]" in nb_runner.get_output(2)

    def test_product_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import product\ncolors = ['red', 'blue']\nsizes = ['S', 'M']",
            "combos = list(product(colors, sizes))\nprint(f'combos={combos}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "('red', 'S')" in nb_runner.get_output(2)
        assert "('blue', 'M')" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "from itertools import product\ncolors = ['green']\nsizes = ['L', 'XL']")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "('green', 'L')" in out
        assert "('green', 'XL')" in out

    def test_starmap(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import starmap\npairs = [(2, 3), (4, 5), (6, 7)]",
            "products = list(starmap(lambda a, b: a * b, pairs))\nprint(f'products={products}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "products=[6, 20, 42]" in nb_runner.get_output(2)
