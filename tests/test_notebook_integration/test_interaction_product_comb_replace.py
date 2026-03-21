"""
Interaction test: itertools product and combinations_with_replacement.
Tests cartesian products, combinations with replacement,
and cross-cell combinatorial analysis.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestProductCombReplace:
    """Test itertools product and combinations_with_replacement across cells."""

    def test_product_comb_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: product
            "from itertools import product, combinations_with_replacement\ncolors = ['R', 'G', 'B']\nsizes = ['S', 'M', 'L']\ncombos = list(product(colors, sizes))\nprint(f'product_count={len(combos)}')\nprint(f'first={combos[0]}')\nprint(f'last={combos[-1]}')",
            # Cell 2: combinations with replacement
            "coins = [1, 5, 10]\nways = list(combinations_with_replacement(coins, 2))\nprint(f'ways_count={len(ways)}')\nfor w in ways:\n    print(f'pair={w} sum={sum(w)}')",
            # Cell 3: filter products
            "matching = [(c, s) for c, s in combos if c == 'R' or s == 'L']\nprint(f'matching_count={len(matching)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "product_count=9" in out1
        assert "first=('R', 'S')" in out1
        assert "last=('B', 'L')" in out1
        out2 = nb_runner.get_output(2)
        assert "ways_count=6" in out2
        out3 = nb_runner.get_output(3)
        assert "matching_count=5" in out3

    def test_product_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import product\ndice = [1, 2, 3, 4, 5, 6]\nrolls = list(product(dice, repeat=2))\ntotal = len(rolls)\nprint(f'total={total}')",
            "sevens = [r for r in rolls if sum(r) == 7]\nprint(f'sevens={len(sevens)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=36" in nb_runner.get_output(1)
        assert "sevens=6" in nb_runner.get_output(2)

        # Edit to 3 dice
        nb_runner.set_cell_source(1, "from itertools import product\ndice = [1, 2, 3, 4, 5, 6]\nrolls = list(product(dice, repeat=3))\ntotal = len(rolls)\nprint(f'total={total}')")
        nb_runner.set_cell_source(2, "sevens = [r for r in rolls if sum(r) == 7]  # 3-dice\nprint(f'sevens={len(sevens)}')")
        nb_runner.run_cells([1, 2])
        assert "total=216" in nb_runner.get_output(1)
        assert "sevens=15" in nb_runner.get_output(2)

    def test_product_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import combinations_with_replacement\nitems = ['a', 'b', 'c']\npairs = list(combinations_with_replacement(items, 2))\nprint(f'count={len(pairs)}')",
            "as_strings = ['+'.join(p) for p in pairs]\nprint(f'strings={as_strings}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=6" in nb_runner.get_output(1)

        # Re-run - cache
        nb_runner.run_all()
        assert "count=6" in nb_runner.get_output(1)
