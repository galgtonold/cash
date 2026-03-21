"""
Batch 322: itertools combinatorial patterns with caching.
Tests combinations, permutations, product, chain, and edit propagation.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.timeout(90)]


class TestItertoolsCombinatorial:
    """Test itertools combinatorial operation caching."""

    def test_combinations_basic(self, nb_runner):
        """itertools.combinations with caching."""
        nb_runner.create_notebook([
            "from itertools import combinations",
            "items = ['a', 'b', 'c', 'd']",
            "combos = list(combinations(items, 2))\ncount = len(combos)",
            "print(f'count={count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "count=6" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "count=6" in out2

    def test_permutations_edit(self, nb_runner):
        """Edit input, verify permutations count changes."""
        nb_runner.create_notebook([
            "from itertools import permutations",
            "items = [1, 2, 3]",
            "perms = list(permutations(items))\ncount = len(perms)",
            "print(f'count={count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "count=6" in out

        nb_runner.set_cell_source(2, "items = [1, 2, 3, 4]")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "count=24" in out2

    def test_chain_flatten(self, nb_runner):
        """itertools.chain to flatten nested lists."""
        nb_runner.create_notebook([
            "from itertools import chain",
            "lists = [[1, 2], [3, 4], [5]]",
            "flat = list(chain.from_iterable(lists))\ntotal = sum(flat)",
            "print(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=15" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "total=15" in out2
