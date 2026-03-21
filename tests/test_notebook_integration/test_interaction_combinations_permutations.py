"""Batch 524: itertools combinations permutations."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestCombinationsPermutations:
    def test_combinations(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import combinations",
            "items = ['A', 'B', 'C', 'D']\ncomb2 = list(combinations(items, 2))\ncomb3 = list(combinations(items, 3))\nprint(f'comb2_count={len(comb2)} comb3_count={len(comb3)}')\nprint(f'comb2={comb2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "comb2_count=6" in out
        assert "comb3_count=4" in out

    def test_permutations(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import permutations",
            "items = [1, 2, 3]\nperms = list(permutations(items))\nprint(f'count={len(perms)} first={perms[0]} last={perms[-1]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "count=6" in out
        assert "first=(1, 2, 3)" in out
        assert "last=(3, 2, 1)" in out

    def test_combinations_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import combinations",
            "result = list(combinations([1, 2, 3], 2))\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[(1, 2), (1, 3), (2, 3)]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "result = list(combinations([1, 2, 3, 4], 2))\nprint(f'count={len(result)}')")
        nb_runner.run_all()
        assert "count=6" in nb_runner.get_output(2)
