"""
Interaction test: list comprehension with multiple for-clauses.
Tests nested list comprehensions with multiple iterables,
conditions, and cross-cell flattening patterns.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestMultiForComprehension:
    """Test multi-for comprehensions across cells."""

    def test_multi_for_comp(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: cartesian product via comprehension
            "colors = ['red', 'blue']\nsizes = ['S', 'M', 'L']\ncombos = [(c, s) for c in colors for s in sizes]\nprint(f'combos={combos}')\nprint(f'count={len(combos)}')",
            # Cell 2: filtered cartesian
            "nums1 = range(1, 5)\nnums2 = range(1, 5)\npairs = [(a, b) for a in nums1 for b in nums2 if a < b]\nprint(f'pairs={pairs}')",
            # Cell 3: aggregate
            "total_combos = len(combos)\ntotal_pairs = len(pairs)\nprint(f'combos={total_combos}')\nprint(f'pairs={total_pairs}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "count=6" in out1
        out2 = nb_runner.get_output(2)
        assert "(1, 2)" in out2
        assert "(1, 3)" in out2
        out3 = nb_runner.get_output(3)
        assert "combos=6" in out3
        assert "pairs=6" in out3

    def test_multi_for_edit(self, nb_runner):
        nb_runner.create_notebook([
            "rows = [1, 2, 3]\ncols = ['a', 'b']\ngrid = [(r, c) for r in rows for c in cols]\nprint(f'grid_size={len(grid)}')",
            "first = grid[0]\nlast = grid[-1]\nprint(f'first={first}')\nprint(f'last={last}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "grid_size=6" in nb_runner.get_output(1)
        assert "first=(1, 'a')" in nb_runner.get_output(2)

        # Add more rows
        nb_runner.set_cell_source(1, "rows = [1, 2, 3, 4]\ncols = ['a', 'b']\ngrid = [(r, c) for r in rows for c in cols]\nprint(f'grid_size={len(grid)}')")
        nb_runner.run_cells([1, 2])
        assert "grid_size=8" in nb_runner.get_output(1)
        assert "last=(4, 'b')" in nb_runner.get_output(2)

    def test_multi_for_cache(self, nb_runner):
        nb_runner.create_notebook([
            "matrix = [[i * j for j in range(1, 4)] for i in range(1, 4)]\nprint(f'matrix={matrix}')",
            "flat = [x for row in matrix for x in row]\nprint(f'flat={flat}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "flat=[1, 2, 3, 2, 4, 6, 3, 6, 9]" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "flat=[1, 2, 3, 2, 4, 6, 3, 6, 9]" in nb_runner.get_output(2)
