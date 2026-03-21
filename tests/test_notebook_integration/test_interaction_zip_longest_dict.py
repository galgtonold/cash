"""
Interaction test: zip_longest with fillvalue and dict construction.
Tests zip_longest for unequal iterables, fillvalue parameter,
dict construction from zipped pairs, and cross-cell data alignment.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestZipLongestDict:
    """Test zip_longest with dict construction across cells."""

    def test_zip_longest_fillvalue(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: zip_longest with fill
            "from itertools import zip_longest\nkeys = ['a', 'b', 'c', 'd', 'e']\nvals = [1, 2, 3]\npairs = list(zip_longest(keys, vals, fillvalue=0))\nprint(f'pairs={pairs}')",
            # Cell 2: build dict
            "d = dict(pairs)\nprint(f'dict={d}')\nprint(f'filled={sum(1 for v in d.values() if v == 0)}')",
            # Cell 3: aggregate
            "total = sum(d.values())\nnon_zero = {k: v for k, v in d.items() if v != 0}\nprint(f'total={total}')\nprint(f'non_zero={non_zero}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "('a', 1)" in out1
        assert "('d', 0)" in out1
        out2 = nb_runner.get_output(2)
        assert "filled=2" in out2
        out3 = nb_runner.get_output(3)
        assert "total=6" in out3
        assert "'a': 1" in out3

    def test_zip_longest_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import zip_longest\nnames = ['Alice', 'Bob', 'Charlie']\nages = [30, 25]\npaired = list(zip_longest(names, ages, fillvalue='N/A'))\nprint(f'count={len(paired)}')",
            "result = {n: a for n, a in paired}\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=3" in nb_runner.get_output(1)
        assert "'Charlie': 'N/A'" in nb_runner.get_output(2)

        # Add more ages
        nb_runner.set_cell_source(1, "from itertools import zip_longest\nnames = ['Alice', 'Bob', 'Charlie']\nages = [30, 25, 35, 40]\npaired = list(zip_longest(names, ages, fillvalue='Unknown'))\nprint(f'count={len(paired)}')")
        nb_runner.run_cells([1, 2])
        assert "count=4" in nb_runner.get_output(1)
        assert "'Unknown': 40" in nb_runner.get_output(2)

    def test_zip_longest_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import zip_longest\ncols = ['x', 'y', 'z']\nrow1 = [1, 2]\nrow2 = [4, 5, 6]\naligned = [dict(zip_longest(cols, r, fillvalue=0)) for r in [row1, row2]]\nprint(f'rows={aligned}')",
            "z_vals = [r['z'] for r in aligned]\nprint(f'z_vals={z_vals}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "'z': 0" in out1
        assert "'z': 6" in out1
        assert "z_vals=[0, 6]" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "z_vals=[0, 6]" in nb_runner.get_output(2)
