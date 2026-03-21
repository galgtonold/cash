"""
Interaction test: zip_longest with fillvalue and multi-iterator.
Tests itertools.zip_longest with custom fillvalue,
multiple iterables of different lengths, and cross-cell processing.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestZipLongestFillvalue:
    """Test zip_longest with fillvalue across cells."""

    def test_zip_longest_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: zip_longest with fillvalue
            "from itertools import zip_longest\nnames = ['Alice', 'Bob', 'Charlie']\nscores = [95, 87]\ngrades = ['A', 'B', 'C', 'D']\ncombined = list(zip_longest(names, scores, grades, fillvalue='N/A'))\nprint(f'count={len(combined)}')\nfor name, score, grade in combined:\n    print(f'{name}:{score}:{grade}')",
            # Cell 2: process combined data
            "valid = [(n, s, g) for n, s, g in combined if s != 'N/A' and g != 'N/A']\nprint(f'valid_count={len(valid)}')\nprint(f'first_valid={valid[0]}')",
            # Cell 3: transform
            "result_dict = {n: {'score': s, 'grade': g} for n, s, g in combined if n != 'N/A'}\nprint(f'entries={len(result_dict)}')\nprint(f'alice_score={result_dict[\"Alice\"][\"score\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "count=4" in out1
        assert "Alice:95:A" in out1
        out2 = nb_runner.get_output(2)
        assert "valid_count=2" in out2
        out3 = nb_runner.get_output(3)
        assert "entries=3" in out3
        assert "alice_score=95" in out3

    def test_zip_longest_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import zip_longest\nkeys = ['a', 'b', 'c']\nvals = [1, 2]\npairs = dict(zip_longest(keys, vals, fillvalue=0))\nprint(f'pairs={pairs}')",
            "total = sum(pairs.values())\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=3" in nb_runner.get_output(2)

        # Edit to add more values
        nb_runner.set_cell_source(1, "from itertools import zip_longest\nkeys = ['a', 'b', 'c', 'd']\nvals = [1, 2, 3]\npairs = dict(zip_longest(keys, vals, fillvalue=0))\nprint(f'pairs={pairs}')")
        nb_runner.run_cells([1, 2])
        assert "total=6" in nb_runner.get_output(2)

    def test_zip_longest_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import zip_longest\ncols = ['x', 'y']\nrow1 = [1, 2]\nrow2 = [3]\nmatrix = [dict(zip_longest(cols, r, fillvalue=0)) for r in [row1, row2]]\nprint(f'rows={len(matrix)}')",
            "x_sum = sum(row['x'] for row in matrix)\ny_sum = sum(row['y'] for row in matrix)\nprint(f'x_sum={x_sum}')\nprint(f'y_sum={y_sum}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x_sum=4" in nb_runner.get_output(2)
        assert "y_sum=2" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "x_sum=4" in nb_runner.get_output(2)
