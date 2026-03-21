"""Batch 405: zip with unequal lengths and zip_longest."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestZipLongestPatterns:
    def test_zip_longest_fill(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import zip_longest\na = [1, 2, 3]\nb = ['x', 'y']",
            "pairs = list(zip_longest(a, b, fillvalue='NA'))\nprint(f'pairs={pairs}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "pairs=[(1, 'x'), (2, 'y'), (3, 'NA')]" in nb_runner.get_output(2)

    def test_zip_strict_truncate(self, nb_runner):
        nb_runner.create_notebook([
            "names = ['Alice', 'Bob', 'Charlie']\nscores = [90, 85]",
            "paired = list(zip(names, scores))\ncount = len(paired)\nprint(f'paired={paired} count={count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "count=2" in out
        assert "('Alice', 90)" in out

    def test_zip_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import zip_longest\nk = ['a', 'b']\nv = [1, 2, 3]",
            "result = dict(zip_longest(k, v, fillvalue='?'))\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 1" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from itertools import zip_longest\nk = ['x', 'y', 'z']\nv = [10, 20]")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'x': 10" in out
        assert "'z': '?'" in out
