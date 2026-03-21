"""Batch 361: zip_longest, pairwise, and batched iteration patterns."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestZipLongestPairwise:
    def test_zip_longest(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import zip_longest\na = [1, 2, 3]\nb = ['x', 'y']",
            "result = list(zip_longest(a, b, fillvalue='?'))\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[(1, 'x'), (2, 'y'), (3, '?')]" in nb_runner.get_output(2)

    def test_zip_longest_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import zip_longest\nnames = ['Alice', 'Bob']\nscores = [90, 85, 78]",
            "paired = list(zip_longest(names, scores, fillvalue='N/A'))\nprint(f'paired={paired}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "('N/A', 78)" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "from itertools import zip_longest\nnames = ['X', 'Y', 'Z']\nscores = [10]")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "('X', 10)" in out
        assert "('Y', 'N/A')" in out

    def test_batched_manual(self, nb_runner):
        nb_runner.create_notebook([
            "def batched(iterable, n):\n    from itertools import islice\n    it = iter(iterable)\n    while batch := list(islice(it, n)):\n        yield tuple(batch)",
            "data = list(range(10))\nresult = list(batched(data, 3))\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "(0, 1, 2)" in nb_runner.get_output(2)
        assert "(9,)" in nb_runner.get_output(2)
