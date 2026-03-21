"""Batch 503: itertools groupby with sorted data."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestItertoolsGroupbySorted:
    def test_groupby_basic(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import groupby",
            "data = [('A', 1), ('A', 2), ('B', 3), ('B', 4), ('C', 5)]\ngroups = {k: [v for _, v in g] for k, g in groupby(data, key=lambda x: x[0])}\nprint(f'groups={groups}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'A': [1, 2]" in out
        assert "'B': [3, 4]" in out
        assert "'C': [5]" in out

    def test_groupby_words(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import groupby",
            "words = sorted(['apple', 'ant', 'banana', 'bat', 'cherry'])\ngroups = {k: list(g) for k, g in groupby(words, key=lambda w: w[0])}\ncounts = {k: len(v) for k, v in groups.items()}\nprint(f'counts={counts}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'a': 2" in out
        assert "'b': 2" in out
        assert "'c': 1" in out

    def test_groupby_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from itertools import groupby",
            "nums = [1, 1, 2, 2, 2, 3]\ngroups = [(k, len(list(g))) for k, g in groupby(nums)]\nprint(f'groups={groups}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "groups=[(1, 2), (2, 3), (3, 1)]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "nums = [1, 1, 1, 2, 3, 3]\ngroups = [(k, len(list(g))) for k, g in groupby(nums)]\nprint(f'groups={groups}')")
        nb_runner.run_all()
        assert "groups=[(1, 3), (2, 1), (3, 2)]" in nb_runner.get_output(2)
