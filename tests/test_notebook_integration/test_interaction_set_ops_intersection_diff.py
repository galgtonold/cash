"""Batch 500: set operations intersection difference symmetric."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestSetOpsIntersectionDiff:
    def test_set_operations(self, nb_runner):
        nb_runner.create_notebook([
            "a = {1, 2, 3, 4, 5}\nb = {4, 5, 6, 7, 8}",
            "inter = sorted(a & b)\nunion = sorted(a | b)\ndiff = sorted(a - b)\nsym = sorted(a ^ b)\nprint(f'inter={inter} union={union}')\nprint(f'diff={diff} sym={sym}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "inter=[4, 5]" in out
        assert "union=[1, 2, 3, 4, 5, 6, 7, 8]" in out
        assert "diff=[1, 2, 3]" in out
        assert "sym=[1, 2, 3, 6, 7, 8]" in out

    def test_set_issubset(self, nb_runner):
        nb_runner.create_notebook([
            "small = {1, 2}\nbig = {1, 2, 3, 4}",
            "sub = small.issubset(big)\nsup = big.issuperset(small)\ndisj = small.isdisjoint({5, 6})\nprint(f'sub={sub} sup={sup} disj={disj}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "sub=True" in out
        assert "sup=True" in out
        assert "disj=True" in out

    def test_set_edit(self, nb_runner):
        nb_runner.create_notebook([
            "s = {1, 2, 3}",
            "result = sorted(s & {2, 3, 4})\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[2, 3]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "s = {10, 20, 30}")
        nb_runner.run_all()
        assert "result=[]" in nb_runner.get_output(2)
