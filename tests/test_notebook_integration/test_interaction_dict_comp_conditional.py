"""Batch 507: dict comprehension with conditional logic."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDictComprehensionConditional:
    def test_dict_comp_filter(self, nb_runner):
        nb_runner.create_notebook([
            "scores = {'Alice': 95, 'Bob': 67, 'Carol': 82, 'Dave': 45, 'Eve': 91}",
            "passing = {k: v for k, v in scores.items() if v >= 70}\nfailing = {k: v for k, v in scores.items() if v < 70}\nprint(f'passing={sorted(passing.keys())} failing={sorted(failing.keys())}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "passing=['Alice', 'Carol', 'Eve']" in out
        assert "failing=['Bob', 'Dave']" in out

    def test_dict_comp_transform(self, nb_runner):
        nb_runner.create_notebook([
            "words = ['hello', 'world', 'python', 'code']",
            "lengths = {w: len(w) for w in words}\nuppered = {w: w.upper() for w in words}\nprint(f'lengths={lengths}')\nprint(f'uppered={uppered}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'hello': 5" in out
        assert "'HELLO'" in out

    def test_dict_comp_edit(self, nb_runner):
        nb_runner.create_notebook([
            "nums = [1, 2, 3]",
            "d = {n: n**2 for n in nums}\nprint(f'd={d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d={1: 1, 2: 4, 3: 9}" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "nums = [5, 10, 15]")
        nb_runner.run_all()
        assert "d={5: 25, 10: 100, 15: 225}" in nb_runner.get_output(2)
