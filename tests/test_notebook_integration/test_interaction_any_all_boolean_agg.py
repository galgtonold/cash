"""Batch 520: any all and boolean aggregate checks."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestAnyAllBooleanAgg:
    def test_any_all_basic(self, nb_runner):
        nb_runner.create_notebook([
            "data = [2, 4, 6, 8, 10]",
            "all_even = all(x % 2 == 0 for x in data)\nany_gt5 = any(x > 5 for x in data)\nany_neg = any(x < 0 for x in data)\nprint(f'all_even={all_even} any_gt5={any_gt5} any_neg={any_neg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "all_even=True" in out
        assert "any_gt5=True" in out
        assert "any_neg=False" in out

    def test_any_all_strings(self, nb_runner):
        nb_runner.create_notebook([
            "words = ['hello', 'world', 'python']",
            "all_lower = all(w.islower() for w in words)\nany_long = any(len(w) > 5 for w in words)\nall_alpha = all(w.isalpha() for w in words)\nprint(f'all_lower={all_lower} any_long={any_long} all_alpha={all_alpha}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "all_lower=True" in out
        assert "any_long=True" in out
        assert "all_alpha=True" in out

    def test_any_all_edit(self, nb_runner):
        nb_runner.create_notebook([
            "nums = [1, 3, 5]",
            "result = all(n % 2 == 1 for n in nums)\nprint(f'all_odd={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "all_odd=True" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "nums = [1, 2, 5]")
        nb_runner.run_all()
        assert "all_odd=False" in nb_runner.get_output(2)
