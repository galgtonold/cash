"""Batch 347: multiple return values across cells with edits."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestMultiReturnCrossCell:
    def test_function_multi_return(self, nb_runner):
        nb_runner.create_notebook([
            "def stats(data):\n    return min(data), max(data), sum(data) / len(data)",
            "data = [10, 20, 30, 40, 50]",
            "lo, hi, avg = stats(data)\nprint(f'lo={lo} hi={hi} avg={avg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lo=10 hi=50 avg=30.0" in nb_runner.get_output(3)

    def test_multi_return_edit_data(self, nb_runner):
        nb_runner.create_notebook([
            "def analyze(nums):\n    evens = [n for n in nums if n % 2 == 0]\n    odds = [n for n in nums if n % 2 != 0]\n    return evens, odds",
            "nums = [1, 2, 3, 4, 5, 6]",
            "evens, odds = analyze(nums)\nprint(f'evens={evens} odds={odds}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "evens=[2, 4, 6]" in nb_runner.get_output(3)
        assert "odds=[1, 3, 5]" in nb_runner.get_output(3)
        # Edit data
        nb_runner.set_cell_source(2, "nums = [10, 15, 20, 25]")
        nb_runner.run_all()
        assert "evens=[10, 20]" in nb_runner.get_output(3)
        assert "odds=[15, 25]" in nb_runner.get_output(3)

    def test_multi_return_edit_function(self, nb_runner):
        nb_runner.create_notebook([
            "def transform(x):\n    return x * 2, x + 10",
            "a, b = transform(5)\nprint(f'a={a} b={b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=10 b=15" in nb_runner.get_output(2)
        # Edit function
        nb_runner.set_cell_source(1, "def transform(x):\n    return x * 3, x + 100")
        nb_runner.run_all()
        assert "a=15 b=105" in nb_runner.get_output(2)
