"""Batch 528: walrus operator in while and comprehension."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestWalrusWhileComprehension:
    def test_walrus_in_while(self, nb_runner):
        nb_runner.create_notebook([
            "data = [1, 5, 3, 8, 2, 9]",
            "results = []\ni = 0\nwhile (val := data[i] if i < len(data) else None) is not None:\n    if val > 4:\n        results.append(val)\n    i += 1\nprint(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[5, 8, 9]" in nb_runner.get_output(2)

    def test_walrus_in_comprehension(self, nb_runner):
        nb_runner.create_notebook([
            "nums = [2, 8, 3, 12, 5, 1]",
            "filtered = [(y, n) for n in nums if (y := n * 2) > 6]\nprint(f'filtered={filtered}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "(16, 8)" in nb_runner.get_output(2)

    def test_walrus_edit(self, nb_runner):
        nb_runner.create_notebook([
            "data = [10, 20, 30]",
            "total = 0\nresult = [(total := total + x) for x in data]\nprint(f'result={result} total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[10, 30, 60]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "data = [5, 15, 25]")
        nb_runner.run_all()
        assert "result=[5, 20, 45]" in nb_runner.get_output(2)
