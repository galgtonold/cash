"""Batch 515: map filter reduce functional pipeline."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestMapFilterFunctionalPipeline:
    def test_map_filter_pipeline(self, nb_runner):
        nb_runner.create_notebook([
            "data = list(range(1, 11))",
            "squared = list(map(lambda x: x**2, data))\nevens = list(filter(lambda x: x % 2 == 0, squared))\nprint(f'squared={squared}')\nprint(f'evens={evens}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "squared=[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]" in out
        assert "evens=[4, 16, 36, 64, 100]" in out

    def test_chained_operations(self, nb_runner):
        nb_runner.create_notebook([
            "words = ['Hello', 'WORLD', 'Python', 'CODE']",
            "result = list(map(str.lower, filter(lambda w: len(w) > 4, words)))\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=['hello', 'world', 'python']" in nb_runner.get_output(2)

    def test_pipeline_edit(self, nb_runner):
        nb_runner.create_notebook([
            "nums = [1, 2, 3, 4, 5]",
            "result = list(map(lambda x: x * 10, filter(lambda x: x > 2, nums)))\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[30, 40, 50]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "nums = [10, 20, 30, 40, 50]")
        nb_runner.run_all()
        assert "result=[100, 200, 300, 400, 500]" in nb_runner.get_output(2)
