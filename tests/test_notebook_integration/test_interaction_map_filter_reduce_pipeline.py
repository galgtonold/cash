"""Batch 464: map/filter/reduce pipeline composition."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestMapFilterReducePipeline:
    def test_pipeline(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import reduce\ndata = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
            "result = reduce(lambda a, b: a + b, filter(lambda x: x % 2 == 0, map(lambda x: x ** 2, data)))\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # squares of evens: 4+16+36+64+100 = 220
        assert "result=220" in nb_runner.get_output(2)

    def test_pipeline_strings(self, nb_runner):
        nb_runner.create_notebook([
            "words = ['  Hello ', ' WORLD', 'Python  ', '  foo  ']",
            "cleaned = list(map(str.strip, map(str.lower, words)))\nprint(f'cleaned={cleaned}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cleaned=['hello', 'world', 'python', 'foo']" in nb_runner.get_output(2)

    def test_pipeline_edit(self, nb_runner):
        nb_runner.create_notebook([
            "nums = [10, 20, 30, 40, 50]",
            "doubled_big = list(filter(lambda x: x > 50, map(lambda x: x * 2, nums)))\nprint(f'result={doubled_big}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[60, 80, 100]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "nums = [5, 15, 25, 35]")
        nb_runner.run_all()
        assert "result=[70]" in nb_runner.get_output(2)
