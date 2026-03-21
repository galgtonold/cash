"""Batch 364: dictionary view objects and iteration edits."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDictViewIteration:
    def test_dict_views(self, nb_runner):
        nb_runner.create_notebook([
            "d = {'x': 10, 'y': 20, 'z': 30}",
            "keys = sorted(d.keys())\nvals = sorted(d.values())\nitems = sorted(d.items())\nprint(f'keys={keys} vals={vals}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "keys=['x', 'y', 'z']" in out
        assert "vals=[10, 20, 30]" in out

    def test_dict_iteration_edit(self, nb_runner):
        nb_runner.create_notebook([
            "prices = {'apple': 1.5, 'banana': 0.5, 'cherry': 3.0}",
            "expensive = {k: v for k, v in prices.items() if v > 1.0}\nprint(f'expensive={dict(sorted(expensive.items()))}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "expensive={'apple': 1.5, 'cherry': 3.0}" in nb_runner.get_output(2)
        # Edit prices
        nb_runner.set_cell_source(1, "prices = {'apple': 0.5, 'banana': 2.0, 'cherry': 0.3}")
        nb_runner.run_all()
        assert "expensive={'banana': 2.0}" in nb_runner.get_output(2)

    def test_dict_comprehension_filter(self, nb_runner):
        nb_runner.create_notebook([
            "scores = {'Alice': 85, 'Bob': 42, 'Charlie': 91, 'Diana': 38}",
            "passed = {k: v for k, v in scores.items() if v >= 50}\ncount = len(passed)\nprint(f'passed={dict(sorted(passed.items()))} count={count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "passed={'Alice': 85, 'Charlie': 91}" in nb_runner.get_output(2)
        assert "count=2" in nb_runner.get_output(2)
