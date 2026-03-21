"""Batch 476: dict pipe merge and walrus loop filtering."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDictPipeMergeWalrusLoop:
    def test_dict_pipe_merge(self, nb_runner):
        nb_runner.create_notebook([
            "defaults = {'color': 'blue', 'size': 10, 'font': 'Arial'}",
            "overrides = {'size': 20, 'weight': 'bold'}\nmerged = defaults | overrides\nprint(f'color={merged[\"color\"]} size={merged[\"size\"]} weight={merged[\"weight\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "color=blue" in out
        assert "size=20" in out
        assert "weight=bold" in out

    def test_walrus_filter(self, nb_runner):
        nb_runner.create_notebook([
            "data = [5, 3, 8, 1, 9, 2]",
            "it = iter(data)\nresults = []\nwhile (val := next(it, None)) is not None:\n    if val > 4:\n        results.append(val)\nprint(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[5, 8, 9]" in nb_runner.get_output(2)

    def test_pipe_merge_edit(self, nb_runner):
        nb_runner.create_notebook([
            "a = {'x': 1}",
            "b = {'y': 2}\nc = a | b\nprint(f'keys={sorted(c.keys())}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys=['x', 'y']" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "a = {'x': 1, 'z': 3}")
        nb_runner.run_all()
        assert "keys=['x', 'y', 'z']" in nb_runner.get_output(2)
