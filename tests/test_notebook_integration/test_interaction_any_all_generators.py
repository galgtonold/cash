"""Batch 436: any() and all() with generator expressions."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestAnyAllGenerators:

    def test_any_all_strings(self, nb_runner):
        nb_runner.create_notebook([
            "words = ['hello', 'world', 'foo']",
            "all_alpha = all(w.isalpha() for w in words)\nany_long = any(len(w) > 4 for w in words)\nprint(f'all_alpha={all_alpha} any_long={any_long}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "all_alpha=True" in nb_runner.get_output(2)
        assert "any_long=True" in nb_runner.get_output(2)

    def test_any_all_edit(self, nb_runner):
        nb_runner.create_notebook([
            "values = [1, 2, 3, 4, 5]",
            "result = all(v > 0 for v in values)\nprint(f'all_positive={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "all_positive=True" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "values = [1, -2, 3, 4, 5]")
        nb_runner.run_all()
        assert "all_positive=False" in nb_runner.get_output(2)
