"""Batch 339: chain of comprehensions and transformations across cells."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestComprehensionChain:
    def test_chained_comprehensions(self, nb_runner):
        nb_runner.create_notebook([
            "raw = list(range(20))",
            "evens = [x for x in raw if x % 2 == 0]",
            "squared = [x**2 for x in evens]",
            "result = {x: 'big' if x > 50 else 'small' for x in squared}\nprint(f'result={sorted(result.items())}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=" in nb_runner.get_output(4)
        assert "(64, 'big')" in nb_runner.get_output(4)

    def test_chained_comprehension_edit(self, nb_runner):
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5]",
            "filtered = [x for x in data if x > 2]",
            "doubled = [x * 2 for x in filtered]\nprint(f'doubled={doubled}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "doubled=[6, 8, 10]" in nb_runner.get_output(3)
        # Edit filter threshold
        nb_runner.set_cell_source(2, "filtered = [x for x in data if x > 3]")
        nb_runner.run_all()
        assert "doubled=[8, 10]" in nb_runner.get_output(3)

    def test_nested_dict_comprehension(self, nb_runner):
        nb_runner.create_notebook([
            "keys = ['a', 'b', 'c']\nvals = [1, 2, 3]",
            "mapping = {k: v * 10 for k, v in zip(keys, vals)}\nprint(f'mapping={mapping}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mapping={'a': 10, 'b': 20, 'c': 30}" in nb_runner.get_output(2)
