"""Batch 450: filter with None and lambda predicates."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestFilterPredicates:
    def test_filter_none(self, nb_runner):
        nb_runner.create_notebook([
            "data = [0, 1, '', 'hello', None, False, 42, [], [1]]",
            "truthy = list(filter(None, data))\nprint(f'truthy={truthy}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "truthy=[1, 'hello', 42, [1]]" in nb_runner.get_output(2)

    def test_filter_lambda(self, nb_runner):
        nb_runner.create_notebook([
            "nums = list(range(1, 21))",
            "evens = list(filter(lambda x: x % 2 == 0, nums))\nprint(f'evens={evens}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "evens=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20]" in nb_runner.get_output(2)

    def test_filter_edit(self, nb_runner):
        nb_runner.create_notebook([
            "words = ['apple', 'banana', 'cherry', 'date', 'elderberry']",
            "long_words = list(filter(lambda w: len(w) > 5, words))\nprint(f'long={long_words}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "long=['banana', 'cherry', 'elderberry']" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "words = ['fig', 'grape', 'honeydew', 'kiwi']")
        nb_runner.run_all()
        assert "long=['honeydew']" in nb_runner.get_output(2)
