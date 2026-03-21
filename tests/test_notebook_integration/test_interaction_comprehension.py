"""
Batch 301: Comprehension variants interaction tests.
Tests dict comprehension, set comprehension, and nested comprehension
patterns with cache invalidation.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestComprehensionVariantsInteraction:
    """Test various comprehension patterns with cache invalidation."""

    def test_dict_comprehension_edit(self, nb_runner):
        """Editing data used in dict comprehension should propagate."""
        nb_runner.create_notebook([
            "names = ['alice', 'bob', 'charlie']",
            "name_lengths = {n: len(n) for n in names}",
            "result = sorted(name_lengths.items())",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "alice" in out
        assert "bob" in out

        nb_runner.set_cell_source(1, "names = ['x', 'hello', 'world']")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "x" in out
        assert "hello" in out
        assert "world" in out

    def test_set_comprehension_edit(self, nb_runner):
        """Editing data used in set comprehension should propagate."""
        nb_runner.create_notebook([
            "nums = [1, 2, 2, 3, 3, 3, 4]",
            "unique_doubled = {x * 2 for x in nums}",
            "result = sorted(unique_doubled)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=[2, 4, 6, 8]" in out

        nb_runner.set_cell_source(1, "nums = [5, 5, 10, 10, 15]")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=[10, 20, 30]" in out

    def test_nested_comprehension_edit(self, nb_runner):
        """Editing data in nested comprehension should propagate."""
        nb_runner.create_notebook([
            "matrix = [[1, 2], [3, 4], [5, 6]]",
            "flat = [x for row in matrix for x in row]",
            "total = sum(flat)",
            "print(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=21" in out

        nb_runner.set_cell_source(1, "matrix = [[10, 20], [30, 40]]")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=100" in out
