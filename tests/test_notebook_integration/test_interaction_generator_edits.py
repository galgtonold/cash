"""Batch 156 – Generator, iterator, and lazy evaluation interaction tests.

Tests where generators and iterators are created in cells,
consumed downstream, and cell edits affect the generation logic.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(45)]


class TestGeneratorEdits:
    """Generator function edits."""

    def test_edit_generator_function(self, nb_runner):
        """Edit generator function, verify consumer updates."""
        nb_runner.create_notebook([
            "def gen_range(n):\n    for i in range(n):\n        yield i * 2",
            "result = list(gen_range(5))\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [0, 2, 4, 6, 8]" in nb_runner.get_output(2)

        # Edit generator to yield squares
        nb_runner.set_cell_source(
            1, "def gen_range(n):\n    for i in range(n):\n        yield i ** 2"
        )
        nb_runner.run_all()
        assert "result = [0, 1, 4, 9, 16]" in nb_runner.get_output(2)

    def test_edit_generator_param(self, nb_runner):
        """Edit parameter passed to generator."""
        nb_runner.create_notebook([
            "count = 3",
            "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b",
            "fibs = list(fib(count))\nprint(f'fibs = {fibs}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "fibs = [0, 1, 1]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "count = 8")
        nb_runner.run_all()
        assert "fibs = [0, 1, 1, 2, 3, 5, 8, 13]" in nb_runner.get_output(3)


class TestMapFilterEdits:
    """Map/filter patterns with edits."""

    def test_edit_map_function(self, nb_runner):
        """Edit function used in map."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5]",
            "mapped = list(map(lambda x: x * 2, data))",
            "total = sum(mapped)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 30" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2, "mapped = list(map(lambda x: x ** 2, data))"
        )
        nb_runner.run_all()
        assert "total = 55" in nb_runner.get_output(3)

    def test_edit_filter_predicate(self, nb_runner):
        """Edit filter predicate."""
        nb_runner.create_notebook([
            "nums = list(range(20))",
            "filtered = list(filter(lambda x: x % 2 == 0, nums))",
            "count = len(filtered)\nprint(f'count = {count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 10" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2, "filtered = list(filter(lambda x: x % 5 == 0, nums))"
        )
        nb_runner.run_all()
        assert "count = 4" in nb_runner.get_output(3)

    def test_chain_map_filter_edit(self, nb_runner):
        """Chain map then filter, edit map."""
        nb_runner.create_notebook([
            "raw = list(range(1, 11))",
            "doubled = [x * 2 for x in raw]",
            "big = [x for x in doubled if x > 10]",
            "total = sum(big)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # doubled = [2,4,6,8,10,12,14,16,18,20], big = [12,14,16,18,20] -> 80
        assert "total = 80" in nb_runner.get_output(4)

        # Change map to triple
        nb_runner.set_cell_source(2, "doubled = [x * 3 for x in raw]")
        nb_runner.run_all()
        # tripled = [3,6,9,12,15,18,21,24,27,30], big = [12,15,18,21,24,27,30] -> 147
        assert "total = 147" in nb_runner.get_output(4)
