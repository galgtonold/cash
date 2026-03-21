"""Batch 170 – Generator and iterator interaction tests.

Tests editing generator functions, iterator protocols,
and lazy evaluation patterns.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestGeneratorEdits:
    """Editing generator function definitions."""

    def test_edit_generator_yield(self, nb_runner):
        """Edit what a generator yields."""
        nb_runner.create_notebook([
            "def gen_nums(n):\n    for i in range(n):\n        yield i",
            "result = list(gen_nums(5))\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [0, 1, 2, 3, 4]" in nb_runner.get_output(2)

        # Change to yield squares
        nb_runner.set_cell_source(
            1,
            "def gen_nums(n):\n    for i in range(n):\n        yield i ** 2",
        )
        nb_runner.run_all()
        assert "result = [0, 1, 4, 9, 16]" in nb_runner.get_output(2)

    def test_edit_generator_filter(self, nb_runner):
        """Edit the filter condition in a generator."""
        nb_runner.create_notebook([
            "def even_gen(n):\n    for i in range(n):\n        if i % 2 == 0:\n            yield i",
            "result = list(even_gen(10))\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [0, 2, 4, 6, 8]" in nb_runner.get_output(2)

        # Change to odd
        nb_runner.set_cell_source(
            1,
            "def even_gen(n):\n    for i in range(n):\n        if i % 2 == 1:\n            yield i",
        )
        nb_runner.run_all()
        assert "result = [1, 3, 5, 7, 9]" in nb_runner.get_output(2)

    def test_edit_generator_range(self, nb_runner):
        """Edit the range of a generator call."""
        nb_runner.create_notebook([
            "def countdown(n):\n    while n > 0:\n        yield n\n        n -= 1",
            "result = list(countdown(3))\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [3, 2, 1]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            2, "result = list(countdown(6))\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = [6, 5, 4, 3, 2, 1]" in nb_runner.get_output(2)


class TestIteratorProtocol:
    """Iterator protocol with edits."""

    def test_edit_iterator_class(self, nb_runner):
        """Edit an iterator class __next__ method."""
        nb_runner.create_notebook([
            "class Counter:\n    def __init__(self, n):\n        self.n = n\n        self.i = 0\n    def __iter__(self):\n        return self\n    def __next__(self):\n        if self.i >= self.n:\n            raise StopIteration\n        val = self.i\n        self.i += 1\n        return val",
            "result = list(Counter(4))\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [0, 1, 2, 3]" in nb_runner.get_output(2)

        # Change to yield squares
        nb_runner.set_cell_source(
            1,
            "class Counter:\n    def __init__(self, n):\n        self.n = n\n        self.i = 0\n    def __iter__(self):\n        return self\n    def __next__(self):\n        if self.i >= self.n:\n            raise StopIteration\n        val = self.i ** 2\n        self.i += 1\n        return val",
        )
        nb_runner.run_all()
        assert "result = [0, 1, 4, 9]" in nb_runner.get_output(2)

    def test_generator_expression_edit(self, nb_runner):
        """Edit a generator expression."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5]  # source data for gen",
            "total = sum(x * 2 for x in data)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 30" in nb_runner.get_output(2)

        # Change to cubed
        nb_runner.set_cell_source(
            2, "total = sum(x ** 3 for x in data)\nprint(f'total = {total}')"
        )
        nb_runner.run_all()
        assert "total = 225" in nb_runner.get_output(2)
