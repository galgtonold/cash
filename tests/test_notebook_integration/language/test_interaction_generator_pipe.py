"""Generator and iterator pipeline patterns.

Tests generator functions, chaining, and edit propagation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestGeneratorPipeline:
    """Generator pipeline patterns with edits."""

    def test_chained_generators(self, nb_runner):
        """Edit first generator in chain, final result updates."""
        nb_runner.create_notebook(
            [
                "def source(n):\n    for i in range(1, n + 1):\n        yield i",
                "def transform(gen):\n    for x in gen:\n        yield x * 10",
                "result = list(transform(source(4)))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [10, 20, 30, 40]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            1,
            "def source(n):\n    for i in range(1, n + 1):\n        yield i * 2",
        )
        nb_runner.run_all()
        assert "result = [20, 40, 60, 80]" in nb_runner.get_output(3)

    def test_generator_expression_edit(self, nb_runner):
        """Edit data fed into generator expression."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5]",
                "squares = list(x**2 for x in data if x > 2)\nprint(f'squares = {squares}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "squares = [9, 16, 25]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "data = [10, 20, 30]")
        nb_runner.run_all()
        assert "squares = [100, 400, 900]" in nb_runner.get_output(2)
