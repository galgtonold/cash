"""Batch 258 – Partial application and functools patterns.

Tests functools.partial, lru_cache, and related patterns with edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestFunctoolsPatterns:
    """functools pattern edit propagation."""

    def test_partial_edit(self, nb_runner):
        """Edit partial application, downstream updates."""
        nb_runner.create_notebook([
            "from functools import partial\ndef power(base, exp):\n    return base ** exp",
            "square = partial(power, exp=2)",
            "results = [square(x) for x in [2, 3, 4, 5]]\nprint(f'results = {results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results = [4, 9, 16, 25]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "square = partial(power, exp=3)")
        nb_runner.run_all()
        assert "results = [8, 27, 64, 125]" in nb_runner.get_output(3)

    def test_partial_base_function_edit(self, nb_runner):
        """Edit the base function used in partial."""
        nb_runner.create_notebook([
            "from functools import partial\ndef combine(a, b, sep):\n    return f'{a}{sep}{b}'",
            "dash_join = partial(combine, sep='-')",
            "result = dash_join('hello', 'world')\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = hello-world" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            1,
            "from functools import partial\ndef combine(a, b, sep):\n    return f'{a.upper()}{sep}{b.upper()}'",
        )
        nb_runner.run_all()
        assert "result = HELLO-WORLD" in nb_runner.get_output(3)

    def test_cached_function_edit(self, nb_runner):
        """Edit function logic, even with lru_cache behavior changes."""
        nb_runner.create_notebook([
            "def expensive(n):\n    return sum(range(n))",
            "results = [expensive(x) for x in [10, 100, 1000]]\nprint(f'results = {results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results = [45, 4950, 499500]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def expensive(n):\n    return sum(range(n)) * 2",
        )
        nb_runner.run_all()
        assert "results = [90, 9900, 999000]" in nb_runner.get_output(2)
