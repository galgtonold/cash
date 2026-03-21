"""Batch 273 – Async/await patterns with edits.

Tests asyncio-based patterns in notebook cells.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestAsyncPatterns:
    """Async/await edit propagation."""

    def test_async_function_edit(self, nb_runner):
        """Edit async function, await result changes."""
        nb_runner.create_notebook([
            "import asyncio\nasync def compute(x):\n    return x * 2",
            "result = await compute(21)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 42" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "import asyncio\nasync def compute(x):\n    return x ** 2",
        )
        nb_runner.run_all()
        assert "result = 441" in nb_runner.get_output(2)

    def test_async_gather_edit(self, nb_runner):
        """Edit async tasks gathered together."""
        nb_runner.create_notebook([
            "import asyncio\nasync def task(n):\n    return n + 1",
            "result = list(await asyncio.gather(task(1), task(2), task(3)))\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 3, 4]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "import asyncio\nasync def task(n):\n    return n * 10",
        )
        nb_runner.run_all()
        assert "result = [10, 20, 30]" in nb_runner.get_output(2)
