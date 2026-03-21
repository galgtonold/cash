"""Batch 199 – Multi-return function with unpacking interaction tests.

Tests editing functions that return multiple values and
various unpacking patterns.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestMultiReturnEdits:
    """Editing multi-return function patterns."""

    def test_edit_returned_values(self, nb_runner):
        """Edit the values returned by a multi-return function."""
        nb_runner.create_notebook([
            "def compute(x):\n    return x * 2, x ** 2, x + 10",
            "a, b, c = compute(5)\nprint(f'a={a} b={b} c={c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=10 b=25 c=15" in nb_runner.get_output(2)

        # Edit function
        nb_runner.set_cell_source(
            1, "def compute(x):\n    return x * 3, x ** 3, x + 100"
        )
        nb_runner.run_all()
        assert "a=15 b=125 c=105" in nb_runner.get_output(2)

    def test_edit_unpacking_target(self, nb_runner):
        """Edit which returned values are used."""
        nb_runner.create_notebook([
            "def stats(data):\n    return min(data), max(data), sum(data)",
            "lo, hi, total = stats([3, 1, 4, 1, 5])\nprint(f'lo={lo} hi={hi} total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lo=1 hi=5 total=14" in nb_runner.get_output(2)

        # Change input data
        nb_runner.set_cell_source(
            2, "lo, hi, total = stats([10, 20, 30])\nprint(f'lo={lo} hi={hi} total={total}')"
        )
        nb_runner.run_all()
        assert "lo=10 hi=30 total=60" in nb_runner.get_output(2)

    def test_edit_star_unpacking(self, nb_runner):
        """Edit star unpacking patterns."""
        nb_runner.create_notebook([
            "def get_items():\n    return 1, 2, 3, 4, 5",
            "first, *rest = get_items()\nprint(f'first={first} rest={rest}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first=1 rest=[2, 3, 4, 5]" in nb_runner.get_output(2)

        # Change function
        nb_runner.set_cell_source(
            1, "def get_items():\n    return 10, 20, 30"
        )
        nb_runner.run_all()
        assert "first=10 rest=[20, 30]" in nb_runner.get_output(2)

    def test_edit_namedtuple_return(self, nb_runner):
        """Edit function returning namedtuple."""
        nb_runner.create_notebook([
            "from collections import namedtuple\nResult = namedtuple('Result', ['value', 'status'])",
            "def process(x):\n    if x > 0:\n        return Result(x * 2, 'ok')\n    return Result(0, 'error')",
            "r = process(5)\nprint(f'value={r.value} status={r.status}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "value=10 status=ok" in nb_runner.get_output(3)

        # Edit function
        nb_runner.set_cell_source(
            2,
            "def process(x):\n    if x > 0:\n        return Result(x ** 2, 'success')\n    return Result(-1, 'fail')",
        )
        nb_runner.run_all()
        assert "value=25 status=success" in nb_runner.get_output(3)
