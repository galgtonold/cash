"""Batch 243 – Multiple return value patterns.

Tests tuple unpacking from function returns and editing the function.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestMultiReturnEdit:
    """Functions returning multiple values, edit propagation."""

    def test_tuple_return_edit(self, nb_runner):
        """Edit function that returns tuple, unpacked downstream."""
        nb_runner.create_notebook([
            "def stats(data):\n    return min(data), max(data), sum(data) / len(data)",
            "lo, hi, avg = stats([10, 20, 30, 40, 50])\nprint(f'lo={lo} hi={hi} avg={avg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lo=10 hi=50 avg=30.0" in nb_runner.get_output(2)

        # Edit to return range instead of avg
        nb_runner.set_cell_source(
            1,
            "def stats(data):\n    return min(data), max(data), max(data) - min(data)",
        )
        nb_runner.run_all()
        assert "lo=10 hi=50 avg=40" in nb_runner.get_output(2)

    def test_dict_return_edit(self, nb_runner):
        """Edit function returning dict, downstream uses keys."""
        nb_runner.create_notebook([
            "def analyze(items):\n    return {'count': len(items), 'total': sum(items)}",
            "info = analyze([5, 10, 15])\nprint(f'count={info[\"count\"]} total={info[\"total\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=3 total=30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def analyze(items):\n    return {'count': len(items), 'total': sum(items) * 2}",
        )
        nb_runner.run_all()
        assert "count=3 total=60" in nb_runner.get_output(2)

    def test_named_tuple_return_edit(self, nb_runner):
        """Edit function returning namedtuple fields."""
        nb_runner.create_notebook([
            "from collections import namedtuple\nResult = namedtuple('Result', ['value', 'label'])",
            "def compute(x):\n    return Result(value=x*2, label='doubled')",
            "r = compute(7)\nprint(f'value={r.value} label={r.label}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "value=14 label=doubled" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2,
            "def compute(x):\n    return Result(value=x**2, label='squared')",
        )
        nb_runner.run_all()
        assert "value=49 label=squared" in nb_runner.get_output(3)
