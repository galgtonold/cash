"""Batch 264 – Numeric precision and math computation edits.

Tests math operations, rounding, precision with edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestNumericPrecisionEdits:
    """Numeric computation edit patterns."""

    def test_rounding_edit(self, nb_runner):
        """Edit rounding precision, result changes."""
        nb_runner.create_notebook([
            "import math\nval = math.pi",
            "precision = 2",
            "result = round(val, precision)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 3.14" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "precision = 5")
        nb_runner.run_all()
        assert "result = 3.14159" in nb_runner.get_output(3)

    def test_formula_edit(self, nb_runner):
        """Edit formula, downstream result updates."""
        nb_runner.create_notebook([
            "a = 3\nb = 4",
            "import math\nhyp = math.sqrt(a**2 + b**2)\nprint(f'hyp = {hyp}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hyp = 5.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "a = 5\nb = 12")
        nb_runner.run_all()
        assert "hyp = 13.0" in nb_runner.get_output(2)

    def test_statistics_edit(self, nb_runner):
        """Edit data, statistical measures update."""
        nb_runner.create_notebook([
            "data = [10, 20, 30, 40, 50]",
            "import statistics\nmean = statistics.mean(data)\nstdev = round(statistics.stdev(data), 2)\nprint(f'mean={mean} stdev={stdev}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean=30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "data = [100, 100, 100, 100, 100]")
        nb_runner.run_all()
        assert "mean=100" in nb_runner.get_output(2)
        assert "stdev=0" in nb_runner.get_output(2)
