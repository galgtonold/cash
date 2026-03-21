"""Batch 176 – Complex real-world data analysis simulation tests.

Tests simulating real data analysis workflows with multiple
edit cycles, variable reuse, and result verification.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestStatisticalAnalysis:
    """Statistical analysis workflow with edits."""

    def test_mean_calculation_edit(self, nb_runner):
        """Compute mean, then edit the data source."""
        nb_runner.create_notebook([
            "data = [10, 20, 30, 40, 50]  # sample data",
            "mean_val = sum(data) / len(data)\nprint(f'mean = {mean_val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean = 30.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "data = [100, 200, 300]  # new sample data")
        nb_runner.run_all()
        assert "mean = 200.0" in nb_runner.get_output(2)

    def test_data_pipeline_multiple_stats(self, nb_runner):
        """Compute multiple statistics, edit the dataset."""
        nb_runner.create_notebook([
            "nums = [4, 8, 15, 16, 23, 42]  # dataset",
            "n = len(nums)\nmean = sum(nums) / n",
            "variance = sum((x - mean) ** 2 for x in nums) / n",
            "import math\nstd = math.sqrt(variance)\nprint(f'mean={mean:.1f} std={std:.1f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean=18.0" in nb_runner.get_output(4)

        # Change dataset
        nb_runner.set_cell_source(1, "nums = [10, 10, 10, 10]  # uniform dataset")
        nb_runner.run_all()
        assert "mean=10.0" in nb_runner.get_output(4)
        assert "std=0.0" in nb_runner.get_output(4)


class TestDataTransformWorkflow:
    """Data transformation workflows."""

    def test_filter_and_aggregate(self, nb_runner):
        """Filter data then aggregate, edit the filter."""
        nb_runner.create_notebook([
            "records = [('A', 10), ('B', 20), ('A', 30), ('B', 40)]",
            "filtered = [v for k, v in records if k == 'A']",
            "total = sum(filtered)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 40" in nb_runner.get_output(3)

        # Change filter
        nb_runner.set_cell_source(
            2, "filtered = [v for k, v in records if k == 'B']"
        )
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(3)

    def test_sort_and_rank(self, nb_runner):
        """Sort data and compute ranks, edit sorting order."""
        nb_runner.create_notebook([
            "scores = [85, 92, 78, 95, 88]  # student scores",
            "ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)",
            "top = ranked[0]\nprint(f'top student={top[0]} score={top[1]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "top student=3 score=95" in nb_runner.get_output(3)

        # Change scores
        nb_runner.set_cell_source(
            1, "scores = [50, 99, 78, 60, 88]  # student scores updated"
        )
        nb_runner.run_all()
        assert "top student=1 score=99" in nb_runner.get_output(3)


class TestReportGeneration:
    """Report generation workflow."""

    def test_build_report_string(self, nb_runner):
        """Build a report string from data, edit the data."""
        nb_runner.create_notebook([
            "title = 'Sales Report'",
            "items = {'Widget A': 100, 'Widget B': 200}",
            "lines = [title, '=' * len(title)]\nfor name, count in items.items():\n    lines.append(f'{name}: {count}')\nreport = '\\n'.join(lines)\nprint(report)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "Sales Report" in out
        assert "Widget A: 100" in out

        # Edit title
        nb_runner.set_cell_source(1, "title = 'Q4 Sales'")
        nb_runner.run_all()
        assert "Q4 Sales" in nb_runner.get_output(3)

    def test_summary_metrics(self, nb_runner):
        """Compute summary metrics, edit the input."""
        nb_runner.create_notebook([
            "values = [10, 20, 30, 40, 50]  # input values",
            "metrics = {\n    'count': len(values),\n    'sum': sum(values),\n    'min': min(values),\n    'max': max(values),\n}",
            "for k, v in metrics.items():\n    print(f'{k}: {v}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "count: 5" in out
        assert "sum: 150" in out

        nb_runner.set_cell_source(
            1, "values = [1, 2, 3]  # input values shorter"
        )
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "count: 3" in out
        assert "sum: 6" in out
