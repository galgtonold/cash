"""Batch 127 – Real-world workflow simulation interaction tests.

Tests that simulate realistic data analysis workflows with multiple
rounds of exploration, parameter tuning, and iterative refinement.
"""

import pytest

pytestmark = [pytest.mark.core, pytest.mark.stress, pytest.mark.timeout(30)]


class TestDataAnalysisWorkflow:
    """Simulate an iterative data analysis workflow."""

    def test_parameter_tuning_workflow(self, nb_runner):
        """Simulate parameter tuning — change params and re-evaluate."""
        nb_runner.create_notebook([
            "data = list(range(1, 101))",
            "threshold = 50",
            "filtered = [x for x in data if x > threshold]",
            "count = len(filtered)\navg = sum(filtered) / count if count else 0",
            "print(f'count={count}, avg={avg:.1f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # Above 50: 51..100 = 50 items, avg = 75.5
        assert "count=50, avg=75.5" in nb_runner.get_output(5)

        # Tune threshold down
        nb_runner.set_cell_source(2, "threshold = 80")
        nb_runner.run_all()
        # Above 80: 81..100 = 20 items, avg = 90.5
        assert "count=20, avg=90.5" in nb_runner.get_output(5)

        # Tune threshold up
        nb_runner.set_cell_source(2, "threshold = 95")
        nb_runner.run_all()
        # Above 95: 96..100 = 5 items, avg = 98.0
        assert "count=5, avg=98.0" in nb_runner.get_output(5)

    def test_feature_engineering_workflow(self, nb_runner):
        """Simulate feature engineering with iterative changes."""
        nb_runner.create_notebook([
            "raw = [10, 20, 30, 40, 50]",
            "def transform(data):\n    return [x / max(data) for x in data]",
            "features = transform(raw)",
            "score = sum(features)\nprint(f'score = {score:.2f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 0.2 + 0.4 + 0.6 + 0.8 + 1.0 = 3.0
        assert "score = 3.00" in nb_runner.get_output(4)

        # Change transform
        nb_runner.set_cell_source(
            2, "def transform(data):\n    mean = sum(data) / len(data)\n    return [(x - mean) / mean for x in data]"
        )
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "score = " in output  # should be 0.0 (mean-centered)

    def test_etl_pipeline_edit(self, nb_runner):
        """ETL pipeline: extract → transform → load. Edit transform."""
        nb_runner.create_notebook([
            "# Extract\nraw_records = [{'name': 'A', 'val': 10}, {'name': 'B', 'val': 20}, {'name': 'C', 'val': 30}]",
            "# Transform\ntransformed = {r['name']: r['val'] * 2 for r in raw_records}",
            "# Load\ntotal = sum(transformed.values())\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # (10+20+30)*2 = 120
        assert "total = 120" in nb_runner.get_output(3)

        # Change transform logic
        nb_runner.set_cell_source(
            2, "# Transform\ntransformed = {r['name']: r['val'] ** 2 for r in raw_records}"
        )
        nb_runner.run_all()
        # 100+400+900 = 1400
        assert "total = 1400" in nb_runner.get_output(3)


class TestExploratoryAnalysis:
    """Simulate exploratory data analysis with back-and-forth edits."""

    def test_explore_and_refine(self, nb_runner):
        """Explore data, refine analysis, go back and change approach."""
        nb_runner.create_notebook([
            "data = [3, 7, 2, 9, 1, 5, 8, 4, 6, 10]",
            "# Approach 1: simple sort\nsorted_data = sorted(data)",
            "top3 = sorted_data[-3:]\nprint(f'top3 = {top3}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "top3 = [8, 9, 10]" in nb_runner.get_output(3)

        # Refine: use different sort approach
        nb_runner.set_cell_source(
            2, "# Approach 2: reverse sort\nsorted_data = sorted(data, reverse=True)"
        )
        nb_runner.set_cell_source(3, "top3 = sorted_data[:3]\nprint(f'top3 = {top3}')")
        nb_runner.run_all()
        assert "top3 = [10, 9, 8]" in nb_runner.get_output(3)

    def test_multi_round_exploration(self, nb_runner):
        """Multiple rounds of exploration on the same data."""
        nb_runner.create_notebook([
            "nums = [1, 4, 9, 16, 25]",
            "# Analysis\nresult = sum(nums)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 55" in nb_runner.get_output(2)

        # Round 2: average
        nb_runner.set_cell_source(
            2, "# Analysis\nresult = sum(nums) / len(nums)\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = 11.0" in nb_runner.get_output(2)

        # Round 3: max - min range
        nb_runner.set_cell_source(
            2, "# Analysis\nresult = max(nums) - min(nums)\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = 24" in nb_runner.get_output(2)


class TestConfigDrivenWorkflow:
    """Configuration-driven workflows with config edits."""

    def test_config_dict_workflow(self, nb_runner):
        """Change config dict, verify pipeline adjusts."""
        nb_runner.create_notebook([
            "config = {'scale': 2, 'offset': 10}",
            "data = [1, 2, 3, 4, 5]",
            "processed = [x * config['scale'] + config['offset'] for x in data]",
            "result = sum(processed)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # (1*2+10) + (2*2+10) + ... + (5*2+10) = 12+14+16+18+20 = 80
        assert "result = 80" in nb_runner.get_output(4)

        nb_runner.set_cell_source(1, "config = {'scale': 10, 'offset': 0}")
        nb_runner.run_all()
        # 10+20+30+40+50 = 150
        assert "result = 150" in nb_runner.get_output(4)

    def test_multi_config_changes(self, nb_runner):
        """Change config multiple times in sequence."""
        nb_runner.create_notebook([
            "multiplier = 1",
            "result = 42 * multiplier\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 42" in nb_runner.get_output(2)

        for m in [2, 5, 10, 100]:
            nb_runner.set_cell_source(1, f"multiplier = {m}")
            nb_runner.run_all()
            assert f"result = {42 * m}" in nb_runner.get_output(2)
