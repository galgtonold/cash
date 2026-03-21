"""Batch 221 – Data pipeline chain interaction tests.

Tests editing cells in multi-stage data pipelines
where each stage transforms the data.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestPipelineChainEdits:
    """Editing multi-stage pipeline patterns."""

    def test_edit_pipeline_source(self, nb_runner):
        """Edit source data in a 3-stage pipeline."""
        nb_runner.create_notebook([
            "raw = [1, -2, 3, -4, 5, -6]",
            "positives = [x for x in raw if x > 0]\ndoubled = [x * 2 for x in positives]\nresult = sum(doubled)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 18" in nb_runner.get_output(2)

        # Change source
        nb_runner.set_cell_source(1, "raw = [10, -1, 20, -2]")
        nb_runner.run_all()
        assert "result = 60" in nb_runner.get_output(2)

    def test_edit_pipeline_middle_stage(self, nb_runner):
        """Edit a middle stage of the pipeline."""
        nb_runner.create_notebook([
            "data = ['  Alice  ', '  Bob  ', '  Charlie  ']",
            "cleaned = [s.strip() for s in data]\nresult = [s.lower() for s in cleaned]\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = ['alice', 'bob', 'charlie']" in nb_runner.get_output(2)

        # Change middle stage to uppercase
        nb_runner.set_cell_source(2, "cleaned = [s.strip() for s in data]\nresult = [s.upper() for s in cleaned]\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = ['ALICE', 'BOB', 'CHARLIE']" in nb_runner.get_output(2)

    def test_edit_pipeline_aggregation(self, nb_runner):
        """Edit the final aggregation step."""
        nb_runner.create_notebook([
            "sales = [100, 200, 150, 300, 250]",
            "above_avg = [s for s in sales if s > sum(sales)/len(sales)]\ntotal = sum(above_avg)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 550" in nb_runner.get_output(2)

        # Change sales
        nb_runner.set_cell_source(1, "sales = [500, 100, 200, 600]")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "total = 1100" in out

    def test_edit_pipeline_filter_and_source(self, nb_runner):
        """Edit both source and filter in pipeline."""
        nb_runner.create_notebook([
            "records = [('A', 10), ('B', 20), ('C', 30), ('D', 40)]",
            "filtered = [(k, v) for k, v in records if v > 15]\nkeys = [k for k, v in filtered]\nprint(f'keys = {keys}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys = ['B', 'C', 'D']" in nb_runner.get_output(2)

        # Change records
        nb_runner.set_cell_source(1, "records = [('X', 5), ('Y', 50), ('Z', 25)]")
        nb_runner.run_all()
        assert "keys = ['Y', 'Z']" in nb_runner.get_output(2)
