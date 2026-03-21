"""Batch 261 – Complex multi-step data pipeline patterns.

Tests multi-cell data transformation pipelines with edits at different stages.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDataPipelineEdits:
    """Multi-step pipeline with edits at different stages."""

    def test_filter_transform_aggregate(self, nb_runner):
        """Three-stage pipeline: filter → transform → aggregate."""
        nb_runner.create_notebook([
            "raw = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10]",
            "filtered = [x for x in raw if x > 0]",
            "transformed = [x ** 2 for x in filtered]",
            "total = sum(transformed)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 1+9+25+49+81 = 165
        assert "total = 165" in nb_runner.get_output(4)

        # Edit filter stage
        nb_runner.set_cell_source(2, "filtered = [x for x in raw if x < 0]")
        nb_runner.run_all()
        # (-2)^2+(-4)^2+(-6)^2+(-8)^2+(-10)^2 = 4+16+36+64+100 = 220
        assert "total = 220" in nb_runner.get_output(4)

    def test_edit_transform_stage(self, nb_runner):
        """Edit the transformation in the middle of a pipeline."""
        nb_runner.create_notebook([
            "prices = [10.0, 20.0, 30.0, 40.0]",
            "discounted = [p * 0.9 for p in prices]",
            "with_tax = [p * 1.1 for p in discounted]",
            "total = round(sum(with_tax), 2)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 10*.9*1.1 + 20*.9*1.1 + ... = 9.9+19.8+29.7+39.6 = 99.0
        assert "total = 99.0" in nb_runner.get_output(4)

        # Change discount rate
        nb_runner.set_cell_source(2, "discounted = [p * 0.5 for p in prices]")
        nb_runner.run_all()
        # 10*.5*1.1 + 20*.5*1.1 + 30*.5*1.1 + 40*.5*1.1 = 5.5+11.0+16.5+22.0 = 55.0
        assert "total = 55.0" in nb_runner.get_output(4)

    def test_dict_pipeline_edit(self, nb_runner):
        """Dict-based pipeline: lookup → transform → format."""
        nb_runner.create_notebook([
            "inventory = {'apple': 50, 'banana': 30, 'cherry': 20}",
            "threshold = 25",
            "low_stock = {k: v for k, v in inventory.items() if v <= threshold}",
            "report = ', '.join(f'{k}:{v}' for k, v in sorted(low_stock.items()))\nprint(f'report = {report}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "report = cherry:20" in nb_runner.get_output(4)

        nb_runner.set_cell_source(2, "threshold = 35")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "banana:30" in out
        assert "cherry:20" in out
