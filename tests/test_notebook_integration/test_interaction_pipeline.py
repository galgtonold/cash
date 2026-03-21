"""
Batch 297: Multi-cell data pipeline interaction tests.
Tests complex data transformations spanning multiple cells where
edits at different pipeline stages propagate correctly.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestMultiCellPipelineInteraction:
    """Test multi-cell pipeline patterns with cache invalidation."""

    def test_etl_pipeline_edit_extract(self, nb_runner):
        """Editing the extract stage should propagate through transform and load."""
        nb_runner.create_notebook([
            "# Extract\nraw_data = [{'name': 'Alice', 'score': 85}, {'name': 'Bob', 'score': 92}]",
            "# Transform\nfiltered = [d for d in raw_data if d['score'] >= 90]",
            "# Load\nresult = ', '.join(d['name'] for d in filtered)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=Bob" in out

        nb_runner.set_cell_source(1, "# Extract\nraw_data = [{'name': 'Alice', 'score': 95}, {'name': 'Bob', 'score': 92}, {'name': 'Charlie', 'score': 98}]")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "Alice" in out
        assert "Bob" in out
        assert "Charlie" in out

    def test_etl_pipeline_edit_transform(self, nb_runner):
        """Editing the transform stage should propagate to load only."""
        nb_runner.create_notebook([
            "raw = [10, 20, 30, 40, 50]",
            "# Transform: filter\nprocessed = [x for x in raw if x > 20]",
            "# Aggregate\ntotal = sum(processed)",
            "print(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=120" in out

        # Change transform threshold
        nb_runner.set_cell_source(2, "# Transform: filter\nprocessed = [x for x in raw if x > 35]")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=90" in out

    def test_pipeline_three_stage_edit_middle(self, nb_runner):
        """Editing the middle of a 5-cell pipeline."""
        nb_runner.create_notebook([
            "data = list(range(1, 11))",
            "squared = [x**2 for x in data]",
            "filtered = [x for x in squared if x > 25]",
            "total = sum(filtered)",
            "print(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        # squared: 1,4,9,16,25,36,49,64,81,100 → filtered: 36,49,64,81,100 → total=330
        assert "total=330" in out

        # Change to cubed instead of squared
        nb_runner.set_cell_source(2, "squared = [x**3 for x in data]")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        # cubed: 1,8,27,64,125,216,343,512,729,1000 → filtered: 27,64,...,1000 → total=3016
        assert "total=3016" in out
