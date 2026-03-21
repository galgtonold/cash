"""
Batch 311: Assertion and validation interaction tests.
Tests that editing validation logic or asserted conditions
properly invalidates downstream cells.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestValidationInteraction:
    """Test validation/assertion patterns with cache invalidation."""

    def test_validation_function_edit(self, nb_runner):
        """Editing a validation function should propagate."""
        nb_runner.create_notebook([
            "def validate(x):\n    return 0 <= x <= 100",
            "values = [10, 50, 150, -5, 80]",
            "valid = [v for v in values if validate(v)]",
            "count = len(valid)",
            "print(f'count={count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "count=3" in out

        # Widen validation range
        nb_runner.set_cell_source(1, "def validate(x):\n    return -10 <= x <= 200")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "count=5" in out

    def test_schema_validation_edit(self, nb_runner):
        """Editing schema/rules should propagate to validated results."""
        nb_runner.create_notebook([
            "required_keys = {'name', 'age'}",
            "records = [{'name': 'Alice', 'age': 30}, {'name': 'Bob'}, {'age': 25}]",
            "valid_records = [r for r in records if required_keys.issubset(r.keys())]",
            "count = len(valid_records)",
            "print(f'count={count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "count=1" in out

        # Relax requirements
        nb_runner.set_cell_source(1, "required_keys = {'name'}")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "count=2" in out
