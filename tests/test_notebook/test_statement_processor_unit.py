from cash.notebook.cache_status import CacheStatus
"""Direct import tests for statement_processor module.

Verifies that the StatementProcessor class and its TypedDicts can be
imported and instantiated directly, improving test coverage visibility.
"""

from cash.notebook.statement_processor import (
    ProcessResult,
    StatementProcessor,
    _ProcessResultRequired,
)
from cash.notebook._protocols import TrackingState


class TestProcessResultTypes:
    """Test the ProcessResult TypedDict structure."""

    def test_required_keys_present(self):
        """_ProcessResultRequired defines the mandatory keys."""
        required: _ProcessResultRequired = {
            "status": CacheStatus.COMPUTED,
            "code": "x = 1",
            "outputs": ["x"],
            "execution_time": 0.01,
            "saved_time": 0.0,
            "storage": 0,
            "total_time": 0.01,
            "cached": False,
        }
        assert required["status"] == CacheStatus.COMPUTED
        assert required["code"] == "x = 1"

    def test_optional_keys_allowed(self):
        """ProcessResult extends _ProcessResultRequired with optional keys."""
        result: ProcessResult = {
            "status": CacheStatus.RESTORED,
            "code": "y = x + 1",
            "outputs": ["y"],
            "execution_time": 0.0,
            "saved_time": 0.5,
            "storage": 128,
            "total_time": 0.5,
            "cached": True,
            "file_dependencies": {"data.csv": 1234567890.0},
            "output_lineages": {"y": "abc123"},
            "input_lineages": {"x": "def456"},
        }
        assert result["cached"] is True
        assert "file_dependencies" in result


class TestStatementProcessorImport:
    """Verify StatementProcessor is importable and inspectable."""

    def test_class_exists(self):
        assert StatementProcessor is not None

    def test_has_process_statement_method(self):
        assert hasattr(StatementProcessor, "process_statement")

    def test_has_expected_attributes(self):
        """The class should define key methods for statement processing."""
        expected_methods = [
            "process_statement",
            "_execute_statement",
            "_capture_and_track_variables",
            "_analyze_and_hash",
        ]
        for method_name in expected_methods:
            assert hasattr(StatementProcessor, method_name), (
                f"StatementProcessor missing method {method_name}"
            )

    def test_has_set_tracking_state(self):
        """StatementProcessor should have the new set_tracking_state method."""
        assert hasattr(StatementProcessor, "set_tracking_state")
        assert callable(StatementProcessor.set_tracking_state)

    def test_set_tracking_dicts_removed(self):
        """set_tracking_dicts was removed; only set_tracking_state remains."""
        assert not hasattr(StatementProcessor, "set_tracking_dicts")


class TestTrackingState:
    """Test the TrackingState dataclass."""

    def test_default_fields_are_empty(self):
        state = TrackingState()
        assert state.executed_cell_codes == {}
        assert state.executed_cell_hashes == {}
        assert state.variable_lineage == {}
        assert state.executed_file_deps == {}
        assert state.variable_hashes == {}
        assert state.variable_sources == {}
        assert state.current_session_hashes == {}
        assert state.vars_with_mutation_lineage == set()
        assert state.executed_input_lineages == {}

    def test_independent_instances(self):
        """Each TrackingState instance should have independent containers."""
        state1 = TrackingState()
        state2 = TrackingState()
        state1.variable_lineage["x"] = "hash1"
        assert "x" not in state2.variable_lineage

    def test_mutation_through_reference(self):
        """Mutating a dict from TrackingState should be visible via the original."""
        state = TrackingState()
        codes_ref = state.executed_cell_codes
        codes_ref["y"] = "y = 42"
        assert state.executed_cell_codes["y"] == "y = 42"

    def test_all_fields_present(self):
        """All documented fields should exist."""
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(TrackingState)}
        expected = {
            "executed_cell_codes",
            "executed_cell_hashes",
            "variable_lineage",
            "executed_file_deps",
            "variable_hashes",
            "variable_sources",
            "current_session_hashes",
            "vars_with_mutation_lineage",
            "executed_input_lineages",
        }
        assert expected.issubset(field_names)
