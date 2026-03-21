"""
Batch 327: enum patterns with caching.
Tests Enum, IntEnum, Flag operations, and edit propagation.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.timeout(90)]


class TestEnumPatterns:
    """Test Enum operation caching."""

    def test_enum_basic(self, nb_runner):
        """Basic Enum creation and comparison with caching."""
        nb_runner.create_notebook([
            "from enum import Enum",
            "class Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3",
            "c = Color.GREEN\nresult = c.name",
            "print(f'result={result} value={c.value}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=GREEN" in out
        assert "value=2" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "result=GREEN" in out2

    def test_enum_edit_selection(self, nb_runner):
        """Edit enum member selection, verify propagation."""
        nb_runner.create_notebook([
            "from enum import Enum",
            "class Status(Enum):\n    PENDING = 'pending'\n    ACTIVE = 'active'\n    CLOSED = 'closed'",
            "current = Status.PENDING",
            "is_open = current != Status.CLOSED\nprint(f'status={current.value} is_open={is_open}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "status=pending" in out
        assert "is_open=True" in out

        nb_runner.set_cell_source(3, "current = Status.CLOSED")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "status=closed" in out2
        assert "is_open=False" in out2

    def test_int_enum_arithmetic(self, nb_runner):
        """IntEnum used in arithmetic."""
        nb_runner.create_notebook([
            "from enum import IntEnum",
            "class Priority(IntEnum):\n    LOW = 1\n    MEDIUM = 2\n    HIGH = 3\n    CRITICAL = 4",
            "tasks = [Priority.HIGH, Priority.LOW, Priority.MEDIUM]\navg = sum(tasks) / len(tasks)",
            "print(f'avg={avg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "avg=2.0" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "avg=2.0" in out2
