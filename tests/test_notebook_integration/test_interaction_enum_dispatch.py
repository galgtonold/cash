"""Batch 256 – Enum member editing and dispatch patterns.

Tests enum-based dispatch/mapping with edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestEnumDispatchEdits:
    """Enum-based dispatch patterns with edits."""

    def test_enum_dispatch_edit(self, nb_runner):
        """Edit enum dispatch mapping."""
        nb_runner.create_notebook([
            "from enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3",
            "dispatch = {Color.RED: 'stop', Color.GREEN: 'go', Color.BLUE: 'info'}",
            "result = dispatch[Color.RED]\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = stop" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2,
            "dispatch = {Color.RED: 'danger', Color.GREEN: 'safe', Color.BLUE: 'neutral'}",
        )
        nb_runner.run_all()
        assert "result = danger" in nb_runner.get_output(3)

    def test_enum_value_function(self, nb_runner):
        """Edit function using enum values."""
        nb_runner.create_notebook([
            "from enum import Enum\nclass Priority(Enum):\n    LOW = 1\n    MED = 5\n    HIGH = 10",
            "def score(p):\n    return p.value * 2",
            "s = score(Priority.HIGH)\nprint(f's = {s}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "s = 20" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2,
            "def score(p):\n    return p.value ** 2",
        )
        nb_runner.run_all()
        assert "s = 100" in nb_runner.get_output(3)

    def test_enum_class_edit(self, nb_runner):
        """Edit enum class itself, downstream dispatch updates."""
        nb_runner.create_notebook([
            "from enum import Enum\nclass Status(Enum):\n    ACTIVE = 'active'\n    INACTIVE = 'inactive'",
            "labels = {s: s.value.upper() for s in Status}\nprint(f'labels = {labels}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "ACTIVE" in out
        assert "INACTIVE" in out

        nb_runner.set_cell_source(
            1,
            "from enum import Enum\nclass Status(Enum):\n    ACTIVE = 'active'\n    INACTIVE = 'inactive'\n    PENDING = 'pending'",
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "PENDING" in out2
