"""Batch 177 – Enum and constant pattern interaction tests.

Tests editing enum definitions, constant values, and
patterns that use them across cells.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestEnumEdits:
    """Editing enum definitions."""

    def test_edit_enum_member(self, nb_runner):
        """Edit an enum member value."""
        nb_runner.create_notebook([
            "from enum import Enum",
            "class Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3",
            "c = Color.RED\nprint(f'color = {c.name}, value = {c.value}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "color = RED, value = 1" in nb_runner.get_output(3)

        # Change value
        nb_runner.set_cell_source(
            2,
            "class Color(Enum):\n    RED = 10\n    GREEN = 20\n    BLUE = 30",
        )
        nb_runner.set_cell_source(3, "c = Color.RED\nprint(f'color = {c.name}, value = {c.value}')")
        nb_runner.run_all()
        assert "color = RED, value = 10" in nb_runner.get_output(3)

    def test_add_enum_member(self, nb_runner):
        """Add a new member to an enum."""
        nb_runner.create_notebook([
            "from enum import Enum",
            "class Status(Enum):\n    ACTIVE = 'active'\n    INACTIVE = 'inactive'",
            "statuses = [s.value for s in Status]\nprint(f'statuses = {statuses}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "active" in out
        assert "inactive" in out

        # Add PENDING
        nb_runner.set_cell_source(
            2,
            "class Status(Enum):\n    ACTIVE = 'active'\n    INACTIVE = 'inactive'\n    PENDING = 'pending'",
        )
        nb_runner.set_cell_source(3, "statuses = [s.value for s in Status]\nprint(f'statuses = {statuses}')")
        nb_runner.run_all()
        assert "pending" in nb_runner.get_output(3)


class TestConstantEdits:
    """Editing constants used across cells."""

    def test_edit_config_constant(self, nb_runner):
        """Edit a config constant that affects computation."""
        nb_runner.create_notebook([
            "MAX_RETRIES = 3  # config constant",
            "retries = list(range(MAX_RETRIES))\nprint(f'retries = {retries}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "retries = [0, 1, 2]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "MAX_RETRIES = 5  # config constant increased")
        nb_runner.run_all()
        assert "retries = [0, 1, 2, 3, 4]" in nb_runner.get_output(2)

    def test_edit_multiple_constants(self, nb_runner):
        """Edit multiple constants at once."""
        nb_runner.create_notebook([
            "WIDTH = 10\nHEIGHT = 5",
            "area = WIDTH * HEIGHT\nperim = 2 * (WIDTH + HEIGHT)\nprint(f'area={area} perim={perim}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=50 perim=30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "WIDTH = 20\nHEIGHT = 10")
        nb_runner.run_all()
        assert "area=200 perim=60" in nb_runner.get_output(2)

    def test_frozen_dataclass_constant(self, nb_runner):
        """Use a frozen dataclass as a constant, edit it."""
        nb_runner.create_notebook([
            "from dataclasses import dataclass",
            "@dataclass(frozen=True)\nclass Params:\n    lr: float = 0.01\n    epochs: int = 10",
            "p = Params()\nprint(f'lr={p.lr} epochs={p.epochs}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lr=0.01 epochs=10" in nb_runner.get_output(3)

        # Edit defaults
        nb_runner.set_cell_source(
            2,
            "@dataclass(frozen=True)\nclass Params:\n    lr: float = 0.001\n    epochs: int = 100",
        )
        nb_runner.set_cell_source(3, "p = Params()\nprint(f'lr={p.lr} epochs={p.epochs}')")
        nb_runner.run_all()
        assert "lr=0.001 epochs=100" in nb_runner.get_output(3)
