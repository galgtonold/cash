"""Batch 161 – Dataclass and namedtuple interaction tests.

Tests editing dataclass/namedtuple definitions, field changes,
and downstream usage after modifications.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestDataclassEdits:
    """Editing dataclass definitions and usage."""

    def test_edit_dataclass_field(self, nb_runner):
        """Add a field to a dataclass, verify downstream updates."""
        nb_runner.create_notebook([
            "from dataclasses import dataclass",
            "@dataclass\nclass Point:\n    x: float\n    y: float",
            "p = Point(1.0, 2.0)\nprint(f'p = {p}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Point(x=1.0, y=2.0)" in nb_runner.get_output(3)

        # Add a z field
        nb_runner.set_cell_source(
            2,
            "@dataclass\nclass Point:\n    x: float\n    y: float\n    z: float = 0.0",
        )
        nb_runner.set_cell_source(3, "p = Point(1.0, 2.0, 3.0)\nprint(f'p = {p}')")
        nb_runner.run_all()
        assert "Point(x=1.0, y=2.0, z=3.0)" in nb_runner.get_output(3)

    def test_edit_dataclass_method(self, nb_runner):
        """Edit a method on a dataclass."""
        nb_runner.create_notebook([
            "from dataclasses import dataclass",
            "@dataclass\nclass Rect:\n    w: float\n    h: float\n    def area(self):\n        return self.w * self.h",
            "r = Rect(3.0, 4.0)\nprint(f'area = {r.area()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 12.0" in nb_runner.get_output(3)

        # Change method to perimeter
        nb_runner.set_cell_source(
            2,
            "@dataclass\nclass Rect:\n    w: float\n    h: float\n    def area(self):\n        return 2 * (self.w + self.h)",
        )
        nb_runner.run_all()
        assert "area = 14.0" in nb_runner.get_output(3)

    def test_dataclass_default_edit(self, nb_runner):
        """Edit default values on a dataclass."""
        nb_runner.create_notebook([
            "from dataclasses import dataclass, field",
            "@dataclass\nclass Config:\n    name: str = 'default'\n    value: int = 0",
            "c = Config()\nprint(f'name={c.name} value={c.value}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "name=default value=0" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2,
            "@dataclass\nclass Config:\n    name: str = 'updated'\n    value: int = 42",
        )
        nb_runner.set_cell_source(3, "c = Config()\nprint(f'name={c.name} value={c.value}')")
        nb_runner.run_all()
        assert "name=updated value=42" in nb_runner.get_output(3)


class TestNamedtupleEdits:
    """Editing namedtuple definitions."""

    def test_namedtuple_add_field(self, nb_runner):
        """Add a field to a namedtuple."""
        nb_runner.create_notebook([
            "from collections import namedtuple",
            "Color = namedtuple('Color', ['r', 'g', 'b'])",
            "c = Color(255, 128, 0)\nprint(f'color = {c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "color = Color(r=255, g=128, b=0)" in nb_runner.get_output(3)

        # Add alpha field
        nb_runner.set_cell_source(
            2, "Color = namedtuple('Color', ['r', 'g', 'b', 'a'])"
        )
        nb_runner.set_cell_source(3, "c = Color(255, 128, 0, 1.0)\nprint(f'color = {c}')")
        nb_runner.run_all()
        assert "color = Color(r=255, g=128, b=0, a=1.0)" in nb_runner.get_output(3)

    def test_namedtuple_to_dataclass(self, nb_runner):
        """Convert from namedtuple to dataclass."""
        nb_runner.create_notebook([
            "from collections import namedtuple",
            "Point = namedtuple('Point', ['x', 'y'])\np = Point(1, 2)\nprint(f'p = {p}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "p = Point(x=1, y=2)" in nb_runner.get_output(2)

        # Switch to dataclass
        nb_runner.set_cell_source(1, "from dataclasses import dataclass")
        nb_runner.set_cell_source(
            2,
            "@dataclass\nclass Point:\n    x: int\n    y: int\np = Point(1, 2)\nprint(f'p = {p}')",
        )
        nb_runner.run_all()
        assert "p = Point(x=1, y=2)" in nb_runner.get_output(2)
