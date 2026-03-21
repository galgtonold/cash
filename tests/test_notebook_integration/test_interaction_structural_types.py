"""Batch 235 – Dataclass and namedtuple structural change tests.

Tests editing dataclass field definitions and namedtuple structures
to verify changes propagate through the cache.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestStructuralTypeEdits:
    """Editing dataclass and namedtuple structures."""

    def test_edit_dataclass_add_field(self, nb_runner):
        """Add a field to a dataclass and verify downstream."""
        nb_runner.create_notebook([
            "from dataclasses import dataclass\n@dataclass\nclass Item:\n    name: str\n    price: float = 0.0",
            "item = Item('Widget', 9.99)\nprint(f'item = {item}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Widget" in nb_runner.get_output(2)
        assert "9.99" in nb_runner.get_output(2)

        # Add a quantity field
        nb_runner.set_cell_source(1, "from dataclasses import dataclass\n@dataclass\nclass Item:\n    name: str\n    price: float = 0.0\n    qty: int = 1")
        nb_runner.set_cell_source(2, "item = Item('Widget', 9.99, qty=5)\nprint(f'item = {item}')")
        nb_runner.run_all()
        assert "qty=5" in nb_runner.get_output(2)

    def test_edit_namedtuple_add_field(self, nb_runner):
        """Add a field to a namedtuple."""
        nb_runner.create_notebook([
            "from collections import namedtuple\nPoint = namedtuple('Point', ['x', 'y'])",
            "p = Point(3, 4)\nprint(f'p = {p}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Point(x=3, y=4)" in nb_runner.get_output(2)

        # Add z field
        nb_runner.set_cell_source(1, "from collections import namedtuple\nPoint = namedtuple('Point', ['x', 'y', 'z'])")
        nb_runner.set_cell_source(2, "p = Point(3, 4, 5)\nprint(f'p = {p}')")
        nb_runner.run_all()
        assert "Point(x=3, y=4, z=5)" in nb_runner.get_output(2)

    def test_edit_dataclass_method_behavior(self, nb_runner):
        """Edit a method on a dataclass."""
        nb_runner.create_notebook([
            "from dataclasses import dataclass\n@dataclass\nclass Circle:\n    radius: float\n    def area(self):\n        return 3.14159 * self.radius ** 2",
            "c = Circle(5.0)\nprint(f'area = {c.area():.2f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 78.54" in nb_runner.get_output(2)

        # Change to circumference calculation
        nb_runner.set_cell_source(1, "from dataclasses import dataclass\n@dataclass\nclass Circle:\n    radius: float\n    def area(self):\n        return 2 * 3.14159 * self.radius")
        nb_runner.run_all()
        assert "area = 31.42" in nb_runner.get_output(2)
