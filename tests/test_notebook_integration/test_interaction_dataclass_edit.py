"""Batch 259 – Dataclass and structured data edit patterns.

Tests dataclass definitions and instances with edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDataclassEdits:
    """Dataclass edit propagation patterns."""

    def test_dataclass_field_edit(self, nb_runner):
        """Edit dataclass definition, instance creation updates."""
        nb_runner.create_notebook([
            "from dataclasses import dataclass\n@dataclass\nclass Point:\n    x: float\n    y: float\n    def distance(self):\n        return (self.x**2 + self.y**2)**0.5",
            "p = Point(3.0, 4.0)\ndist = p.distance()\nprint(f'dist = {dist}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "dist = 5.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "from dataclasses import dataclass\n@dataclass\nclass Point:\n    x: float\n    y: float\n    def distance(self):\n        return abs(self.x) + abs(self.y)",
        )
        nb_runner.run_all()
        assert "dist = 7.0" in nb_runner.get_output(2)

    def test_dataclass_default_edit(self, nb_runner):
        """Edit dataclass defaults, downstream reflects."""
        nb_runner.create_notebook([
            "from dataclasses import dataclass, field\n@dataclass\nclass Config:\n    name: str = 'default'\n    scale: int = 1\n    tags: list = field(default_factory=list)",
            "c = Config()\nprint(f'name={c.name} scale={c.scale} tags={c.tags}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "name=default scale=1 tags=[]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "from dataclasses import dataclass, field\n@dataclass\nclass Config:\n    name: str = 'production'\n    scale: int = 10\n    tags: list = field(default_factory=lambda: ['v2'])",
        )
        nb_runner.run_all()
        assert "name=production scale=10 tags=['v2']" in nb_runner.get_output(2)

    def test_dataclass_instance_edit(self, nb_runner):
        """Edit instance creation, downstream computation updates."""
        nb_runner.create_notebook([
            "from dataclasses import dataclass\n@dataclass\nclass Rectangle:\n    width: float\n    height: float\n    def area(self):\n        return self.width * self.height",
            "r = Rectangle(5.0, 3.0)",
            "a = r.area()\nprint(f'area = {a}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 15.0" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "r = Rectangle(10.0, 7.0)")
        nb_runner.run_all()
        assert "area = 70.0" in nb_runner.get_output(3)
