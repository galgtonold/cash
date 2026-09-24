"""Editing dataclass and namedtuple definitions and the cells that use them."""

import pytest

pytestmark = [pytest.mark.stress]


class TestDataclassEdits:
    """Editing dataclass definitions and usage."""

    @pytest.mark.upstream
    @pytest.mark.timeout(90)
    def test_edit_dataclass_field(self, nb_runner):
        """Add a field to a dataclass, verify downstream updates."""
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass",
                "@dataclass\nclass Point:\n    x: float\n    y: float",
                "p = Point(1.0, 2.0)\nprint(f'p = {p}')",
            ]
        )
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

    @pytest.mark.upstream
    @pytest.mark.timeout(90)
    def test_edit_dataclass_method(self, nb_runner):
        """Edit a method on a dataclass."""
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass",
                "@dataclass\nclass Rect:\n    w: float\n    h: float\n    def area(self):\n        return self.w * self.h",
                "r = Rect(3.0, 4.0)\nprint(f'area = {r.area()}')",
            ]
        )
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

    @pytest.mark.upstream
    @pytest.mark.timeout(90)
    def test_dataclass_default_edit(self, nb_runner):
        """Edit default values on a dataclass."""
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass, field",
                "@dataclass\nclass Config:\n    name: str = 'default'\n    value: int = 0",
                "c = Config()\nprint(f'name={c.name} value={c.value}')",
            ]
        )
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

    # Dataclass and structured data edit patterns.
    #
    # Tests dataclass definitions and instances with edits.
    @pytest.mark.timeout(90)
    def test_dataclass_field_edit(self, nb_runner):
        """Edit dataclass definition, instance creation updates."""
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass\n@dataclass\nclass Point:\n    x: float\n    y: float\n    def distance(self):\n        return (self.x**2 + self.y**2)**0.5",
                "p = Point(3.0, 4.0)\ndist = p.distance()\nprint(f'dist = {dist}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "dist = 5.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "from dataclasses import dataclass\n@dataclass\nclass Point:\n    x: float\n    y: float\n    def distance(self):\n        return abs(self.x) + abs(self.y)",
        )
        nb_runner.run_all()
        assert "dist = 7.0" in nb_runner.get_output(2)

    @pytest.mark.timeout(90)
    def test_edit_defaults_and_default_factory_in_the_import_cell(self, nb_runner):
        """Edit dataclass defaults, downstream reflects."""
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass, field\n@dataclass\nclass Config:\n    name: str = 'default'\n    scale: int = 1\n    tags: list = field(default_factory=list)",
                "c = Config()\nprint(f'name={c.name} scale={c.scale} tags={c.tags}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "name=default scale=1 tags=[]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "from dataclasses import dataclass, field\n@dataclass\nclass Config:\n    name: str = 'production'\n    scale: int = 10\n    tags: list = field(default_factory=lambda: ['v2'])",
        )
        nb_runner.run_all()
        assert "name=production scale=10 tags=['v2']" in nb_runner.get_output(2)

    @pytest.mark.timeout(90)
    def test_dataclass_instance_edit(self, nb_runner):
        """Edit instance creation, downstream computation updates."""
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass\n@dataclass\nclass Rectangle:\n    width: float\n    height: float\n    def area(self):\n        return self.width * self.height",
                "r = Rectangle(5.0, 3.0)",
                "a = r.area()\nprint(f'area = {a}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 15.0" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "r = Rectangle(10.0, 7.0)")
        nb_runner.run_all()
        assert "area = 70.0" in nb_runner.get_output(3)


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestNamedtupleEdits:
    """Editing namedtuple definitions."""

    def test_namedtuple_add_field(self, nb_runner):
        """Add a field to a namedtuple."""
        nb_runner.create_notebook(
            [
                "from collections import namedtuple",
                "Color = namedtuple('Color', ['r', 'g', 'b'])",
                "c = Color(255, 128, 0)\nprint(f'color = {c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "color = Color(r=255, g=128, b=0)" in nb_runner.get_output(3)

        # Add alpha field
        nb_runner.set_cell_source(2, "Color = namedtuple('Color', ['r', 'g', 'b', 'a'])")
        nb_runner.set_cell_source(3, "c = Color(255, 128, 0, 1.0)\nprint(f'color = {c}')")
        nb_runner.run_all()
        assert "color = Color(r=255, g=128, b=0, a=1.0)" in nb_runner.get_output(3)

    def test_namedtuple_to_dataclass(self, nb_runner):
        """Convert from namedtuple to dataclass."""
        nb_runner.create_notebook(
            [
                "from collections import namedtuple",
                "Point = namedtuple('Point', ['x', 'y'])\np = Point(1, 2)\nprint(f'p = {p}')",
            ]
        )
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


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestStructuralTypeEdits:
    """Editing dataclass and namedtuple structures."""

    def test_edit_dataclass_add_field(self, nb_runner):
        """Add a field to a dataclass and verify downstream."""
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass\n@dataclass\nclass Item:\n    name: str\n    price: float = 0.0",
                "item = Item('Widget', 9.99)\nprint(f'item = {item}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Widget" in nb_runner.get_output(2)
        assert "9.99" in nb_runner.get_output(2)

        # Add a quantity field
        nb_runner.set_cell_source(
            1,
            "from dataclasses import dataclass\n@dataclass\nclass Item:\n    name: str\n    price: float = 0.0\n    qty: int = 1",
        )
        nb_runner.set_cell_source(2, "item = Item('Widget', 9.99, qty=5)\nprint(f'item = {item}')")
        nb_runner.run_all()
        assert "qty=5" in nb_runner.get_output(2)

    def test_edit_namedtuple_add_field(self, nb_runner):
        """Add a field to a namedtuple."""
        nb_runner.create_notebook(
            [
                "from collections import namedtuple\nPoint = namedtuple('Point', ['x', 'y'])",
                "p = Point(3, 4)\nprint(f'p = {p}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass\n@dataclass\nclass Circle:\n    radius: float\n    def area(self):\n        return 3.14159 * self.radius ** 2",
                "c = Circle(5.0)\nprint(f'area = {c.area():.2f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 78.54" in nb_runner.get_output(2)

        # Change to circumference calculation
        nb_runner.set_cell_source(
            1,
            "from dataclasses import dataclass\n@dataclass\nclass Circle:\n    radius: float\n    def area(self):\n        return 2 * 3.14159 * self.radius",
        )
        nb_runner.run_all()
        assert "area = 31.42" in nb_runner.get_output(2)
