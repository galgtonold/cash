"""Batch 226 – Property/classmethod/staticmethod interaction tests.

Tests editing cells containing class features like properties,
classmethods, and staticmethods to verify cache invalidation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestClassFeatureEdits:
    """Editing class properties, classmethods, and staticmethods."""

    def test_edit_staticmethod(self, nb_runner):
        """Edit a staticmethod and verify downstream uses new version."""
        nb_runner.create_notebook([
            "class MathHelper:\n    @staticmethod\n    def double(x):\n        return x * 2",
            "result = MathHelper.double(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        # Edit to triple instead of double
        nb_runner.set_cell_source(1, "class MathHelper:\n    @staticmethod\n    def double(x):\n        return x * 3")
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(2)

    def test_edit_classmethod_factory(self, nb_runner):
        """Edit a classmethod factory and verify downstream."""
        nb_runner.create_notebook([
            "class Config:\n    def __init__(self, val):\n        self.val = val\n    @classmethod\n    def default(cls):\n        return cls(42)",
            "cfg = Config.default()\nprint(f'val = {cfg.val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 42" in nb_runner.get_output(2)

        # Change default value
        nb_runner.set_cell_source(1, "class Config:\n    def __init__(self, val):\n        self.val = val\n    @classmethod\n    def default(cls):\n        return cls(99)")
        nb_runner.run_all()
        assert "val = 99" in nb_runner.get_output(2)

    def test_edit_property_getter(self, nb_runner):
        """Edit a property getter and verify downstream."""
        nb_runner.create_notebook([
            "class Box:\n    def __init__(self, w, h):\n        self.w = w\n        self.h = h\n    @property\n    def area(self):\n        return self.w * self.h",
            "b = Box(3, 4)\nprint(f'area = {b.area}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 12" in nb_runner.get_output(2)

        # Change to perimeter
        nb_runner.set_cell_source(1, "class Box:\n    def __init__(self, w, h):\n        self.w = w\n        self.h = h\n    @property\n    def area(self):\n        return 2 * (self.w + self.h)")
        nb_runner.run_all()
        assert "area = 14" in nb_runner.get_output(2)

    def test_edit_dunder_repr(self, nb_runner):
        """Edit __repr__ and verify output changes."""
        nb_runner.create_notebook([
            "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def __repr__(self):\n        return f'({self.x}, {self.y})'",
            "p = Point(1, 2)\nprint(f'p = {p}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "p = (1, 2)" in nb_runner.get_output(2)

        # Edit __repr__
        nb_runner.set_cell_source(1, "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def __repr__(self):\n        return f'Point({self.x}, {self.y})'")
        nb_runner.run_all()
        assert "p = Point(1, 2)" in nb_runner.get_output(2)
