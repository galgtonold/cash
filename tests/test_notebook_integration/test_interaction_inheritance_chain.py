"""Batch 239 – Class inheritance chain edit tests.

Tests editing base/parent classes and verifying that changes
propagate through inheritance hierarchies.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestInheritanceChainEdits:
    """Editing classes in an inheritance hierarchy."""


    def test_edit_derived_class_override(self, nb_runner):
        """Edit a derived class to override a base method."""
        nb_runner.create_notebook([
            "class Shape:\n    def describe(self):\n        return 'shape'",
            "class Circle(Shape):\n    pass",
            "c = Circle()\nresult = c.describe()\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = shape" in nb_runner.get_output(3)

        # Add override
        nb_runner.set_cell_source(2, "class Circle(Shape):\n    def describe(self):\n        return 'circle'")
        nb_runner.run_all()
        assert "result = circle" in nb_runner.get_output(3)

    def test_edit_super_call(self, nb_runner):
        """Edit a class that uses super()."""
        nb_runner.create_notebook([
            "class Base:\n    def value(self):\n        return 10",
            "class Child(Base):\n    def value(self):\n        return super().value() + 5",
            "obj = Child()\nresult = obj.value()\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(3)

        # Edit base value
        nb_runner.set_cell_source(1, "class Base:\n    def value(self):\n        return 100")
        nb_runner.run_all()
        assert "result = 105" in nb_runner.get_output(3)
