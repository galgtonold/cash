"""Batch 121 – Class/OOP + cell edit interaction tests.

Tests that exercise class definitions, inheritance, method changes,
and how cash tracks class-related dependencies through cell edits.
"""

import pytest

pytestmark = [pytest.mark.core, pytest.mark.stress, pytest.mark.timeout(30)]


class TestClassDefinitionEdits:
    """Basic class definition + edit scenarios."""

    def test_edit_class_attribute(self, nb_runner):
        """Edit a class attribute and verify downstream update."""
        nb_runner.create_notebook([
            "class Config:\n    value = 10",
            "result = Config.value * 2\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "class Config:\n    value = 50")
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(2)


    def test_add_method_to_class(self, nb_runner):
        """Add a new method to a class."""
        nb_runner.create_notebook([
            "class Ops:\n    def add(self, a, b):\n        return a + b",
            "o = Ops()\nresult = o.add(3, 4)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 7" in nb_runner.get_output(2)

        # Add multiply method and use it
        nb_runner.set_cell_source(
            1,
            "class Ops:\n    def add(self, a, b):\n        return a + b\n    def mul(self, a, b):\n        return a * b",
        )
        nb_runner.set_cell_source(
            2,
            "o = Ops()\nresult = o.mul(3, 4)\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = 12" in nb_runner.get_output(2)


class TestInheritanceEdits:
    """Class inheritance + cell edit scenarios."""

    def test_edit_parent_class(self, nb_runner):
        """Edit parent class, verify child reflects the change."""
        nb_runner.create_notebook([
            "class Base:\n    factor = 2",
            "class Child(Base):\n    def compute(self, x):\n        return x * self.factor",
            "c = Child()\nresult = c.compute(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "class Base:\n    factor = 10")
        nb_runner.run_all()
        assert "result = 50" in nb_runner.get_output(3)

    def test_edit_child_class(self, nb_runner):
        """Edit child class, keep parent unchanged."""
        nb_runner.create_notebook([
            "class Base:\n    def greet(self):\n        return 'hello'",
            "class Child(Base):\n    def greet(self):\n        return 'hi from child'",
            "c = Child()\nresult = c.greet()\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = hi from child" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2,
            "class Child(Base):\n    def greet(self):\n        return super().greet() + ' world'",
        )
        nb_runner.run_all()
        assert "result = hello world" in nb_runner.get_output(3)



class TestClassInstanceEdits:
    """Class instance state + cell edits."""

    def test_edit_constructor_args(self, nb_runner):
        """Edit constructor arguments for a class instance."""
        nb_runner.create_notebook([
            "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def dist(self):\n        return (self.x ** 2 + self.y ** 2) ** 0.5",
            "p = Point(3, 4)",
            "d = p.dist()\nprint(f'd = {d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d = 5.0" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "p = Point(5, 12)")
        nb_runner.run_all()
        assert "d = 13.0" in nb_runner.get_output(3)

    def test_edit_class_then_instantiation(self, nb_runner):
        """Edit both class and its instantiation."""
        nb_runner.create_notebook([
            "class Msg:\n    def __init__(self, text):\n        self.text = text\n    def show(self):\n        return self.text.upper()",
            "m = Msg('hello')",
            "result = m.show()\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = HELLO" in nb_runner.get_output(3)

        # Edit both class and instantiation
        nb_runner.set_cell_source(
            1,
            "class Msg:\n    def __init__(self, text):\n        self.text = text\n    def show(self):\n        return self.text.lower()",
        )
        nb_runner.set_cell_source(2, "m = Msg('WORLD')")
        nb_runner.run_all()
        assert "result = world" in nb_runner.get_output(3)
