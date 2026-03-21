"""Batch 167 – Class hierarchy interaction tests.

Tests editing base classes, overriding methods, adding/removing
inheritance, and multiple inheritance scenarios.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestInheritanceEdits:
    """Editing class hierarchies."""

    def test_edit_base_class_method(self, nb_runner):
        """Edit a method in the base class, verify subclass picks it up."""
        nb_runner.create_notebook([
            "class Animal:\n    def speak(self):\n        return 'generic sound'",
            "class Dog(Animal):\n    pass",
            "d = Dog()\nresult = d.speak()\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = generic sound" in nb_runner.get_output(3)

        # Edit base class
        nb_runner.set_cell_source(
            1, "class Animal:\n    def speak(self):\n        return 'LOUD sound'"
        )
        nb_runner.run_all()
        assert "result = LOUD sound" in nb_runner.get_output(3)

    def test_add_method_override(self, nb_runner):
        """Add a method override to a subclass."""
        nb_runner.create_notebook([
            "class Shape:\n    def area(self):\n        return 0",
            "class Circle(Shape):\n    def __init__(self, r):\n        self.r = r",
            "c = Circle(5)\nresult = c.area()\nprint(f'area = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 0" in nb_runner.get_output(3)

        # Override area in Circle
        nb_runner.set_cell_source(
            2,
            "import math as _math_mod\nclass Circle(Shape):\n    def __init__(self, r):\n        self.r = r\n    def area(self):\n        return _math_mod.pi * self.r ** 2",
        )
        nb_runner.set_cell_source(3, "c = Circle(5)\nresult = c.area()\nprint(f'area = {result:.2f}')")
        nb_runner.run_all()
        assert "area = 78.54" in nb_runner.get_output(3)

    def test_change_parent_class(self, nb_runner):
        """Change which class a subclass inherits from."""
        nb_runner.create_notebook([
            "class Base1:\n    val = 10",
            "class Base2:\n    val = 20",
            "class Child(Base1):\n    pass",
            "result = Child.val\nprint(f'val = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 10" in nb_runner.get_output(4)

        # Change parent
        nb_runner.set_cell_source(3, "class Child(Base2):\n    pass")
        nb_runner.run_all()
        assert "val = 20" in nb_runner.get_output(4)


class TestMultipleInheritance:
    """Multiple inheritance edits."""

    def test_mixin_edit(self, nb_runner):
        """Edit a mixin class in a multiple inheritance chain."""
        nb_runner.create_notebook([
            "class LogMixin:\n    def log(self):\n        return 'LOG:base'",
            "class Service:\n    name = 'svc'",
            "class App(LogMixin, Service):\n    pass",
            "a = App()\nprint(f'log={a.log()} name={a.name}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "log=LOG:base" in nb_runner.get_output(4)
        assert "name=svc" in nb_runner.get_output(4)

        # Edit mixin
        nb_runner.set_cell_source(
            1, "class LogMixin:\n    def log(self):\n        return 'LOG:v2'"
        )
        nb_runner.run_all()
        assert "log=LOG:v2" in nb_runner.get_output(4)

    def test_add_class_attribute(self, nb_runner):
        """Add a class attribute to a parent, use in child."""
        nb_runner.create_notebook([
            "class Config:\n    debug = False",
            "class App(Config):\n    name = 'myapp'",
            "a = App()\nprint(f'debug={a.debug} name={a.name}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "debug=False" in nb_runner.get_output(3)

        # Enable debug in parent
        nb_runner.set_cell_source(1, "class Config:\n    debug = True")
        nb_runner.run_all()
        assert "debug=True" in nb_runner.get_output(3)
