"""Batch 145 – Class and OOP interaction tests.

Tests where users define classes in cells, edit methods/attributes,
instantiate objects, and verify caching tracks OOP changes.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(45)]


class TestClassDefinitionEdits:
    """Edit class definitions and verify instances update."""

    def test_edit_class_method(self, nb_runner):
        """Edit a method in a class."""
        nb_runner.create_notebook([
            "class Calculator:\n    def compute(self, x):\n        return x * 2",
            "calc = Calculator()\nresult = calc.compute(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        # Edit method
        nb_runner.set_cell_source(
            1, "class Calculator:\n    def compute(self, x):\n        return x ** 2"
        )
        nb_runner.run_all()
        assert "result = 25" in nb_runner.get_output(2)

    def test_add_method_to_class(self, nb_runner):
        """Add a new method to a class."""
        nb_runner.create_notebook([
            "class Stats:\n    def mean(self, data):\n        return sum(data) / len(data)",
            "s = Stats()\nresult = s.mean([10, 20, 30])\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20.0" in nb_runner.get_output(2)

        # Add method and use it
        nb_runner.set_cell_source(
            1,
            "class Stats:\n    def mean(self, data):\n        return sum(data) / len(data)\n    def total(self, data):\n        return sum(data)",
        )
        nb_runner.set_cell_source(
            2,
            "s = Stats()\nresult = s.total([10, 20, 30])\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = 60" in nb_runner.get_output(2)

    def test_edit_class_init(self, nb_runner):
        """Edit __init__ of a class."""
        nb_runner.create_notebook([
            "class Multiplier:\n    def __init__(self, factor):\n        self.factor = factor\n    def apply(self, x):\n        return x * self.factor",
            "m = Multiplier(3)\nresult = m.apply(10)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(2)

        # Edit the instantiation
        nb_runner.set_cell_source(
            2,
            "m = Multiplier(5)\nresult = m.apply(10)\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = 50" in nb_runner.get_output(2)


class TestInheritanceEdits:
    """Inheritance patterns with edits."""

    def test_edit_base_class(self, nb_runner):
        """Edit base class, verify subclass updates."""
        nb_runner.create_notebook([
            "class Base:\n    def value(self):\n        return 10",
            "class Child(Base):\n    def doubled(self):\n        return self.value() * 2",
            "c = Child()\nresult = c.doubled()\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(3)

        # Edit base
        nb_runner.set_cell_source(
            1, "class Base:\n    def value(self):\n        return 100"
        )
        nb_runner.run_all()
        assert "result = 200" in nb_runner.get_output(3)

    def test_edit_child_class(self, nb_runner):
        """Edit child class only."""
        nb_runner.create_notebook([
            "class Base:\n    def value(self):\n        return 5",
            "class Child(Base):\n    def compute(self):\n        return self.value() + 1",
            "c = Child()\nresult = c.compute()\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 6" in nb_runner.get_output(3)

        # Edit child
        nb_runner.set_cell_source(
            2, "class Child(Base):\n    def compute(self):\n        return self.value() * 10"
        )
        nb_runner.run_all()
        assert "result = 50" in nb_runner.get_output(3)
