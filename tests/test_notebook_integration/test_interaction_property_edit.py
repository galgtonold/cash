"""Batch 271 – Property and descriptor edit patterns.

Tests @property, computed properties with class edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestPropertyDescriptorEdits:
    """Property and descriptor patterns."""

    def test_property_getter_edit(self, nb_runner):
        """Edit property getter, downstream reflects."""
        nb_runner.create_notebook([
            "class Circle:\n    def __init__(self, r):\n        self._r = r\n    @property\n    def area(self):\n        return 3.14 * self._r ** 2",
            "c = Circle(5)\nprint(f'area = {c.area}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 78.5" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "class Circle:\n    def __init__(self, r):\n        self._r = r\n    @property\n    def area(self):\n        return 3.14159 * self._r ** 2",
        )
        nb_runner.run_all()
        assert "78.539" in nb_runner.get_output(2)

    def test_computed_property_edit(self, nb_runner):
        """Edit computed property formula."""
        nb_runner.create_notebook([
            "class BMI:\n    def __init__(self, weight, height):\n        self.weight = weight\n        self.height = height\n    @property\n    def value(self):\n        return round(self.weight / (self.height ** 2), 1)",
            "bmi = BMI(70, 1.75)\nprint(f'bmi = {bmi.value}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "bmi = 22.9" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "class BMI:\n    def __init__(self, weight, height):\n        self.weight = weight\n        self.height = height\n    @property\n    def value(self):\n        return round(self.weight / (self.height ** 2) * 100, 1)",
        )
        nb_runner.run_all()
        assert "bmi = 2285.7" in nb_runner.get_output(2)

    def test_property_with_setter_edit(self, nb_runner):
        """Edit class with property setter."""
        nb_runner.create_notebook([
            "class Temperature:\n    def __init__(self, celsius):\n        self._c = celsius\n    @property\n    def fahrenheit(self):\n        return self._c * 9/5 + 32",
            "t = Temperature(100)\nprint(f'f = {t.fahrenheit}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "f = 212.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "t = Temperature(0)\nprint(f'f = {t.fahrenheit}')")
        nb_runner.run_all()
        assert "f = 32.0" in nb_runner.get_output(2)
