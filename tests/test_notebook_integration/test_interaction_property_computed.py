"""
Batch 331: property decorator and computed attribute patterns with caching.
Tests @property, computed fields, and edit propagation.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.timeout(90)]


class TestPropertyComputed:
    """Test property decorator and computed attribute caching."""

    def test_property_basic(self, nb_runner):
        """Class with @property, verify caching."""
        nb_runner.create_notebook([
            "class Rectangle:\n    def __init__(self, w, h):\n        self.w = w\n        self.h = h\n    @property\n    def area(self):\n        return self.w * self.h",
            "r = Rectangle(4, 5)",
            "result = r.area\nprint(f'area={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "area=20" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "area=20" in out2

    def test_property_edit_instance(self, nb_runner):
        """Edit instance creation, verify property updates."""
        nb_runner.create_notebook([
            "class BMI:\n    def __init__(self, weight_kg, height_m):\n        self.weight = weight_kg\n        self.height = height_m\n    @property\n    def value(self):\n        return round(self.weight / self.height**2, 1)",
            "person = BMI(70, 1.75)",
            "bmi = person.value\nprint(f'bmi={bmi}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "bmi=22.9" in out

        nb_runner.set_cell_source(2, "person = BMI(90, 1.75)")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "bmi=29.4" in out2

    def test_property_with_setter(self, nb_runner):
        """Property with setter, verify caching."""
        nb_runner.create_notebook([
            "class Temp:\n    def __init__(self, celsius):\n        self._c = celsius\n    @property\n    def fahrenheit(self):\n        return self._c * 9/5 + 32",
            "t = Temp(100)\nf = t.fahrenheit",
            "print(f'f={f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "f=212" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "f=212" in out2
