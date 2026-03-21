"""Batch 391: class property setter/deleter and computed attrs."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestPropertySetterDeleter:
    def test_property_getter_setter(self, nb_runner):
        nb_runner.create_notebook([
            "class Temperature:\n    def __init__(self, celsius):\n        self._celsius = celsius\n    @property\n    def fahrenheit(self):\n        return self._celsius * 9/5 + 32\n    @fahrenheit.setter\n    def fahrenheit(self, value):\n        self._celsius = (value - 32) * 5/9",
            "t = Temperature(100)\nf = t.fahrenheit\nt.fahrenheit = 32\nc = t._celsius\nprint(f'f={f} c={c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "f=212.0" in nb_runner.get_output(2)
        assert "c=0.0" in nb_runner.get_output(2)

    def test_property_edit_class(self, nb_runner):
        nb_runner.create_notebook([
            "class Circle:\n    def __init__(self, radius):\n        self.radius = radius\n    @property\n    def area(self):\n        import math\n        return round(math.pi * self.radius ** 2, 2)",
            "c = Circle(5)\nresult = c.area\nprint(f'area={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=78.54" in nb_runner.get_output(2)
        # Edit to add circumference
        nb_runner.set_cell_source(1, "class Circle:\n    def __init__(self, radius):\n        self.radius = radius\n    @property\n    def area(self):\n        import math\n        return round(math.pi * self.radius ** 2, 2)\n    @property\n    def circumference(self):\n        import math\n        return round(2 * math.pi * self.radius, 2)")
        nb_runner.set_cell_source(2, "c = Circle(5)\nresult = c.circumference\nprint(f'circ={result}')")
        nb_runner.run_all()
        assert "circ=31.42" in nb_runner.get_output(2)

    def test_property_validation(self, nb_runner):
        nb_runner.create_notebook([
            "class Age:\n    def __init__(self, value):\n        self.value = value\n    @property\n    def value(self):\n        return self._value\n    @value.setter\n    def value(self, v):\n        self._value = max(0, min(150, v))",
            "a = Age(200)\nresult = a.value\nprint(f'clamped={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "clamped=150" in nb_runner.get_output(2)
