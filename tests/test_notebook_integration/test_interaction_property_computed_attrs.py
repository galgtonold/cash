"""Batch 407: property decorators and computed attributes."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestPropertyComputedAttrs:
    def test_property_getter(self, nb_runner):
        nb_runner.create_notebook([
            "class Circle:\n    def __init__(self, radius):\n        self._radius = radius\n    @property\n    def area(self):\n        import math\n        return round(math.pi * self._radius ** 2, 2)\n    @property\n    def circumference(self):\n        import math\n        return round(2 * math.pi * self._radius, 2)",
            "c = Circle(5)\nprint(f'area={c.area} circ={c.circumference}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=78.54" in nb_runner.get_output(2)
        assert "circ=31.42" in nb_runner.get_output(2)

    def test_property_setter(self, nb_runner):
        nb_runner.create_notebook([
            "class Temp:\n    def __init__(self, celsius):\n        self._c = celsius\n    @property\n    def fahrenheit(self):\n        return self._c * 9/5 + 32\n    @fahrenheit.setter\n    def fahrenheit(self, f):\n        self._c = (f - 32) * 5/9",
            "t = Temp(100)\nf1 = t.fahrenheit\nt.fahrenheit = 32\nc1 = t._c\nprint(f'f1={f1} c1={c1}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "f1=212.0" in nb_runner.get_output(2)
        assert "c1=0.0" in nb_runner.get_output(2)

    def test_property_edit(self, nb_runner):
        nb_runner.create_notebook([
            "class Box:\n    def __init__(self, w, h):\n        self.w = w\n        self.h = h\n    @property\n    def area(self):\n        return self.w * self.h",
            "b = Box(3, 4)\nprint(f'area={b.area}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=12" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "class Box:\n    def __init__(self, w, h):\n        self.w = w\n        self.h = h\n    @property\n    def area(self):\n        return self.w * self.h")
        nb_runner.set_cell_source(2, "b = Box(10, 5)\nprint(f'area={b.area}')")
        nb_runner.run_all()
        assert "area=50" in nb_runner.get_output(2)
