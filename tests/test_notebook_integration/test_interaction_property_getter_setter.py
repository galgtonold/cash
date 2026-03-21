"""Batch 466: property decorators getters and setters."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestPropertyGetterSetter:
    def test_property_basic(self, nb_runner):
        nb_runner.create_notebook([
            "class Circle:\n    def __init__(self, r): self._r = r\n    @property\n    def radius(self): return self._r\n    @radius.setter\n    def radius(self, val):\n        if val < 0: raise ValueError\n        self._r = val\n    @property\n    def area(self): return 3.14159 * self._r ** 2",
            "c = Circle(5)\na1 = round(c.area, 2)\nc.radius = 10\na2 = round(c.area, 2)\nprint(f'a1={a1} a2={a2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a1=78.54" in nb_runner.get_output(2)
        assert "a2=314.16" in nb_runner.get_output(2)

    def test_property_validation(self, nb_runner):
        nb_runner.create_notebook([
            "class Temp:\n    def __init__(self, c): self._c = c\n    @property\n    def fahrenheit(self): return self._c * 9/5 + 32\n    @property\n    def celsius(self): return self._c",
            "t = Temp(100)\nprint(f'c={t.celsius} f={t.fahrenheit}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c=100" in nb_runner.get_output(2)
        assert "f=212.0" in nb_runner.get_output(2)

    def test_property_edit(self, nb_runner):
        nb_runner.create_notebook([
            "class Box:\n    def __init__(self, w, h): self.w, self.h = w, h\n    @property\n    def area(self): return self.w * self.h",
            "b = Box(3, 4)\nprint(f'area={b.area}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=12" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "b = Box(10, 20)\nprint(f'area={b.area}')")
        nb_runner.run_all()
        assert "area=200" in nb_runner.get_output(2)
