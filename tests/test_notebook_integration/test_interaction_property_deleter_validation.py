"""
Interaction test: property with deleter and validation.
Tests property getter/setter/deleter with validation logic,
AttributeError handling, and cross-cell state management.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestPropertyDeleterValidation:
    """Test property with deleter and validation across cells."""

    def test_property_validation(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: define class with validated property
            "class Temperature:\n    def __init__(self, celsius):\n        self.celsius = celsius\n    @property\n    def celsius(self):\n        return self._celsius\n    @celsius.setter\n    def celsius(self, value):\n        if value < -273.15:\n            raise ValueError('Below absolute zero')\n        self._celsius = value\n    @celsius.deleter\n    def celsius(self):\n        self._celsius = 0.0\n    @property\n    def fahrenheit(self):\n        return self._celsius * 9/5 + 32\nprint('Temperature defined')",
            # Cell 2: use property
            "t = Temperature(100)\nprint(f'c={t.celsius}')\nprint(f'f={t.fahrenheit}')\nt.celsius = 0\nprint(f'freezing_f={t.fahrenheit}')",
            # Cell 3: deleter and validation
            "del t.celsius\nprint(f'after_del={t.celsius}')\ntry:\n    t.celsius = -300\n    print('no_error')\nexcept ValueError as e:\n    print(f'error={e}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "c=100" in out2
        assert "f=212.0" in out2
        assert "freezing_f=32.0" in out2
        out3 = nb_runner.get_output(3)
        assert "after_del=0.0" in out3
        assert "error=Below absolute zero" in out3

    def test_property_edit(self, nb_runner):
        nb_runner.create_notebook([
            "class Box:\n    def __init__(self, w, h):\n        self.w = w\n        self.h = h\n    @property\n    def area(self):\n        return self.w * self.h\nprint('Box defined')",
            "b = Box(5, 3)\nprint(f'area={b.area}')",
            "double_area = b.area * 2\nprint(f'double={double_area}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=15" in nb_runner.get_output(2)
        assert "double=30" in nb_runner.get_output(3)

        # Edit box dimensions
        nb_runner.set_cell_source(2, "b = Box(10, 7)\nprint(f'area={b.area}')")
        nb_runner.run_cells([2, 3])
        assert "area=70" in nb_runner.get_output(2)
        assert "double=140" in nb_runner.get_output(3)

    def test_property_cache(self, nb_runner):
        nb_runner.create_notebook([
            "class Circle:\n    def __init__(self, r):\n        self._r = r\n    @property\n    def radius(self):\n        return self._r\n    @radius.setter\n    def radius(self, val):\n        if val < 0:\n            raise ValueError('Negative')\n        self._r = val\nprint('Circle defined')",
            "c = Circle(5)\nprint(f'r={c.radius}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r=5" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "r=5" in nb_runner.get_output(2)
