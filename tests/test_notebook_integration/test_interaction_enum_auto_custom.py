"""Batch 473: enum auto and custom value methods."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestEnumAutoCustom:
    def test_auto_values(self, nb_runner):
        nb_runner.create_notebook([
            "from enum import Enum, auto",
            "class Color(Enum):\n    RED = auto()\n    GREEN = auto()\n    BLUE = auto()\nprint(f'red={Color.RED.value} green={Color.GREEN.value} blue={Color.BLUE.value}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "red=1" in out
        assert "green=2" in out
        assert "blue=3" in out

    def test_custom_enum_method(self, nb_runner):
        nb_runner.create_notebook([
            "from enum import Enum",
            "class Planet(Enum):\n    EARTH = (5.97e24, 6.37e6)\n    MARS = (6.42e23, 3.39e6)\n    def __init__(self, mass, radius):\n        self.mass = mass\n        self.radius = radius\n    @property\n    def surface_gravity(self):\n        G = 6.674e-11\n        return G * self.mass / self.radius**2\nearth_g = round(Planet.EARTH.surface_gravity, 2)\nmars_g = round(Planet.MARS.surface_gravity, 2)\nprint(f'earth={earth_g} mars={mars_g}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "earth=9.82" in out
        assert "mars=3.73" in out

    def test_enum_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from enum import Enum, auto",
            "class Dir(Enum):\n    N = auto()\n    S = auto()\nprint(f'count={len(Dir)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=2" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "class Dir(Enum):\n    N = auto()\n    S = auto()\n    E = auto()\n    W = auto()\nprint(f'count={len(Dir)}')")
        nb_runner.run_all()
        assert "count=4" in nb_runner.get_output(2)
