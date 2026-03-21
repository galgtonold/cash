"""
Interaction test: enum with custom methods and class attributes.
Tests Enum with custom methods, classmethods, properties,
and cross-cell enum-based dispatch.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestEnumCustomMethods:
    """Test Enum with custom methods across cells."""

    def test_enum_methods(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: define enum with methods
            "from enum import Enum\nclass HttpStatus(Enum):\n    OK = 200\n    NOT_FOUND = 404\n    SERVER_ERROR = 500\n    @property\n    def is_error(self):\n        return self.value >= 400\n    @classmethod\n    def from_code(cls, code):\n        for member in cls:\n            if member.value == code:\n                return member\n        return None\n    def describe(self):\n        descriptions = {200: 'Success', 404: 'Not Found', 500: 'Internal Error'}\n        return descriptions.get(self.value, 'Unknown')\nprint('HttpStatus defined')",
            # Cell 2: use enum methods
            "s1 = HttpStatus.OK\ns2 = HttpStatus.from_code(404)\nprint(f's1_err={s1.is_error}')\nprint(f's2_err={s2.is_error}')\nprint(f's1_desc={s1.describe()}')\nprint(f's2_desc={s2.describe()}')",
            # Cell 3: iterate and filter
            "errors = [s for s in HttpStatus if s.is_error]\nprint(f'errors={[e.name for e in errors]}')\ncodes = [s.value for s in HttpStatus]\nprint(f'codes={codes}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "s1_err=False" in out2
        assert "s2_err=True" in out2
        assert "s1_desc=Success" in out2
        assert "s2_desc=Not Found" in out2
        out3 = nb_runner.get_output(3)
        assert "errors=['NOT_FOUND', 'SERVER_ERROR']" in out3
        assert "codes=[200, 404, 500]" in out3

    def test_enum_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from enum import Enum\nclass Color(Enum):\n    RED = 'red'\n    GREEN = 'green'\n    BLUE = 'blue'\n    def hex_code(self):\n        codes = {'red': '#FF0000', 'green': '#00FF00', 'blue': '#0000FF'}\n        return codes[self.value]\nprint('Color defined')",
            "colors = [Color.RED, Color.BLUE]\nhexes = [c.hex_code() for c in colors]\nprint(f'hexes={hexes}')",
            "all_hexes = [c.hex_code() for c in Color]\nprint(f'all={all_hexes}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hexes=['#FF0000', '#0000FF']" in nb_runner.get_output(2)

        # Edit selection
        nb_runner.set_cell_source(2, "colors = [Color.GREEN, Color.RED]\nhexes = [c.hex_code() for c in colors]\nprint(f'hexes={hexes}')")
        nb_runner.run_cells([2, 3])
        assert "hexes=['#00FF00', '#FF0000']" in nb_runner.get_output(2)

    def test_enum_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from enum import Enum\nclass Direction(Enum):\n    NORTH = (0, 1)\n    SOUTH = (0, -1)\n    EAST = (1, 0)\n    WEST = (-1, 0)\n    @property\n    def dx(self): return self.value[0]\n    @property\n    def dy(self): return self.value[1]\nprint('Direction defined')",
            "path = [Direction.NORTH, Direction.NORTH, Direction.EAST, Direction.EAST, Direction.SOUTH]\nfinal_x = sum(d.dx for d in path)\nfinal_y = sum(d.dy for d in path)\nprint(f'pos=({final_x},{final_y})')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "pos=(2,1)" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "pos=(2,1)" in nb_runner.get_output(2)
