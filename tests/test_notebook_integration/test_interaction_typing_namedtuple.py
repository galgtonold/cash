"""
Interaction test: typing.NamedTuple with methods and defaults.
Tests typing.NamedTuple with default values, custom methods,
and cross-cell tuple-based computations.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestTypingNamedTuple:
    """Test typing.NamedTuple with methods across cells."""

    def test_namedtuple_methods(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: define typed NamedTuple
            "from typing import NamedTuple\nclass Point(NamedTuple):\n    x: float\n    y: float\n    label: str = 'unnamed'\n    def distance_to(self, other):\n        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5\n    def shifted(self, dx, dy):\n        return Point(self.x + dx, self.y + dy, self.label)\nprint('Point defined')",
            # Cell 2: create and use
            "p1 = Point(0, 0, 'origin')\np2 = Point(3, 4, 'target')\ndist = p1.distance_to(p2)\nprint(f'p1={p1}')\nprint(f'p2={p2}')\nprint(f'dist={dist}')",
            # Cell 3: shift and measure
            "p3 = p1.shifted(1, 1)\nnew_dist = p3.distance_to(p2)\nprint(f'p3={p3}')\nprint(f'new_dist={new_dist:.2f}')\nprint(f'is_tuple={isinstance(p3, tuple)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "dist=5.0" in out2
        out3 = nb_runner.get_output(3)
        assert "p3=Point(x=1, y=1, label='origin')" in out3
        assert "new_dist=3.61" in out3
        assert "is_tuple=True" in out3

    def test_namedtuple_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from typing import NamedTuple\nclass Config(NamedTuple):\n    host: str = 'localhost'\n    port: int = 8080\n    debug: bool = False\nprint('Config defined')",
            "cfg = Config(port=3000, debug=True)\nprint(f'host={cfg.host}')\nprint(f'port={cfg.port}')",
            "url = f'http://{cfg.host}:{cfg.port}'\nprint(f'url={url}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "port=3000" in nb_runner.get_output(2)
        assert "url=http://localhost:3000" in nb_runner.get_output(3)

        # Edit config
        nb_runner.set_cell_source(2, "cfg = Config(host='0.0.0.0', port=9090, debug=False)\nprint(f'host={cfg.host}')\nprint(f'port={cfg.port}')")
        nb_runner.run_cells([2, 3])
        assert "port=9090" in nb_runner.get_output(2)
        assert "url=http://0.0.0.0:9090" in nb_runner.get_output(3)

    def test_namedtuple_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from typing import NamedTuple\nclass RGB(NamedTuple):\n    r: int\n    g: int\n    b: int\nprint('RGB defined')",
            "red = RGB(255, 0, 0)\nhex_color = f'#{red.r:02x}{red.g:02x}{red.b:02x}'\nprint(f'hex={hex_color}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hex=#ff0000" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "hex=#ff0000" in nb_runner.get_output(2)
