"""
Interaction test: class __slots__ with inheritance and memory optimization.
Tests __slots__ classes with inheritance, MRO slot resolution,
and cross-cell attribute access patterns.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestSlotsInheritance:
    """Test __slots__ with inheritance across cells."""

    def test_slots_inheritance(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: define base with slots
            "class Point2D:\n    __slots__ = ('x', 'y')\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def __repr__(self):\n        return f'P2({self.x},{self.y})'\nclass Point3D(Point2D):\n    __slots__ = ('z',)\n    def __init__(self, x, y, z):\n        super().__init__(x, y)\n        self.z = z\n    def __repr__(self):\n        return f'P3({self.x},{self.y},{self.z})'\nprint('Point classes defined')",
            # Cell 2: create and use
            "p2 = Point2D(1, 2)\np3 = Point3D(3, 4, 5)\nprint(f'p2={p2}')\nprint(f'p3={p3}')\nprint(f'p2_slots={Point2D.__slots__}')\nprint(f'p3_slots={Point3D.__slots__}')",
            # Cell 3: compute with points
            "import math\ndist = math.sqrt((p3.x - p2.x)**2 + (p3.y - p2.y)**2)\nprint(f'dist_2d={dist:.2f}')\ndist_3d = math.sqrt(p3.x**2 + p3.y**2 + p3.z**2)\nprint(f'dist_origin={dist_3d:.2f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "p2=P2(1,2)" in out2
        assert "p3=P3(3,4,5)" in out2
        assert "p2_slots=('x', 'y')" in out2
        assert "p3_slots=('z',)" in out2
        out3 = nb_runner.get_output(3)
        assert "dist_2d=2.83" in out3
        assert "dist_origin=7.07" in out3

    def test_slots_edit(self, nb_runner):
        nb_runner.create_notebook([
            "class Vec:\n    __slots__ = ('x', 'y')\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def mag(self):\n        return (self.x**2 + self.y**2)**0.5\nprint('Vec defined')",
            "v = Vec(3, 4)\nprint(f'mag={v.mag()}')",
            "scaled_mag = v.mag() * 2\nprint(f'scaled={scaled_mag}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mag=5.0" in nb_runner.get_output(2)
        assert "scaled=10.0" in nb_runner.get_output(3)

        # Edit vector values
        nb_runner.set_cell_source(2, "v = Vec(5, 12)\nprint(f'mag={v.mag()}')")
        nb_runner.run_cells([2, 3])
        assert "mag=13.0" in nb_runner.get_output(2)
        assert "scaled=26.0" in nb_runner.get_output(3)

    def test_slots_cache(self, nb_runner):
        nb_runner.create_notebook([
            "class Coord:\n    __slots__ = ('lat', 'lon')\n    def __init__(self, lat, lon):\n        self.lat = lat\n        self.lon = lon\nprint('Coord defined')",
            "c = Coord(40.7128, -74.0060)\nhas_dict = hasattr(c, '__dict__')\nprint(f'lat={c.lat}')\nprint(f'has_dict={has_dict}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lat=40.7128" in nb_runner.get_output(2)
        assert "has_dict=False" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "has_dict=False" in nb_runner.get_output(2)
