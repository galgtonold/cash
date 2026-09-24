"""Classes with __slots__, and interned strings, across cells."""

import textwrap

import pytest


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestSlotsInheritance:
    """Test __slots__ with inheritance across cells."""

    def test_slots_inheritance(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: define base with slots
                "class Point2D:\n    __slots__ = ('x', 'y')\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def __repr__(self):\n        return f'P2({self.x},{self.y})'\nclass Point3D(Point2D):\n    __slots__ = ('z',)\n    def __init__(self, x, y, z):\n        super().__init__(x, y)\n        self.z = z\n    def __repr__(self):\n        return f'P3({self.x},{self.y},{self.z})'\nprint('Point classes defined')",
                # Cell 2: create and use
                "p2 = Point2D(1, 2)\np3 = Point3D(3, 4, 5)\nprint(f'p2={p2}')\nprint(f'p3={p3}')\nprint(f'p2_slots={Point2D.__slots__}')\nprint(f'p3_slots={Point3D.__slots__}')",
                # Cell 3: compute with points
                "import math\ndist = math.sqrt((p3.x - p2.x)**2 + (p3.y - p2.y)**2)\nprint(f'dist_2d={dist:.2f}')\ndist_3d = math.sqrt(p3.x**2 + p3.y**2 + p3.z**2)\nprint(f'dist_origin={dist_3d:.2f}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "class Vec:\n    __slots__ = ('x', 'y')\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def mag(self):\n        return (self.x**2 + self.y**2)**0.5\nprint('Vec defined')",
                "v = Vec(3, 4)\nprint(f'mag={v.mag()}')",
                "scaled_mag = v.mag() * 2\nprint(f'scaled={scaled_mag}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "class Coord:\n    __slots__ = ('lat', 'lon')\n    def __init__(self, lat, lon):\n        self.lat = lat\n        self.lon = lon\nprint('Coord defined')",
                "c = Coord(40.7128, -74.0060)\nhas_dict = hasattr(c, '__dict__')\nprint(f'lat={c.lat}')\nprint(f'has_dict={has_dict}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lat=40.7128" in nb_runner.get_output(2)
        assert "has_dict=False" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "has_dict=False" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestSlotsMemory:
    """class __slots__ for memory efficiency."""

    def test_slots_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Point:\n    __slots__ = ('x', 'y')\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y",
                "p = Point(3, 4)\nhas_dict = hasattr(p, '__dict__')\nhas_slots = hasattr(p, '__slots__')\nprint(f'x={p.x} y={p.y} has_dict={has_dict} has_slots={has_slots}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "x=3" in out
        assert "has_dict=False" in out
        assert "has_slots=True" in out

    def test_slots_no_extra_attr(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Pair:\n    __slots__ = ('a', 'b')\n    def __init__(self, a, b): self.a, self.b = a, b",
                "p = Pair(1, 2)\ntry:\n    p.c = 3\n    result = 'no_error'\nexcept AttributeError:\n    result = 'attr_error'\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=attr_error" in nb_runner.get_output(2)

    def test_slots_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Vec:\n    __slots__ = ('x', 'y')\n    def __init__(self, x, y): self.x, self.y = x, y\n    def mag_sq(self): return self.x**2 + self.y**2",
                "v = Vec(3, 4)\nprint(f'mag_sq={v.mag_sq()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mag_sq=25" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "v = Vec(5, 12)\nprint(f'mag_sq={v.mag_sq()}')")
        nb_runner.run_all()
        assert "mag_sq=169" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestSlotsOptimization:
    """class slots optimization and attribute access."""

    def test_slots_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Point:\n    __slots__ = ('x', 'y')\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y",
                "p = Point(3, 4)\nresult = f'{p.x},{p.y}'\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=3,4" in nb_runner.get_output(2)

    def test_slots_edit_class(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Vec:\n    __slots__ = ('x', 'y')\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def mag(self):\n        return (self.x**2 + self.y**2) ** 0.5",
                "v = Vec(3, 4)\nm = v.mag()\nprint(f'm={m}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "m=5.0" in nb_runner.get_output(2)
        # Edit to 3D
        nb_runner.set_cell_source(
            1,
            "class Vec:\n    __slots__ = ('x', 'y', 'z')\n    def __init__(self, x, y, z):\n        self.x = x\n        self.y = y\n        self.z = z\n    def mag(self):\n        return (self.x**2 + self.y**2 + self.z**2) ** 0.5",
        )
        nb_runner.set_cell_source(2, "v = Vec(1, 2, 2)\nm = v.mag()\nprint(f'm={m}')")
        nb_runner.run_all()
        assert "m=3.0" in nb_runner.get_output(2)

    def test_slots_no_dict(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Tiny:\n    __slots__ = ('val',)\n    def __init__(self, v):\n        self.val = v",
                "t = Tiny(42)\nhas_dict = hasattr(t, '__dict__')\nprint(f'val={t.val} has_dict={has_dict}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=42 has_dict=False" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestSlotsInteraction:
    """Test __slots__ class patterns with cache invalidation."""

    def test_slots_class_instance_edit(self, nb_runner):
        """Editing a slots-based class instance should propagate."""
        nb_runner.create_notebook(
            [
                (
                    "class Vector:\n"
                    "    __slots__ = ('x', 'y')\n"
                    "    def __init__(self, x, y):\n"
                    "        self.x = x\n"
                    "        self.y = y\n"
                    "    def magnitude(self):\n"
                    "        return (self.x**2 + self.y**2) ** 0.5"
                ),
                "v = Vector(3, 4)",
                "mag = round(v.magnitude(), 2)",
                "print(f'mag={mag}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "mag=5.0" in out

        nb_runner.set_cell_source(2, "v = Vector(5, 12)")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "mag=13.0" in out

    def test_slots_class_definition_edit(self, nb_runner):
        """Editing the class definition with __slots__ should propagate."""
        nb_runner.create_notebook(
            [
                (
                    "class Record:\n"
                    "    __slots__ = ('name', 'value')\n"
                    "    def __init__(self, name, value):\n"
                    "        self.name = name\n"
                    "        self.value = value\n"
                    "    def display(self):\n"
                    "        return f'{self.name}={self.value}'"
                ),
                "r = Record('temp', 25)",
                "text = r.display()",
                "print(f'text={text}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "text=temp=25" in out

        # Edit class to change display format
        nb_runner.set_cell_source(
            1,
            (
                "class Record:\n"
                "    __slots__ = ('name', 'value')\n"
                "    def __init__(self, name, value):\n"
                "        self.name = name\n"
                "        self.value = value\n"
                "    def display(self):\n"
                "        return f'{self.name}: {self.value}'"
            ),
        )
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "text=temp: 25" in out


@pytest.mark.stress
class TestMemoryOptimization:
    """Test memory optimization patterns."""

    def test_intern_strings(self, nb_runner):
        """sys.intern for string optimization across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                import sys

                # Create interned strings
                categories = [sys.intern(f"cat_{i % 5}") for i in range(100)]
                unique = set(categories)
                print(f"total={len(categories)} unique={len(unique)}")
            """),
                textwrap.dedent("""\
                from collections import Counter
                counts = Counter(categories)
                print(f"counts={dict(sorted(counts.items()))}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=100 unique=5" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "cat_0" in out2
        assert "20" in out2  # Each category appears 20 times

    def test_slots_change_propagation(self, nb_runner):
        """Slots class — value extraction propagates changes."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                multiplier = 2
            """),
                textwrap.dedent("""\
                class Config:
                    __slots__ = ('debug', 'level')
                    def __init__(self, debug, level):
                        self.debug = debug
                        self.level = level

                cfg = Config(True, multiplier * 10)
                level_val = cfg.level
                print(f"level={level_val}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "level=20" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            multiplier = 5
        """),
        )
        nb_runner.run_cells([1, 2])
        assert "level=50" in nb_runner.get_output(2)
