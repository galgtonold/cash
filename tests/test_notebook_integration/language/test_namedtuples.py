"""namedtuple and typing.NamedTuple across cells."""

import pytest


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestNamedtupleReplace:
    """namedtuple _replace and _asdict methods."""

    def test_replace(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import namedtuple\nPoint = namedtuple('Point', ['x', 'y'])\np = Point(1, 2)",
                "p2 = p._replace(x=10)\nprint(f'orig={p} new={p2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "orig=Point(x=1, y=2)" in nb_runner.get_output(2)
        assert "new=Point(x=10, y=2)" in nb_runner.get_output(2)

    def test_asdict(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import namedtuple\nColor = namedtuple('Color', 'r g b')\nc = Color(255, 128, 0)",
                "d = c._asdict()\nprint(f'dict={d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'r': 255" in nb_runner.get_output(2)
        assert "'g': 128" in nb_runner.get_output(2)

    def test_namedtuple_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import namedtuple\nRecord = namedtuple('Record', 'name age')\nr = Record('Alice', 30)",
                "info = f'{r.name} is {r.age}'\nprint(f'info={info}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "info=Alice is 30" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            1, "from collections import namedtuple\nRecord = namedtuple('Record', 'name age')\nr = Record('Bob', 25)"
        )
        nb_runner.run_all()
        assert "info=Bob is 25" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestNamedtupleReplaceAsdict:
    """Test namedtuple _replace and _asdict across cells."""

    def test_namedtuple_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: create and use namedtuple
                "from collections import namedtuple\nPoint = namedtuple('Point', ['x', 'y', 'z'])\np1 = Point(1, 2, 3)\nprint(f'p1={p1}')\nprint(f'x={p1.x} y={p1.y} z={p1.z}')",
                # Cell 2: _replace
                "p2 = p1._replace(z=10)\nprint(f'p2={p2}')\nprint(f'p1_unchanged={p1}')",
                # Cell 3: _asdict
                "d = p2._asdict()\nprint(f'dict={dict(d)}')\nprint(f'sum={sum(d.values())}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "p1=Point(x=1, y=2, z=3)" in out1
        out2 = nb_runner.get_output(2)
        assert "p2=Point(x=1, y=2, z=10)" in out2
        assert "p1_unchanged=Point(x=1, y=2, z=3)" in out2
        out3 = nb_runner.get_output(3)
        assert "sum=13" in out3

    def test_namedtuple_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import namedtuple\nColor = namedtuple('Color', 'r g b')\nc = Color(255, 128, 0)\nprint(f'color={c}')",
                "hex_val = f'#{c.r:02x}{c.g:02x}{c.b:02x}'\nprint(f'hex={hex_val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hex=#ff8000" in nb_runner.get_output(2)

        # Edit color
        nb_runner.set_cell_source(
            1,
            "from collections import namedtuple\nColor = namedtuple('Color', 'r g b')\nc = Color(0, 128, 255)\nprint(f'color={c}')",
        )
        nb_runner.run_cells([1, 2])
        assert "hex=#0080ff" in nb_runner.get_output(2)

    def test_namedtuple_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import namedtuple\nStudent = namedtuple('Student', ['name', 'grade', 'gpa'])\ns = Student('Alice', 'A', 3.9)\nprint(f'student={s}')",
                "info = f'{s.name}: {s.grade} ({s.gpa})'\nprint(f'info={info}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "info=Alice: A (3.9)" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "info=Alice: A (3.9)" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestNamedtupleAsdictReplace:
    """namedtuple _asdict _replace operations."""

    def test_namedtuple_asdict(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import namedtuple",
                "Point = namedtuple('Point', ['x', 'y', 'z'])\np = Point(1, 2, 3)\nd = p._asdict()\nprint(f'p={p} d={dict(d)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Point(x=1, y=2, z=3)" in out
        assert "'x': 1" in out

    def test_namedtuple_replace(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import namedtuple",
                "Color = namedtuple('Color', 'r g b')\nc1 = Color(255, 0, 0)\nc2 = c1._replace(g=128)\nc3 = c2._replace(b=255)\nprint(f'c1={c1} c2={c2} c3={c3}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "c1=Color(r=255, g=0, b=0)" in out
        assert "c2=Color(r=255, g=128, b=0)" in out
        assert "c3=Color(r=255, g=128, b=255)" in out

    def test_namedtuple_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import namedtuple",
                "Pair = namedtuple('Pair', 'a b')\np = Pair(10, 20)\nprint(f'sum={p.a + p.b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sum=30" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "Pair = namedtuple('Pair', 'a b')\np = Pair(100, 200)\nprint(f'sum={p.a + p.b}')")
        nb_runner.run_all()
        assert "sum=300" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestTypingNamedTuple:
    """Test typing.NamedTuple with methods across cells."""

    def test_namedtuple_methods(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: define typed NamedTuple
                "from typing import NamedTuple\nclass Point(NamedTuple):\n    x: float\n    y: float\n    label: str = 'unnamed'\n    def distance_to(self, other):\n        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5\n    def shifted(self, dx, dy):\n        return Point(self.x + dx, self.y + dy, self.label)\nprint('Point defined')",
                # Cell 2: create and use
                "p1 = Point(0, 0, 'origin')\np2 = Point(3, 4, 'target')\ndist = p1.distance_to(p2)\nprint(f'p1={p1}')\nprint(f'p2={p2}')\nprint(f'dist={dist}')",
                # Cell 3: shift and measure
                "p3 = p1.shifted(1, 1)\nnew_dist = p3.distance_to(p2)\nprint(f'p3={p3}')\nprint(f'new_dist={new_dist:.2f}')\nprint(f'is_tuple={isinstance(p3, tuple)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "dist=5.0" in out2
        out3 = nb_runner.get_output(3)
        assert "p3=Point(x=1, y=1, label='origin')" in out3
        assert "new_dist=3.61" in out3
        assert "is_tuple=True" in out3

    def test_namedtuple_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from typing import NamedTuple\nclass Config(NamedTuple):\n    host: str = 'localhost'\n    port: int = 8080\n    debug: bool = False\nprint('Config defined')",
                "cfg = Config(port=3000, debug=True)\nprint(f'host={cfg.host}')\nprint(f'port={cfg.port}')",
                "url = f'http://{cfg.host}:{cfg.port}'\nprint(f'url={url}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "port=3000" in nb_runner.get_output(2)
        assert "url=http://localhost:3000" in nb_runner.get_output(3)

        # Edit config
        nb_runner.set_cell_source(
            2,
            "cfg = Config(host='0.0.0.0', port=9090, debug=False)\nprint(f'host={cfg.host}')\nprint(f'port={cfg.port}')",
        )
        nb_runner.run_cells([2, 3])
        assert "port=9090" in nb_runner.get_output(2)
        assert "url=http://0.0.0.0:9090" in nb_runner.get_output(3)

    def test_namedtuple_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from typing import NamedTuple\nclass RGB(NamedTuple):\n    r: int\n    g: int\n    b: int\nprint('RGB defined')",
                "red = RGB(255, 0, 0)\nhex_color = f'#{red.r:02x}{red.g:02x}{red.b:02x}'\nprint(f'hex={hex_color}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hex=#ff0000" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "hex=#ff0000" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestNamedtupleDataclass:
    """Test namedtuple and dataclass caching."""

    def test_namedtuple_basic(self, nb_runner):
        """Create and use namedtuple, verify caching."""
        nb_runner.create_notebook(
            [
                "from collections import namedtuple",
                "Point = namedtuple('Point', ['x', 'y'])",
                "p = Point(3, 4)\ndist = (p.x**2 + p.y**2)**0.5",
                "print(f'dist={dist}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "dist=5.0" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "dist=5.0" in out2

    def test_namedtuple_edit_values(self, nb_runner):
        """Edit namedtuple values, verify propagation."""
        nb_runner.create_notebook(
            [
                "from collections import namedtuple",
                "RGB = namedtuple('RGB', 'r g b')\ncolor = RGB(255, 0, 0)",
                "brightness = (color.r + color.g + color.b) // 3",
                "print(f'brightness={brightness}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "brightness=85" in out

        nb_runner.set_cell_source(2, "RGB = namedtuple('RGB', 'r g b')\ncolor = RGB(0, 255, 0)")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "brightness=85" in out2

    def test_dataclass_pattern(self, nb_runner):
        """Dataclass creation and field access with caching."""
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass",
                "@dataclass\nclass Item:\n    name: str\n    price: float\n    qty: int = 1",
                "item = Item('Widget', 9.99, 5)\ntotal = item.price * item.qty",
                "print(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=49.95" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "total=49.95" in out2
