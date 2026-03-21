"""Batch 502: namedtuple _asdict _replace operations."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestNamedtupleAsdictReplace:
    def test_namedtuple_asdict(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import namedtuple",
            "Point = namedtuple('Point', ['x', 'y', 'z'])\np = Point(1, 2, 3)\nd = p._asdict()\nprint(f'p={p} d={dict(d)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Point(x=1, y=2, z=3)" in out
        assert "'x': 1" in out

    def test_namedtuple_replace(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import namedtuple",
            "Color = namedtuple('Color', 'r g b')\nc1 = Color(255, 0, 0)\nc2 = c1._replace(g=128)\nc3 = c2._replace(b=255)\nprint(f'c1={c1} c2={c2} c3={c3}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "c1=Color(r=255, g=0, b=0)" in out
        assert "c2=Color(r=255, g=128, b=0)" in out
        assert "c3=Color(r=255, g=128, b=255)" in out

    def test_namedtuple_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import namedtuple",
            "Pair = namedtuple('Pair', 'a b')\np = Pair(10, 20)\nprint(f'sum={p.a + p.b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sum=30" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "Pair = namedtuple('Pair', 'a b')\np = Pair(100, 200)\nprint(f'sum={p.a + p.b}')")
        nb_runner.run_all()
        assert "sum=300" in nb_runner.get_output(2)
