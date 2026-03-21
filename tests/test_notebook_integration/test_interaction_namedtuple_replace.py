"""Batch 443: namedtuple _replace and _asdict methods."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestNamedtupleReplace:
    def test_replace(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import namedtuple\nPoint = namedtuple('Point', ['x', 'y'])\np = Point(1, 2)",
            "p2 = p._replace(x=10)\nprint(f'orig={p} new={p2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "orig=Point(x=1, y=2)" in nb_runner.get_output(2)
        assert "new=Point(x=10, y=2)" in nb_runner.get_output(2)

    def test_asdict(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import namedtuple\nColor = namedtuple('Color', 'r g b')\nc = Color(255, 128, 0)",
            "d = c._asdict()\nprint(f'dict={d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'r': 255" in nb_runner.get_output(2)
        assert "'g': 128" in nb_runner.get_output(2)

    def test_namedtuple_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import namedtuple\nRecord = namedtuple('Record', 'name age')\nr = Record('Alice', 30)",
            "info = f'{r.name} is {r.age}'\nprint(f'info={info}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "info=Alice is 30" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from collections import namedtuple\nRecord = namedtuple('Record', 'name age')\nr = Record('Bob', 25)")
        nb_runner.run_all()
        assert "info=Bob is 25" in nb_runner.get_output(2)
