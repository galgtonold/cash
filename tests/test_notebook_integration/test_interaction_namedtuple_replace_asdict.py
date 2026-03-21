"""
Interaction test: namedtuple with _replace and _asdict.
Tests namedtuple creation, _replace for immutable update,
_asdict conversion, and cross-cell typed tuple pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestNamedtupleReplaceAsdict:
    """Test namedtuple _replace and _asdict across cells."""

    def test_namedtuple_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: create and use namedtuple
            "from collections import namedtuple\nPoint = namedtuple('Point', ['x', 'y', 'z'])\np1 = Point(1, 2, 3)\nprint(f'p1={p1}')\nprint(f'x={p1.x} y={p1.y} z={p1.z}')",
            # Cell 2: _replace
            "p2 = p1._replace(z=10)\nprint(f'p2={p2}')\nprint(f'p1_unchanged={p1}')",
            # Cell 3: _asdict
            "d = p2._asdict()\nprint(f'dict={dict(d)}')\nprint(f'sum={sum(d.values())}')",
        ])
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
        nb_runner.create_notebook([
            "from collections import namedtuple\nColor = namedtuple('Color', 'r g b')\nc = Color(255, 128, 0)\nprint(f'color={c}')",
            "hex_val = f'#{c.r:02x}{c.g:02x}{c.b:02x}'\nprint(f'hex={hex_val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hex=#ff8000" in nb_runner.get_output(2)

        # Edit color
        nb_runner.set_cell_source(1, "from collections import namedtuple\nColor = namedtuple('Color', 'r g b')\nc = Color(0, 128, 255)\nprint(f'color={c}')")
        nb_runner.run_cells([1, 2])
        assert "hex=#0080ff" in nb_runner.get_output(2)

    def test_namedtuple_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import namedtuple\nStudent = namedtuple('Student', ['name', 'grade', 'gpa'])\ns = Student('Alice', 'A', 3.9)\nprint(f'student={s}')",
            "info = f'{s.name}: {s.grade} ({s.gpa})'\nprint(f'info={info}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "info=Alice: A (3.9)" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "info=Alice: A (3.9)" in nb_runner.get_output(2)
