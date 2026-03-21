"""Batch 409: enum class definitions and usage."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestEnumClassUsage:
    def test_enum_basic(self, nb_runner):
        nb_runner.create_notebook([
            "from enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3",
            "c = Color.RED\nname = c.name\nval = c.value\nprint(f'name={name} val={val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "name=RED" in nb_runner.get_output(2)
        assert "val=1" in nb_runner.get_output(2)

    def test_enum_iteration(self, nb_runner):
        nb_runner.create_notebook([
            "from enum import Enum\nclass Status(Enum):\n    OPEN = 'open'\n    CLOSED = 'closed'\n    PENDING = 'pending'",
            "names = [s.name for s in Status]\nvalues = [s.value for s in Status]\nprint(f'names={names} values={values}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "OPEN" in out
        assert "'open'" in out

    def test_enum_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from enum import Enum\nclass Dir(Enum):\n    NORTH = 0\n    SOUTH = 1",
            "count = len(Dir)\nfirst = Dir(0).name\nprint(f'count={count} first={first}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=2" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from enum import Enum\nclass Dir(Enum):\n    NORTH = 0\n    SOUTH = 1\n    EAST = 2\n    WEST = 3")
        nb_runner.run_all()
        assert "count=4" in nb_runner.get_output(2)
