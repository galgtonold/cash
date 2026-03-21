"""Batch 471: tuple unpacking and starred assignment."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestTupleUnpackingStarred:
    def test_basic_unpack(self, nb_runner):
        nb_runner.create_notebook([
            "data = (10, 20, 30, 40, 50)",
            "first, second, *rest = data\nprint(f'first={first} second={second} rest={rest}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "first=10" in out
        assert "second=20" in out
        assert "rest=[30, 40, 50]" in out

    def test_nested_unpack(self, nb_runner):
        nb_runner.create_notebook([
            "records = [('Alice', 90), ('Bob', 85), ('Carol', 95)]",
            "names = []\nscores = []\nfor name, score in records:\n    names.append(name)\n    scores.append(score)\navg = sum(scores) / len(scores)\nprint(f'names={names} avg={avg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Alice" in out
        assert "avg=90.0" in out

    def test_unpack_edit(self, nb_runner):
        nb_runner.create_notebook([
            "coords = (1, 2, 3)",
            "x, y, z = coords\nprint(f'x={x} y={y} z={z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=1 y=2 z=3" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "coords = (100, 200, 300)")
        nb_runner.run_all()
        assert "x=100 y=200 z=300" in nb_runner.get_output(2)
