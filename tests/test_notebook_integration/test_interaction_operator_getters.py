"""Batch 434: operator module itemgetter and attrgetter."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestOperatorGetters:
    def test_itemgetter(self, nb_runner):
        nb_runner.create_notebook([
            "from operator import itemgetter\ndata = [('Alice', 85), ('Bob', 92), ('Charlie', 78)]",
            "by_score = sorted(data, key=itemgetter(1), reverse=True)\ntop = by_score[0]\nprint(f'top={top}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "top=('Bob', 92)" in nb_runner.get_output(2)

    def test_attrgetter(self, nb_runner):
        nb_runner.create_notebook([
            "from operator import attrgetter\nclass Student:\n    def __init__(self, name, gpa):\n        self.name = name\n        self.gpa = gpa\nstudents = [Student('X', 3.5), Student('Y', 3.9), Student('Z', 3.2)]",
            "best = max(students, key=attrgetter('gpa'))\nprint(f'best={best.name} gpa={best.gpa}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "best=Y" in nb_runner.get_output(2)
        assert "gpa=3.9" in nb_runner.get_output(2)

    def test_itemgetter_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from operator import itemgetter\nrecords = [{'name': 'a', 'val': 3}, {'name': 'b', 'val': 1}, {'name': 'c', 'val': 2}]",
            "ordered = sorted(records, key=itemgetter('val'))\nfirst = ordered[0]['name']\nprint(f'first={first}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first=b" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from operator import itemgetter\nrecords = [{'name': 'x', 'val': 5}, {'name': 'y', 'val': 2}, {'name': 'z', 'val': 8}]")
        nb_runner.run_all()
        assert "first=y" in nb_runner.get_output(2)
