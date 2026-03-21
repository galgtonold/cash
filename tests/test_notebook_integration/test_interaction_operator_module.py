"""Batch 359: operator module and itemgetter/attrgetter patterns."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestOperatorModule:
    def test_itemgetter_sort(self, nb_runner):
        nb_runner.create_notebook([
            "from operator import itemgetter\ndata = [('Alice', 85), ('Bob', 92), ('Charlie', 78)]",
            "by_score = sorted(data, key=itemgetter(1), reverse=True)\nnames = [x[0] for x in by_score]\nprint(f'names={names}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "names=['Bob', 'Alice', 'Charlie']" in nb_runner.get_output(2)

    def test_attrgetter_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from operator import attrgetter\nclass Student:\n    def __init__(self, name, gpa):\n        self.name = name\n        self.gpa = gpa\n    def __repr__(self):\n        return f'{self.name}:{self.gpa}'",
            "students = [Student('A', 3.5), Student('B', 3.9), Student('C', 3.2)]",
            "ranked = sorted(students, key=attrgetter('gpa'), reverse=True)\nprint(f'ranked={ranked}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "ranked=[B:3.9, A:3.5, C:3.2]" in nb_runner.get_output(3)
        # Edit students
        nb_runner.set_cell_source(2, "students = [Student('X', 4.0), Student('Y', 2.8)]")
        nb_runner.run_all()
        assert "ranked=[X:4.0, Y:2.8]" in nb_runner.get_output(3)

    def test_methodcaller(self, nb_runner):
        nb_runner.create_notebook([
            "from operator import methodcaller\nwords = ['hello', 'WORLD', 'Python']",
            "upper_words = list(map(methodcaller('upper'), words))\nprint(f'upper={upper_words}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "upper=['HELLO', 'WORLD', 'PYTHON']" in nb_runner.get_output(2)
