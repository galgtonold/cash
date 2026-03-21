"""Batch 496: operator itemgetter attrgetter sort patterns."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestOperatorItemgetterAttrgetter:
    def test_itemgetter_sort(self, nb_runner):
        nb_runner.create_notebook([
            "from operator import itemgetter",
            "data = [('Alice', 90), ('Bob', 85), ('Carol', 95)]\nby_score = sorted(data, key=itemgetter(1))\nby_name = sorted(data, key=itemgetter(0))\nprint(f'by_score={[n for n,s in by_score]}')\nprint(f'by_name={[n for n,s in by_name]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "by_score=['Bob', 'Alice', 'Carol']" in out
        assert "by_name=['Alice', 'Bob', 'Carol']" in out

    def test_attrgetter_sort(self, nb_runner):
        nb_runner.create_notebook([
            "from operator import attrgetter",
            "class Student:\n    def __init__(self, name, gpa): self.name, self.gpa = name, gpa\nstudents = [Student('A', 3.5), Student('B', 3.9), Student('C', 3.2)]\nby_gpa = sorted(students, key=attrgetter('gpa'), reverse=True)\nnames = [s.name for s in by_gpa]\nprint(f'names={names}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "names=['B', 'A', 'C']" in nb_runner.get_output(2)

    def test_itemgetter_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from operator import itemgetter",
            "data = [(1, 'a'), (3, 'c'), (2, 'b')]\nresult = sorted(data, key=itemgetter(0))\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[(1, 'a'), (2, 'b'), (3, 'c')]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "data = [(1, 'z'), (3, 'a'), (2, 'm')]\nresult = sorted(data, key=itemgetter(1))\nprint(f'result={result}')")
        nb_runner.run_all()
        assert "result=[(3, 'a'), (2, 'm'), (1, 'z')]" in nb_runner.get_output(2)
