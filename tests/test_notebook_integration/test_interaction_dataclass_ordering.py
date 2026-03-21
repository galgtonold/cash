"""Batch 385: dataclass ordering and comparison methods."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDataclassOrdering:
    def test_order_basic(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass\n@dataclass(order=True)\nclass Priority:\n    level: int\n    name: str",
            "items = [Priority(3, 'low'), Priority(1, 'high'), Priority(2, 'mid')]\nsorted_items = sorted(items)\nresult = [(p.level, p.name) for p in sorted_items]\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[(1, 'high'), (2, 'mid'), (3, 'low')]" in nb_runner.get_output(2)

    def test_order_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass, field\n@dataclass(order=True)\nclass Student:\n    gpa: float\n    name: str = field(compare=False)",
            "students = [Student(3.5, 'A'), Student(3.9, 'B'), Student(3.2, 'C')]\nbest = max(students)\nprint(f'best={best.name}:{best.gpa}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "best=B:3.9" in nb_runner.get_output(2)
        # Edit students
        nb_runner.set_cell_source(2, "students = [Student(4.0, 'X'), Student(3.8, 'Y')]\nbest = max(students)\nprint(f'best={best.name}:{best.gpa}')")
        nb_runner.run_all()
        assert "best=X:4.0" in nb_runner.get_output(2)

    def test_order_reverse(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass\n@dataclass(order=True)\nclass Score:\n    value: int",
            "scores = [Score(80), Score(95), Score(70), Score(88)]\ntop2 = sorted(scores, reverse=True)[:2]\nresult = [s.value for s in top2]\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[95, 88]" in nb_runner.get_output(2)
