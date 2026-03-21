"""Batch 338: dataclass field defaults, post_init, and frozen edits."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDataclassFieldEdits:
    def test_dataclass_post_init(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass, field\n@dataclass\nclass Rectangle:\n    width: float\n    height: float\n    area: float = field(init=False)\n    def __post_init__(self):\n        self.area = self.width * self.height",
            "r = Rectangle(3.0, 4.0)\nprint(f'area={r.area}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=12.0" in nb_runner.get_output(2)

    def test_dataclass_frozen_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass Point:\n    x: int\n    y: int",
            "p = Point(1, 2)\nresult = f'{p.x},{p.y}'\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=1,2" in nb_runner.get_output(2)
        # Edit class
        nb_runner.set_cell_source(1, "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass Point:\n    x: int\n    y: int\n    z: int = 0")
        nb_runner.set_cell_source(2, "p = Point(10, 20, 30)\nresult = f'{p.x},{p.y},{p.z}'\nprint(f'result={result}')")
        nb_runner.run_all()
        assert "result=10,20,30" in nb_runner.get_output(2)

    def test_dataclass_default_factory(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass, field\n@dataclass\nclass Config:\n    name: str = 'default'\n    tags: list = field(default_factory=list)",
            "c1 = Config('test', ['a', 'b'])\nc2 = Config()\nprint(f'c1={c1.name},{c1.tags}')\nprint(f'c2={c2.name},{c2.tags}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "c1=test,['a', 'b']" in out
        assert "c2=default,[]" in out
