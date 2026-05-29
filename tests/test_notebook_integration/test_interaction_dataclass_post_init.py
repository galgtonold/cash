"""Batch 468: dataclass post init and field defaults."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDataclassPostInit:

    def test_field_default_factory(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass, field",
            "@dataclass\nclass Config:\n    name: str = 'default'\n    tags: list = field(default_factory=list)\nc1 = Config()\nc2 = Config('custom', ['a', 'b'])\nprint(f'c1={c1.name}:{c1.tags} c2={c2.name}:{c2.tags}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "c1=default:[]" in out
        assert "c2=custom:['a', 'b']" in out

    def test_dataclass_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass, field",
            "@dataclass\nclass Point:\n    x: float = 0\n    y: float = 0\np = Point(1, 2)\nprint(f'p={p.x},{p.y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "p=1,2" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "@dataclass\nclass Point:\n    x: float = 0\n    y: float = 0\np = Point(100, 200)\nprint(f'p={p.x},{p.y}')")
        nb_runner.run_all()
        assert "p=100,200" in nb_runner.get_output(2)
