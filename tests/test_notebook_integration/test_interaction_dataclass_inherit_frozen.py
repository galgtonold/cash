"""Batch 523: dataclass inheritance and frozen."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDataclassInheritanceFrozen:
    def test_dataclass_inheritance(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass",
            "@dataclass\nclass Animal:\n    name: str\n    sound: str\n@dataclass\nclass Pet(Animal):\n    owner: str\np = Pet('Rex', 'Woof', 'Alice')\nprint(f'name={p.name} sound={p.sound} owner={p.owner}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "name=Rex" in out
        assert "sound=Woof" in out
        assert "owner=Alice" in out

    def test_frozen_dataclass(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass",
            "@dataclass(frozen=True)\nclass Point:\n    x: float\n    y: float\np = Point(1.0, 2.0)\ntry:\n    p.x = 5.0\n    msg = 'mutable'\nexcept AttributeError:\n    msg = 'frozen'\nprint(f'msg={msg} x={p.x} y={p.y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "msg=frozen" in out  # FrozenInstanceError is subclass of AttributeError
        assert "x=1.0" in out

    def test_dataclass_inh_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass",
            "@dataclass\nclass Base:\n    val: int = 10\nclass Child(Base):\n    pass\nc = Child()\nprint(f'val={c.val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=10" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "@dataclass\nclass Base:\n    val: int = 99\nclass Child(Base):\n    pass\nc = Child()\nprint(f'val={c.val}')")
        nb_runner.run_all()
        assert "val=99" in nb_runner.get_output(2)
