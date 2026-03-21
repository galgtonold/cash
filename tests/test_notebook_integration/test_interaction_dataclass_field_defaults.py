"""Batch 422: dataclass with field defaults and post_init."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDataclassFieldDefaults:
    def test_dataclass_defaults(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass, field\n@dataclass\nclass Config:\n    name: str\n    debug: bool = False\n    tags: list = field(default_factory=list)",
            "c = Config('test')\nc.tags.append('v1')\nprint(f'name={c.name} debug={c.debug} tags={c.tags}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "name=test" in out
        assert "debug=False" in out
        assert "tags=['v1']" in out

    def test_dataclass_post_init(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass\n@dataclass\nclass Rect:\n    width: float\n    height: float\n    area: float = 0\n    def __post_init__(self):\n        self.area = self.width * self.height",
            "r = Rect(3.0, 4.0)\nprint(f'area={r.area}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=12.0" in nb_runner.get_output(2)

    def test_dataclass_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass\n@dataclass\nclass Point:\n    x: int\n    y: int",
            "p = Point(1, 2)\ndist_sq = p.x**2 + p.y**2\nprint(f'dist_sq={dist_sq}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "dist_sq=5" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "p = Point(3, 4)\ndist_sq = p.x**2 + p.y**2\nprint(f'dist_sq={dist_sq}')")
        nb_runner.run_all()
        assert "dist_sq=25" in nb_runner.get_output(2)
