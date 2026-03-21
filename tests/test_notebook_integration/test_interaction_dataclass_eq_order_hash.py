"""
Interaction test: dataclass with eq, order, and hash customization.
Tests dataclass with eq=True, order=True for comparison, frozen for
hashability, and cross-cell sorted/set operations.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDataclassEqOrderHash:
    """Test dataclass eq/order/hash across cells."""

    def test_dataclass_ordering(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: define ordered dataclass
            "from dataclasses import dataclass\n@dataclass(order=True, frozen=True)\nclass Priority:\n    level: int\n    name: str\nprint('Priority defined')",
            # Cell 2: create and sort
            "tasks = [\n    Priority(3, 'low'),\n    Priority(1, 'critical'),\n    Priority(2, 'medium'),\n    Priority(1, 'urgent'),\n]\nsorted_tasks = sorted(tasks)\nprint(f'sorted={sorted_tasks}')",
            # Cell 3: set operations (frozen=True makes hashable)
            "unique = set(tasks)\nprint(f'unique_count={len(unique)}')\nmin_task = min(tasks)\nmax_task = max(tasks)\nprint(f'min={min_task}')\nprint(f'max={max_task}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        # Sorted by (level, name)
        assert "Priority(level=1, name='critical')" in out2
        out3 = nb_runner.get_output(3)
        assert "unique_count=4" in out3
        assert "min=Priority(level=1, name='critical')" in out3
        assert "max=Priority(level=3, name='low')" in out3

    def test_dataclass_order_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass\n@dataclass(order=True, frozen=True)\nclass Score:\n    value: int\n    name: str\nprint('Score defined')",
            "scores = [Score(90, 'A'), Score(80, 'B'), Score(95, 'A+')]\nbest = max(scores)\nprint(f'best={best}')",
            "ranking = [s.name for s in sorted(scores, reverse=True)]\nprint(f'ranking={ranking}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "best=Score(value=95, name='A+')" in nb_runner.get_output(2)
        assert "ranking=['A+', 'A', 'B']" in nb_runner.get_output(3)

        # Add more scores
        nb_runner.set_cell_source(2, "scores = [Score(90, 'A'), Score(80, 'B'), Score(95, 'A+'), Score(100, 'S')]\nbest = max(scores)\nprint(f'best={best}')")
        nb_runner.run_cells([2, 3])
        assert "best=Score(value=100, name='S')" in nb_runner.get_output(2)
        assert "ranking=['S', 'A+', 'A', 'B']" in nb_runner.get_output(3)

    def test_dataclass_order_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass Coord:\n    x: int\n    y: int\nprint('Coord defined')",
            "points = {Coord(1, 2), Coord(3, 4), Coord(1, 2)}\nprint(f'unique={len(points)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "unique=2" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "unique=2" in nb_runner.get_output(2)
