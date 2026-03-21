"""Batch 246 – Class composition patterns.

Tests composition (has-a) relationships with edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestClassComposition:
    """Composition-based class patterns with edit propagation."""

    def test_engine_in_car(self, nb_runner):
        """Edit composed engine class, car reflects change."""
        nb_runner.create_notebook([
            "class Engine:\n    def __init__(self, hp):\n        self.hp = hp\n    def describe(self):\n        return f'{self.hp}hp'",
            "class Car:\n    def __init__(self, name, engine):\n        self.name = name\n        self.engine = engine\n    def spec(self):\n        return f'{self.name}: {self.engine.describe()}'",
            "e = Engine(200)\nc = Car('Sedan', e)\nprint(f'spec = {c.spec()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "spec = Sedan: 200hp" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            1,
            "class Engine:\n    def __init__(self, hp):\n        self.hp = hp\n    def describe(self):\n        return f'{self.hp}HP turbo'",
        )
        nb_runner.run_all()
        assert "spec = Sedan: 200HP turbo" in nb_runner.get_output(3)

    def test_strategy_pattern(self, nb_runner):
        """Edit strategy object, context reflects new behavior."""
        nb_runner.create_notebook([
            "class AddStrategy:\n    def execute(self, a, b):\n        return a + b",
            "strategy = AddStrategy()\nresult = strategy.execute(10, 20)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "class AddStrategy:\n    def execute(self, a, b):\n        return a * b",
        )
        nb_runner.run_all()
        assert "result = 200" in nb_runner.get_output(2)

    def test_nested_composition(self, nb_runner):
        """Three-level composition: department -> team -> member."""
        nb_runner.create_notebook([
            "class Member:\n    def __init__(self, name):\n        self.name = name",
            "class Team:\n    def __init__(self, members):\n        self.members = members\n    def names(self):\n        return [m.name for m in self.members]",
            "t = Team([Member('Alice'), Member('Bob')])\nprint(f'names = {t.names()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "names = ['Alice', 'Bob']" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            3,
            "t = Team([Member('Charlie'), Member('Diana'), Member('Eve')])\nprint(f'names = {t.names()}')",
        )
        nb_runner.run_all()
        assert "names = ['Charlie', 'Diana', 'Eve']" in nb_runner.get_output(3)
