"""
Batch 286: Custom __hash__/__eq__ interaction tests.
Tests that objects with custom hashing/equality properly interact
with cash's caching when their definitions or data change.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestCustomHashEqInteraction:
    """Test custom __hash__/__eq__ with cache invalidation."""

    def test_hashable_object_edit(self, nb_runner):
        """Editing a class with custom __hash__ should invalidate downstream."""
        nb_runner.create_notebook([
            (
                "class Point:\n"
                "    def __init__(self, x, y):\n"
                "        self.x = x\n"
                "        self.y = y\n"
                "    def __hash__(self):\n"
                "        return hash((self.x, self.y))\n"
                "    def __eq__(self, other):\n"
                "        return self.x == other.x and self.y == other.y\n"
                "    def __repr__(self):\n"
                "        return f'Point({self.x},{self.y})'"
            ),
            "p = Point(3, 4)",
            "s = {p, Point(1, 2)}\ncount = len(s)",
            "print(f'count={count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "count=2" in out

        # Change to create duplicate (same hash)
        nb_runner.set_cell_source(2, "p = Point(1, 2)")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "count=1" in out

    def test_dict_key_custom_hash_edit(self, nb_runner):
        """Editing objects used as dict keys with custom hash."""
        nb_runner.create_notebook([
            (
                "class Key:\n"
                "    def __init__(self, name):\n"
                "        self.name = name\n"
                "    def __hash__(self):\n"
                "        return hash(self.name)\n"
                "    def __eq__(self, other):\n"
                "        return self.name == other.name"
            ),
            "k = Key('alpha')\nmapping = {k: 100, Key('beta'): 200}",
            "val = mapping[Key('alpha')]",
            "print(f'val={val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "val=100" in out

        nb_runner.set_cell_source(2, "k = Key('alpha')\nmapping = {k: 999, Key('beta'): 200}")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "val=999" in out

    def test_frozen_dataclass_hash_edit(self, nb_runner):
        """Frozen dataclass (auto-hashing) edit should propagate."""
        nb_runner.create_notebook([
            "from dataclasses import dataclass",
            (
                "@dataclass(frozen=True)\n"
                "class Config:\n"
                "    name: str\n"
                "    version: int"
            ),
            "c1 = Config('app', 1)\nc2 = Config('app', 2)\nconfigs = {c1, c2}",
            "count = len(configs)",
            "print(f'count={count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "count=2" in out

        # Make them the same
        nb_runner.set_cell_source(3, "c1 = Config('app', 1)\nc2 = Config('app', 1)\nconfigs = {c1, c2}")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "count=1" in out
