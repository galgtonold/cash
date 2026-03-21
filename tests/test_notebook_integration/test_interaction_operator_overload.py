"""
Batch 294: Operator overloading interaction tests.
Tests that editing classes with overloaded operators properly invalidates
downstream computations using those operators.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestOperatorOverloadInteraction:
    """Test operator overloading patterns with cache invalidation."""

    def test_add_mul_overload_edit(self, nb_runner):
        """Editing vector class with __add__ and __mul__ should propagate."""
        nb_runner.create_notebook([
            (
                "class Vec:\n"
                "    def __init__(self, x, y):\n"
                "        self.x = x\n"
                "        self.y = y\n"
                "    def __add__(self, other):\n"
                "        return Vec(self.x + other.x, self.y + other.y)\n"
                "    def __mul__(self, scalar):\n"
                "        return Vec(self.x * scalar, self.y * scalar)\n"
                "    def __repr__(self):\n"
                "        return f'Vec({self.x},{self.y})'"
            ),
            "a = Vec(1, 2)\nb = Vec(3, 4)",
            "c = a + b\nd = c * 2",
            "print(f'c={c},d={d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "c=Vec(4,6)" in out
        assert "d=Vec(8,12)" in out

        nb_runner.set_cell_source(2, "a = Vec(10, 20)\nb = Vec(30, 40)")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "c=Vec(40,60)" in out
        assert "d=Vec(80,120)" in out

    def test_comparison_overload_edit(self, nb_runner):
        """Editing comparison overloads should propagate sorting."""
        nb_runner.create_notebook([
            (
                "class Score:\n"
                "    def __init__(self, name, val):\n"
                "        self.name = name\n"
                "        self.val = val\n"
                "    def __lt__(self, other):\n"
                "        return self.val < other.val\n"
                "    def __repr__(self):\n"
                "        return f'{self.name}:{self.val}'"
            ),
            "scores = [Score('A', 30), Score('B', 10), Score('C', 20)]",
            "ranked = sorted(scores)",
            "result = ','.join(str(s) for s in ranked)",
            "print(f'ranked={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "ranked=B:10,C:20,A:30" in out

        nb_runner.set_cell_source(2, "scores = [Score('A', 5), Score('B', 50), Score('C', 25)]")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "ranked=A:5,C:25,B:50" in out

    def test_contains_overload_edit(self, nb_runner):
        """Editing a class with __contains__ should propagate membership checks."""
        nb_runner.create_notebook([
            (
                "class WordSet:\n"
                "    def __init__(self, words):\n"
                "        self.words = set(w.lower() for w in words)\n"
                "    def __contains__(self, item):\n"
                "        return item.lower() in self.words"
            ),
            "ws = WordSet(['Hello', 'World'])",
            "checks = ['hello' in ws, 'python' in ws, 'WORLD' in ws]",
            "print(f'checks={checks}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "checks=[True, False, True]" in out

        nb_runner.set_cell_source(2, "ws = WordSet(['Python', 'Code'])")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "checks=[False, True, False]" in out
