"""Batch 203 – Object identity and equality interaction tests.

Tests editing equality vs identity checks, hash-based
comparisons, and their cache implications.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestEqualityEdits:
    """Editing equality and identity checks."""

    def test_edit_eq_implementation(self, nb_runner):
        """Edit a custom __eq__ implementation."""
        nb_runner.create_notebook([
            "class Pt:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def __eq__(self, other):\n        return self.x == other.x and self.y == other.y\n    def __repr__(self):\n        return f'Pt({self.x},{self.y})'",
            "a = Pt(1, 2)\nb = Pt(1, 2)\nprint(f'eq = {a == b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "eq = True" in nb_runner.get_output(2)

        # Change eq to only compare x
        nb_runner.set_cell_source(
            1,
            "class Pt:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def __eq__(self, other):\n        return self.x == other.x\n    def __repr__(self):\n        return f'Pt({self.x},{self.y})'",
        )
        nb_runner.set_cell_source(
            2, "a = Pt(1, 2)\nb = Pt(1, 99)\nprint(f'eq = {a == b}')"
        )
        nb_runner.run_all()
        assert "eq = True" in nb_runner.get_output(2)

    def test_edit_hash_implementation(self, nb_runner):
        """Edit a custom __hash__ implementation."""
        nb_runner.create_notebook([
            "class Token:\n    def __init__(self, val):\n        self.val = val\n    def __hash__(self):\n        return hash(self.val)\n    def __eq__(self, other):\n        return self.val == other.val",
            "s = {Token('a'), Token('b'), Token('a')}\nprint(f'count = {len(s)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 2" in nb_runner.get_output(2)

        # Change to make all equal
        nb_runner.set_cell_source(
            1,
            "class Token:\n    def __init__(self, val):\n        self.val = val\n    def __hash__(self):\n        return 42\n    def __eq__(self, other):\n        return True",
        )
        nb_runner.run_all()
        assert "count = 1" in nb_runner.get_output(2)


class TestComparisonEdits:
    """Editing comparison operations."""

    def test_edit_comparison_operator(self, nb_runner):
        """Edit comparison operators."""
        nb_runner.create_notebook([
            "a = 5\nb = 10  # comparison source",
            "result = a < b\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = True" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            2, "result = a > b\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = False" in nb_runner.get_output(2)
