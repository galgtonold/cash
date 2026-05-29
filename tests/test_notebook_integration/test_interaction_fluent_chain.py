"""Batch 257 – Chained method calls and fluent interface patterns.

Tests method chaining / builder patterns with edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestFluentInterface:
    """Fluent/builder interface edit patterns."""

    def test_builder_pattern_edit(self, nb_runner):
        """Edit builder class method, chained result changes."""
        nb_runner.create_notebook([
            "class Query:\n    def __init__(self):\n        self.parts = []\n    def select(self, cols):\n        self.parts.append(f'SELECT {cols}')\n        return self\n    def where(self, cond):\n        self.parts.append(f'WHERE {cond}')\n        return self\n    def build(self):\n        return ' '.join(self.parts)",
            "q = Query().select('*').where('id > 5').build()\nprint(f'q = {q}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "q = SELECT * WHERE id > 5" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "class Query:\n    def __init__(self):\n        self.parts = []\n    def select(self, cols):\n        self.parts.append(f'SELECT {cols}')\n        return self\n    def where(self, cond):\n        self.parts.append(f'WHERE {cond}')\n        return self\n    def build(self):\n        return ' | '.join(self.parts)",
        )
        nb_runner.run_all()
        assert "q = SELECT * | WHERE id > 5" in nb_runner.get_output(2)

    def test_chain_edit_usage(self, nb_runner):
        """Edit the chain usage, builder class stays same."""
        nb_runner.create_notebook([
            "class Pipe:\n    def __init__(self, val):\n        self.val = val\n    def add(self, n):\n        self.val += n\n        return self\n    def mul(self, n):\n        self.val *= n\n        return self\n    def result(self):\n        return self.val",
            "r = Pipe(10).add(5).mul(2).result()\nprint(f'r = {r}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r = 30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "r = Pipe(10).mul(3).add(5).result()\nprint(f'r = {r}')")
        nb_runner.run_all()
        assert "r = 35" in nb_runner.get_output(2)

