"""Batch 135 – Type conversion + structural change interaction tests.

Tests that exercise patterns where variable types change between
runs (int→str, list→dict, etc.) and structural changes that
could confuse the cache system.
"""

import pytest

pytestmark = [pytest.mark.core, pytest.mark.stress, pytest.mark.timeout(30)]


class TestTypeChanges:
    """Variable type changes between runs."""


    def test_list_to_dict(self, nb_runner):
        """Variable changes from list to dict."""
        nb_runner.create_notebook([
            "data = [1, 2, 3]",
            "result = len(data)\nprint(f'len = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "len = 3" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "data = {'a': 1, 'b': 2}")
        nb_runner.run_all()
        assert "len = 2" in nb_runner.get_output(2)

    def test_scalar_to_collection(self, nb_runner):
        """Variable changes from scalar to collection."""
        nb_runner.create_notebook([
            "val = 10",
            "result = type(val).__name__\nprint(f'type = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "type = int" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "val = [10, 20, 30]")
        nb_runner.run_all()
        assert "type = list" in nb_runner.get_output(2)

    def test_none_to_value(self, nb_runner):
        """Variable changes from None to a value."""
        nb_runner.create_notebook([
            "x = None",
            "result = x is None\nprint(f'is_none = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "is_none = True" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "x = 42")
        nb_runner.run_all()
        assert "is_none = False" in nb_runner.get_output(2)


class TestStructuralChanges:
    """Structural changes to cells."""

    def test_single_to_multi_statement(self, nb_runner):
        """Single statement becomes multiple statements."""
        nb_runner.create_notebook([
            "x = 10",
            "print(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 10" in nb_runner.get_output(2)

        # Expand to multi-statement
        nb_runner.set_cell_source(1, "x = 10\ny = 20")
        nb_runner.set_cell_source(2, "print(f'x = {x}, y = {y}')")
        nb_runner.run_all()
        assert "x = 10, y = 20" in nb_runner.get_output(2)

    def test_multi_to_single_statement(self, nb_runner):
        """Multiple statements collapses to single."""
        nb_runner.create_notebook([
            "x = 10\ny = 20",
            "result = x + y\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(2)

        # Collapse to single
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.set_cell_source(2, "result = x\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(2)

    def test_change_number_of_outputs(self, nb_runner):
        """Cell changes from producing 1 output to 3 outputs."""
        nb_runner.create_notebook([
            "x = 10",
            "result = x\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        # Now produce 3 outputs
        nb_runner.set_cell_source(1, "x = 10\ny = 20\nz = 30")
        nb_runner.set_cell_source(
            2, "result = x + y + z\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = 60" in nb_runner.get_output(2)


class TestExpressionPatterns:
    """Different expression patterns."""

    def test_comprehension_to_loop(self, nb_runner):
        """Change from list comprehension to explicit loop."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5]",
            "result = [x * 2 for x in data]",
            "total = sum(result)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 30" in nb_runner.get_output(3)

        # Change to explicit loop
        nb_runner.set_cell_source(
            2,
            "result = []\nfor x in data:\n    result.append(x * 3)",
        )
        nb_runner.run_all()
        assert "total = 45" in nb_runner.get_output(3)

    def test_inline_to_function(self, nb_runner):
        """Change from inline expression to function call."""
        nb_runner.create_notebook([
            "x = 10",
            "result = x * 2 + 1",
            "print(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 21" in nb_runner.get_output(3)

        # Change to function
        nb_runner.set_cell_source(
            2,
            "def calc(v):\n    return v * 3 + 1\nresult = calc(x)",
        )
        nb_runner.run_all()
        assert "result = 31" in nb_runner.get_output(3)
