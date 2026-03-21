"""Batch 262 – Multi-cell class instantiation with method edits.

Tests class defined in one cell, instantiated in another, method called in third.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestMultiCellClassEdits:
    """Class spread across cells with edits."""

    def test_class_method_edit_propagates(self, nb_runner):
        """Edit class method, instantiation and usage cells reflect."""
        nb_runner.create_notebook([
            "class Calculator:\n    def __init__(self, val=0):\n        self.val = val\n    def add(self, n):\n        return Calculator(self.val + n)\n    def result(self):\n        return self.val",
            "c = Calculator(10).add(5).add(3)",
            "r = c.result()\nprint(f'r = {r}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r = 18" in nb_runner.get_output(3)

        # Edit add to multiply instead
        nb_runner.set_cell_source(
            1,
            "class Calculator:\n    def __init__(self, val=0):\n        self.val = val\n    def add(self, n):\n        return Calculator(self.val * n)\n    def result(self):\n        return self.val",
        )
        nb_runner.run_all()
        # 10 * 5 * 3 = 150
        assert "r = 150" in nb_runner.get_output(3)

    def test_edit_instantiation_params(self, nb_runner):
        """Edit instantiation parameters, method call reflects."""
        nb_runner.create_notebook([
            "class Formatter:\n    def __init__(self, prefix, suffix):\n        self.prefix = prefix\n        self.suffix = suffix\n    def wrap(self, text):\n        return f'{self.prefix}{text}{self.suffix}'",
            "fmt = Formatter('[', ']')",
            "out = fmt.wrap('hello')\nprint(f'out = {out}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "out = [hello]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "fmt = Formatter('<<', '>>')")
        nb_runner.run_all()
        assert "out = <<hello>>" in nb_runner.get_output(3)

    def test_edit_both_class_and_usage(self, nb_runner):
        """Edit class and usage cell simultaneously."""
        nb_runner.create_notebook([
            "class Scaler:\n    def __init__(self, factor):\n        self.factor = factor\n    def scale(self, x):\n        return x * self.factor",
            "s = Scaler(2)",
            "result = s.scale(10)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(3)

        # Edit both class and instantiation
        nb_runner.set_cell_source(
            1,
            "class Scaler:\n    def __init__(self, factor):\n        self.factor = factor\n    def scale(self, x):\n        return x * self.factor + 1",
        )
        nb_runner.set_cell_source(2, "s = Scaler(5)")
        nb_runner.run_all()
        assert "result = 51" in nb_runner.get_output(3)
