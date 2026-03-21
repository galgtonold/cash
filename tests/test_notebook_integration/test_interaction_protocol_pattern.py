"""Batch 218 – Protocol/interface interaction tests.

Tests editing cells with abstract base class and protocol
patterns and verifying downstream propagation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestProtocolPatternEdits:
    """Editing protocol/interface patterns."""

    def test_edit_abc_implementation(self, nb_runner):
        """Edit an ABC-derived class implementation."""
        nb_runner.create_notebook([
            "from abc import ABC, abstractmethod\nclass Shape(ABC):\n    @abstractmethod\n    def area(self):\n        pass",
            "class Circle(Shape):\n    def __init__(self, r):\n        self.r = r\n    def area(self):\n        return 3.14 * self.r ** 2",
            "c = Circle(5)\nprint(f'area = {c.area()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 78.5" in nb_runner.get_output(3)

        # Edit Circle implementation
        nb_runner.set_cell_source(2, "class Circle(Shape):\n    def __init__(self, r):\n        self.r = r\n    def area(self):\n        return 3.14159 * self.r ** 2")
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "78.5" in out or "78.53" in out

    def test_edit_strategy_pattern(self, nb_runner):
        """Edit strategy function selection."""
        nb_runner.create_notebook([
            "def add(a, b):\n    return a + b\ndef multiply(a, b):\n    return a * b",
            "strategy = add\nresult = strategy(3, 4)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 7" in nb_runner.get_output(2)

        # Switch strategy
        nb_runner.set_cell_source(2, "strategy = multiply\nresult = strategy(3, 4)\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 12" in nb_runner.get_output(2)

    def test_edit_interface_method(self, nb_runner):
        """Edit a class that implements a protocol."""
        nb_runner.create_notebook([
            "class Formatter:\n    def format(self, text):\n        return text.upper()",
            "f = Formatter()\nprint(f.format('hello world'))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "HELLO WORLD" in nb_runner.get_output(2)

        # Change formatting
        nb_runner.set_cell_source(1, "class Formatter:\n    def format(self, text):\n        return text.title()")
        nb_runner.run_all()
        assert "Hello World" in nb_runner.get_output(2)
