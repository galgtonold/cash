"""Batch 248 – Abstract base class patterns.

Tests ABC with concrete implementations, edit propagation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestABCPatterns:
    """Abstract base class interaction patterns."""

    def test_abc_concrete_edit(self, nb_runner):
        """Edit concrete implementation of ABC."""
        nb_runner.create_notebook([
            "from abc import ABC, abstractmethod\nclass Shape(ABC):\n    @abstractmethod\n    def area(self):\n        pass",
            "class Circle(Shape):\n    def __init__(self, r):\n        self.r = r\n    def area(self):\n        return 3.14 * self.r ** 2",
            "c = Circle(5)\nprint(f'area = {c.area()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 78.5" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2,
            "class Circle(Shape):\n    def __init__(self, r):\n        self.r = r\n    def area(self):\n        return 3.14159 * self.r ** 2",
        )
        nb_runner.run_all()
        assert "78.539" in nb_runner.get_output(3)

    def test_switch_implementation(self, nb_runner):
        """Switch between different concrete implementations."""
        nb_runner.create_notebook([
            "from abc import ABC, abstractmethod\nclass Formatter(ABC):\n    @abstractmethod\n    def format(self, text):\n        pass",
            "class UpperFormatter(Formatter):\n    def format(self, text):\n        return text.upper()",
            "fmt = UpperFormatter()\nresult = fmt.format('hello world')\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = HELLO WORLD" in nb_runner.get_output(3)

        # Switch to a different implementation
        nb_runner.set_cell_source(
            2,
            "class UpperFormatter(Formatter):\n    def format(self, text):\n        return text.title()",
        )
        nb_runner.run_all()
        assert "result = Hello World" in nb_runner.get_output(3)

    def test_abc_with_default_method(self, nb_runner):
        """ABC with default method, override in subclass."""
        nb_runner.create_notebook([
            "from abc import ABC, abstractmethod\nclass Processor(ABC):\n    @abstractmethod\n    def process(self, data):\n        pass\n    def describe(self):\n        return 'base processor'",
            "class Doubler(Processor):\n    def process(self, data):\n        return [x * 2 for x in data]",
            "p = Doubler()\nout = p.process([1, 2, 3])\ndesc = p.describe()\nprint(f'out={out} desc={desc}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "out=[2, 4, 6]" in nb_runner.get_output(3)
        assert "desc=base processor" in nb_runner.get_output(3)

        # Add describe override
        nb_runner.set_cell_source(
            2,
            "class Doubler(Processor):\n    def process(self, data):\n        return [x * 2 for x in data]\n    def describe(self):\n        return 'doubler v2'",
        )
        nb_runner.run_all()
        assert "desc=doubler v2" in nb_runner.get_output(3)
