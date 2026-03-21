"""
Batch 330: abstract base class patterns with caching.
Tests ABC, abstractmethod, and edit propagation.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.timeout(90)]


class TestABCAbstractPatterns:
    """Test abstract base class caching."""

    def test_abc_basic(self, nb_runner):
        """ABC with concrete implementation, verify caching."""
        nb_runner.create_notebook([
            "from abc import ABC, abstractmethod",
            "class Shape(ABC):\n    @abstractmethod\n    def area(self):\n        pass\n\nclass Circle(Shape):\n    def __init__(self, r):\n        self.r = r\n    def area(self):\n        return 3.14159 * self.r ** 2",
            "c = Circle(5)\nresult = round(c.area(), 2)",
            "print(f'area={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "area=78.54" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "area=78.54" in out2

    def test_abc_edit_implementation(self, nb_runner):
        """Edit concrete implementation, verify propagation."""
        nb_runner.create_notebook([
            "from abc import ABC, abstractmethod",
            "class Greeter(ABC):\n    @abstractmethod\n    def greet(self, name):\n        pass\n\nclass Formal(Greeter):\n    def greet(self, name):\n        return f'Good day, {name}.'",
            "g = Formal()\nmsg = g.greet('Alice')",
            "print(f'msg={msg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "msg=Good day, Alice." in out

        nb_runner.set_cell_source(2, "class Greeter(ABC):\n    @abstractmethod\n    def greet(self, name):\n        pass\n\nclass Casual(Greeter):\n    def greet(self, name):\n        return f'Hey {name}!'")
        nb_runner.set_cell_source(3, "g = Casual()\nmsg = g.greet('Bob')")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "msg=Hey Bob!" in out2
