"""Batch 440: abstract base classes with abc module."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestAbcAbstractBase:
    def test_abc_basic(self, nb_runner):
        nb_runner.create_notebook([
            "from abc import ABC, abstractmethod\nclass Shape(ABC):\n    @abstractmethod\n    def area(self): pass\nclass Circle(Shape):\n    def __init__(self, r): self.r = r\n    def area(self): return 3.14159 * self.r ** 2",
            "c = Circle(5)\nresult = round(c.area(), 2)\nprint(f'area={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=78.54" in nb_runner.get_output(2)

    def test_abc_cant_instantiate(self, nb_runner):
        nb_runner.create_notebook([
            "from abc import ABC, abstractmethod\nclass Base(ABC):\n    @abstractmethod\n    def process(self): pass",
            "try:\n    b = Base()\n    result = 'no_error'\nexcept TypeError:\n    result = 'type_error'\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=type_error" in nb_runner.get_output(2)

    def test_abc_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from abc import ABC, abstractmethod\nclass Converter(ABC):\n    @abstractmethod\n    def convert(self, val): pass\nclass DoubleConverter(Converter):\n    def convert(self, val): return val * 2",
            "dc = DoubleConverter()\nresult = dc.convert(21)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=42" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "dc = DoubleConverter()\nresult = dc.convert(50)\nprint(f'result={result}')")
        nb_runner.run_all()
        assert "result=100" in nb_runner.get_output(2)
