"""Batch 481: abstract base class with abc module."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestAbcAbstractMethods:
    def test_abstract_enforcement(self, nb_runner):
        nb_runner.create_notebook([
            "from abc import ABC, abstractmethod",
            "class Shape(ABC):\n    @abstractmethod\n    def area(self): pass\nclass Square(Shape):\n    def __init__(self, s): self.s = s\n    def area(self): return self.s ** 2\nsq = Square(5)\nprint(f'area={sq.area()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=25" in nb_runner.get_output(2)

    def test_abstract_with_default(self, nb_runner):
        nb_runner.create_notebook([
            "from abc import ABC, abstractmethod",
            "class Animal(ABC):\n    @abstractmethod\n    def speak(self): pass\n    def describe(self): return f'I am {type(self).__name__}'\nclass Dog(Animal):\n    def speak(self): return 'Woof'\nclass Cat(Animal):\n    def speak(self): return 'Meow'\nd = Dog()\nc = Cat()\nprint(f'd={d.speak()} c={c.speak()} desc={d.describe()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "d=Woof" in out
        assert "c=Meow" in out
        assert "desc=I am Dog" in out

    def test_abc_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from abc import ABC, abstractmethod",
            "class Op(ABC):\n    @abstractmethod\n    def run(self, x): pass\nclass Double(Op):\n    def run(self, x): return x * 2\nresult = Double().run(5)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=10" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "class Op(ABC):\n    @abstractmethod\n    def run(self, x): pass\nclass Triple(Op):\n    def run(self, x): return x * 3\nresult = Triple().run(5)\nprint(f'result={result}')")
        nb_runner.run_all()
        assert "result=15" in nb_runner.get_output(2)
