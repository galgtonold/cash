"""Batch 431: class inheritance and method resolution order."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestClassInheritanceMRO:
    def test_single_inheritance(self, nb_runner):
        nb_runner.create_notebook([
            "class Animal:\n    def speak(self): return 'generic'\nclass Dog(Animal):\n    def speak(self): return 'woof'\nclass Cat(Animal):\n    def speak(self): return 'meow'",
            "d = Dog()\nc = Cat()\nprint(f'd={d.speak()} c={c.speak()} is_animal={isinstance(d, Animal)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "d=woof" in out
        assert "c=meow" in out
        assert "is_animal=True" in out

    def test_mro(self, nb_runner):
        nb_runner.create_notebook([
            "class A:\n    val = 'A'\nclass B(A):\n    val = 'B'\nclass C(A):\n    val = 'C'\nclass D(B, C):\n    pass",
            "mro = [cls.__name__ for cls in D.__mro__]\nval = D.val\nprint(f'mro={mro} val={val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mro=['D', 'B', 'C', 'A', 'object']" in nb_runner.get_output(2)
        assert "val=B" in nb_runner.get_output(2)

    def test_inheritance_edit(self, nb_runner):
        nb_runner.create_notebook([
            "class Shape:\n    def area(self): return 0\nclass Square(Shape):\n    def __init__(self, s): self.s = s\n    def area(self): return self.s ** 2",
            "sq = Square(5)\nprint(f'area={sq.area()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=25" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "sq = Square(10)\nprint(f'area={sq.area()}')")
        nb_runner.run_all()
        assert "area=100" in nb_runner.get_output(2)
