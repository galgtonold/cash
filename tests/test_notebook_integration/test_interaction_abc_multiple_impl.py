"""
Interaction test: abstract base class with multiple implementations.
Tests ABC with abstractmethod, concrete methods, isinstance checks,
and cross-cell polymorphic dispatch.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestAbcMultipleImpl:
    """Test ABC with multiple implementations across cells."""

    def test_abc_polymorphism(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: define ABC
            "from abc import ABC, abstractmethod\nclass Shape(ABC):\n    @abstractmethod\n    def area(self): ...\n    @abstractmethod\n    def perimeter(self): ...\n    def describe(self):\n        return f'{type(self).__name__}: area={self.area():.1f}, perim={self.perimeter():.1f}'\nprint('Shape ABC defined')",
            # Cell 2: implement subclasses
            "import math\nclass Circle(Shape):\n    def __init__(self, r):\n        self.r = r\n    def area(self):\n        return math.pi * self.r ** 2\n    def perimeter(self):\n        return 2 * math.pi * self.r\nclass Rect(Shape):\n    def __init__(self, w, h):\n        self.w, self.h = w, h\n    def area(self):\n        return self.w * self.h\n    def perimeter(self):\n        return 2 * (self.w + self.h)\nprint('Circle and Rect defined')",
            # Cell 3: polymorphic usage
            "shapes = [Circle(5), Rect(3, 4), Circle(1), Rect(10, 2)]\nfor s in shapes:\n    print(s.describe())\ntotal_area = sum(s.area() for s in shapes)\nprint(f'total_area={total_area:.1f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out3 = nb_runner.get_output(3)
        assert "Circle: area=78.5" in out3
        assert "Rect: area=12.0" in out3
        assert "total_area=" in out3

    def test_abc_edit_implementation(self, nb_runner):
        nb_runner.create_notebook([
            "from abc import ABC, abstractmethod\nclass Animal(ABC):\n    @abstractmethod\n    def speak(self): ...\nprint('Animal defined')",
            "class Dog(Animal):\n    def speak(self):\n        return 'Woof'\nclass Cat(Animal):\n    def speak(self):\n        return 'Meow'\nprint('Dog, Cat defined')",
            "animals = [Dog(), Cat(), Dog()]\nsounds = [a.speak() for a in animals]\nprint(f'sounds={sounds}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sounds=['Woof', 'Meow', 'Woof']" in nb_runner.get_output(3)

        # Edit Dog's implementation
        nb_runner.set_cell_source(2, "class Dog(Animal):\n    def speak(self):\n        return 'Bark'\nclass Cat(Animal):\n    def speak(self):\n        return 'Hiss'\nprint('Dog, Cat redefined')")
        nb_runner.run_cells([2, 3])
        assert "sounds=['Bark', 'Hiss', 'Bark']" in nb_runner.get_output(3)

    def test_abc_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from abc import ABC, abstractmethod\nclass Converter(ABC):\n    @abstractmethod\n    def convert(self, val): ...\nclass CelsiusToF(Converter):\n    def convert(self, val):\n        return val * 9/5 + 32\nprint('Converter defined')",
            "c = CelsiusToF()\nresult = c.convert(100)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=212.0" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "result=212.0" in nb_runner.get_output(2)
