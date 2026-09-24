"""abc: abstract bases, abstract methods and virtual subclasses across cells."""

import textwrap

import pytest


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestAbcAbstractBase:
    """abstract base classes with abc module."""

    def test_abc_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from abc import ABC, abstractmethod\nclass Shape(ABC):\n    @abstractmethod\n    def area(self): pass\nclass Circle(Shape):\n    def __init__(self, r): self.r = r\n    def area(self): return 3.14159 * self.r ** 2",
                "c = Circle(5)\nresult = round(c.area(), 2)\nprint(f'area={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=78.54" in nb_runner.get_output(2)

    def test_abc_cant_instantiate(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from abc import ABC, abstractmethod\nclass Base(ABC):\n    @abstractmethod\n    def process(self): pass",
                "try:\n    b = Base()\n    result = 'no_error'\nexcept TypeError:\n    result = 'type_error'\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=type_error" in nb_runner.get_output(2)

    def test_abc_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from abc import ABC, abstractmethod\nclass Converter(ABC):\n    @abstractmethod\n    def convert(self, val): pass\nclass DoubleConverter(Converter):\n    def convert(self, val): return val * 2",
                "dc = DoubleConverter()\nresult = dc.convert(21)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=42" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "dc = DoubleConverter()\nresult = dc.convert(50)\nprint(f'result={result}')")
        nb_runner.run_all()
        assert "result=100" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestAbcAbstractMethods:
    """abstract base class with abc module."""

    def test_abstract_enforcement(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from abc import ABC, abstractmethod",
                "class Shape(ABC):\n    @abstractmethod\n    def area(self): pass\nclass Square(Shape):\n    def __init__(self, s): self.s = s\n    def area(self): return self.s ** 2\nsq = Square(5)\nprint(f'area={sq.area()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area=25" in nb_runner.get_output(2)

    def test_abstract_with_default(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from abc import ABC, abstractmethod",
                "class Animal(ABC):\n    @abstractmethod\n    def speak(self): pass\n    def describe(self): return f'I am {type(self).__name__}'\nclass Dog(Animal):\n    def speak(self): return 'Woof'\nclass Cat(Animal):\n    def speak(self): return 'Meow'\nd = Dog()\nc = Cat()\nprint(f'd={d.speak()} c={c.speak()} desc={d.describe()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "d=Woof" in out
        assert "c=Meow" in out
        assert "desc=I am Dog" in out

    def test_abc_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from abc import ABC, abstractmethod",
                "class Op(ABC):\n    @abstractmethod\n    def run(self, x): pass\nclass Double(Op):\n    def run(self, x): return x * 2\nresult = Double().run(5)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=10" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2,
            "class Op(ABC):\n    @abstractmethod\n    def run(self, x): pass\nclass Triple(Op):\n    def run(self, x): return x * 3\nresult = Triple().run(5)\nprint(f'result={result}')",
        )
        nb_runner.run_all()
        assert "result=15" in nb_runner.get_output(2)


# abstract base class patterns with caching.
# Tests ABC, abstractmethod, and edit propagation.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestABCAbstractPatterns:
    """Test abstract base class caching."""

    def test_abc_basic(self, nb_runner):
        """ABC with concrete implementation, verify caching."""
        nb_runner.create_notebook(
            [
                "from abc import ABC, abstractmethod",
                "class Shape(ABC):\n    @abstractmethod\n    def area(self):\n        pass\n\nclass Circle(Shape):\n    def __init__(self, r):\n        self.r = r\n    def area(self):\n        return 3.14159 * self.r ** 2",
                "c = Circle(5)\nresult = round(c.area(), 2)",
                "print(f'area={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "area=78.54" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "area=78.54" in out2

    def test_abc_edit_implementation(self, nb_runner):
        """Edit concrete implementation, verify propagation."""
        nb_runner.create_notebook(
            [
                "from abc import ABC, abstractmethod",
                "class Greeter(ABC):\n    @abstractmethod\n    def greet(self, name):\n        pass\n\nclass Formal(Greeter):\n    def greet(self, name):\n        return f'Good day, {name}.'",
                "g = Formal()\nmsg = g.greet('Alice')",
                "print(f'msg={msg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "msg=Good day, Alice." in out

        nb_runner.set_cell_source(
            2,
            "class Greeter(ABC):\n    @abstractmethod\n    def greet(self, name):\n        pass\n\nclass Casual(Greeter):\n    def greet(self, name):\n        return f'Hey {name}!'",
        )
        nb_runner.set_cell_source(3, "g = Casual()\nmsg = g.greet('Bob')")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "msg=Hey Bob!" in out2


# Abstract base class patterns.
#
# Tests ABC with concrete implementations, edit propagation.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestABCPatterns:
    """Abstract base class interaction patterns."""

    def test_abc_concrete_edit(self, nb_runner):
        """Edit concrete implementation of ABC."""
        nb_runner.create_notebook(
            [
                "from abc import ABC, abstractmethod\nclass Shape(ABC):\n    @abstractmethod\n    def area(self):\n        pass",
                "class Circle(Shape):\n    def __init__(self, r):\n        self.r = r\n    def area(self):\n        return 3.14 * self.r ** 2",
                "c = Circle(5)\nprint(f'area = {c.area()}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "from abc import ABC, abstractmethod\nclass Formatter(ABC):\n    @abstractmethod\n    def format(self, text):\n        pass",
                "class UpperFormatter(Formatter):\n    def format(self, text):\n        return text.upper()",
                "fmt = UpperFormatter()\nresult = fmt.format('hello world')\nprint(f'result = {result}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "from abc import ABC, abstractmethod\nclass Processor(ABC):\n    @abstractmethod\n    def process(self, data):\n        pass\n    def describe(self):\n        return 'base processor'",
                "class Doubler(Processor):\n    def process(self, data):\n        return [x * 2 for x in data]",
                "p = Doubler()\nout = p.process([1, 2, 3])\ndesc = p.describe()\nprint(f'out={out} desc={desc}')",
            ]
        )
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


# Interaction test: abstract base class with multiple implementations.
# Tests ABC with abstractmethod, concrete methods, isinstance checks,
# and cross-cell polymorphic dispatch.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestAbcMultipleImpl:
    """Test ABC with multiple implementations across cells."""

    def test_abc_polymorphism(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: define ABC
                "from abc import ABC, abstractmethod\nclass Shape(ABC):\n    @abstractmethod\n    def area(self): ...\n    @abstractmethod\n    def perimeter(self): ...\n    def describe(self):\n        return f'{type(self).__name__}: area={self.area():.1f}, perim={self.perimeter():.1f}'\nprint('Shape ABC defined')",
                # Cell 2: implement subclasses
                "import math\nclass Circle(Shape):\n    def __init__(self, r):\n        self.r = r\n    def area(self):\n        return math.pi * self.r ** 2\n    def perimeter(self):\n        return 2 * math.pi * self.r\nclass Rect(Shape):\n    def __init__(self, w, h):\n        self.w, self.h = w, h\n    def area(self):\n        return self.w * self.h\n    def perimeter(self):\n        return 2 * (self.w + self.h)\nprint('Circle and Rect defined')",
                # Cell 3: polymorphic usage
                "shapes = [Circle(5), Rect(3, 4), Circle(1), Rect(10, 2)]\nfor s in shapes:\n    print(s.describe())\ntotal_area = sum(s.area() for s in shapes)\nprint(f'total_area={total_area:.1f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out3 = nb_runner.get_output(3)
        assert "Circle: area=78.5" in out3
        assert "Rect: area=12.0" in out3
        assert "total_area=" in out3

    def test_abc_edit_implementation(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from abc import ABC, abstractmethod\nclass Animal(ABC):\n    @abstractmethod\n    def speak(self): ...\nprint('Animal defined')",
                "class Dog(Animal):\n    def speak(self):\n        return 'Woof'\nclass Cat(Animal):\n    def speak(self):\n        return 'Meow'\nprint('Dog, Cat defined')",
                "animals = [Dog(), Cat(), Dog()]\nsounds = [a.speak() for a in animals]\nprint(f'sounds={sounds}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sounds=['Woof', 'Meow', 'Woof']" in nb_runner.get_output(3)

        # Edit Dog's implementation
        nb_runner.set_cell_source(
            2,
            "class Dog(Animal):\n    def speak(self):\n        return 'Bark'\nclass Cat(Animal):\n    def speak(self):\n        return 'Hiss'\nprint('Dog, Cat redefined')",
        )
        nb_runner.run_cells([2, 3])
        assert "sounds=['Bark', 'Hiss', 'Bark']" in nb_runner.get_output(3)

    def test_abc_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from abc import ABC, abstractmethod\nclass Converter(ABC):\n    @abstractmethod\n    def convert(self, val): ...\nclass CelsiusToF(Converter):\n    def convert(self, val):\n        return val * 9/5 + 32\nprint('Converter defined')",
                "c = CelsiusToF()\nresult = c.convert(100)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=212.0" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "result=212.0" in nb_runner.get_output(2)


# Interaction test: ABC with virtual subclass registration.
# Tests abc.ABC with register() for virtual subclasses,
# __subclasshook__, and cross-cell polymorphism patterns.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestAbcVirtualSubclass:
    """Test ABC virtual subclass registration across cells."""

    def test_abc_virtual(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: define ABC and register virtual subclass
                "from abc import ABC, abstractmethod\n\nclass Serializable(ABC):\n    @abstractmethod\n    def serialize(self):\n        pass\n\nclass JsonData:\n    def __init__(self, data):\n        self.data = data\n    def serialize(self):\n        import json\n        return json.dumps(self.data)\n\nSerializable.register(JsonData)\nj = JsonData({'key': 'value'})\nprint(f'is_serializable={isinstance(j, Serializable)}')\nprint(f'serialized={j.serialize()}')",
                # Cell 2: another registered class
                "class CsvData:\n    def __init__(self, rows):\n        self.rows = rows\n    def serialize(self):\n        return '\\n'.join(','.join(str(c) for c in row) for row in self.rows)\n\nSerializable.register(CsvData)\ncsv = CsvData([[1, 2], [3, 4]])\nprint(f'csv_is_ser={isinstance(csv, Serializable)}')\nprint(f'csv_out={csv.serialize()}')",
                # Cell 3: polymorphic use
                "items = [j, csv]\nfor item in items:\n    print(f'type={type(item).__name__} check={isinstance(item, Serializable)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "is_serializable=True" in out1
        out2 = nb_runner.get_output(2)
        assert "csv_is_ser=True" in out2
        out3 = nb_runner.get_output(3)
        assert "type=JsonData check=True" in out3
        assert "type=CsvData check=True" in out3

    def test_abc_virtual_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from abc import ABC, abstractmethod\nclass Printable(ABC):\n    @abstractmethod\n    def display(self):\n        pass\n\nclass Report:\n    def __init__(self, title):\n        self.title = title\n    def display(self):\n        return f'Report: {self.title}'\n\nPrintable.register(Report)\nrep = Report('Q1')\nprint(f'display={rep.display()}')",
                "output = rep.display()\nprint(f'output={output}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "output=Report: Q1" in nb_runner.get_output(2)

        # Edit title
        nb_runner.set_cell_source(
            1,
            "from abc import ABC, abstractmethod\nclass Printable(ABC):\n    @abstractmethod\n    def display(self):\n        pass\n\nclass Report:\n    def __init__(self, title):\n        self.title = title\n    def display(self):\n        return f'Report: {self.title}'\n\nPrintable.register(Report)\nrep = Report('Q2 Summary')\nprint(f'display={rep.display()}')",
        )
        nb_runner.run_cells([1, 2])
        assert "output=Report: Q2 Summary" in nb_runner.get_output(2)

    def test_abc_virtual_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from abc import ABC\nclass Container(ABC):\n    pass\n\nContainer.register(list)\nContainer.register(tuple)\nresults = [isinstance([], Container), isinstance((), Container), isinstance({}, Container)]\nprint(f'results={results}')",
                "all_true = all(results[:2])\nprint(f'lists_tuples_are_containers={all_true}')\nprint(f'dict_is_container={results[2]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[True, True, False]" in nb_runner.get_output(1)
        assert "lists_tuples_are_containers=True" in nb_runner.get_output(2)
        assert "dict_is_container=False" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "lists_tuples_are_containers=True" in nb_runner.get_output(2)


# Abstract base classes & mixins — cash caching with ABC patterns.
@pytest.mark.stress
class TestMixinPatterns:
    """Test mixin patterns across cells."""

    def test_mixin_composition(self, nb_runner):
        """Multiple mixins composed into a class."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class JsonMixin:
                    def to_json(self):
                        import json
                        data = {k: v for k, v in vars(self).items() if not k.startswith('_cash')}
                        return json.dumps(data)

                class ValidateMixin:
                    def validate(self):
                        for k, v in vars(self).items():
                            if k.startswith('_cash'):
                                continue
                            if v is None:
                                return False
                        return True

                class User(JsonMixin, ValidateMixin):
                    def __init__(self, name, email):
                        self.name = name
                        self.email = email

                u = User('Alice', 'alice@test.com')
                print(f"valid={u.validate()}")
                print(f"json={u.to_json()}")
            """),
                textwrap.dedent("""\
                u2 = User('Bob', None)
                print(f"u2_valid={u2.validate()}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "valid=True" in nb_runner.get_output(1)
        assert '"name": "Alice"' in nb_runner.get_output(1)
        assert "u2_valid=False" in nb_runner.get_output(2)

    def test_mixin_change_propagation(self, nb_runner):
        """Mixin behavior change propagates downstream."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class FormatMixin:
                    sep = ', '
                    def format_items(self, items):
                        return self.sep.join(str(i) for i in items)

                class Report(FormatMixin):
                    def __init__(self, data):
                        self.data = data
                    def summary(self):
                        return self.format_items(self.data)

                report = Report([1, 2, 3, 4, 5])
            """),
                textwrap.dedent("""\
                text = report.summary()
                print(f"summary={text}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "summary=1, 2, 3, 4, 5" in nb_runner.get_output(2)

        # Change separator
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            class FormatMixin:
                sep = ' | '
                def format_items(self, items):
                    return self.sep.join(str(i) for i in items)

            class Report(FormatMixin):
                def __init__(self, data):
                    self.data = data
                def summary(self):
                    return self.format_items(self.data)

            report = Report([1, 2, 3, 4, 5])
        """),
        )
        nb_runner.run_cells([1, 2])
        assert "summary=1 | 2 | 3 | 4 | 5" in nb_runner.get_output(2)
