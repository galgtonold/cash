"""Editing class definitions: methods, properties, dataclasses, hierarchies."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress]


# Multi-cell class evolution and refactoring patterns, complex
# re-execution scenarios, out-of-order execution, and selective cell re-runs.
#
# Tests patterns where users iteratively develop and refine code across cells,
# re-run subsets of cells, and evolve class hierarchies during a session.
class TestClassEvolution:
    """Test iterative class development patterns common in notebooks."""

    @pytest.mark.integration
    def test_add_method_to_class(self, nb_runner):
        """Define class, use it, then add a method and re-use."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class Calculator:
                    def __init__(self):
                        self.history = []
                    def add(self, a, b):
                        result = a + b
                        self.history.append(result)
                        return result
            """),
                textwrap.dedent("""\
                calc = Calculator()
                r1 = calc.add(3, 4)
                print(r1)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "7" in nb_runner.get_output(2)

        # Add multiply method
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            class Calculator:
                def __init__(self):
                    self.history = []
                def add(self, a, b):
                    result = a + b
                    self.history.append(result)
                    return result
                def multiply(self, a, b):
                    result = a * b
                    self.history.append(result)
                    return result
        """),
        )
        nb_runner.set_cell_source(
            2,
            textwrap.dedent("""\
            calc = Calculator()
            r1 = calc.add(3, 4)
            r2 = calc.multiply(5, 6)
            print(r1, r2)
        """),
        )
        nb_runner.run_all()
        assert "7 30" in nb_runner.get_output(2)

    @pytest.mark.integration
    def test_refine_function_signature(self, nb_runner):
        """Iteratively refine a function signature."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def process(data):
                    return sum(data)
            """),
                textwrap.dedent("""\
                result = process([1, 2, 3])
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "6" in nb_runner.get_output(2)

        # Refine with default parameter
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            def process(data, multiplier=1):
                return sum(data) * multiplier
        """),
        )
        nb_runner.set_cell_source(
            2,
            textwrap.dedent("""\
            result = process([1, 2, 3], multiplier=10)
            print(result)
        """),
        )
        nb_runner.run_all()
        assert "60" in nb_runner.get_output(2)

    @pytest.mark.timeout(90)
    def test_class_add_method(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Calculator:\n    def __init__(self):\n        self.result = 0\n    def add(self, x):\n        self.result += x\n        return self",
                "c = Calculator().add(5).add(3)\nprint(f'result={c.result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=8" in nb_runner.get_output(2)
        # Add multiply method
        nb_runner.set_cell_source(
            1,
            "class Calculator:\n    def __init__(self):\n        self.result = 0\n    def add(self, x):\n        self.result += x\n        return self\n    def multiply(self, x):\n        self.result *= x\n        return self",
        )
        nb_runner.set_cell_source(2, "c = Calculator().add(5).multiply(3)\nprint(f'result={c.result}')")
        nb_runner.run_all()
        assert "result=15" in nb_runner.get_output(2)

    @pytest.mark.timeout(90)
    def test_class_rename_attr(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Config:\n    def __init__(self, host, port):\n        self.host = host\n        self.port = port",
                "cfg = Config('localhost', 8080)\naddr = f'{cfg.host}:{cfg.port}'\nprint(f'addr={addr}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "addr=localhost:8080" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(
            1,
            "class Config:\n    def __init__(self, host, port):\n        self.host = host\n        self.port = port\n    def url(self):\n        return f'http://{self.host}:{self.port}'",
        )
        nb_runner.set_cell_source(2, "cfg = Config('example.com', 443)\naddr = cfg.url()\nprint(f'addr={addr}')")
        nb_runner.run_all()
        assert "addr=http://example.com:443" in nb_runner.get_output(2)

    @pytest.mark.timeout(90)
    def test_class_default_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "class Greeter:\n    def __init__(self, greeting='Hello'):\n        self.greeting = greeting\n    def greet(self, name):\n        return f'{self.greeting}, {name}!'",
                "g = Greeter()\nmsg = g.greet('World')\nprint(f'msg={msg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg=Hello, World!" in nb_runner.get_output(2)


@pytest.mark.integration
class TestSelectiveCellReexecution:
    """Test running subsets of cells (common notebook interaction pattern)."""

    def test_rerun_subset_of_cells(self, nb_runner):
        """Re-run only cells 2 and 3 out of 4."""
        nb_runner.create_notebook(
            [
                "a = 5",
                "b = a + 10",
                "c = b * 2",
                "print(c)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "30" in nb_runner.get_output(4)

        # Re-run cells 2-4
        nb_runner.run_cells([2, 3, 4])
        assert "30" in nb_runner.get_output(4)

    def test_modify_and_rerun_downstream(self, nb_runner):
        """Modify a cell and only re-run it and downstream cells."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x + 1",
                "z = y + 1",
                "print(z)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "3" in nb_runner.get_output(4)

        # Modify cell 2 and run cells 2-4
        nb_runner.set_cell_source(2, "y = x + 100")
        nb_runner.run_cells([2, 3, 4])
        assert "102" in nb_runner.get_output(4)


@pytest.mark.integration
class TestVariableReassignment:
    """Test variable reassignment and shadow patterns."""

    def test_progressive_refinement(self, nb_runner):
        """Progressively refine a variable across cells."""
        nb_runner.create_notebook(
            [
                "data = list(range(10))",
                "data = [x for x in data if x % 2 == 0]",  # filter evens
                "data = [x ** 2 for x in data]",  # square
                "print(data)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[0, 4, 16, 36, 64]" in nb_runner.get_output(4)

    def test_conditional_variable_assignment(self, nb_runner):
        """Variable assigned conditionally across cells."""
        nb_runner.create_notebook(
            [
                "threshold = 50",
                "value = 75",
                textwrap.dedent("""\
                if value > threshold:
                    status = 'HIGH'
                else:
                    status = 'LOW'
                print(status)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "HIGH" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "value = 25")
        nb_runner.run_all()
        assert "LOW" in nb_runner.get_output(3)


@pytest.mark.integration
class TestDatetimePatterns:
    """Test datetime handling across cells."""

    def test_datetime_creation_and_formatting(self, nb_runner):
        """Create and format datetime objects across cells."""
        nb_runner.create_notebook(
            [
                "from datetime import datetime, timedelta",
                textwrap.dedent("""\
                start = datetime(2024, 1, 15, 10, 30)
                end = start + timedelta(days=7, hours=3)
            """),
                textwrap.dedent("""\
                diff = end - start
                formatted = end.strftime('%Y-%m-%d %H:%M')
                print(formatted, diff.days)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "2024-01-22 13:30" in output
        assert "7" in output

    def test_timedelta_arithmetic(self, nb_runner):
        """Timedelta arithmetic across cells."""
        nb_runner.create_notebook(
            [
                "from datetime import timedelta",
                textwrap.dedent("""\
                work_day = timedelta(hours=8)
                work_week = work_day * 5
                work_month = work_week * 4
            """),
                textwrap.dedent("""\
                total_hours = work_month.total_seconds() / 3600
                print(int(total_hours))
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "160" in nb_runner.get_output(3)


# Class/OOP + cell edit interaction tests.
#
# Tests that exercise class definitions, inheritance, method changes,
# and how cash tracks class-related dependencies through cell edits.
class TestClassDefinitionEdits:
    """Basic class definition + edit scenarios."""

    @pytest.mark.core
    @pytest.mark.timeout(30)
    def test_edit_class_attribute(self, nb_runner):
        """Edit a class attribute and verify downstream update."""
        nb_runner.create_notebook(
            [
                "class Config:\n    value = 10",
                "result = Config.value * 2\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "class Config:\n    value = 50")
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(2)

    @pytest.mark.core
    @pytest.mark.timeout(30)
    def test_add_method_to_class(self, nb_runner):
        """Add a new method to a class."""
        nb_runner.create_notebook(
            [
                "class Ops:\n    def add(self, a, b):\n        return a + b",
                "o = Ops()\nresult = o.add(3, 4)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 7" in nb_runner.get_output(2)

        # Add multiply method and use it
        nb_runner.set_cell_source(
            1,
            "class Ops:\n    def add(self, a, b):\n        return a + b\n    def mul(self, a, b):\n        return a * b",
        )
        nb_runner.set_cell_source(
            2,
            "o = Ops()\nresult = o.mul(3, 4)\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = 12" in nb_runner.get_output(2)

    # Class and OOP interaction tests.
    #
    # Tests where users define classes in cells, edit methods/attributes,
    # instantiate objects, and verify caching tracks OOP changes.
    @pytest.mark.upstream
    @pytest.mark.timeout(45)
    def test_edit_class_method(self, nb_runner):
        """Edit a method in a class."""
        nb_runner.create_notebook(
            [
                "class Calculator:\n    def compute(self, x):\n        return x * 2",
                "calc = Calculator()\nresult = calc.compute(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        # Edit method
        nb_runner.set_cell_source(1, "class Calculator:\n    def compute(self, x):\n        return x ** 2")
        nb_runner.run_all()
        assert "result = 25" in nb_runner.get_output(2)

    @pytest.mark.upstream
    @pytest.mark.timeout(45)
    def test_add_method_taking_a_list(self, nb_runner):
        """Add a new method to a class."""
        nb_runner.create_notebook(
            [
                "class Stats:\n    def mean(self, data):\n        return sum(data) / len(data)",
                "s = Stats()\nresult = s.mean([10, 20, 30])\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20.0" in nb_runner.get_output(2)

        # Add method and use it
        nb_runner.set_cell_source(
            1,
            "class Stats:\n    def mean(self, data):\n        return sum(data) / len(data)\n    def total(self, data):\n        return sum(data)",
        )
        nb_runner.set_cell_source(
            2,
            "s = Stats()\nresult = s.total([10, 20, 30])\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = 60" in nb_runner.get_output(2)

    @pytest.mark.upstream
    @pytest.mark.timeout(45)
    def test_edit_class_init(self, nb_runner):
        """Edit __init__ of a class."""
        nb_runner.create_notebook(
            [
                "class Multiplier:\n    def __init__(self, factor):\n        self.factor = factor\n    def apply(self, x):\n        return x * self.factor",
                "m = Multiplier(3)\nresult = m.apply(10)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(2)

        # Edit the instantiation
        nb_runner.set_cell_source(
            2,
            "m = Multiplier(5)\nresult = m.apply(10)\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = 50" in nb_runner.get_output(2)


class TestInheritanceEdits:
    """Class inheritance + cell edit scenarios."""

    @pytest.mark.core
    @pytest.mark.timeout(30)
    def test_edit_parent_class(self, nb_runner):
        """Edit parent class, verify child reflects the change."""
        nb_runner.create_notebook(
            [
                "class Base:\n    factor = 2",
                "class Child(Base):\n    def compute(self, x):\n        return x * self.factor",
                "c = Child()\nresult = c.compute(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "class Base:\n    factor = 10")
        nb_runner.run_all()
        assert "result = 50" in nb_runner.get_output(3)

    @pytest.mark.core
    @pytest.mark.timeout(30)
    def test_edit_child_class(self, nb_runner):
        """Edit child class, keep parent unchanged."""
        nb_runner.create_notebook(
            [
                "class Base:\n    def greet(self):\n        return 'hello'",
                "class Child(Base):\n    def greet(self):\n        return 'hi from child'",
                "c = Child()\nresult = c.greet()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = hi from child" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2,
            "class Child(Base):\n    def greet(self):\n        return super().greet() + ' world'",
        )
        nb_runner.run_all()
        assert "result = hello world" in nb_runner.get_output(3)

    @pytest.mark.upstream
    @pytest.mark.timeout(45)
    def test_edit_base_class(self, nb_runner):
        """Edit base class, verify subclass updates."""
        nb_runner.create_notebook(
            [
                "class Base:\n    def value(self):\n        return 10",
                "class Child(Base):\n    def doubled(self):\n        return self.value() * 2",
                "c = Child()\nresult = c.doubled()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(3)

        # Edit base
        nb_runner.set_cell_source(1, "class Base:\n    def value(self):\n        return 100")
        nb_runner.run_all()
        assert "result = 200" in nb_runner.get_output(3)

    @pytest.mark.upstream
    @pytest.mark.timeout(45)
    def test_edit_a_child_method_that_calls_the_parent(self, nb_runner):
        """Edit child class only."""
        nb_runner.create_notebook(
            [
                "class Base:\n    def value(self):\n        return 5",
                "class Child(Base):\n    def compute(self):\n        return self.value() + 1",
                "c = Child()\nresult = c.compute()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 6" in nb_runner.get_output(3)

        # Edit child
        nb_runner.set_cell_source(2, "class Child(Base):\n    def compute(self):\n        return self.value() * 10")
        nb_runner.run_all()
        assert "result = 50" in nb_runner.get_output(3)


@pytest.mark.core
@pytest.mark.timeout(30)
class TestClassInstanceEdits:
    """Class instance state + cell edits."""

    def test_edit_constructor_args(self, nb_runner):
        """Edit constructor arguments for a class instance."""
        nb_runner.create_notebook(
            [
                "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def dist(self):\n        return (self.x ** 2 + self.y ** 2) ** 0.5",
                "p = Point(3, 4)",
                "d = p.dist()\nprint(f'd = {d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d = 5.0" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "p = Point(5, 12)")
        nb_runner.run_all()
        assert "d = 13.0" in nb_runner.get_output(3)

    def test_edit_class_then_instantiation(self, nb_runner):
        """Edit both class and its instantiation."""
        nb_runner.create_notebook(
            [
                "class Msg:\n    def __init__(self, text):\n        self.text = text\n    def show(self):\n        return self.text.upper()",
                "m = Msg('hello')",
                "result = m.show()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = HELLO" in nb_runner.get_output(3)

        # Edit both class and instantiation
        nb_runner.set_cell_source(
            1,
            "class Msg:\n    def __init__(self, text):\n        self.text = text\n    def show(self):\n        return self.text.lower()",
        )
        nb_runner.set_cell_source(2, "m = Msg('WORLD')")
        nb_runner.run_all()
        assert "result = world" in nb_runner.get_output(3)


# Dataclass and namedtuple interaction tests.
#
# Tests editing dataclass/namedtuple definitions, field changes,
# and downstream usage after modifications.
class TestDataclassEdits:
    """Editing dataclass definitions and usage."""

    @pytest.mark.upstream
    @pytest.mark.timeout(90)
    def test_edit_dataclass_field(self, nb_runner):
        """Add a field to a dataclass, verify downstream updates."""
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass",
                "@dataclass\nclass Point:\n    x: float\n    y: float",
                "p = Point(1.0, 2.0)\nprint(f'p = {p}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Point(x=1.0, y=2.0)" in nb_runner.get_output(3)

        # Add a z field
        nb_runner.set_cell_source(
            2,
            "@dataclass\nclass Point:\n    x: float\n    y: float\n    z: float = 0.0",
        )
        nb_runner.set_cell_source(3, "p = Point(1.0, 2.0, 3.0)\nprint(f'p = {p}')")
        nb_runner.run_all()
        assert "Point(x=1.0, y=2.0, z=3.0)" in nb_runner.get_output(3)

    @pytest.mark.upstream
    @pytest.mark.timeout(90)
    def test_edit_dataclass_method(self, nb_runner):
        """Edit a method on a dataclass."""
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass",
                "@dataclass\nclass Rect:\n    w: float\n    h: float\n    def area(self):\n        return self.w * self.h",
                "r = Rect(3.0, 4.0)\nprint(f'area = {r.area()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 12.0" in nb_runner.get_output(3)

        # Change method to perimeter
        nb_runner.set_cell_source(
            2,
            "@dataclass\nclass Rect:\n    w: float\n    h: float\n    def area(self):\n        return 2 * (self.w + self.h)",
        )
        nb_runner.run_all()
        assert "area = 14.0" in nb_runner.get_output(3)

    @pytest.mark.upstream
    @pytest.mark.timeout(90)
    def test_dataclass_default_edit(self, nb_runner):
        """Edit default values on a dataclass."""
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass, field",
                "@dataclass\nclass Config:\n    name: str = 'default'\n    value: int = 0",
                "c = Config()\nprint(f'name={c.name} value={c.value}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "name=default value=0" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2,
            "@dataclass\nclass Config:\n    name: str = 'updated'\n    value: int = 42",
        )
        nb_runner.set_cell_source(3, "c = Config()\nprint(f'name={c.name} value={c.value}')")
        nb_runner.run_all()
        assert "name=updated value=42" in nb_runner.get_output(3)

    # Dataclass and structured data edit patterns.
    #
    # Tests dataclass definitions and instances with edits.
    @pytest.mark.timeout(90)
    def test_dataclass_field_edit(self, nb_runner):
        """Edit dataclass definition, instance creation updates."""
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass\n@dataclass\nclass Point:\n    x: float\n    y: float\n    def distance(self):\n        return (self.x**2 + self.y**2)**0.5",
                "p = Point(3.0, 4.0)\ndist = p.distance()\nprint(f'dist = {dist}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "dist = 5.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "from dataclasses import dataclass\n@dataclass\nclass Point:\n    x: float\n    y: float\n    def distance(self):\n        return abs(self.x) + abs(self.y)",
        )
        nb_runner.run_all()
        assert "dist = 7.0" in nb_runner.get_output(2)

    @pytest.mark.timeout(90)
    def test_edit_defaults_and_default_factory_in_the_import_cell(self, nb_runner):
        """Edit dataclass defaults, downstream reflects."""
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass, field\n@dataclass\nclass Config:\n    name: str = 'default'\n    scale: int = 1\n    tags: list = field(default_factory=list)",
                "c = Config()\nprint(f'name={c.name} scale={c.scale} tags={c.tags}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "name=default scale=1 tags=[]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "from dataclasses import dataclass, field\n@dataclass\nclass Config:\n    name: str = 'production'\n    scale: int = 10\n    tags: list = field(default_factory=lambda: ['v2'])",
        )
        nb_runner.run_all()
        assert "name=production scale=10 tags=['v2']" in nb_runner.get_output(2)

    @pytest.mark.timeout(90)
    def test_dataclass_instance_edit(self, nb_runner):
        """Edit instance creation, downstream computation updates."""
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass\n@dataclass\nclass Rectangle:\n    width: float\n    height: float\n    def area(self):\n        return self.width * self.height",
                "r = Rectangle(5.0, 3.0)",
                "a = r.area()\nprint(f'area = {a}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 15.0" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "r = Rectangle(10.0, 7.0)")
        nb_runner.run_all()
        assert "area = 70.0" in nb_runner.get_output(3)


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestNamedtupleEdits:
    """Editing namedtuple definitions."""

    def test_namedtuple_add_field(self, nb_runner):
        """Add a field to a namedtuple."""
        nb_runner.create_notebook(
            [
                "from collections import namedtuple",
                "Color = namedtuple('Color', ['r', 'g', 'b'])",
                "c = Color(255, 128, 0)\nprint(f'color = {c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "color = Color(r=255, g=128, b=0)" in nb_runner.get_output(3)

        # Add alpha field
        nb_runner.set_cell_source(2, "Color = namedtuple('Color', ['r', 'g', 'b', 'a'])")
        nb_runner.set_cell_source(3, "c = Color(255, 128, 0, 1.0)\nprint(f'color = {c}')")
        nb_runner.run_all()
        assert "color = Color(r=255, g=128, b=0, a=1.0)" in nb_runner.get_output(3)

    def test_namedtuple_to_dataclass(self, nb_runner):
        """Convert from namedtuple to dataclass."""
        nb_runner.create_notebook(
            [
                "from collections import namedtuple",
                "Point = namedtuple('Point', ['x', 'y'])\np = Point(1, 2)\nprint(f'p = {p}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "p = Point(x=1, y=2)" in nb_runner.get_output(2)

        # Switch to dataclass
        nb_runner.set_cell_source(1, "from dataclasses import dataclass")
        nb_runner.set_cell_source(
            2,
            "@dataclass\nclass Point:\n    x: int\n    y: int\np = Point(1, 2)\nprint(f'p = {p}')",
        )
        nb_runner.run_all()
        assert "p = Point(x=1, y=2)" in nb_runner.get_output(2)


# Class __repr__/__str__ edit propagation.
#
# Tests editing dunder methods on classes.
@pytest.mark.timeout(90)
class TestDunderMethodEdits:
    """Dunder method edit patterns."""

    def test_str_edit(self, nb_runner):
        """Edit __str__, string representation changes."""
        nb_runner.create_notebook(
            [
                "class Item:\n    def __init__(self, name, qty):\n        self.name = name\n        self.qty = qty\n    def __str__(self):\n        return f'{self.name}({self.qty})'",
                "item = Item('widget', 5)\nprint(f'item = {item}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "item = widget(5)" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "class Item:\n    def __init__(self, name, qty):\n        self.name = name\n        self.qty = qty\n    def __str__(self):\n        return f'{self.name} x{self.qty}'",
        )
        nb_runner.run_all()
        assert "item = widget x5" in nb_runner.get_output(2)

    def test_eq_edit(self, nb_runner):
        """Edit __eq__, comparison results change."""
        nb_runner.create_notebook(
            [
                "class Box:\n    def __init__(self, size):\n        self.size = size\n    def __eq__(self, other):\n        return self.size == other.size",
                "b1 = Box(10)\nb2 = Box(10)\nresult = (b1 == b2)\nprint(f'equal = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "equal = True" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "class Box:\n    def __init__(self, size):\n        self.size = size\n    def __eq__(self, other):\n        return self.size > other.size",
        )
        nb_runner.run_all()
        assert "equal = False" in nb_runner.get_output(2)

    def test_len_edit(self, nb_runner):
        """Edit __len__, len() call result changes."""
        nb_runner.create_notebook(
            [
                "class Container:\n    def __init__(self, items):\n        self.items = items\n    def __len__(self):\n        return len(self.items)",
                "c = Container([1, 2, 3, 4, 5])\nsize = len(c)\nprint(f'size = {size}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "size = 5" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "class Container:\n    def __init__(self, items):\n        self.items = items\n    def __len__(self):\n        return len(self.items) * 2",
        )
        nb_runner.run_all()
        assert "size = 10" in nb_runner.get_output(2)


# Dynamic class creation interaction tests.
#
# Tests editing dynamic class creation with type(),
# class factories, and mixin patterns.
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestDynamicClassEdits:
    """Editing dynamic class patterns."""

    def test_edit_type_creation(self, nb_runner):
        """Edit dynamic class created with type()."""
        nb_runner.create_notebook(
            [
                "MyClass = type('MyClass', (), {'value': 42, 'describe': lambda self: f'val={self.value}'})",
                "obj = MyClass()\nprint(f'result = {obj.describe()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = val=42" in nb_runner.get_output(2)

        # Change value
        nb_runner.set_cell_source(
            1,
            "MyClass = type('MyClass', (), {'value': 99, 'describe': lambda self: f'val={self.value}'})",
        )
        nb_runner.run_all()
        assert "result = val=99" in nb_runner.get_output(2)

    def test_edit_class_factory(self, nb_runner):
        """Edit a class factory function."""
        nb_runner.create_notebook(
            [
                "def make_class(prefix):\n    class Cls:\n        def greet(self):\n            return f'{prefix} World'\n    return Cls",
                "Hello = make_class('Hello')\nobj = Hello()\nprint(f'result = {obj.greet()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = Hello World" in nb_runner.get_output(2)

        # Change factory
        nb_runner.set_cell_source(
            1,
            "def make_class(prefix):\n    class Cls:\n        def greet(self):\n            return f'{prefix}!!!'\n    return Cls",
        )
        nb_runner.run_all()
        assert "result = Hello!!!" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestMixinEdits:
    """Editing mixin patterns."""

    def test_edit_mixin_combined(self, nb_runner):
        """Edit a class method that determines output."""
        nb_runner.create_notebook(
            [
                "class Greeter:\n    def greet(self, name):\n        return 'Hello ' + name",
                "g = Greeter()\nprint(g.greet('alice'))",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(2)
        assert "Hello alice" in out1

        # Edit class method
        nb_runner.set_cell_source(1, "class Greeter:\n    def greet(self, name):\n        return 'Hi ' + name")
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "Hi alice" in out2


# Multi-cell class instantiation with method edits.
#
# Tests class defined in one cell, instantiated in another, method called in third.
@pytest.mark.timeout(90)
class TestMultiCellClassEdits:
    """Class spread across cells with edits."""

    def test_class_method_edit_propagates(self, nb_runner):
        """Edit class method, instantiation and usage cells reflect."""
        nb_runner.create_notebook(
            [
                "class Calculator:\n    def __init__(self, val=0):\n        self.val = val\n    def add(self, n):\n        return Calculator(self.val + n)\n    def result(self):\n        return self.val",
                "c = Calculator(10).add(5).add(3)",
                "r = c.result()\nprint(f'r = {r}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "class Formatter:\n    def __init__(self, prefix, suffix):\n        self.prefix = prefix\n        self.suffix = suffix\n    def wrap(self, text):\n        return f'{self.prefix}{text}{self.suffix}'",
                "fmt = Formatter('[', ']')",
                "out = fmt.wrap('hello')\nprint(f'out = {out}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "out = [hello]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "fmt = Formatter('<<', '>>')")
        nb_runner.run_all()
        assert "out = <<hello>>" in nb_runner.get_output(3)

    def test_edit_both_class_and_usage(self, nb_runner):
        """Edit class and usage cell simultaneously."""
        nb_runner.create_notebook(
            [
                "class Scaler:\n    def __init__(self, factor):\n        self.factor = factor\n    def scale(self, x):\n        return x * self.factor",
                "s = Scaler(2)",
                "result = s.scale(10)\nprint(f'result = {result}')",
            ]
        )
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


# Property and descriptor edit patterns.
#
# Tests @property, computed properties with class edits.
@pytest.mark.timeout(90)
class TestPropertyDescriptorEdits:
    """Property and descriptor patterns."""

    def test_property_getter_edit(self, nb_runner):
        """Edit property getter, downstream reflects."""
        nb_runner.create_notebook(
            [
                "class Circle:\n    def __init__(self, r):\n        self._r = r\n    @property\n    def area(self):\n        return 3.14 * self._r ** 2",
                "c = Circle(5)\nprint(f'area = {c.area}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 78.5" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "class Circle:\n    def __init__(self, r):\n        self._r = r\n    @property\n    def area(self):\n        return 3.14159 * self._r ** 2",
        )
        nb_runner.run_all()
        assert "78.539" in nb_runner.get_output(2)

    def test_computed_property_edit(self, nb_runner):
        """Edit computed property formula."""
        nb_runner.create_notebook(
            [
                "class BMI:\n    def __init__(self, weight, height):\n        self.weight = weight\n        self.height = height\n    @property\n    def value(self):\n        return round(self.weight / (self.height ** 2), 1)",
                "bmi = BMI(70, 1.75)\nprint(f'bmi = {bmi.value}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "bmi = 22.9" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "class BMI:\n    def __init__(self, weight, height):\n        self.weight = weight\n        self.height = height\n    @property\n    def value(self):\n        return round(self.weight / (self.height ** 2) * 100, 1)",
        )
        nb_runner.run_all()
        assert "bmi = 2285.7" in nb_runner.get_output(2)

    def test_property_with_setter_edit(self, nb_runner):
        """Edit class with property setter."""
        nb_runner.create_notebook(
            [
                "class Temperature:\n    def __init__(self, celsius):\n        self._c = celsius\n    @property\n    def fahrenheit(self):\n        return self._c * 9/5 + 32",
                "t = Temperature(100)\nprint(f'f = {t.fahrenheit}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "f = 212.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "t = Temperature(0)\nprint(f'f = {t.fahrenheit}')")
        nb_runner.run_all()
        assert "f = 32.0" in nb_runner.get_output(2)


# Property and descriptor pattern interaction tests.
#
# Tests editing property definitions, getters/setters,
# and descriptor protocols across cells.
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestPropertyEdits:
    """Editing property definitions."""

    def test_edit_property_getter(self, nb_runner):
        """Edit a property getter."""
        nb_runner.create_notebook(
            [
                "class Temp:\n    def __init__(self, c):\n        self._c = c\n    @property\n    def fahrenheit(self):\n        return self._c * 9/5 + 32",
                "t = Temp(100)\nprint(f'f = {t.fahrenheit}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "f = 212.0" in nb_runner.get_output(2)

        # Change to Kelvin
        nb_runner.set_cell_source(
            1,
            "class Temp:\n    def __init__(self, c):\n        self._c = c\n    @property\n    def fahrenheit(self):\n        return self._c + 273.15",
        )
        nb_runner.run_all()
        assert "f = 373.15" in nb_runner.get_output(2)

    def test_add_property_setter(self, nb_runner):
        """Add a property setter to an existing class."""
        nb_runner.create_notebook(
            [
                "class Box:\n    def __init__(self, w, h):\n        self._w = w\n        self._h = h\n    @property\n    def area(self):\n        return self._w * self._h",
                "b = Box(3, 4)\nprint(f'area = {b.area}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 12" in nb_runner.get_output(2)

        # Add volume
        nb_runner.set_cell_source(
            1,
            "class Box:\n    def __init__(self, w, h, d=1):\n        self._w = w\n        self._h = h\n        self._d = d\n    @property\n    def area(self):\n        return self._w * self._h\n    @property\n    def volume(self):\n        return self._w * self._h * self._d",
        )
        nb_runner.set_cell_source(2, "b = Box(3, 4, 5)\nprint(f'area={b.area} vol={b.volume}')")
        nb_runner.run_all()
        assert "area=12 vol=60" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestClassMethodEdits:
    """Editing class/static methods."""

    def test_edit_classmethod(self, nb_runner):
        """Edit a classmethod."""
        nb_runner.create_notebook(
            [
                "class Counter:\n    _count = 0\n    @classmethod\n    def increment(cls):\n        cls._count += 1\n        return cls._count",
                "a = Counter.increment()\nb = Counter.increment()\nprint(f'a={a} b={b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=1 b=2" in nb_runner.get_output(2)

        # Change to increment by 10
        nb_runner.set_cell_source(
            1,
            "class Counter:\n    _count = 0\n    @classmethod\n    def increment(cls):\n        cls._count += 10\n        return cls._count",
        )
        nb_runner.run_all()
        assert "a=10 b=20" in nb_runner.get_output(2)


# Dataclass and namedtuple structural change tests.
#
# Tests editing dataclass field definitions and namedtuple structures
# to verify changes propagate through the cache.
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestStructuralTypeEdits:
    """Editing dataclass and namedtuple structures."""

    def test_edit_dataclass_add_field(self, nb_runner):
        """Add a field to a dataclass and verify downstream."""
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass\n@dataclass\nclass Item:\n    name: str\n    price: float = 0.0",
                "item = Item('Widget', 9.99)\nprint(f'item = {item}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Widget" in nb_runner.get_output(2)
        assert "9.99" in nb_runner.get_output(2)

        # Add a quantity field
        nb_runner.set_cell_source(
            1,
            "from dataclasses import dataclass\n@dataclass\nclass Item:\n    name: str\n    price: float = 0.0\n    qty: int = 1",
        )
        nb_runner.set_cell_source(2, "item = Item('Widget', 9.99, qty=5)\nprint(f'item = {item}')")
        nb_runner.run_all()
        assert "qty=5" in nb_runner.get_output(2)

    def test_edit_namedtuple_add_field(self, nb_runner):
        """Add a field to a namedtuple."""
        nb_runner.create_notebook(
            [
                "from collections import namedtuple\nPoint = namedtuple('Point', ['x', 'y'])",
                "p = Point(3, 4)\nprint(f'p = {p}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Point(x=3, y=4)" in nb_runner.get_output(2)

        # Add z field
        nb_runner.set_cell_source(1, "from collections import namedtuple\nPoint = namedtuple('Point', ['x', 'y', 'z'])")
        nb_runner.set_cell_source(2, "p = Point(3, 4, 5)\nprint(f'p = {p}')")
        nb_runner.run_all()
        assert "Point(x=3, y=4, z=5)" in nb_runner.get_output(2)

    def test_edit_dataclass_method_behavior(self, nb_runner):
        """Edit a method on a dataclass."""
        nb_runner.create_notebook(
            [
                "from dataclasses import dataclass\n@dataclass\nclass Circle:\n    radius: float\n    def area(self):\n        return 3.14159 * self.radius ** 2",
                "c = Circle(5.0)\nprint(f'area = {c.area():.2f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "area = 78.54" in nb_runner.get_output(2)

        # Change to circumference calculation
        nb_runner.set_cell_source(
            1,
            "from dataclasses import dataclass\n@dataclass\nclass Circle:\n    radius: float\n    def area(self):\n        return 2 * 3.14159 * self.radius",
        )
        nb_runner.run_all()
        assert "area = 31.42" in nb_runner.get_output(2)


# complex class interactions across multiple cells.
@pytest.mark.integration
class TestCrossCellClassInteractions:
    """Classes defined in one cell, used in another."""

    def test_observer_pattern_cross_cell(self, nb_runner):
        """Observer pattern across cells with change propagation."""
        nb_runner.create_notebook(
            [
                "event_name = 'click'",
                textwrap.dedent("""\
                class EventBus:
                    def __init__(self):
                        self.listeners = {}
                        self.log = []
                    def on(self, event, fn):
                        self.listeners.setdefault(event, []).append(fn)
                    def emit(self, event, data=None):
                        for fn in self.listeners.get(event, []):
                            result = fn(data)
                            self.log.append(result)

                bus = EventBus()
                bus.on(event_name, lambda d: f"handler1: {d}")
                bus.on(event_name, lambda d: f"handler2: {d}")
                bus.emit(event_name, "test_data")
                logged = bus.log[:]
            """),
                "print(f'logged={logged}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "handler1: test_data" in out
        assert "handler2: test_data" in out

        nb_runner.set_cell_source(1, "event_name = 'submit'")
        nb_runner.run_cells([1, 2, 3])
        out2 = nb_runner.get_output(3)
        assert "handler1: test_data" in out2
