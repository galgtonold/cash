"""Functions, classes and closures defined in one cell and used in cells below."""

import textwrap

import pytest


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestMultiCellFunctionPatterns:
    """Test function definition and usage across multiple cells."""

    @pytest.mark.core
    def test_function_change_propagates_to_composition(self, nb_runner):
        """Changing a helper function should invalidate composed function usage."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def process(x):
                    return x * 2"""),
                "result = process(10)\nprint(f'Result: {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(2)
        assert "Result: 20" in out1

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            def process(x):
                return x * 3"""),
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "Result: 30" in out2

    @pytest.mark.core
    def test_closure_captures_cell_variable(self, nb_runner):
        """Closure that captures a variable from a previous cell."""
        nb_runner.create_notebook(
            [
                "multiplier = 5",
                textwrap.dedent("""\
                def make_multiplier():
                    return lambda x: x * multiplier"""),
                "fn = make_multiplier()",
                "result = fn(10)\nprint(f'Result: {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(4)
        assert "Result: 50" in out1

        # Change multiplier
        nb_runner.set_cell_source(1, "multiplier = 10")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "Result: 100" in out2


@pytest.mark.core
class TestFunctionRedefinition:
    """Test that redefining a function triggers downstream recomputation."""

    def test_redefine_function_invalidates_downstream(self, nb_runner):
        """
        Define a function in cell 1, use it in cell 2.
        Then redefine the function and re-run cell 2.
        """
        nb_runner.create_notebook(
            [
                "def transform(x):\n    return x * 2",
                "result = transform(5)\nprint(f'Result: {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "Result: 10" in nb_runner.get_output(2)

        # Redefine the function
        nb_runner.set_cell_source(1, "def transform(x):\n    return x * 3")
        nb_runner.run_cells([1, 2])

        out = nb_runner.get_output(2)
        assert "Result: 15" in out, f"Expected Result: 15 after redefine, got: {out}"

    def test_redefine_function_only_downstream_runs(self, nb_runner):
        """
        Redefine function in cell 1, run only downstream cell 2.
        The upstream simulation should detect the change and re-execute cell 1.
        """
        nb_runner.create_notebook(
            [
                "def compute(x):\n    return x + 100",
                "val = compute(5)\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "val = 105" in nb_runner.get_output(2)

        # Modify function definition
        nb_runner.set_cell_source(1, "def compute(x):\n    return x + 200")
        # Only run cell 2 - upstream should auto-execute cell 1
        nb_runner.run_cell(2)

        out = nb_runner.get_output(2)
        assert "val = 205" in out, f"Expected val=205, got: {out}"


@pytest.mark.integration
@pytest.mark.timeout(30)
class TestNestedFunctionClosures:
    """Tests for nested functions and closure patterns across cells."""

    @pytest.mark.core
    def test_higher_order_function_composition(self, nb_runner):
        """Function composition with higher-order functions."""
        nb_runner.create_notebook(
            [
                "def compose(f, g):\n    def composed(x):\n        return f(g(x))\n    return composed",
                "double = lambda x: x * 2\nadd_one = lambda x: x + 1",
                "double_then_add = compose(add_one, double)",
                "result = double_then_add(5)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        output = nb_runner.get_output(5)
        assert "result=11" in output  # double(5)=10, add_one(10)=11

        # Change composition order
        nb_runner.set_cell_source(3, "add_then_double = compose(double, add_one)")
        nb_runner.set_cell_source(4, "result = add_then_double(5)")
        nb_runner.run_all()

        output2 = nb_runner.get_output(5)
        assert "result=12" in output2  # add_one(5)=6, double(6)=12


# Advanced cross-cell interaction torture tests.
#
# Tests that combine multiple features simultaneously: function definitions
# referencing external variables, class hierarchies with file dependencies,
# decorator + module reload combos, and long multi-cell computation chains.
@pytest.mark.integration
@pytest.mark.stress
class TestCrossCellFunctionState:
    """Test functions that capture state from other cells."""

    def test_function_reads_global_config(self, nb_runner):
        """Function in cell 2 reads config dict from cell 1."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                config = {
                    'multiplier': 3,
                    'offset': 10,
                    'precision': 2
                }
            """),
                textwrap.dedent("""\
                def transform(x):
                    result = x * config['multiplier'] + config['offset']
                    return round(result, config['precision'])
            """),
                textwrap.dedent("""\
                values = [1.0, 2.5, 3.7]
                results = [transform(v) for v in values]
                print(results)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        # 1*3+10=13, 2.5*3+10=17.5, 3.7*3+10=21.1
        assert "13" in output
        assert "17.5" in output
        assert "21.1" in output

        # Change config -> function results should change
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            config = {
                'multiplier': 10,
                'offset': 0,
                'precision': 1
            }
        """),
        )
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        # 1*10+0=10, 2.5*10+0=25, 3.7*10+0=37
        assert "10" in output
        assert "25" in output
        assert "37" in output

    def test_recursive_function_across_cells(self, nb_runner):
        """Recursive function defined in one cell, called in another."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                memo = {}
                def fib(n):
                    if n in memo:
                        return memo[n]
                    if n < 2:
                        return n
                    result = fib(n-1) + fib(n-2)
                    memo[n] = result
                    return result
            """),
                textwrap.dedent("""\
                result = fib(20)
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "6765" in nb_runner.get_output(2)


@pytest.mark.core
class TestGlobalStateInteraction:
    """Test caching with global/module-level state modifications."""

    def test_memoization_pattern(self, nb_runner):
        """Test a memoized function pattern."""
        nb_runner.create_notebook(
            [
                "def memoize(f):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = f(*args)\n        return cache[args]\n    return wrapper",
                "@memoize\ndef fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)",
                "r = fib(10)\nprint(f'fib(10) = {r}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "fib(10) = 55" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.integration
class TestDecoratorStacking:
    """Test functions with multiple decorators and decorator interactions."""

    def test_decorator_change_invalidation(self, nb_runner):
        """Changing a decorator definition should invalidate decorated functions."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def multiplier(func):
                    def wrapper(*args, **kwargs):
                        return func(*args, **kwargs) * 2
                    return wrapper
            """),
                textwrap.dedent("""\
                @multiplier
                def calc(x):
                    return x + 10

                result = calc(5)
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # calc(5) -> 15, *2 = 30
        assert "30" in nb_runner.get_output(2)

        # Change decorator
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            def multiplier(func):
                def wrapper(*args, **kwargs):
                    return func(*args, **kwargs) * 10
                return wrapper
        """),
        )
        nb_runner.run_all()
        # calc(5) -> 15, *10 = 150
        assert "150" in nb_runner.get_output(2)

    def test_class_decorator(self, nb_runner):
        """Class used as a decorator."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class Cache:
                    def __init__(self, func):
                        self.func = func
                        self._cache = {}
                    def __call__(self, *args):
                        if args not in self._cache:
                            self._cache[args] = self.func(*args)
                        return self._cache[args]
            """),
                textwrap.dedent("""\
                @Cache
                def expensive(n):
                    return n * n

                r1 = expensive(5)
                r2 = expensive(5)
                print(r1, r2)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "25 25" in nb_runner.get_output(2)

    def test_parametrized_decorator(self, nb_runner):
        """Decorator with arguments (decorator factory)."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def repeat(n):
                    def decorator(func):
                        def wrapper(*args, **kwargs):
                            return [func(*args, **kwargs) for _ in range(n)]
                        return wrapper
                    return decorator
            """),
                textwrap.dedent("""\
                @repeat(3)
                def greet(name):
                    return f"Hello {name}"

                result = greet("World")
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "Hello World" in output


@pytest.mark.stress
@pytest.mark.integration
class TestContextManagers:
    """Test context manager patterns and their caching behavior."""

    def test_custom_context_manager_class(self, nb_runner):
        """Custom context manager using __enter__/__exit__."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class Timer:
                    def __init__(self):
                        self.elapsed = 0
                    def __enter__(self):
                        self.elapsed = 0
                        return self
                    def __exit__(self, *args):
                        self.elapsed = 42  # fake timing
                        return False
            """),
                textwrap.dedent("""\
                with Timer() as t:
                    result = sum(range(100))
                elapsed = t.elapsed
                print(result, elapsed)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "4950" in nb_runner.get_output(2)
        assert "42" in nb_runner.get_output(2)

    def test_file_context_manager_tracking(self, nb_runner, tmp_path):
        """File opened with context manager should be tracked."""
        data_file = tmp_path / "ctx_data.txt"
        data_file.write_text("hello context", encoding="utf-8")
        path_str = str(data_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"path = '{path_str}'",
                textwrap.dedent("""\
                with open(path, 'r') as f:
                    content = f.read()
                print(content)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hello context" in nb_runner.get_output(2)


# Cross-cell patterns, class instances, generators,
# exception recovery, partial re-runs, multi-module cascades.
#
# Tests focus on complex real-world usage patterns that span multiple cells
# and exercise the caching framework's ability to track dependencies across
# execution boundaries.
@pytest.mark.core
class TestClassInstanceCrossCells:
    """Test class instance creation, method calls, and mutation across cells."""

    def test_class_redefinition_invalidates_instances(self, nb_runner):
        """Redefining a class should invalidate cells using instances of it."""
        nb_runner.create_notebook(
            [
                "class Greeter:\n    def greet(self):\n        return 'hello'",
                "g = Greeter()\nprint(g.greet())",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "hello" in nb_runner.get_output(2)

        # Change the class
        nb_runner.set_cell_source(1, "class Greeter:\n    def greet(self):\n        return 'hi there'")
        nb_runner.run_all()

        out = nb_runner.get_output(2)
        assert "hi there" in out, f"Expected 'hi there', got: {out}"

    def test_inheritance_chain_across_cells(self, nb_runner):
        """Base class in cell 1, child in cell 2, usage in cell 3."""
        nb_runner.create_notebook(
            [
                "class Animal:\n    def speak(self):\n        return 'generic sound'",
                "class Dog(Animal):\n    def speak(self):\n        return 'woof'",
                "d = Dog()\nprint(d.speak())",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "woof" in nb_runner.get_output(3)

        # Modify base class — doesn't change Dog.speak(), but tests revalidation
        nb_runner.set_cell_source(
            1,
            "class Animal:\n    def speak(self):\n        return 'roar'\n    def name(self):\n        return 'animal'",
        )
        nb_runner.run_all()

        # Dog.speak still returns woof
        assert "woof" in nb_runner.get_output(3)


@pytest.mark.core
class TestClassInheritancePatterns:
    """Test class inheritance and method caching."""

    def test_modify_base_class_invalidates_child(self, nb_runner):
        """Modifying base class should invalidate derived class usage."""
        nb_runner.create_notebook(
            [
                """class Base:
    def greet(self):
        return "Hello"
""",
                """class Child(Base):
    def greet(self):
        return super().greet() + " World"
""",
                "obj = Child()\nprint(obj.greet())",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "Hello World" in nb_runner.get_output(3)

        # Modify the base class
        nb_runner.set_cell_source(
            1,
            """class Base:
    def greet(self):
        return "Hi"
""",
        )
        nb_runner.run_cells([1, 2, 3])

        out = nb_runner.get_output(3)
        assert "Hi World" in out, f"Expected 'Hi World', got: {out}"


@pytest.mark.stress
@pytest.mark.integration
class TestClassInheritanceMRO:
    """Test class hierarchies, MRO, and method resolution caching."""

    def test_multiple_inheritance_mro(self, nb_runner):
        """Diamond inheritance with MRO resolution."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class A:
                    def who(self):
                        return "A"

                class B(A):
                    def who(self):
                        return "B"

                class C(A):
                    def who(self):
                        return "C"

                class D(B, C):
                    pass
            """),
                textwrap.dedent("""\
                d = D()
                print(d.who())
                print([cls.__name__ for cls in D.__mro__])
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "B" in output  # MRO: D -> B -> C -> A
        assert "['D', 'B', 'C', 'A'" in output

    def test_base_class_change_propagation(self, nb_runner):
        """Changing a base class should invalidate subclass instances."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                class Config:
                    DEFAULT = 10
            """),
                textwrap.dedent("""\
                class AppConfig(Config):
                    def get_value(self):
                        return self.DEFAULT * 2
            """),
                textwrap.dedent("""\
                cfg = AppConfig()
                print(cfg.get_value())
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "20" in nb_runner.get_output(3)

        # Change base class default
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            class Config:
                DEFAULT = 50
        """),
        )
        nb_runner.run_all()
        assert "100" in nb_runner.get_output(3)


@pytest.mark.core
class TestDataclassCaching:
    """Test caching behavior with dataclass objects."""

    def test_dataclass_mutation_detection(self, nb_runner):
        """Test that mutating a dataclass field is detected."""
        nb_runner.create_notebook(
            [
                """from dataclasses import dataclass, field
from typing import List

@dataclass
class Accumulator:
    items: List[int] = field(default_factory=list)
    
    def add(self, val):
        self.items.append(val)
        return self""",
                "acc = Accumulator()",
                "acc.add(10)\nacc.add(20)\nprint(f'Items: {acc.items}')",
                "total = sum(acc.items)\nprint(f'Total: {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        out3 = nb_runner.get_output(3)
        assert "Items: [10, 20]" in out3, f"Expected [10, 20], got: {out3}"

        out4 = nb_runner.get_output(4)
        assert "Total: 30" in out4, f"Expected Total: 30, got: {out4}"


@pytest.mark.stress
@pytest.mark.integration
class TestNamespaceScopeEdgeCases:
    """Test subtle namespace and scope interactions."""

    def test_closure_variable_update(self, nb_runner):
        """Closure captures variable, variable changes, closure re-created."""
        nb_runner.create_notebook(
            [
                "scale = 2",
                textwrap.dedent("""\
                def create_scaler():
                    return lambda x: x * scale
                scaler = create_scaler()
            """),
                "print(scaler(10))",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "20" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "scale = 5")
        nb_runner.run_all()
        # After change, scaler should be re-created with new scale
        assert "50" in nb_runner.get_output(3)

    def test_builtin_shadowing(self, nb_runner):
        """Shadowing a builtin name and using it."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                list = [1, 2, 3]  # shadows builtin list
                result = len(list)
                print(result)
            """),
                textwrap.dedent("""\
                # Restore builtin
                import builtins
                list = builtins.list
                result = list(range(5))
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "3" in nb_runner.get_output(1)
        assert "[0, 1, 2, 3, 4]" in nb_runner.get_output(2)
