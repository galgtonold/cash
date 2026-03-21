"""
Batch 13: Complex dependency chains, decorator stacking, context managers,
class inheritance MRO, and namespace/scope edge cases.

Tests deep multi-cell dependency propagation, complex decorator interactions,
context manager state tracking, MRO-based method resolution caching, and
subtle namespace scoping issues.
"""
import pytest
import textwrap


pytestmark = [pytest.mark.integration, pytest.mark.stress]


# ============================================================
# Test Group 1: Deep Dependency Chains
# ============================================================

class TestDeepDependencyChains:
    """Test multi-level variable dependency propagation across many cells."""

    def test_linear_chain_10_cells(self, nb_runner):
        """10-cell linear dependency chain: each cell uses previous cell's output."""
        cells = [f"v{i} = v{i-1} + 1" if i > 0 else "v0 = 1" for i in range(10)]
        cells.append("print(v9)")
        nb_runner.create_notebook(cells)
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "10" in nb_runner.get_output(len(cells))

        # Second run — all should be cached
        nb_runner.reset_cash_state()
        nb_runner.run_all()
        assert "10" in nb_runner.get_output(len(cells))

    def test_linear_chain_change_root(self, nb_runner):
        """Change root of a dependency chain and verify propagation."""
        nb_runner.create_notebook([
            "a = 10",
            "b = a * 2",
            "c = b + 5",
            "d = c ** 2",
            "print(d)"
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "625" in nb_runner.get_output(5)  # (10*2+5)^2 = 625

        # Change root
        nb_runner.set_cell_source(1, "a = 20")
        nb_runner.run_all()
        assert "2025" in nb_runner.get_output(5)  # (20*2+5)^2 = 2025

    def test_diamond_dependency(self, nb_runner):
        """Diamond pattern: A -> B, A -> C, B+C -> D."""
        nb_runner.create_notebook([
            "a = 5",
            "b = a * 2",
            "c = a + 3",
            "d = b + c",
            "print(d)"
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "18" in nb_runner.get_output(5)  # 5*2 + 5+3 = 18

        # Change root -> both branches update
        nb_runner.set_cell_source(1, "a = 10")
        nb_runner.run_all()
        assert "33" in nb_runner.get_output(5)  # 10*2 + 10+3 = 33

    def test_fan_out_dependency(self, nb_runner):
        """One variable feeds many downstream cells."""
        nb_runner.create_notebook([
            "base = 100",
            "x1 = base + 1",
            "x2 = base + 2",
            "x3 = base + 3",
            "total = x1 + x2 + x3",
            "print(total)"
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "306" in nb_runner.get_output(6)  # 101+102+103

        nb_runner.set_cell_source(1, "base = 200")
        nb_runner.run_all()
        assert "606" in nb_runner.get_output(6)  # 201+202+203

    def test_fan_in_dependency(self, nb_runner):
        """Multiple independent sources converge into one cell."""
        nb_runner.create_notebook([
            "x = 10",
            "y = 20",
            "z = 30",
            "total = x + y + z",
            "print(total)"
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "60" in nb_runner.get_output(5)

        # Change only one source
        nb_runner.set_cell_source(2, "y = 200")
        nb_runner.run_all()
        assert "240" in nb_runner.get_output(5)


# ============================================================
# Test Group 2: Decorator Stacking
# ============================================================

class TestDecoratorStacking:
    """Test functions with multiple decorators and decorator interactions."""

    def test_stacked_decorators(self, nb_runner):
        """Function with multiple decorators stacked."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def double_result(func):
                    def wrapper(*args, **kwargs):
                        return func(*args, **kwargs) * 2
                    return wrapper

                def add_one(func):
                    def wrapper(*args, **kwargs):
                        return func(*args, **kwargs) + 1
                    return wrapper
            """),
            textwrap.dedent("""\
                @double_result
                @add_one
                def compute(x):
                    return x * 3

                result = compute(5)
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # compute(5) -> 5*3=15, add_one->16, double->32
        assert "32" in nb_runner.get_output(2)

    def test_decorator_change_invalidation(self, nb_runner):
        """Changing a decorator definition should invalidate decorated functions."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # calc(5) -> 15, *2 = 30
        assert "30" in nb_runner.get_output(2)

        # Change decorator
        nb_runner.set_cell_source(1, textwrap.dedent("""\
            def multiplier(func):
                def wrapper(*args, **kwargs):
                    return func(*args, **kwargs) * 10
                return wrapper
        """))
        nb_runner.run_all()
        # calc(5) -> 15, *10 = 150
        assert "150" in nb_runner.get_output(2)

    def test_class_decorator(self, nb_runner):
        """Class used as a decorator."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "25 25" in nb_runner.get_output(2)

    def test_parametrized_decorator(self, nb_runner):
        """Decorator with arguments (decorator factory)."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "Hello World" in output


# ============================================================
# Test Group 3: Context Managers
# ============================================================

class TestContextManagers:
    """Test context manager patterns and their caching behavior."""

    def test_custom_context_manager_class(self, nb_runner):
        """Custom context manager using __enter__/__exit__."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "4950" in nb_runner.get_output(2)
        assert "42" in nb_runner.get_output(2)

    def test_contextlib_contextmanager(self, nb_runner):
        """Context manager from contextlib.contextmanager."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from contextlib import contextmanager

                @contextmanager
                def managed_list():
                    lst = []
                    yield lst
                    lst.sort()
            """),
            textwrap.dedent("""\
                with managed_list() as items:
                    items.extend([3, 1, 2])
                print(items)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[1, 2, 3]" in nb_runner.get_output(2)

    def test_nested_context_managers(self, nb_runner):
        """Multiple nested context managers."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from contextlib import contextmanager

                @contextmanager
                def tag(name):
                    yield f"<{name}>"
            """),
            textwrap.dedent("""\
                with tag("div") as outer:
                    with tag("span") as inner:
                        combined = f"{outer}{inner}content"
                print(combined)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "<div><span>content" in nb_runner.get_output(2)

    def test_file_context_manager_tracking(self, nb_runner, tmp_path):
        """File opened with context manager should be tracked."""
        data_file = tmp_path / "ctx_data.txt"
        data_file.write_text("hello context")
        path_str = str(data_file).replace('\\', '/')

        nb_runner.create_notebook([
            f"path = '{path_str}'",
            textwrap.dedent("""\
                with open(path, 'r') as f:
                    content = f.read()
                print(content)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hello context" in nb_runner.get_output(2)


# ============================================================
# Test Group 4: Class Inheritance & MRO
# ============================================================

class TestClassInheritanceMRO:
    """Test class hierarchies, MRO, and method resolution caching."""

    def test_simple_inheritance(self, nb_runner):
        """Single inheritance with method override."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Animal:
                    def speak(self):
                        return "..."

                class Dog(Animal):
                    def speak(self):
                        return "Woof"
            """),
            textwrap.dedent("""\
                d = Dog()
                print(d.speak())
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Woof" in nb_runner.get_output(2)

    def test_multi_level_inheritance(self, nb_runner):
        """Three-level inheritance chain."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Base:
                    def value(self):
                        return 1

                class Middle(Base):
                    def value(self):
                        return super().value() + 10

                class Child(Middle):
                    def value(self):
                        return super().value() + 100
            """),
            textwrap.dedent("""\
                obj = Child()
                print(obj.value())
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "111" in nb_runner.get_output(2)

    def test_multiple_inheritance_mro(self, nb_runner):
        """Diamond inheritance with MRO resolution."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "B" in output  # MRO: D -> B -> C -> A
        assert "['D', 'B', 'C', 'A'" in output

    def test_base_class_change_propagation(self, nb_runner):
        """Changing a base class should invalidate subclass instances."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "20" in nb_runner.get_output(3)

        # Change base class default
        nb_runner.set_cell_source(1, textwrap.dedent("""\
            class Config:
                DEFAULT = 50
        """))
        nb_runner.run_all()
        assert "100" in nb_runner.get_output(3)

    def test_mixin_pattern(self, nb_runner):
        """Mixin classes combined via multiple inheritance."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class JsonMixin:
                    def to_json(self):
                        import json
                        return json.dumps(self.__dict__)

                class PrintMixin:
                    def display(self):
                        return f"<{self.__class__.__name__}: {self.__dict__}>"
            """),
            textwrap.dedent("""\
                class User(JsonMixin, PrintMixin):
                    def __init__(self, name, age):
                        self.name = name
                        self.age = age

                u = User("Alice", 30)
                print(u.to_json())
                print(u.display())
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert '"Alice"' in output
        assert "User:" in output


# ============================================================
# Test Group 5: Namespace & Scope Edge Cases
# ============================================================

class TestNamespaceScopeEdgeCases:
    """Test subtle namespace and scope interactions."""

    def test_closure_captures_cell_variable(self, nb_runner):
        """Closure capturing a variable defined in another cell."""
        nb_runner.create_notebook([
            "multiplier = 5",
            textwrap.dedent("""\
                def make_multiplier():
                    return lambda x: x * multiplier
                fn = make_multiplier()
            """),
            "print(fn(10))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "50" in nb_runner.get_output(3)

    def test_closure_variable_update(self, nb_runner):
        """Closure captures variable, variable changes, closure re-created."""
        nb_runner.create_notebook([
            "scale = 2",
            textwrap.dedent("""\
                def create_scaler():
                    return lambda x: x * scale
                scaler = create_scaler()
            """),
            "print(scaler(10))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "20" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "scale = 5")
        nb_runner.run_all()
        # After change, scaler should be re-created with new scale
        assert "50" in nb_runner.get_output(3)

    def test_class_defined_across_cells(self, nb_runner):
        """Base class in one cell, subclass in another, instance in third."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Shape:
                    def area(self):
                        return 0
            """),
            textwrap.dedent("""\
                class Circle(Shape):
                    def __init__(self, r):
                        self.r = r
                    def area(self):
                        return 3.14159 * self.r ** 2
            """),
            textwrap.dedent("""\
                c = Circle(5)
                print(f"{c.area():.2f}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "78.54" in nb_runner.get_output(3)

    def test_exec_scope_isolation(self, nb_runner):
        """Variables created via exec() in a namespace dict."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                ns = {}
                exec("x = 42", ns)
                val = ns['x']
                print(val)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "42" in nb_runner.get_output(1)

    def test_builtin_shadowing(self, nb_runner):
        """Shadowing a builtin name and using it."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "3" in nb_runner.get_output(1)
        assert "[0, 1, 2, 3, 4]" in nb_runner.get_output(2)

    def test_walrus_operator(self, nb_runner):
        """Walrus operator (:=) scope behavior."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                filtered = [y for x in data if (y := x * 2) > 10]
                print(filtered)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(1)
        assert "12" in output
        assert "14" in output


# ============================================================
# Test Group 6: Complex Data Transformations
# ============================================================

class TestComplexDataTransformations:
    """Test complex data pipeline patterns."""

    def test_reduce_pattern(self, nb_runner):
        """functools.reduce across cells."""
        nb_runner.create_notebook([
            "from functools import reduce",
            "numbers = [1, 2, 3, 4, 5]",
            textwrap.dedent("""\
                total = reduce(lambda a, b: a + b, numbers)
                print(total)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "15" in nb_runner.get_output(3)

    def test_itertools_chain(self, nb_runner):
        """itertools combinations across cells."""
        nb_runner.create_notebook([
            "import itertools",
            textwrap.dedent("""\
                groups = [[1, 2], [3, 4], [5, 6]]
                flat = list(itertools.chain.from_iterable(groups))
                print(flat)
            """),
            textwrap.dedent("""\
                pairs = list(itertools.combinations(flat, 2))
                print(len(pairs))
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[1, 2, 3, 4, 5, 6]" in nb_runner.get_output(2)
        assert "15" in nb_runner.get_output(3)  # C(6,2) = 15

    def test_nested_dict_transformation(self, nb_runner):
        """Deep nested dict transformation pipeline."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                data = {
                    'users': [
                        {'name': 'Alice', 'scores': [90, 85, 92]},
                        {'name': 'Bob', 'scores': [78, 82, 88]},
                    ]
                }
            """),
            textwrap.dedent("""\
                averages = {
                    u['name']: sum(u['scores']) / len(u['scores'])
                    for u in data['users']
                }
                print(averages)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "Alice" in output
        assert "Bob" in output

    def test_pipeline_with_map_filter(self, nb_runner):
        """Map-filter pipeline across cells."""
        nb_runner.create_notebook([
            "raw = list(range(1, 21))",
            "squared = list(map(lambda x: x**2, raw))",
            "big = list(filter(lambda x: x > 100, squared))",
            "print(big)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "121" in output  # 11^2
        assert "400" in output  # 20^2
