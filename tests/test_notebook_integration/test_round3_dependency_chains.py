"""
Complex dependency chains, decorator stacking, context managers,
class inheritance MRO, and namespace/scope edge cases.

Tests deep multi-cell dependency propagation, complex decorator interactions,
context manager state tracking, MRO-based method resolution caching, and
subtle namespace scoping issues.
"""

import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress]


# ============================================================
# Test Group 1: Deep Dependency Chains
# ============================================================


class TestDeepDependencyChains:
    """Test multi-level variable dependency propagation across many cells."""

    def test_linear_chain_10_cells(self, nb_runner):
        """10-cell linear dependency chain: each cell uses previous cell's output."""
        cells = [f"v{i} = v{i - 1} + 1" if i > 0 else "v0 = 1" for i in range(10)]
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
        nb_runner.create_notebook(["a = 10", "b = a * 2", "c = b + 5", "d = c ** 2", "print(d)"])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "625" in nb_runner.get_output(5)  # (10*2+5)^2 = 625

        # Change root
        nb_runner.set_cell_source(1, "a = 20")
        nb_runner.run_all()
        assert "2025" in nb_runner.get_output(5)  # (20*2+5)^2 = 2025

    def test_diamond_dependency(self, nb_runner):
        """Diamond pattern: A -> B, A -> C, B+C -> D."""
        nb_runner.create_notebook(["a = 5", "b = a * 2", "c = a + 3", "d = b + c", "print(d)"])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "18" in nb_runner.get_output(5)  # 5*2 + 5+3 = 18

        # Change root -> both branches update
        nb_runner.set_cell_source(1, "a = 10")
        nb_runner.run_all()
        assert "33" in nb_runner.get_output(5)  # 10*2 + 10+3 = 33

    def test_fan_out_dependency(self, nb_runner):
        """One variable feeds many downstream cells."""
        nb_runner.create_notebook(
            ["base = 100", "x1 = base + 1", "x2 = base + 2", "x3 = base + 3", "total = x1 + x2 + x3", "print(total)"]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "306" in nb_runner.get_output(6)  # 101+102+103

        nb_runner.set_cell_source(1, "base = 200")
        nb_runner.run_all()
        assert "606" in nb_runner.get_output(6)  # 201+202+203

    def test_fan_in_dependency(self, nb_runner):
        """Multiple independent sources converge into one cell."""
        nb_runner.create_notebook(["x = 10", "y = 20", "z = 30", "total = x + y + z", "print(total)"])
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


# ============================================================
# Test Group 3: Context Managers
# ============================================================


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
        data_file.write_text("hello context")
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


# ============================================================
# Test Group 4: Class Inheritance & MRO
# ============================================================


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


# ============================================================
# Test Group 5: Namespace & Scope Edge Cases
# ============================================================


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


# ============================================================
# Test Group 6: Complex Data Transformations
# ============================================================
