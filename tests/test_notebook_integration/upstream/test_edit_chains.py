"""Long dependency chains edited at different points."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress]


# Complex dependency chains, decorator stacking, context managers,
# class inheritance MRO, and namespace/scope edge cases.
#
# Tests deep multi-cell dependency propagation, complex decorator interactions,
# context manager state tracking, MRO-based method resolution caching, and
# subtle namespace scoping issues.
@pytest.mark.integration
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


# Long cascade workflow interaction tests.
#
# Tests that exercise multi-cell workflows with cascading edits,
# partial reruns, and complex dependency chains.
@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestCascadingEdits:
    """Edit one cell in a chain, verify all downstream update."""

    def test_five_cell_chain_edit_root(self, nb_runner):
        """5-cell chain: a -> b -> c -> d -> e. Edit a."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a + 1",
                "c = b + 1",
                "d = c + 1",
                "e = d + 1\nprint(f'e = {e}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "e = 5" in nb_runner.get_output(5)

        nb_runner.set_cell_source(1, "a = 10")
        nb_runner.run_all()
        assert "e = 14" in nb_runner.get_output(5)

    def test_five_cell_chain_edit_middle(self, nb_runner):
        """5-cell chain, edit the middle cell."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a + 1",
                "c = b * 10",
                "d = c + 1",
                "e = d + 1\nprint(f'e = {e}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "e = 22" in nb_runner.get_output(5)

        nb_runner.set_cell_source(3, "c = b * 100")
        nb_runner.run_all()
        assert "e = 202" in nb_runner.get_output(5)

    def test_chain_edit_two_cells(self, nb_runner):
        """Edit two non-adjacent cells in a chain."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a * 2",
                "c = b + 10",
                "d = c * 3",
                "print(f'd = {d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d = 36" in nb_runner.get_output(5)

        # Edit a and c
        nb_runner.set_cell_source(1, "a = 5")
        nb_runner.set_cell_source(3, "c = b + 100")
        nb_runner.run_all()
        assert "d = 330" in nb_runner.get_output(5)

    def test_diamond_edit_one_branch(self, nb_runner):
        """Diamond dependency, edit one branch."""
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = a * 2",
                "c = a * 3",
                "d = b + c\nprint(f'd = {d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d = 50" in nb_runner.get_output(4)

        # Edit only the b branch
        nb_runner.set_cell_source(2, "b = a * 10")
        nb_runner.run_all()
        assert "d = 130" in nb_runner.get_output(4)


@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestPartialReruns:
    """Run only some cells after edits."""

    def test_edit_middle_run_from_middle(self, nb_runner):
        """Edit cell 2, run cells 2-3 only."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "y = x * 2",
                "z = y + 1\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 11" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "y = x * 10")
        nb_runner.run_cells([2, 3])
        assert "z = 51" in nb_runner.get_output(3)

    def test_edit_root_run_only_leaf(self, nb_runner):
        """Edit root cell but only run the leaf. Upstream should trigger."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "y = x * 2",
                "z = y + 1\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 11" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_cell(3)
        assert "z = 201" in nb_runner.get_output(3)

    def test_edit_leaf_only(self, nb_runner):
        """Edit only the leaf cell, run it."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "y = x * 2",
                "z = y + 1\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 11" in nb_runner.get_output(3)

        nb_runner.set_cell_source(3, "z = y + 100\nprint(f'z = {z}')")
        nb_runner.run_cell(3)
        assert "z = 110" in nb_runner.get_output(3)


@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestMultiRoundWorkflows:
    """Multiple rounds of edits and reruns."""

    def test_three_rounds_of_edits(self, nb_runner):
        """Three successive rounds of editing the same cell."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x + 1\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 2" in nb_runner.get_output(2)

        # Edit 1
        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.run_all()
        assert "y = 11" in nb_runner.get_output(2)

        # Edit 2
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_all()
        assert "y = 101" in nb_runner.get_output(2)

        # Edit 3
        nb_runner.set_cell_source(1, "x = 1000")
        nb_runner.run_all()
        assert "y = 1001" in nb_runner.get_output(2)

    def test_alternating_cell_edits(self, nb_runner):
        """Alternate editing two different cells."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = 2",
                "c = a + b\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 3" in nb_runner.get_output(3)

        # Edit a
        nb_runner.set_cell_source(1, "a = 10")
        nb_runner.run_all()
        assert "c = 12" in nb_runner.get_output(3)

        # Edit b
        nb_runner.set_cell_source(2, "b = 20")
        nb_runner.run_all()
        assert "c = 30" in nb_runner.get_output(3)

        # Edit a again
        nb_runner.set_cell_source(1, "a = 100")
        nb_runner.run_all()
        assert "c = 120" in nb_runner.get_output(3)

    def test_edit_with_intermediate_restart(self, nb_runner):
        """Edit, restart, edit again."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x * 2\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 2" in nb_runner.get_output(2)

        # Edit and run
        nb_runner.set_cell_source(1, "x = 5")
        nb_runner.run_all()
        assert "y = 10" in nb_runner.get_output(2)

        # Restart
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 10" in nb_runner.get_output(2)

        # Edit again
        nb_runner.set_cell_source(1, "x = 50")
        nb_runner.run_all()
        assert "y = 100" in nb_runner.get_output(2)

    def test_progressive_notebook_building(self, nb_runner):
        """Build a notebook progressively: run cells as they're added.
        This simulates typical notebook usage patterns."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3]",
                "total = sum(data)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(2)

        # Now edit cell 1 to add more data
        nb_runner.set_cell_source(1, "data = [1, 2, 3, 4, 5]")
        nb_runner.run_all()
        assert "total = 15" in nb_runner.get_output(2)

    def test_revert_all_changes(self, nb_runner):
        """Make edits, then revert everything back to original."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x + 1",
                "z = y + 1\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 3" in nb_runner.get_output(3)

        # Edit all cells
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.set_cell_source(2, "y = x * 2")
        nb_runner.set_cell_source(3, "z = y * 3\nprint(f'z = {z}')")
        nb_runner.run_all()
        assert "z = 600" in nb_runner.get_output(3)

        # Revert all
        nb_runner.set_cell_source(1, "x = 1")
        nb_runner.set_cell_source(2, "y = x + 1")
        nb_runner.set_cell_source(3, "z = y + 1\nprint(f'z = {z}')")
        nb_runner.run_all()
        assert "z = 3" in nb_runner.get_output(3)


# Cross-cell data dependency interaction tests.
#
# Tests that exercise complex cross-cell data flows, transitive
# dependencies, diamond dependencies, and dependency chain changes.
@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestTransitiveDependencies:
    """Long transitive dependency chains + edits."""

    def test_six_level_chain_edit_root(self, nb_runner):
        """6-level transitive chain, edit the root."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a * 2",
                "c = b * 2",
                "d = c * 2",
                "e = d * 2",
                "f = e * 2\nprint(f'f = {f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "f = 32" in nb_runner.get_output(6)

        nb_runner.set_cell_source(1, "a = 3")
        nb_runner.run_all()
        assert "f = 96" in nb_runner.get_output(6)

    def test_six_level_chain_edit_middle(self, nb_runner):
        """6-level chain, edit a middle node."""
        nb_runner.create_notebook(
            [
                "a = 2",
                "b = a + 1",
                "c = b + 1",
                "d = c + 1",
                "e = d + 1",
                "f = e + 1\nprint(f'f = {f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # a=2, b=3, c=4, d=5, e=6, f=7
        assert "f = 7" in nb_runner.get_output(6)

        nb_runner.set_cell_source(3, "c = b * 10")
        nb_runner.run_all()
        # a=2, b=3, c=30, d=31, e=32, f=33
        assert "f = 33" in nb_runner.get_output(6)


@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestDiamondDependencies:
    """Diamond-shaped dependency graphs + edits."""

    def test_diamond_edit_one_branch(self, nb_runner):
        """Diamond: edit only one branch."""
        nb_runner.create_notebook(
            [
                "base = 10",
                "branch_a = base + 1",
                "branch_b = base + 2",
                "result = branch_a * branch_b\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 11 * 12 = 132
        assert "result = 132" in nb_runner.get_output(4)

        nb_runner.set_cell_source(2, "branch_a = base * 10")
        nb_runner.run_all()
        # 100 * 12 = 1200
        assert "result = 1200" in nb_runner.get_output(4)

    def test_double_diamond(self, nb_runner):
        """Double diamond: two merge points."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "a = x + 1",
                "b = x + 2",
                "mid = a + b",
                "c = mid * 2",
                "d = mid * 3",
                "final = c + d\nprint(f'final = {final}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # a=6, b=7, mid=13, c=26, d=39, final=65
        assert "final = 65" in nb_runner.get_output(7)

        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.run_all()
        # a=11, b=12, mid=23, c=46, d=69, final=115
        assert "final = 115" in nb_runner.get_output(7)


@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestDependencyChainChanges:
    """Change which variables a cell depends on."""

    def test_switch_input_variable(self, nb_runner):
        """Switch which variable a cell reads."""
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = 20",
                "result = a * 3\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        # Switch from a to b
        nb_runner.set_cell_source(3, "result = b * 3\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 60" in nb_runner.get_output(3)

    def test_add_new_dependency(self, nb_runner):
        """Add a new dependency to an existing cell."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "y = 10",
                "result = x\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 5" in nb_runner.get_output(3)

        # Now depend on both x and y
        nb_runner.set_cell_source(3, "result = x + y\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(3)

    def test_remove_dependency(self, nb_runner):
        """Remove a dependency from a cell."""
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = 20",
                "result = a + b\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        # Remove dependency on b
        nb_runner.set_cell_source(3, "result = a * 5\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 50" in nb_runner.get_output(3)

        # Now editing b should NOT affect result
        nb_runner.set_cell_source(2, "b = 999")
        nb_runner.run_all()
        assert "result = 50" in nb_runner.get_output(3)


@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestCyclicLikePatterns:
    """Patterns that look cyclic but aren't (self-assignment chains)."""

    def test_self_assignment_chain(self, nb_runner):
        """x depends on previous x (sequential mutation pattern)."""
        nb_runner.create_notebook(
            [
                "x = [1]",
                "x = x + [2]  # extend step 1",
                "x = x + [3]  # extend step 2",
                "print(f'x = {x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = [1, 2, 3]" in nb_runner.get_output(4)

        nb_runner.set_cell_source(2, "x = x + [20]  # extend step 1 (modified)")
        nb_runner.run_all()
        assert "x = [1, 20, 3]" in nb_runner.get_output(4)

    def test_accumulating_string(self, nb_runner):
        """String accumulation pattern."""
        nb_runner.create_notebook(
            [
                "s = 'hello'",
                "s = s + ' world'  # add world",
                "s = s + '!'  # add exclamation",
                "print(f's = {s}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "s = hello world!" in nb_runner.get_output(4)

        nb_runner.set_cell_source(2, "s = s + ' python'  # add python")
        nb_runner.run_all()
        assert "s = hello python!" in nb_runner.get_output(4)


# Multi-cell dependency chain stress tests.
#
# Tests with longer dependency chains (5-8 cells) where edits
# at various points in the chain verify cache propagation.
@pytest.mark.upstream
@pytest.mark.timeout(60)
class TestLongChainEdits:
    """Long dependency chains with edits at different positions."""

    def test_six_cell_chain_edit_root(self, nb_runner):
        """6-cell chain, edit root."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a + 1",
                "c = b + 1",
                "d = c + 1",
                "e = d + 1",
                "f = e + 1\nprint(f'f = {f}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "f = 6" in nb_runner.get_output(6)

        # Edit root
        nb_runner.set_cell_source(1, "a = 10")
        nb_runner.run_all()
        assert "f = 15" in nb_runner.get_output(6)

    def test_six_cell_chain_edit_middle(self, nb_runner):
        """6-cell chain, edit cell 3 (middle)."""
        nb_runner.create_notebook(
            [
                "a = 5",
                "b = a * 2",
                "c = b + 1",
                "d = c * 3",
                "e = d - 5",
                "print(f'e = {e}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # b=10, c=11, d=33, e=28
        assert "e = 28" in nb_runner.get_output(6)

        # Edit cell 3
        nb_runner.set_cell_source(3, "c = b + 100")
        nb_runner.run_all()
        # b=10, c=110, d=330, e=325
        assert "e = 325" in nb_runner.get_output(6)

    def test_six_cell_chain_edit_near_end(self, nb_runner):
        """6-cell chain, edit second-to-last."""
        nb_runner.create_notebook(
            [
                "a = 2",
                "b = a ** 2",
                "c = b + 3",
                "d = c * 2",
                "e = d + 100",
                "print(f'e = {e}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # b=4, c=7, d=14, e=114
        assert "e = 114" in nb_runner.get_output(6)

        # Edit near end
        nb_runner.set_cell_source(5, "e = d * 100")
        nb_runner.run_all()
        # e = 14*100 = 1400
        assert "e = 1400" in nb_runner.get_output(6)


@pytest.mark.upstream
@pytest.mark.timeout(60)
class TestBranchingChainEdits:
    """Branching dependency chains with edits."""

    def test_diamond_dependency_edit_shared_root(self, nb_runner):
        """Diamond: root -> (left, right) -> merge."""
        nb_runner.create_notebook(
            [
                "root = 10",
                "left = root * 2",
                "right = root + 5",
                "merged = left + right\nprint(f'merged = {merged}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # left=20, right=15, merged=35
        assert "merged = 35" in nb_runner.get_output(4)

        # Edit root
        nb_runner.set_cell_source(1, "root = 100")
        nb_runner.run_all()
        # left=200, right=105, merged=305
        assert "merged = 305" in nb_runner.get_output(4)

    def test_diamond_edit_one_branch(self, nb_runner):
        """Diamond: edit one branch only."""
        nb_runner.create_notebook(
            [
                "root = 5",
                "left = root * 3",
                "right = root + 1",
                "merged = left + right\nprint(f'merged = {merged}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # left=15, right=6, merged=21
        assert "merged = 21" in nb_runner.get_output(4)

        # Edit only left branch
        nb_runner.set_cell_source(2, "left = root * 10")
        nb_runner.run_all()
        # left=50, right=6, merged=56
        assert "merged = 56" in nb_runner.get_output(4)

    def test_two_independent_chains_edit_one(self, nb_runner):
        """Two independent chains, edit one and verify other unchanged."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x * 2\nprint(f'y = {y}')",
                "a = 100",
                "b = a + 50\nprint(f'b = {b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)
        assert "b = 150" in nb_runner.get_output(4)

        # Edit only chain 1
        nb_runner.set_cell_source(1, "x = 50")
        nb_runner.run_all()
        assert "y = 100" in nb_runner.get_output(2)
        assert "b = 150" in nb_runner.get_output(4)


@pytest.mark.upstream
@pytest.mark.timeout(60)
class TestMultipleEditsInSequence:
    """Multiple sequential edits to the same chain."""

    def test_three_edits_to_root(self, nb_runner):
        """Edit root three times in sequence."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x * 10",
                "z = y + 5\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 15" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 2")
        nb_runner.run_all()
        assert "z = 25" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 5")
        nb_runner.run_all()
        assert "z = 55" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.run_all()
        assert "z = 105" in nb_runner.get_output(3)

    def test_edit_different_cells_alternating(self, nb_runner):
        """Alternate between editing cell 1 and cell 2."""
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = a + 5",
                "c = b * 2\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # c = (10+5)*2 = 30
        assert "c = 30" in nb_runner.get_output(3)

        # Edit cell 1
        nb_runner.set_cell_source(1, "a = 20")
        nb_runner.run_all()
        # c = (20+5)*2 = 50
        assert "c = 50" in nb_runner.get_output(3)

        # Edit cell 2
        nb_runner.set_cell_source(2, "b = a + 100")
        nb_runner.run_all()
        # c = (20+100)*2 = 240
        assert "c = 240" in nb_runner.get_output(3)

        # Edit cell 1 again
        nb_runner.set_cell_source(1, "a = 0")
        nb_runner.run_all()
        # c = (0+100)*2 = 200
        assert "c = 200" in nb_runner.get_output(3)


# Deep dependency chain interaction tests.
#
# Tests with long chains of cells (5+ cells) where a change at any
# point in the chain must properly propagate through all downstream cells.
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestDeepChainPropagation:
    """Deep dependency chains with edits at various points."""

    def test_edit_head_of_chain(self, nb_runner):
        """5-cell chain, edit the first cell."""
        nb_runner.create_notebook(
            [
                "a = 1  # head",
                "b = a + 1  # step 2",
                "c = b * 2  # step 3",
                "d = c + 10  # step 4",
                "result = d * 3\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # a=1, b=2, c=4, d=14, result=42
        assert "result = 42" in nb_runner.get_output(5)

        # Edit head
        nb_runner.set_cell_source(1, "a = 10  # head changed")
        nb_runner.run_all()
        # a=10, b=11, c=22, d=32, result=96
        assert "result = 96" in nb_runner.get_output(5)

    def test_edit_middle_of_chain(self, nb_runner):
        """5-cell chain, edit the middle cell."""
        nb_runner.create_notebook(
            [
                "x = 2  # start",
                "y = x * 3  # middle1",
                "z = y + 1  # middle2",
                "w = z ** 2  # step4",
                "out = w - 1\nprint(f'out = {out}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # x=2, y=6, z=7, w=49, out=48
        assert "out = 48" in nb_runner.get_output(5)

        # Edit middle
        nb_runner.set_cell_source(3, "z = y + 100  # middle2 boosted")
        nb_runner.run_all()
        # x=2, y=6, z=106, w=11236, out=11235
        assert "out = 11235" in nb_runner.get_output(5)

    def test_edit_tail_of_chain(self, nb_runner):
        """5-cell chain, edit the last cell."""
        nb_runner.create_notebook(
            [
                "p = 5  # start",
                "q = p * 2  # step2",
                "r = q + 3  # step3",
                "s = r - 1  # step4",
                "final = s\nprint(f'final = {final}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # p=5, q=10, r=13, s=12, final=12
        assert "final = 12" in nb_runner.get_output(5)

        # Edit output cell
        nb_runner.set_cell_source(5, "final = s * 100\nprint(f'final = {final}')")
        nb_runner.run_all()
        assert "final = 1200" in nb_runner.get_output(5)

    def test_edit_two_points_in_chain(self, nb_runner):
        """Edit two non-adjacent cells in a chain simultaneously."""
        nb_runner.create_notebook(
            [
                "a = 1  # chain start",
                "b = a + 10  # chain step2",
                "c = b * 2  # chain step3",
                "d = c + 5  # chain step4",
                "e = d * 3\nprint(f'e = {e}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # a=1, b=11, c=22, d=27, e=81
        assert "e = 81" in nb_runner.get_output(5)

        # Edit cells 1 and 4
        nb_runner.set_cell_source(1, "a = 100  # chain start big")
        nb_runner.set_cell_source(4, "d = c + 1000  # chain step4 big")
        nb_runner.run_all()
        # a=100, b=110, c=220, d=1220, e=3660
        assert "e = 3660" in nb_runner.get_output(5)


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestChainWithFunctions:
    """Deep chains involving function definitions."""

    def test_function_chain_edit(self, nb_runner):
        """Chain where each cell defines a function using the previous."""
        nb_runner.create_notebook(
            [
                "def step1(x):\n    return x + 1",
                "def step2(x):\n    return step1(x) * 2",
                "def step3(x):\n    return step2(x) + 10",
                "result = step3(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # step1(5)=6, step2(5)=12, step3(5)=22
        assert "result = 22" in nb_runner.get_output(4)

        # Edit step1
        nb_runner.set_cell_source(1, "def step1(x):\n    return x + 100")
        nb_runner.run_all()
        # step1(5)=105, step2(5)=210, step3(5)=220
        assert "result = 220" in nb_runner.get_output(4)

    def test_lambda_chain_edit(self, nb_runner):
        """Chain of lambda functions with edits."""
        nb_runner.create_notebook(
            [
                "fn1 = lambda x: x * 2",
                "fn2 = lambda x: fn1(x) + 3",
                "fn3 = lambda x: fn2(x) ** 2",
                "out = fn3(4)\nprint(f'out = {out}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # fn1(4)=8, fn2(4)=11, fn3(4)=121
        assert "out = 121" in nb_runner.get_output(4)

        # Edit fn1
        nb_runner.set_cell_source(1, "fn1 = lambda x: x * 10")
        nb_runner.run_all()
        # fn1(4)=40, fn2(4)=43, fn3(4)=1849
        assert "out = 1849" in nb_runner.get_output(4)


# Multiple cell chain edit interaction tests.
#
# Tests editing a cell in the middle of a multi-cell pipeline to
# verify both upstream restoration and downstream propagation work.
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestMultiCellChainEdits:
    """Editing cells in multi-step pipelines."""

    def test_edit_middle_of_3_cell_chain(self, nb_runner):
        """Edit the middle cell in a 3-cell chain."""
        nb_runner.create_notebook(
            [
                "raw = [1, 2, 3, 4, 5]",
                "processed = [x ** 2 for x in raw]",
                "total = sum(processed)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 55" in nb_runner.get_output(3)

        # Edit middle cell to cube instead of square
        nb_runner.set_cell_source(2, "processed = [x ** 3 for x in raw]")
        nb_runner.run_all()
        assert "total = 225" in nb_runner.get_output(3)

    def test_edit_first_of_4_cell_chain(self, nb_runner):
        """Edit the first cell in a 4-cell chain — all downstream recompute."""
        nb_runner.create_notebook(
            [
                "a = 2",
                "b = a * 3",
                "c = b + 1",
                "print(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 7" in nb_runner.get_output(4)

        # Edit first cell
        nb_runner.set_cell_source(1, "a = 10")
        nb_runner.run_all()
        assert "c = 31" in nb_runner.get_output(4)

    def test_edit_adds_intermediate_step(self, nb_runner):
        """Edit middle cell to add an extra processing step in same cell."""
        nb_runner.create_notebook(
            [
                "data = [3, 1, 4, 1, 5, 9]",
                "result = sorted(data)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [1, 1, 3, 4, 5, 9]" in nb_runner.get_output(2)

        # Add dedup step after sort
        nb_runner.set_cell_source(2, "result = sorted(set(data))\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = [1, 3, 4, 5, 9]" in nb_runner.get_output(2)

    def test_edit_preserves_unrelated_cells(self, nb_runner):
        """Editing one branch shouldn't affect an unrelated branch."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = 20",
                "sum_xy = x + y\nprint(f'sum = {sum_xy}')",
                "prod_xy = x * y\nprint(f'prod = {prod_xy}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sum = 30" in nb_runner.get_output(3)
        assert "prod = 200" in nb_runner.get_output(4)

        # Edit y — both downstream cells should update
        nb_runner.set_cell_source(2, "y = 30")
        nb_runner.run_all()
        assert "sum = 40" in nb_runner.get_output(3)
        assert "prod = 300" in nb_runner.get_output(4)


# Long chain dependency propagation (5+ cells).
#
# Tests editing early cell in long chain, verifying final cell updates.
@pytest.mark.timeout(90)
class TestLongChainPropagation:
    """Long dependency chain edit patterns."""

    def test_six_cell_chain(self, nb_runner):
        """Edit cell 1 in 6-cell chain, cell 6 reflects."""
        nb_runner.create_notebook(
            [
                "base = 10",
                "step1 = base + 5",
                "step2 = step1 * 2",
                "step3 = step2 - 3",
                "step4 = step3 // 4",
                "result = step4\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 10+5=15, 15*2=30, 30-3=27, 27//4=6
        assert "result = 6" in nb_runner.get_output(6)

        nb_runner.set_cell_source(1, "base = 100")
        nb_runner.run_all()
        # 100+5=105, 105*2=210, 210-3=207, 207//4=51
        assert "result = 51" in nb_runner.get_output(6)

    def test_edit_middle_of_long_chain(self, nb_runner):
        """Edit middle cell (3 of 5), tail updates."""
        nb_runner.create_notebook(
            [
                "x = 2",
                "y = x * 3",
                "z = y + 10",
                "w = z ** 2",
                "final = w - 1\nprint(f'final = {final}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # x=2, y=6, z=16, w=256, final=255
        assert "final = 255" in nb_runner.get_output(5)

        nb_runner.set_cell_source(3, "z = y + 100")
        nb_runner.run_all()
        # x=2, y=6, z=106, w=11236, final=11235
        assert "final = 11235" in nb_runner.get_output(5)

    def test_branching_chain(self, nb_runner):
        """Two branches merge in final cell."""
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = 20",
                "left = a * 2",
                "right = b * 3",
                "combined = left + right\nprint(f'combined = {combined}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 10*2 + 20*3 = 20 + 60 = 80
        assert "combined = 80" in nb_runner.get_output(5)

        # Edit one branch source
        nb_runner.set_cell_source(1, "a = 50")
        nb_runner.run_all()
        # 50*2 + 20*3 = 100 + 60 = 160
        assert "combined = 160" in nb_runner.get_output(5)
