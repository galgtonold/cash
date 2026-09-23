"""Editing function definitions and the calls downstream of them."""

import pytest

pytestmark = [pytest.mark.stress]


# Function & class redefinition interaction tests.
#
# Tests that exercise function/class redefinition across cells,
# combined with cell edits, reruns, and kernel restarts.
@pytest.mark.mutations
@pytest.mark.timeout(30)
class TestFunctionRedefinition:
    """Redefine a function in one cell, observe downstream effects."""

    def test_redefine_function_downstream_updates(self, nb_runner):
        """Change function body, downstream should recompute."""
        nb_runner.create_notebook(
            [
                "def f(x):\n    return x * 2",
                "result = f(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "def f(x):\n    return x * 3")
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(2)

    def test_redefine_function_twice(self, nb_runner):
        """Redefine a function two times sequentially."""
        nb_runner.create_notebook(
            [
                "def compute(a, b):\n    return a + b",
                "val = compute(3, 4)\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 7" in nb_runner.get_output(2)

        # First edit: multiply
        nb_runner.set_cell_source(1, "def compute(a, b):\n    return a * b")
        nb_runner.run_all()
        assert "val = 12" in nb_runner.get_output(2)

        # Second edit: subtract
        nb_runner.set_cell_source(1, "def compute(a, b):\n    return a - b")
        nb_runner.run_all()
        assert "val = -1" in nb_runner.get_output(2)

    def test_redefine_function_revert(self, nb_runner):
        """Change function, then revert to original. Cache should still work."""
        nb_runner.create_notebook(
            [
                "def double(x):\n    return x * 2",
                "r = double(7)\nprint(f'r = {r}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r = 14" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "def double(x):\n    return x * 3")
        nb_runner.run_all()
        assert "r = 21" in nb_runner.get_output(2)

        # Revert
        nb_runner.set_cell_source(1, "def double(x):\n    return x * 2")
        nb_runner.run_all()
        assert "r = 14" in nb_runner.get_output(2)

    def test_function_used_in_multiple_cells(self, nb_runner):
        """Function redefined, used in multiple downstream cells."""
        nb_runner.create_notebook(
            [
                "def inc(x):\n    return x + 1",
                "a = inc(10)\nprint(f'a = {a}')",
                "b = inc(20)\nprint(f'b = {b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 11" in nb_runner.get_output(2)
        assert "b = 21" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "def inc(x):\n    return x + 10")
        nb_runner.run_all()
        assert "a = 20" in nb_runner.get_output(2)
        assert "b = 30" in nb_runner.get_output(3)

    def test_function_redefined_after_restart(self, nb_runner):
        """Restart kernel, redefine function, downstream should recompute."""
        nb_runner.create_notebook(
            [
                "def square(x):\n    return x ** 2",
                "val = square(4)\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 16" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.set_cell_source(1, "def square(x):\n    return x ** 3")
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 64" in nb_runner.get_output(2)


@pytest.mark.mutations
@pytest.mark.timeout(30)
class TestClassRedefinition:
    """Redefine a class, observe downstream effects."""

    def test_redefine_class_method(self, nb_runner):
        """Change a class method, downstream should reflect."""
        nb_runner.create_notebook(
            [
                "class Calculator:\n    def add(self, a, b):\n        return a + b",
                "c = Calculator()\nresult = c.add(3, 4)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 7" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "class Calculator:\n    def add(self, a, b):\n        return a + b + 100",
        )
        nb_runner.run_all()
        assert "result = 107" in nb_runner.get_output(2)

    def test_add_method_to_class(self, nb_runner):
        """Add a new method to an existing class."""
        nb_runner.create_notebook(
            [
                "class MyClass:\n    def foo(self):\n        return 1",
                "obj = MyClass()\nprint(f'foo = {obj.foo()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "foo = 1" in nb_runner.get_output(2)

        # Add bar method and use it
        nb_runner.set_cell_source(
            1,
            "class MyClass:\n    def foo(self):\n        return 1\n    def bar(self):\n        return 2",
        )
        nb_runner.set_cell_source(
            2,
            "obj = MyClass()\nprint(f'foo = {obj.foo()}, bar = {obj.bar()}')",
        )
        nb_runner.run_all()
        assert "foo = 1, bar = 2" in nb_runner.get_output(2)


@pytest.mark.mutations
@pytest.mark.timeout(30)
class TestFunctionCallingFunction:
    """Function that calls another function — redefinition chains."""

    def test_inner_function_redefined(self, nb_runner):
        """Redefine an inner function used by an outer function."""
        nb_runner.create_notebook(
            [
                "def helper(x):\n    return x + 1",
                "def main(x):\n    return helper(x) * 2",
                "result = main(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 12" in nb_runner.get_output(3)

        # Redefine helper
        nb_runner.set_cell_source(1, "def helper(x):\n    return x + 10")
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

    def test_outer_function_redefined(self, nb_runner):
        """Redefine the outer function while helper stays the same."""
        nb_runner.create_notebook(
            [
                "def helper(x):\n    return x + 1",
                "def main(x):\n    return helper(x) * 2",
                "result = main(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 12" in nb_runner.get_output(3)

        # Redefine main
        nb_runner.set_cell_source(2, "def main(x):\n    return helper(x) * 3")
        nb_runner.run_all()
        assert "result = 18" in nb_runner.get_output(3)

    def test_both_functions_redefined(self, nb_runner):
        """Redefine both inner and outer functions simultaneously."""
        nb_runner.create_notebook(
            [
                "def helper(x):\n    return x + 1",
                "def main(x):\n    return helper(x) * 2",
                "result = main(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 12" in nb_runner.get_output(3)

        # Redefine both
        nb_runner.set_cell_source(1, "def helper(x):\n    return x * 10")
        nb_runner.set_cell_source(2, "def main(x):\n    return helper(x) + 1")
        nb_runner.run_all()
        assert "result = 51" in nb_runner.get_output(3)

    def test_recursive_function_redefined(self, nb_runner):
        """Redefine a recursive function."""
        nb_runner.create_notebook(
            [
                "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)",
                "val = fib(6)\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 8" in nb_runner.get_output(2)

        # Change to factorial
        nb_runner.set_cell_source(
            1,
            "def fib(n):\n    if n <= 1:\n        return 1\n    return n * fib(n-1)",
        )
        nb_runner.run_all()
        assert "val = 720" in nb_runner.get_output(2)

    def test_lambda_redefined(self, nb_runner):
        """Redefine a lambda used downstream."""
        nb_runner.create_notebook(
            [
                "transform = lambda x: x * 2",
                "result = transform(7)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 14" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "transform = lambda x: x ** 2")
        nb_runner.run_all()
        assert "result = 49" in nb_runner.get_output(2)


# Nested function and closure interaction tests.
#
# Tests where users define functions in one cell and call them in another,
# then modify the function definition and verify downstream cells update.
@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestFunctionDefinitionEdits:
    """Edit function definitions and verify callers update."""

    def test_edit_function_body(self, nb_runner):
        """Change function body, verify callers get new result."""
        nb_runner.create_notebook(
            [
                "def compute(x):\n    return x * 2",
                "result = compute(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        # Edit function body
        nb_runner.set_cell_source(1, "def compute(x):\n    return x ** 2")
        nb_runner.run_all()
        assert "result = 25" in nb_runner.get_output(2)

    def test_add_parameter_to_function(self, nb_runner):
        """Add parameter to function definition."""
        nb_runner.create_notebook(
            [
                "def greet(name):\n    return f'Hello {name}'",
                "msg = greet('World')\nprint(msg)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Hello World" in nb_runner.get_output(2)

        # Add second parameter
        nb_runner.set_cell_source(
            1,
            "def greet(name, greeting='Hi'):\n    return f'{greeting} {name}'",
        )
        nb_runner.set_cell_source(2, "msg = greet('World', 'Hey')\nprint(msg)")
        nb_runner.run_all()
        assert "Hey World" in nb_runner.get_output(2)

    def test_chain_of_functions_edit_middle(self, nb_runner):
        """Chain: f -> g -> h, edit g."""
        nb_runner.create_notebook(
            [
                "def f(x):\n    return x + 1",
                "def g(x):\n    return f(x) * 2",
                "def h(x):\n    return g(x) + 10",
                "result = h(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # h(5) = g(5) + 10 = f(5)*2 + 10 = 6*2+10 = 22
        assert "result = 22" in nb_runner.get_output(4)

        # Edit g
        nb_runner.set_cell_source(2, "def g(x):\n    return f(x) * 3")
        nb_runner.run_all()
        # h(5) = g(5) + 10 = f(5)*3 + 10 = 6*3+10 = 28
        assert "result = 28" in nb_runner.get_output(4)


@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestClosureEdits:
    """Edit closures and verify behavior updates."""

    def test_edit_closure_variable(self, nb_runner):
        """Closure captures a variable, edit the variable."""
        nb_runner.create_notebook(
            [
                "factor = 3",
                "def scale(x):\n    return x * factor",
                "result = scale(10)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        # Edit the captured variable
        nb_runner.set_cell_source(1, "factor = 7")
        nb_runner.run_all()
        assert "result = 70" in nb_runner.get_output(3)

    def test_function_with_default_arg_edit(self, nb_runner):
        """Function with default arg, edit the default."""
        nb_runner.create_notebook(
            [
                "default_power = 2",
                "def raise_to(x, p=None):\n    if p is None:\n        p = default_power\n    return x ** p",
                "result = raise_to(3)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 9" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "default_power = 3")
        nb_runner.run_all()
        assert "result = 27" in nb_runner.get_output(3)


@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestLambdaEdits:
    """Edit lambda expressions."""

    def test_edit_lambda(self, nb_runner):
        """Change lambda, verify new behavior."""
        nb_runner.create_notebook(
            [
                "transform = lambda x: x + 10",
                "vals = [transform(i) for i in range(5)]",
                "total = sum(vals)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # [10, 11, 12, 13, 14] -> 60
        assert "total = 60" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "transform = lambda x: x * 10")
        nb_runner.run_all()
        # [0, 10, 20, 30, 40] -> 100
        assert "total = 100" in nb_runner.get_output(3)

    def test_lambda_to_function(self, nb_runner):
        """Replace lambda with full function def."""
        nb_runner.create_notebook(
            [
                "double = lambda x: x * 2",
                "result = double(15)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def double(x):\n    return x * 2 + 1  # with offset",
        )
        nb_runner.run_all()
        assert "result = 31" in nb_runner.get_output(2)


# Map/filter/reduce pattern interaction tests.
#
# Tests editing functional programming patterns like map, filter,
# reduce, and their combinations.
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestMapEdits:
    """Editing map operations."""

    def test_edit_map_function(self, nb_runner):
        """Edit the function passed to map."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5]  # map source data",
                "result = list(map(lambda x: x * 2, data))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 4, 6, 8, 10]" in nb_runner.get_output(2)

        # Change map function
        nb_runner.set_cell_source(
            2,
            "result = list(map(lambda x: x ** 2, data))\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = [1, 4, 9, 16, 25]" in nb_runner.get_output(2)

    def test_edit_map_source(self, nb_runner):
        """Edit the source data for map."""
        nb_runner.create_notebook(
            [
                "nums = [10, 20, 30]  # map source",
                "result = list(map(str, nums))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = ['10', '20', '30']" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "nums = [100, 200]  # map source smaller")
        nb_runner.run_all()
        assert "result = ['100', '200']" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestFilterEdits:
    """Editing filter operations."""

    def test_edit_filter_predicate(self, nb_runner):
        """Edit the predicate passed to filter."""
        nb_runner.create_notebook(
            [
                "items = list(range(10))  # filter source",
                "result = list(filter(lambda x: x > 5, items))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [6, 7, 8, 9]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            2,
            "result = list(filter(lambda x: x % 3 == 0, items))\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = [0, 3, 6, 9]" in nb_runner.get_output(2)

    def test_chain_map_filter(self, nb_runner):
        """Chain map and filter, edit one operation."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5, 6]  # chain source",
                "doubled = map(lambda x: x * 2, data)",
                "result = list(filter(lambda x: x > 6, doubled))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [8, 10, 12]" in nb_runner.get_output(3)

        # Change map to triple
        nb_runner.set_cell_source(2, "doubled = map(lambda x: x * 3, data)")
        nb_runner.run_all()
        assert "result = [9, 12, 15, 18]" in nb_runner.get_output(3)


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestReduceEdits:
    """Editing reduce operations."""

    def test_edit_reduce_function(self, nb_runner):
        """Edit the function passed to reduce."""
        nb_runner.create_notebook(
            [
                "from functools import reduce",
                "nums = [1, 2, 3, 4]  # reduce source",
                "result = reduce(lambda a, b: a + b, nums)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(3)

        # Change to multiply
        nb_runner.set_cell_source(
            3,
            "result = reduce(lambda a, b: a * b, nums)\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = 24" in nb_runner.get_output(3)

    def test_reduce_with_initial(self, nb_runner):
        """Edit reduce initial value."""
        nb_runner.create_notebook(
            [
                "from functools import reduce",
                "vals = [1, 2, 3]  # reduce initial source",
                "result = reduce(lambda a, b: a + b, vals, 0)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 6" in nb_runner.get_output(3)

        # Change initial value
        nb_runner.set_cell_source(
            3,
            "result = reduce(lambda a, b: a + b, vals, 100)\nprint(f'result = {result}')",
        )
        nb_runner.run_all()
        assert "result = 106" in nb_runner.get_output(3)


# Global constant and config pattern interaction tests.
#
# Tests where shared constants/config are defined in an early cell
# and used by many downstream cells. Edit the config and verify
# all downstream cells update correctly.
@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestConfigDrivenWorkflow:
    """Config dict driving multiple downstream cells."""

    def test_edit_config_value(self, nb_runner):
        """Edit config value, verify all downstream updates."""
        nb_runner.create_notebook(
            [
                "config = {'scale': 2, 'offset': 10}",
                "a = 5 * config['scale']\nprint(f'a = {a}')",
                "b = 100 + config['offset']\nprint(f'b = {b}')",
                "c = a + b\nprint(f'c = {c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 10" in nb_runner.get_output(2)
        assert "b = 110" in nb_runner.get_output(3)
        assert "c = 120" in nb_runner.get_output(4)

        # Edit config
        nb_runner.set_cell_source(1, "config = {'scale': 10, 'offset': 0}")
        nb_runner.run_all()
        assert "a = 50" in nb_runner.get_output(2)
        assert "b = 100" in nb_runner.get_output(3)
        assert "c = 150" in nb_runner.get_output(4)

    def test_add_config_key(self, nb_runner):
        """Add a new key to config, use it downstream."""
        nb_runner.create_notebook(
            [
                "params = {'lr': 0.01}",
                "effective_lr = params['lr'] * 10\nprint(f'lr = {effective_lr}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lr = 0.1" in nb_runner.get_output(2)

        # Add key and use it
        nb_runner.set_cell_source(1, "params = {'lr': 0.01, 'decay': 0.5}")
        nb_runner.set_cell_source(
            2,
            "effective_lr = params['lr'] * params['decay']\nprint(f'lr = {effective_lr}')",
        )
        nb_runner.run_all()
        assert "lr = 0.005" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestConstantEdits:
    """Edit shared constants used by many cells."""

    def test_edit_constant_three_consumers(self, nb_runner):
        """One constant used by three cells."""
        nb_runner.create_notebook(
            [
                "PI = 3.14",
                "circumference = 2 * PI * 5\nprint(f'circ = {circumference}')",
                "area = PI * 5 ** 2\nprint(f'area = {area}')",
                "volume = (4/3) * PI * 5 ** 3\nprint(f'vol = {volume}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "circ = " in nb_runner.get_output(2)
        assert "area = " in nb_runner.get_output(3)
        assert "vol = " in nb_runner.get_output(4)

        # Use more precise PI
        nb_runner.set_cell_source(1, "PI = 3.14159")
        nb_runner.run_all()
        # All downstream should update
        out2 = nb_runner.get_output(2)
        out3 = nb_runner.get_output(3)
        out4 = nb_runner.get_output(4)
        assert "3.14159" in out2 or "31.4159" in out2
        assert "area = " in out3
        assert "vol = " in out4

    def test_edit_constant_with_restart(self, nb_runner):
        """Edit constant, restart, verify restored correctly."""
        nb_runner.create_notebook(
            [
                "MULTIPLIER = 5",
                "result = MULTIPLIER * 20\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(2)

        # Edit and run
        nb_runner.set_cell_source(1, "MULTIPLIER = 50")
        nb_runner.run_all()
        assert "result = 1000" in nb_runner.get_output(2)

        # Restart - should restore
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 1000" in nb_runner.get_output(2)


# Decorator pattern interaction tests.
#
# Tests where decorators are defined in one cell, applied to functions
# in another, and edits to either the decorator or the function
# verify correct cache behavior.
class TestDecoratorEdits:
    """Decorator definition and application edits."""

    @pytest.mark.upstream
    @pytest.mark.timeout(90)
    def test_edit_decorator_logic(self, nb_runner):
        """Edit decorator logic, verify function behavior changes."""
        nb_runner.create_notebook(
            [
                "def double_result(fn):\n    def wrapper(*args):\n        return fn(*args) * 2\n    return wrapper",
                "@double_result\ndef add(a, b):\n    return a + b",
                "result = add(3, 4)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 14" in nb_runner.get_output(3)

        # Change decorator to triple
        nb_runner.set_cell_source(
            1,
            "def double_result(fn):\n    def wrapper(*args):\n        return fn(*args) * 3\n    return wrapper",
        )
        nb_runner.run_all()
        assert "result = 21" in nb_runner.get_output(3)

    @pytest.mark.upstream
    @pytest.mark.timeout(90)
    def test_edit_decorated_function(self, nb_runner):
        """Edit the decorated function itself."""
        nb_runner.create_notebook(
            [
                "def negate(fn):\n    def wrapper(*args):\n        return -fn(*args)\n    return wrapper",
                "@negate\ndef compute(x):\n    return x * 2",
                "result = compute(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = -10" in nb_runner.get_output(3)

        # Edit the base function
        nb_runner.set_cell_source(2, "@negate\ndef compute(x):\n    return x ** 2")
        nb_runner.run_all()
        assert "result = -25" in nb_runner.get_output(3)

    @pytest.mark.upstream
    @pytest.mark.timeout(90)
    def test_remove_decorator(self, nb_runner):
        """Remove decorator from function."""
        nb_runner.create_notebook(
            [
                "def add_ten(fn):\n    def wrapper(*args):\n        return fn(*args) + 10\n    return wrapper",
                "@add_ten\ndef square(x):\n    return x * x",
                "result = square(3)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 19" in nb_runner.get_output(3)

        # Remove decorator
        nb_runner.set_cell_source(2, "def square(x):\n    return x * x")
        nb_runner.run_all()
        assert "result = 9" in nb_runner.get_output(3)

    # Decorator patterns with edits.
    #
    # Tests custom decorators being edited and downstream function behavior.
    @pytest.mark.timeout(90)
    def test_logging_decorator_edit(self, nb_runner):
        """Edit decorator wrapper, decorated function behavior changes."""
        nb_runner.create_notebook(
            [
                "def logged(fn):\n    def wrapper(*args):\n        result = fn(*args)\n        return f'[LOG] {result}'\n    return wrapper",
                "@logged\ndef greet(name):\n    return f'Hello {name}'",
                "msg = greet('Alice')\nprint(f'msg = {msg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg = [LOG] Hello Alice" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            1,
            "def logged(fn):\n    def wrapper(*args):\n        result = fn(*args)\n        return f'[DEBUG] {result}'\n    return wrapper",
        )
        nb_runner.run_all()
        assert "msg = [DEBUG] Hello Alice" in nb_runner.get_output(3)

    @pytest.mark.timeout(90)
    def test_retry_decorator_edit(self, nb_runner):
        """Edit retry count in decorator."""
        nb_runner.create_notebook(
            [
                "def with_prefix(prefix):\n    def decorator(fn):\n        def wrapper(*args):\n            return f'{prefix}: {fn(*args)}'\n        return wrapper\n    return decorator",
                "@with_prefix('INFO')\ndef status(code):\n    return f'code={code}'",
                "result = status(200)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = INFO: code=200" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            1,
            "def with_prefix(prefix):\n    def decorator(fn):\n        def wrapper(*args):\n            return f'[{prefix}] {fn(*args)}'\n        return wrapper\n    return decorator",
        )
        nb_runner.run_all()
        assert "result = [INFO] code=200" in nb_runner.get_output(3)

    @pytest.mark.timeout(90)
    def test_decorator_with_edited_function(self, nb_runner):
        """Edit decorated function body."""
        nb_runner.create_notebook(
            [
                "def double_result(fn):\n    def wrapper(*args):\n        return fn(*args) * 2\n    return wrapper",
                "@double_result\ndef compute(x):\n    return x + 1",
                "val = compute(5)\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 12" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2,
            "@double_result\ndef compute(x):\n    return x * 3",
        )
        nb_runner.run_all()
        assert "val = 30" in nb_runner.get_output(3)


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestDecoratorWithArgs:
    """Decorators with arguments."""

    def test_edit_decorator_argument(self, nb_runner):
        """Edit the argument to a decorator factory."""
        nb_runner.create_notebook(
            [
                "def multiply_by(n):\n    def decorator(fn):\n        def wrapper(*args):\n            return fn(*args) * n\n        return wrapper\n    return decorator",
                "@multiply_by(2)\ndef add(a, b):\n    return a + b",
                "result = add(3, 4)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 14" in nb_runner.get_output(3)

        # Change multiplier
        nb_runner.set_cell_source(2, "@multiply_by(5)\ndef add(a, b):\n    return a + b")
        nb_runner.run_all()
        assert "result = 35" in nb_runner.get_output(3)


# Inter-cell function call interaction tests.
#
# Tests with functions defined in one cell that call functions
# from another cell, with edits at various levels.
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestInterCellCalls:
    """Functions calling functions from other cells."""

    def test_edit_calling_function(self, nb_runner):
        """Edit a function that calls another function."""
        nb_runner.create_notebook(
            [
                "def square(x):\n    return x ** 2",
                "def process(x):\n    return square(x) + 1",
                "result = process(4)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # square(4)=16, process(4)=17
        assert "result = 17" in nb_runner.get_output(3)

        # Edit process
        nb_runner.set_cell_source(2, "def process(x):\n    return square(x) * 10")
        nb_runner.run_all()
        assert "result = 160" in nb_runner.get_output(3)

    def test_three_level_call_chain(self, nb_runner):
        """Three functions calling each other across cells."""
        nb_runner.create_notebook(
            [
                "def level1(x):\n    return x + 1",
                "def level2(x):\n    return level1(x) * 2",
                "def level3(x):\n    return level2(x) + 10",
                "result = level3(3)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # level1(3)=4, level2(3)=8, level3(3)=18
        assert "result = 18" in nb_runner.get_output(4)

        # Edit bottom of chain
        nb_runner.set_cell_source(1, "def level1(x):\n    return x + 100")
        nb_runner.run_all()
        # level1(3)=103, level2(3)=206, level3(3)=216
        assert "result = 216" in nb_runner.get_output(4)


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestCallbackEdits:
    """Callback/strategy patterns with edits."""

    def test_edit_callback_function(self, nb_runner):
        """Edit a callback function passed to another function."""
        nb_runner.create_notebook(
            [
                "def apply_fn(fn, data):\n    return [fn(x) for x in data]",
                "def transform(x):\n    return x * 2",
                "result = apply_fn(transform, [1, 2, 3])\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 4, 6]" in nb_runner.get_output(3)

        # Edit transform
        nb_runner.set_cell_source(2, "def transform(x):\n    return x ** 3")
        nb_runner.run_all()
        assert "result = [1, 8, 27]" in nb_runner.get_output(3)

    def test_edit_apply_function(self, nb_runner):
        """Edit the higher-order function."""
        nb_runner.create_notebook(
            [
                "def processor(fn, data):\n    return [fn(x) for x in data]",
                "def double(x):\n    return x * 2",
                "result = processor(double, [5, 10])\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [10, 20]" in nb_runner.get_output(3)

        # Change processor to also sum
        nb_runner.set_cell_source(
            1,
            "def processor(fn, data):\n    mapped = [fn(x) for x in data]\n    return sum(mapped)",
        )
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)


# Default parameter and keyword argument edit patterns.
#
# Tests functions with default/keyword args, editing defaults.
@pytest.mark.timeout(90)
class TestDefaultParamEdits:
    """Default parameter edit propagation."""

    def test_default_param_edit(self, nb_runner):
        """Edit function default parameter."""
        nb_runner.create_notebook(
            [
                "def greet(name, greeting='Hello'):\n    return f'{greeting}, {name}!'",
                "msg = greet('Alice')\nprint(f'msg = {msg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg = Hello, Alice!" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def greet(name, greeting='Hi'):\n    return f'{greeting}, {name}!'",
        )
        nb_runner.run_all()
        assert "msg = Hi, Alice!" in nb_runner.get_output(2)

    def test_kwargs_dict_edit(self, nb_runner):
        """Edit kwargs dict passed to function."""
        nb_runner.create_notebook(
            [
                "def build_config(**kwargs):\n    return {k: v for k, v in sorted(kwargs.items())}",
                "opts = {'color': 'blue', 'size': 10}",
                "cfg = build_config(**opts)\nprint(f'cfg = {cfg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'color': 'blue'" in nb_runner.get_output(3)
        assert "'size': 10" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "opts = {'color': 'red', 'size': 20, 'weight': 5}")
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "'color': 'red'" in out
        assert "'weight': 5" in out

    def test_mixed_args_edit(self, nb_runner):
        """Edit function with mixed positional and keyword args."""
        nb_runner.create_notebook(
            [
                "def format_record(name, age, role='member'):\n    return f'{name} ({age}) - {role}'",
                "record = format_record('Alice', 30)\nprint(f'record = {record}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "record = Alice (30) - member" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def format_record(name, age, role='admin'):\n    return f'{name} ({age}) - {role}'",
        )
        nb_runner.run_all()
        assert "record = Alice (30) - admin" in nb_runner.get_output(2)


# Multiple return value patterns.
#
# Tests tuple unpacking from function returns and editing the function.
@pytest.mark.timeout(90)
class TestMultiReturnEdit:
    """Functions returning multiple values, edit propagation."""

    def test_tuple_return_edit(self, nb_runner):
        """Edit function that returns tuple, unpacked downstream."""
        nb_runner.create_notebook(
            [
                "def stats(data):\n    return min(data), max(data), sum(data) / len(data)",
                "lo, hi, avg = stats([10, 20, 30, 40, 50])\nprint(f'lo={lo} hi={hi} avg={avg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lo=10 hi=50 avg=30.0" in nb_runner.get_output(2)

        # Edit to return range instead of avg
        nb_runner.set_cell_source(
            1,
            "def stats(data):\n    return min(data), max(data), max(data) - min(data)",
        )
        nb_runner.run_all()
        assert "lo=10 hi=50 avg=40" in nb_runner.get_output(2)

    def test_dict_return_edit(self, nb_runner):
        """Edit function returning dict, downstream uses keys."""
        nb_runner.create_notebook(
            [
                "def analyze(items):\n    return {'count': len(items), 'total': sum(items)}",
                'info = analyze([5, 10, 15])\nprint(f\'count={info["count"]} total={info["total"]}\')',
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=3 total=30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def analyze(items):\n    return {'count': len(items), 'total': sum(items) * 2}",
        )
        nb_runner.run_all()
        assert "count=3 total=60" in nb_runner.get_output(2)

    def test_named_tuple_return_edit(self, nb_runner):
        """Edit function returning namedtuple fields."""
        nb_runner.create_notebook(
            [
                "from collections import namedtuple\nResult = namedtuple('Result', ['value', 'label'])",
                "def compute(x):\n    return Result(value=x*2, label='doubled')",
                "r = compute(7)\nprint(f'value={r.value} label={r.label}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "value=14 label=doubled" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2,
            "def compute(x):\n    return Result(value=x**2, label='squared')",
        )
        nb_runner.run_all()
        assert "value=49 label=squared" in nb_runner.get_output(3)


@pytest.mark.timeout(90)
class TestNestedFunctionClosure:
    """nested function definitions and closures with edits."""

    def test_closure_counter(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def make_counter(start=0):\n    count = start\n    def increment():\n        nonlocal count\n        count += 1\n        return count\n    return increment",
                "counter = make_counter(10)\nresults = [counter() for _ in range(3)]\nprint(f'results={results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[11, 12, 13]" in nb_runner.get_output(2)

    def test_closure_edit_factory(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def multiplier(factor):\n    def multiply(x):\n        return x * factor\n    return multiply",
                "double = multiplier(2)\nresult = double(7)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=14" in nb_runner.get_output(2)
        # Edit factory
        nb_runner.set_cell_source(
            1, "def multiplier(factor):\n    def multiply(x):\n        return x * factor + 1\n    return multiply"
        )
        nb_runner.run_all()
        assert "result=15" in nb_runner.get_output(2)

    def test_nested_helper(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def process(items):\n    def clean(s):\n        return s.strip().lower()\n    return [clean(i) for i in items]",
                "data = ['  Hello ', ' WORLD ', '  Python  ']\nresult = process(data)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=['hello', 'world', 'python']" in nb_runner.get_output(2)


# Nested function definition edit patterns.
#
# Tests inner function edits propagating through outer function calls.
@pytest.mark.timeout(90)
class TestNestedFunctionEdits:
    """Nested function definition patterns."""

    def test_inner_function_edit(self, nb_runner):
        """Edit inner function, outer function result changes."""
        nb_runner.create_notebook(
            [
                "def outer(data):\n    def inner(x):\n        return x * 2\n    return [inner(x) for x in data]",
                "result = outer([1, 2, 3, 4])\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 4, 6, 8]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def outer(data):\n    def inner(x):\n        return x ** 2\n    return [inner(x) for x in data]",
        )
        nb_runner.run_all()
        assert "result = [1, 4, 9, 16]" in nb_runner.get_output(2)

    def test_nested_with_accumulator(self, nb_runner):
        """Edit inner accumulator function."""
        nb_runner.create_notebook(
            [
                "def process(items):\n    total = 0\n    def accumulate(x):\n        nonlocal total\n        total += x\n        return total\n    return [accumulate(i) for i in items]",
                "running = process([10, 20, 30])\nprint(f'running = {running}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "running = [10, 30, 60]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def process(items):\n    total = 0\n    def accumulate(x):\n        nonlocal total\n        total += x * 2\n        return total\n    return [accumulate(i) for i in items]",
        )
        nb_runner.run_all()
        assert "running = [20, 60, 120]" in nb_runner.get_output(2)

    def test_factory_with_inner_edit(self, nb_runner):
        """Edit factory function that produces inner functions."""
        nb_runner.create_notebook(
            [
                "def make_processor(op):\n    def process(x):\n        if op == 'double':\n            return x * 2\n        return x + 1\n    return process",
                "proc = make_processor('double')\nresults = [proc(i) for i in range(5)]\nprint(f'results = {results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results = [0, 2, 4, 6, 8]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            2, "proc = make_processor('other')\nresults = [proc(i) for i in range(5)]\nprint(f'results = {results}')"
        )
        nb_runner.run_all()
        assert "results = [1, 2, 3, 4, 5]" in nb_runner.get_output(2)
