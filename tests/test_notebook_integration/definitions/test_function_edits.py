"""Editing function definitions and the calls downstream of them."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress]


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
class TestRecursiveFunctionEdits:
    """Editing recursive function bodies."""

    def test_edit_base_case(self, nb_runner):
        """Edit the base case of a recursive function."""
        nb_runner.create_notebook(
            [
                "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
                "result = factorial(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 120" in nb_runner.get_output(2)

        # Change base case to return 2
        nb_runner.set_cell_source(
            1,
            "def factorial(n):\n    if n <= 1:\n        return 2\n    return n * factorial(n - 1)",
        )
        nb_runner.run_all()
        assert "result = 240" in nb_runner.get_output(2)

    def test_edit_recursive_step(self, nb_runner):
        """Edit the recursive step."""
        nb_runner.create_notebook(
            [
                "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)",
                "result = fib(7)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 13" in nb_runner.get_output(2)

        # Change to tribonacci
        nb_runner.set_cell_source(
            1,
            "def fib(n):\n    if n <= 1:\n        return n\n    if n == 2:\n        return 1\n    return fib(n-1) + fib(n-2) + fib(n-3)",
        )
        nb_runner.run_all()
        # tribonacci(7) = 24
        assert "result = 24" in nb_runner.get_output(2)

    def test_add_memoization(self, nb_runner):
        """Add memoization to a recursive function."""
        nb_runner.create_notebook(
            [
                "def slow_sum(n):\n    if n <= 0:\n        return 0\n    return n + slow_sum(n - 1)",
                "result = slow_sum(100)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 5050" in nb_runner.get_output(2)

        # Add caching
        nb_runner.set_cell_source(
            1,
            "from functools import lru_cache\n@lru_cache(maxsize=None)\ndef slow_sum(n):\n    if n <= 0:\n        return 0\n    return n + slow_sum(n - 1)",
        )
        nb_runner.run_all()
        assert "result = 5050" in nb_runner.get_output(2)

    def test_recursive_with_helper(self, nb_runner):
        """Edit a recursive function that calls a helper."""
        nb_runner.create_notebook(
            [
                "def double(x):\n    return x * 2",
                "def recurse(n):\n    if n <= 0:\n        return 0\n    return double(n) + recurse(n - 1)",
                "result = recurse(4)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # double(4)+double(3)+double(2)+double(1) = 8+6+4+2 = 20
        assert "result = 20" in nb_runner.get_output(3)

        # Edit helper
        nb_runner.set_cell_source(1, "def double(x):\n    return x * 3")
        nb_runner.run_all()
        # 12+9+6+3 = 30
        assert "result = 30" in nb_runner.get_output(3)


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


@pytest.mark.integration
class TestMutableDefaultArguments:
    """Test caching with mutable default arguments."""

    def test_mutable_default_list(self, nb_runner):
        """Classic mutable default argument gotcha."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def append_to(item, lst=None):
                    if lst is None:
                        lst = []
                    lst.append(item)
                    return lst
            """),
                textwrap.dedent("""\
                r1 = append_to(1)
                r2 = append_to(2)
                r3 = append_to(3, [10, 20])
                print(f"r1={r1} r2={r2} r3={r3}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "r1=[1]" in output
        assert "r2=[2]" in output
        assert "r3=[10, 20, 3]" in output


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
