"""Lambdas, higher-order functions, composition and currying across cells."""

import textwrap

import pytest


@pytest.mark.core
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestLambdaEdits:
    """Lambda expressions + cell edits."""

    def test_edit_lambda(self, nb_runner):
        """Edit a lambda expression."""
        nb_runner.create_notebook(
            [
                "fn = lambda x: x * 2",
                "result = fn(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "fn = lambda x: x ** 2")
        nb_runner.run_all()
        assert "result = 25" in nb_runner.get_output(2)

    def test_lambda_with_closure(self, nb_runner):
        """Lambda capturing a variable, edit the captured value."""
        nb_runner.create_notebook(
            [
                "factor = 3",
                "scale = lambda x: x * factor",
                "result = scale(10)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "factor = 10")
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(3)

    def test_lambda_as_argument(self, nb_runner):
        """Lambda used as argument, edit the lambda."""
        nb_runner.create_notebook(
            [
                "data = [3, 1, 4, 1, 5, 9, 2, 6]",
                "sorted_data = sorted(data, key=lambda x: x)",
                "print(f'sorted = {sorted_data}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sorted = [1, 1, 2, 3, 4, 5, 6, 9]" in nb_runner.get_output(3)

        # Sort in reverse
        nb_runner.set_cell_source(2, "sorted_data = sorted(data, key=lambda x: -x)")
        nb_runner.run_all()
        assert "sorted = [9, 6, 5, 4, 3, 2, 1, 1]" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestLambdaHigherOrder:
    """Lambda and higher-order function edit patterns."""

    def test_lambda_in_map(self, nb_runner):
        """Edit lambda used in map."""
        nb_runner.create_notebook(
            [
                "transform = lambda x: x * 2",
                "data = [1, 2, 3, 4, 5]\nresult = list(map(transform, data))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 4, 6, 8, 10]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "transform = lambda x: x ** 3")
        nb_runner.run_all()
        assert "result = [1, 8, 27, 64, 125]" in nb_runner.get_output(2)

    def test_filter_lambda_edit(self, nb_runner):
        """Edit filter criteria lambda."""
        nb_runner.create_notebook(
            [
                "is_valid = lambda x: x > 0",
                "nums = [-3, -1, 0, 2, 5, -4, 8]\nvalid = list(filter(is_valid, nums))\nprint(f'valid = {valid}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "valid = [2, 5, 8]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "is_valid = lambda x: x % 2 == 0")
        nb_runner.run_all()
        assert "valid = [0, 2, -4, 8]" in nb_runner.get_output(2)

    def test_reduce_with_edit(self, nb_runner):
        """Edit reduce function, downstream updates."""
        nb_runner.create_notebook(
            [
                "from functools import reduce\ncombiner = lambda a, b: a + b",
                "data = [1, 2, 3, 4, 5]\nresult = reduce(combiner, data)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "from functools import reduce\ncombiner = lambda a, b: a * b",
        )
        nb_runner.run_all()
        assert "result = 120" in nb_runner.get_output(2)

    def test_sorted_key_lambda_edit(self, nb_runner):
        """Edit sort key lambda."""
        nb_runner.create_notebook(
            [
                "key_fn = lambda item: item[1]",
                "records = [('a', 3), ('b', 1), ('c', 2)]\nordered = sorted(records, key=key_fn)\nprint(f'ordered = {ordered}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "ordered = [('b', 1), ('c', 2), ('a', 3)]" in nb_runner.get_output(2)

        # Sort by first element descending
        nb_runner.set_cell_source(1, "key_fn = lambda item: item[0]")
        nb_runner.run_all()
        assert "ordered = [('a', 3), ('b', 1), ('c', 2)]" in nb_runner.get_output(2)


@pytest.mark.core
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestHigherOrderFunctions:
    """Higher-order function patterns + edits."""

    def test_apply_function(self, nb_runner):
        """Apply a function to data, edit the function."""
        nb_runner.create_notebook(
            [
                "def apply_fn(fn, data):\n    return [fn(x) for x in data]",
                "transform = lambda x: x + 1",
                "data = [1, 2, 3]",
                "result = apply_fn(transform, data)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 3, 4]" in nb_runner.get_output(4)

        nb_runner.set_cell_source(2, "transform = lambda x: x * 10")
        nb_runner.run_all()
        assert "result = [10, 20, 30]" in nb_runner.get_output(4)

    def test_compose_functions(self, nb_runner):
        """Compose two functions, edit one."""
        nb_runner.create_notebook(
            [
                "def compose(f, g):\n    return lambda x: f(g(x))",
                "double = lambda x: x * 2",
                "add_one = lambda x: x + 1",
                "pipeline = compose(double, add_one)",
                "result = pipeline(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # add_one(5) = 6, double(6) = 12
        assert "result = 12" in nb_runner.get_output(5)

        nb_runner.set_cell_source(2, "double = lambda x: x * 10")
        nb_runner.run_all()
        # add_one(5) = 6, double(6) = 60
        assert "result = 60" in nb_runner.get_output(5)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestBuiltinHigherOrder:
    """any/all/filter/map builtins with lambdas and edits."""

    def test_any_all(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = [2, 4, 6, 8, 10]",
                "all_even = all(x % 2 == 0 for x in data)\nany_gt5 = any(x > 5 for x in data)\nprint(f'all_even={all_even} any_gt5={any_gt5}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "all_even=True any_gt5=True" in nb_runner.get_output(2)

    def test_filter_map_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
                "evens = list(filter(lambda x: x % 2 == 0, numbers))\nsquared = list(map(lambda x: x ** 2, evens))\nprint(f'squared={squared}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "squared=[4, 16, 36, 64, 100]" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "numbers = [10, 15, 20, 25, 30]")
        nb_runner.run_all()
        assert "squared=[100, 400, 900]" in nb_runner.get_output(2)

    def test_sorted_key_lambda(self, nb_runner):
        nb_runner.create_notebook(
            [
                "items = [('b', 2), ('a', 3), ('c', 1)]",
                "by_val = sorted(items, key=lambda x: x[1])\nby_name = sorted(items, key=lambda x: x[0])\nprint(f'by_val={by_val}')\nprint(f'by_name={by_name}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "by_val=[('c', 1), ('b', 2), ('a', 3)]" in out
        assert "by_name=[('a', 3), ('b', 2), ('c', 1)]" in out


@pytest.mark.stress
class TestFunctionalComposition:
    """Test function composition and piping."""

    def test_compose_change_propagates(self, nb_runner):
        """Changing composed function propagates."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def step1(x):
                    return x + 10
                def step2(x):
                    return x * 2
            """),
                textwrap.dedent("""\
                result = step2(step1(5))
                print(f"result={result}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            def step1(x):
                return x + 100
            def step2(x):
                return x * 2
        """),
        )
        nb_runner.run_all()
        assert "result=210" in nb_runner.get_output(2)


@pytest.mark.stress
class TestCurrying:
    """Test currying and partial application patterns."""

    def test_manual_curry(self, nb_runner):
        """Manual currying implementation."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def curry(fn):
                    import inspect
                    params = inspect.signature(fn).parameters
                    n = len(params)
                    def curried(*args):
                        if len(args) >= n:
                            return fn(*args[:n])
                        return lambda *more: curried(*args, *more)
                    return curried

                @curry
                def add3(a, b, c):
                    return a + b + c
            """),
                textwrap.dedent("""\
                r1 = add3(1, 2, 3)
                r2 = add3(1)(2)(3)
                r3 = add3(1, 2)(3)
                print(f"r1={r1} r2={r2} r3={r3}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=6 r2=6 r3=6" in nb_runner.get_output(2)
