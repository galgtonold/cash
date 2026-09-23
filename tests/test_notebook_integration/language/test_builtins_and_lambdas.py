"""Built-in functions, lambdas, map and filter across cells."""

import textwrap

import pytest


# Lambda & higher-order function interaction tests.
#
# Tests that exercise lambda expressions, map/filter/reduce, and
# higher-order function patterns with cell edits.
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


@pytest.mark.core
@pytest.mark.stress
@pytest.mark.timeout(30)
class TestMapFilterReduce:
    """map/filter/reduce + cell edits."""

    def test_map_edit(self, nb_runner):
        """Edit the function passed to map."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5]",
                "mapped = list(map(lambda x: x * 2, data))",
                "result = sum(mapped)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "mapped = list(map(lambda x: x ** 2, data))")
        nb_runner.run_all()
        assert "result = 55" in nb_runner.get_output(3)

    def test_filter_edit(self, nb_runner):
        """Edit the filter condition."""
        nb_runner.create_notebook(
            [
                "data = list(range(1, 11))",
                "filtered = list(filter(lambda x: x > 5, data))",
                "result = len(filtered)\nprint(f'count = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 5" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "filtered = list(filter(lambda x: x % 2 == 0, data))")
        nb_runner.run_all()
        assert "count = 5" in nb_runner.get_output(3)

    def test_chain_map_filter(self, nb_runner):
        """Chain map and filter, edit one."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
                "step1 = list(map(lambda x: x * 2, data))",
                "step2 = list(filter(lambda x: x > 10, step1))",
                "result = sum(step2)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # map: [2,4,6,8,10,12,14,16,18,20], filter >10: [12,14,16,18,20] = 80
        assert "result = 80" in nb_runner.get_output(4)

        # Change map
        nb_runner.set_cell_source(2, "step1 = list(map(lambda x: x * 3, data))")
        nb_runner.run_all()
        # map: [3,6,9,12,15,18,21,24,27,30], filter >10: [12,15,18,21,24,27,30] = 147
        assert "result = 147" in nb_runner.get_output(4)


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


# Complex lambda and higher-order function patterns.
#
# Tests lambdas, map/filter/reduce with edits.
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


# Map/reduce/filter interaction tests.
#
# Tests editing cells that use functional programming patterns
# (map, filter, reduce) and verifying cache invalidation.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestMapReduceEdits:
    """Editing map/reduce/filter patterns."""

    def test_edit_map_function(self, nb_runner):
        """Edit the mapping function."""
        nb_runner.create_notebook(
            [
                "nums = [1, 2, 3, 4, 5]",
                "result = list(map(lambda x: x * 2, nums))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 4, 6, 8, 10]" in nb_runner.get_output(2)

        # Change mapping
        nb_runner.set_cell_source(2, "result = list(map(lambda x: x ** 2, nums))\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = [1, 4, 9, 16, 25]" in nb_runner.get_output(2)

    def test_edit_filter_predicate(self, nb_runner):
        """Edit the filter predicate."""
        nb_runner.create_notebook(
            [
                "nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
                "evens = list(filter(lambda x: x % 2 == 0, nums))\nprint(f'evens = {evens}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "evens = [2, 4, 6, 8, 10]" in nb_runner.get_output(2)

        # Change to filter multiples of 3
        nb_runner.set_cell_source(2, "threes = list(filter(lambda x: x % 3 == 0, nums))\nprint(f'threes = {threes}')")
        nb_runner.run_all()
        assert "threes = [3, 6, 9]" in nb_runner.get_output(2)

    def test_edit_reduce_source(self, nb_runner):
        """Edit data fed into reduce."""
        nb_runner.create_notebook(
            [
                "from functools import reduce\nnums = [1, 2, 3, 4, 5]",
                "product = reduce(lambda a, b: a * b, nums)\nprint(f'product = {product}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "product = 120" in nb_runner.get_output(2)

        # Change data
        nb_runner.set_cell_source(1, "from functools import reduce\nnums = [2, 3, 4]")
        nb_runner.run_all()
        assert "product = 24" in nb_runner.get_output(2)

    def test_edit_chained_map_filter(self, nb_runner):
        """Edit source for chained map then filter."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5, 6]",
                "doubled = list(map(lambda x: x * 2, data))\nbig = list(filter(lambda x: x > 6, doubled))\nprint(f'big = {big}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "big = [8, 10, 12]" in nb_runner.get_output(2)

        # Change data
        nb_runner.set_cell_source(1, "data = [1, 2, 3, 10, 20]")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "20" in out
        assert "40" in out


# Built-in function patterns with edits.
#
# Tests map, filter, reduce with function/data edits.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestBuiltinFuncEdits:
    """Map/filter/reduce edit patterns."""

    def test_map_function_edit(self, nb_runner):
        """Edit mapping function, result changes."""
        nb_runner.create_notebook(
            [
                "def transform(x):\n    return x * 2",
                "data = [1, 2, 3, 4, 5]\nresult = list(map(transform, data))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 4, 6, 8, 10]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "def transform(x):\n    return x ** 2")
        nb_runner.run_all()
        assert "result = [1, 4, 9, 16, 25]" in nb_runner.get_output(2)

    def test_filter_predicate_edit(self, nb_runner):
        """Edit filter predicate, filtered set changes."""
        nb_runner.create_notebook(
            [
                "def pred(x):\n    return x % 2 == 0",
                "data = [1, 2, 3, 4, 5, 6]\nresult = list(filter(pred, data))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 4, 6]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "def pred(x):\n    return x > 3")
        nb_runner.run_all()
        assert "result = [4, 5, 6]" in nb_runner.get_output(2)

    def test_reduce_operator_edit(self, nb_runner):
        """Edit reduce operation."""
        nb_runner.create_notebook(
            [
                "from functools import reduce\ndef op(a, b):\n    return a + b",
                "data = [1, 2, 3, 4]\nresult = reduce(op, data)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "from functools import reduce\ndef op(a, b):\n    return a * b",
        )
        nb_runner.run_all()
        assert "result = 24" in nb_runner.get_output(2)


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
@pytest.mark.timeout(90)
class TestFilterPredicates:
    """filter with None and lambda predicates."""

    def test_filter_none(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = [0, 1, '', 'hello', None, False, 42, [], [1]]",
                "truthy = list(filter(None, data))\nprint(f'truthy={truthy}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "truthy=[1, 'hello', 42, [1]]" in nb_runner.get_output(2)

    def test_filter_lambda(self, nb_runner):
        nb_runner.create_notebook(
            [
                "nums = list(range(1, 21))",
                "evens = list(filter(lambda x: x % 2 == 0, nums))\nprint(f'evens={evens}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "evens=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20]" in nb_runner.get_output(2)

    def test_filter_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "words = ['apple', 'banana', 'cherry', 'date', 'elderberry']",
                "long_words = list(filter(lambda w: len(w) > 5, words))\nprint(f'long={long_words}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "long=['banana', 'cherry', 'elderberry']" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "words = ['fig', 'grape', 'honeydew', 'kiwi']")
        nb_runner.run_all()
        assert "long=['honeydew']" in nb_runner.get_output(2)


# Mapping and filtering with functions interaction tests.
# Tests that editing filter/map functions or data properly invalidates downstream.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestMapFilterInteraction:
    """Test map/filter function patterns with cache invalidation."""

    def test_map_function_edit(self, nb_runner):
        """Editing the mapping function should propagate."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5]",
                "def transform(x):\n    return x * 2",
                "mapped = list(map(transform, data))",
                "result = sum(mapped)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=30" in out

        nb_runner.set_cell_source(2, "def transform(x):\n    return x ** 2")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=55" in out

    def test_filter_predicate_edit(self, nb_runner):
        """Editing the filter predicate should propagate."""
        nb_runner.create_notebook(
            [
                "numbers = list(range(1, 11))",
                "def is_valid(x):\n    return x % 2 == 0",
                "filtered = list(filter(is_valid, numbers))",
                "result = ','.join(str(x) for x in filtered)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=2,4,6,8,10" in out

        nb_runner.set_cell_source(2, "def is_valid(x):\n    return x % 3 == 0")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=3,6,9" in out

    def test_reduce_edit(self, nb_runner):
        """Editing the reduce function should propagate."""
        nb_runner.create_notebook(
            [
                "from functools import reduce\ndata = [1, 2, 3, 4]",
                "def combine(acc, x):\n    return acc + x",
                "result = reduce(combine, data, 0)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=10" in out

        nb_runner.set_cell_source(2, "def combine(acc, x):\n    return acc * x")
        nb_runner.set_cell_source(3, "result = reduce(combine, data, 1)")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=24" in out


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestMapFilterFunctionalPipeline:
    """map filter reduce functional pipeline."""

    def test_map_filter_pipeline(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = list(range(1, 11))",
                "squared = list(map(lambda x: x**2, data))\nevens = list(filter(lambda x: x % 2 == 0, squared))\nprint(f'squared={squared}')\nprint(f'evens={evens}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "squared=[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]" in out
        assert "evens=[4, 16, 36, 64, 100]" in out

    def test_chained_operations(self, nb_runner):
        nb_runner.create_notebook(
            [
                "words = ['Hello', 'WORLD', 'Python', 'CODE']",
                "result = list(map(str.lower, filter(lambda w: len(w) > 4, words)))\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=['hello', 'world', 'python']" in nb_runner.get_output(2)

    def test_pipeline_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "nums = [1, 2, 3, 4, 5]",
                "result = list(map(lambda x: x * 10, filter(lambda x: x > 2, nums)))\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[30, 40, 50]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "nums = [10, 20, 30, 40, 50]")
        nb_runner.run_all()
        assert "result=[100, 200, 300, 400, 500]" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestMapFilterReducePipeline:
    """map/filter/reduce pipeline composition."""

    def test_pipeline(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import reduce\ndata = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
                "result = reduce(lambda a, b: a + b, filter(lambda x: x % 2 == 0, map(lambda x: x ** 2, data)))\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # squares of evens: 4+16+36+64+100 = 220
        assert "result=220" in nb_runner.get_output(2)

    def test_pipeline_strings(self, nb_runner):
        nb_runner.create_notebook(
            [
                "words = ['  Hello ', ' WORLD', 'Python  ', '  foo  ']",
                "cleaned = list(map(str.strip, map(str.lower, words)))\nprint(f'cleaned={cleaned}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cleaned=['hello', 'world', 'python', 'foo']" in nb_runner.get_output(2)

    def test_pipeline_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "nums = [10, 20, 30, 40, 50]",
                "doubled_big = list(filter(lambda x: x > 50, map(lambda x: x * 2, nums)))\nprint(f'result={doubled_big}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[60, 80, 100]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "nums = [5, 15, 25, 35]")
        nb_runner.run_all()
        assert "result=[70]" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestMapMultiIter:
    """map with multiple iterables and starmap patterns."""

    def test_map_two_lists(self, nb_runner):
        nb_runner.create_notebook(
            [
                "a = [1, 2, 3]\nb = [10, 20, 30]",
                "sums = list(map(lambda x, y: x + y, a, b))\nprint(f'sums={sums}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sums=[11, 22, 33]" in nb_runner.get_output(2)

    def test_map_edit_lists(self, nb_runner):
        nb_runner.create_notebook(
            [
                "xs = [1, 2, 3]\nys = [4, 5, 6]",
                "products = list(map(lambda x, y: x * y, xs, ys))\ntotal = sum(products)\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=32" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "xs = [10, 20]\nys = [3, 4]")
        nb_runner.run_all()
        assert "total=110" in nb_runner.get_output(2)

    def test_map_type_convert(self, nb_runner):
        nb_runner.create_notebook(
            [
                "strings = ['1', '2', '3', '4', '5']",
                "ints = list(map(int, strings))\ntotal = sum(ints)\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=15" in nb_runner.get_output(2)


# Functional programming patterns — compose, pipe, curry, partial application.
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
