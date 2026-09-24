"""map, filter and reduce pipelines across cells."""

import textwrap

import pytest


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


@pytest.mark.integration
@pytest.mark.stress
class TestClosurePatterns:
    """Test caching with closures and higher-order functions."""

    def test_higher_order_map_filter(self, nb_runner):
        """Higher-order functions: map + filter."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
                textwrap.dedent("""\
                evens = list(filter(lambda x: x % 2 == 0, data))
                squared = list(map(lambda x: x**2, evens))
                print(squared)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[4, 16, 36, 64, 100]" in nb_runner.get_output(2)
