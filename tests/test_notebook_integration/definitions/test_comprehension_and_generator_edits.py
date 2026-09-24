"""Editing comprehensions, generators, iterators and map/filter/reduce/zip/itertools steps."""

import pytest

pytestmark = [pytest.mark.stress]


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestListComprehensionEdits:
    """List comprehension edits."""

    def test_edit_comprehension_expression(self, nb_runner):
        """Edit the expression in a list comprehension."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5]",
                "result = [x * 2 for x in data]\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [2, 4, 6, 8, 10]" in nb_runner.get_output(2)

        # Change expression
        nb_runner.set_cell_source(2, "result = [x ** 2 for x in data]\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = [1, 4, 9, 16, 25]" in nb_runner.get_output(2)

    def test_edit_comprehension_filter(self, nb_runner):
        """Edit the filter condition in a list comprehension."""
        nb_runner.create_notebook(
            [
                "nums = list(range(10))",
                "evens = [x for x in nums if x % 2 == 0]\nprint(f'evens = {evens}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "evens = [0, 2, 4, 6, 8]" in nb_runner.get_output(2)

        # Change filter to odds
        nb_runner.set_cell_source(2, "evens = [x for x in nums if x % 2 == 1]\nprint(f'evens = {evens}')")
        nb_runner.run_all()
        assert "evens = [1, 3, 5, 7, 9]" in nb_runner.get_output(2)

    def test_edit_comprehension_source(self, nb_runner):
        """Edit the source data of a comprehension."""
        nb_runner.create_notebook(
            [
                "src = [10, 20, 30]  # source data",
                "doubled = [x * 2 for x in src]\nprint(f'doubled = {doubled}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "doubled = [20, 40, 60]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "src = [1, 2, 3]  # source data smaller")
        nb_runner.run_all()
        assert "doubled = [2, 4, 6]" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestDictComprehensionEdits:
    """Dict comprehension edits."""

    def test_edit_dict_comprehension_value(self, nb_runner):
        """Edit the value expression in a dict comprehension."""
        nb_runner.create_notebook(
            [
                "keys = ['a', 'b', 'c']",
                "mapping = {k: len(k) for k in keys}\nprint(f'mapping = {mapping}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 1" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "mapping = {k: k.upper() for k in keys}\nprint(f'mapping = {mapping}')")
        nb_runner.run_all()
        assert "'a': 'A'" in nb_runner.get_output(2)

    def test_nested_comprehension_edit(self, nb_runner):
        """Edit a nested comprehension."""
        nb_runner.create_notebook(
            [
                "matrix = [[1, 2], [3, 4]]",
                "flat = [x for row in matrix for x in row]\nprint(f'flat = {flat}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "flat = [1, 2, 3, 4]" in nb_runner.get_output(2)

        # Change to transform
        nb_runner.set_cell_source(
            2,
            "flat = [x * 10 for row in matrix for x in row]\nprint(f'flat = {flat}')",
        )
        nb_runner.run_all()
        assert "flat = [10, 20, 30, 40]" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestSetComprehensionEdits:
    """Set comprehension edits."""

    def test_edit_set_comprehension(self, nb_runner):
        """Edit a set comprehension expression."""
        nb_runner.create_notebook(
            [
                "words = ['hello', 'world', 'hello', 'python']",
                "lengths = {len(w) for w in words}\nprint(f'lengths = {sorted(lengths)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lengths = [5, 6]" in nb_runner.get_output(2)

        # Change to first chars
        nb_runner.set_cell_source(2, "lengths = {w[0] for w in words}\nprint(f'lengths = {sorted(lengths)}')")
        nb_runner.run_all()
        assert "'h'" in nb_runner.get_output(2)
        assert "'p'" in nb_runner.get_output(2)
        assert "'w'" in nb_runner.get_output(2)


class TestGeneratorEdits:
    """Generator function edits."""

    @pytest.mark.upstream
    @pytest.mark.timeout(45)
    def test_edit_generator_function(self, nb_runner):
        """Edit generator function, verify consumer updates."""
        nb_runner.create_notebook(
            [
                "def gen_range(n):\n    for i in range(n):\n        yield i * 2",
                "result = list(gen_range(5))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [0, 2, 4, 6, 8]" in nb_runner.get_output(2)

        # Edit generator to yield squares
        nb_runner.set_cell_source(1, "def gen_range(n):\n    for i in range(n):\n        yield i ** 2")
        nb_runner.run_all()
        assert "result = [0, 1, 4, 9, 16]" in nb_runner.get_output(2)

    @pytest.mark.upstream
    @pytest.mark.timeout(45)
    def test_edit_generator_param(self, nb_runner):
        """Edit parameter passed to generator."""
        nb_runner.create_notebook(
            [
                "count = 3",
                "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b",
                "fibs = list(fib(count))\nprint(f'fibs = {fibs}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "fibs = [0, 1, 1]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "count = 8")
        nb_runner.run_all()
        assert "fibs = [0, 1, 1, 2, 3, 5, 8, 13]" in nb_runner.get_output(3)

    # Generator and iterator interaction tests.
    #
    # Tests editing generator functions, iterator protocols,
    # and lazy evaluation patterns.
    @pytest.mark.upstream
    @pytest.mark.timeout(90)
    def test_edit_generator_yield(self, nb_runner):
        """Edit what a generator yields."""
        nb_runner.create_notebook(
            [
                "def gen_nums(n):\n    for i in range(n):\n        yield i",
                "result = list(gen_nums(5))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [0, 1, 2, 3, 4]" in nb_runner.get_output(2)

        # Change to yield squares
        nb_runner.set_cell_source(
            1,
            "def gen_nums(n):\n    for i in range(n):\n        yield i ** 2",
        )
        nb_runner.run_all()
        assert "result = [0, 1, 4, 9, 16]" in nb_runner.get_output(2)

    @pytest.mark.upstream
    @pytest.mark.timeout(90)
    def test_edit_generator_filter(self, nb_runner):
        """Edit the filter condition in a generator."""
        nb_runner.create_notebook(
            [
                "def even_gen(n):\n    for i in range(n):\n        if i % 2 == 0:\n            yield i",
                "result = list(even_gen(10))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [0, 2, 4, 6, 8]" in nb_runner.get_output(2)

        # Change to odd
        nb_runner.set_cell_source(
            1,
            "def even_gen(n):\n    for i in range(n):\n        if i % 2 == 1:\n            yield i",
        )
        nb_runner.run_all()
        assert "result = [1, 3, 5, 7, 9]" in nb_runner.get_output(2)

    @pytest.mark.upstream
    @pytest.mark.timeout(90)
    def test_edit_generator_range(self, nb_runner):
        """Edit the range of a generator call."""
        nb_runner.create_notebook(
            [
                "def countdown(n):\n    while n > 0:\n        yield n\n        n -= 1",
                "result = list(countdown(3))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [3, 2, 1]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "result = list(countdown(6))\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = [6, 5, 4, 3, 2, 1]" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestIteratorProtocol:
    """Iterator protocol with edits."""

    def test_edit_iterator_class(self, nb_runner):
        """Edit an iterator class __next__ method."""
        nb_runner.create_notebook(
            [
                "class Counter:\n    def __init__(self, n):\n        self.n = n\n        self.i = 0\n    def __iter__(self):\n        return self\n    def __next__(self):\n        if self.i >= self.n:\n            raise StopIteration\n        val = self.i\n        self.i += 1\n        return val",
                "result = list(Counter(4))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [0, 1, 2, 3]" in nb_runner.get_output(2)

        # Change to yield squares
        nb_runner.set_cell_source(
            1,
            "class Counter:\n    def __init__(self, n):\n        self.n = n\n        self.i = 0\n    def __iter__(self):\n        return self\n    def __next__(self):\n        if self.i >= self.n:\n            raise StopIteration\n        val = self.i ** 2\n        self.i += 1\n        return val",
        )
        nb_runner.run_all()
        assert "result = [0, 1, 4, 9]" in nb_runner.get_output(2)

    def test_generator_expression_edit(self, nb_runner):
        """Edit a generator expression."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5]  # source data for gen",
                "total = sum(x * 2 for x in data)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 30" in nb_runner.get_output(2)

        # Change to cubed
        nb_runner.set_cell_source(2, "total = sum(x ** 3 for x in data)\nprint(f'total = {total}')")
        nb_runner.run_all()
        assert "total = 225" in nb_runner.get_output(2)


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


@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestMapFilterEdits:
    """Map/filter patterns with edits."""

    def test_edit_map_function(self, nb_runner):
        """Edit function used in map."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5]",
                "mapped = list(map(lambda x: x * 2, data))",
                "total = sum(mapped)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 30" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "mapped = list(map(lambda x: x ** 2, data))")
        nb_runner.run_all()
        assert "total = 55" in nb_runner.get_output(3)

    def test_edit_filter_predicate(self, nb_runner):
        """Edit filter predicate."""
        nb_runner.create_notebook(
            [
                "nums = list(range(20))",
                "filtered = list(filter(lambda x: x % 2 == 0, nums))",
                "count = len(filtered)\nprint(f'count = {count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 10" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "filtered = list(filter(lambda x: x % 5 == 0, nums))")
        nb_runner.run_all()
        assert "count = 4" in nb_runner.get_output(3)

    def test_chain_map_filter_edit(self, nb_runner):
        """Chain map then filter, edit map."""
        nb_runner.create_notebook(
            [
                "raw = list(range(1, 11))",
                "doubled = [x * 2 for x in raw]",
                "big = [x for x in doubled if x > 10]",
                "total = sum(big)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # doubled = [2,4,6,8,10,12,14,16,18,20], big = [12,14,16,18,20] -> 80
        assert "total = 80" in nb_runner.get_output(4)

        # Change map to triple
        nb_runner.set_cell_source(2, "doubled = [x * 3 for x in raw]")
        nb_runner.run_all()
        # tripled = [3,6,9,12,15,18,21,24,27,30], big = [12,15,18,21,24,27,30] -> 147
        assert "total = 147" in nb_runner.get_output(4)


@pytest.mark.timeout(90)
class TestZipEnumEdits:
    """Zip/enumerate edit patterns."""

    def test_zip_edit_one_list(self, nb_runner):
        """Edit one of two zipped lists."""
        nb_runner.create_notebook(
            [
                "names = ['Alice', 'Bob', 'Charlie']",
                "scores = [90, 85, 78]",
                "pairs = list(zip(names, scores))\nprint(f'pairs = {pairs}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "('Alice', 90)" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "scores = [100, 95, 88]")
        nb_runner.run_all()
        assert "('Alice', 100)" in nb_runner.get_output(3)
        assert "('Charlie', 88)" in nb_runner.get_output(3)

    def test_enumerate_with_edit(self, nb_runner):
        """Edit list, enumerate indexes correctly reflect."""
        nb_runner.create_notebook(
            [
                "items = ['apple', 'banana', 'cherry']",
                "indexed = list(enumerate(items, start=1))\nprint(f'indexed = {indexed}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "(1, 'apple')" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "items = ['x', 'y']")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "(1, 'x')" in out
        assert "(2, 'y')" in out

    def test_zip_longest_edit(self, nb_runner):
        """Edit data in zip_longest scenario."""
        nb_runner.create_notebook(
            [
                "from itertools import zip_longest\na = [1, 2, 3]\nb = ['x', 'y']",
                "result = list(zip_longest(a, b, fillvalue='?'))\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "(3, '?')" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "from itertools import zip_longest\na = [1]\nb = ['x', 'y', 'z']")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "(1, 'x')" in out
        assert "('?', 'z')" in out


@pytest.mark.timeout(90)
class TestItertoolsEdits:
    """Itertools operation edit patterns."""

    def test_groupby_edit(self, nb_runner):
        """Edit data before groupby."""
        nb_runner.create_notebook(
            [
                "from itertools import groupby\ndata = [('a', 1), ('a', 2), ('b', 3), ('b', 4)]",
                "groups = {k: list(v) for k, v in groupby(data, key=lambda x: x[0])}\nresult = {k: len(v) for k, v in groups.items()}\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'a': 2" in out
        assert "'b': 2" in out

        nb_runner.set_cell_source(
            1,
            "from itertools import groupby\ndata = [('x', 1), ('x', 2), ('x', 3), ('y', 4)]",
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "'x': 3" in out2
        assert "'y': 1" in out2
