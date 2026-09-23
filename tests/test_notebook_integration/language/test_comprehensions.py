"""List, dict and set comprehensions across cells."""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


# Comprehension filter and transform edit tests.
#
# Tests editing filter conditions and transformations in various
# comprehension expressions.
@pytest.mark.upstream
class TestComprehensionFilterEdits:
    """Editing filters and transforms in comprehensions."""

    def test_edit_list_comp_filter(self, nb_runner):
        """Edit the filter condition in a list comprehension."""
        nb_runner.create_notebook(
            [
                "nums = list(range(20))",
                "evens = [x for x in nums if x % 2 == 0]\nprint(f'count = {len(evens)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 10" in nb_runner.get_output(2)

        # Change to multiples of 3
        nb_runner.set_cell_source(2, "multiples = [x for x in nums if x % 3 == 0]\nprint(f'count = {len(multiples)}')")
        nb_runner.run_all()
        assert "count = 7" in nb_runner.get_output(2)

    def test_edit_dict_comp_transform(self, nb_runner):
        """Edit a dict comprehension transformation."""
        nb_runner.create_notebook(
            [
                "words = ['hello', 'world', 'python']",
                "lengths = {w: len(w) for w in words}\nprint(f'lengths = {lengths}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'hello': 5" in nb_runner.get_output(2)

        # Change to upper case keys
        nb_runner.set_cell_source(2, "uppers = {w.upper(): len(w) for w in words}\nprint(f'uppers = {uppers}')")
        nb_runner.run_all()
        assert "'HELLO': 5" in nb_runner.get_output(2)

    def test_edit_nested_flat_comprehension(self, nb_runner):
        """Edit a nested comprehension (matrix flattening)."""
        nb_runner.create_notebook(
            [
                "matrix = [[1, 2], [3, 4], [5, 6]]",
                "flat = [x for row in matrix for x in row]\nprint(f'flat = {flat}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "flat = [1, 2, 3, 4, 5, 6]" in nb_runner.get_output(2)

        # Change matrix
        nb_runner.set_cell_source(1, "matrix = [[10, 20], [30, 40]]")
        nb_runner.run_all()
        assert "flat = [10, 20, 30, 40]" in nb_runner.get_output(2)

    def test_edit_sorted_set_comprehension(self, nb_runner):
        """Edit a set comprehension."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 2, 3, 3, 3]",
                "unique = sorted({x for x in data})\nprint(f'unique = {unique}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "unique = [1, 2, 3]" in nb_runner.get_output(2)

        # Change data
        nb_runner.set_cell_source(1, "data = [4, 4, 5, 6, 6]")
        nb_runner.run_all()
        assert "unique = [4, 5, 6]" in nb_runner.get_output(2)


# Comprehension variants interaction tests.
# Tests dict comprehension, set comprehension, and nested comprehension
# patterns with cache invalidation.
@pytest.mark.integration
class TestComprehensionVariantsInteraction:
    """Test various comprehension patterns with cache invalidation."""

    def test_dict_comprehension_edit(self, nb_runner):
        """Editing data used in dict comprehension should propagate."""
        nb_runner.create_notebook(
            [
                "names = ['alice', 'bob', 'charlie']",
                "name_lengths = {n: len(n) for n in names}",
                "result = sorted(name_lengths.items())",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "alice" in out
        assert "bob" in out

        nb_runner.set_cell_source(1, "names = ['x', 'hello', 'world']")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "x" in out
        assert "hello" in out
        assert "world" in out

    def test_set_comprehension_edit(self, nb_runner):
        """Editing data used in set comprehension should propagate."""
        nb_runner.create_notebook(
            [
                "nums = [1, 2, 2, 3, 3, 3, 4]",
                "unique_doubled = {x * 2 for x in nums}",
                "result = sorted(unique_doubled)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=[2, 4, 6, 8]" in out

        nb_runner.set_cell_source(1, "nums = [5, 5, 10, 10, 15]")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=[10, 20, 30]" in out

    def test_nested_comprehension_edit(self, nb_runner):
        """Editing data in nested comprehension should propagate."""
        nb_runner.create_notebook(
            [
                "matrix = [[1, 2], [3, 4], [5, 6]]",
                "flat = [x for row in matrix for x in row]",
                "total = sum(flat)",
                "print(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=21" in out

        nb_runner.set_cell_source(1, "matrix = [[10, 20], [30, 40]]")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=100" in out


class TestComprehensionChain:
    """chain of comprehensions and transformations across cells."""

    def test_chained_comprehensions(self, nb_runner):
        nb_runner.create_notebook(
            [
                "raw = list(range(20))",
                "evens = [x for x in raw if x % 2 == 0]",
                "squared = [x**2 for x in evens]",
                "result = {x: 'big' if x > 50 else 'small' for x in squared}\nprint(f'result={sorted(result.items())}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=" in nb_runner.get_output(4)
        assert "(64, 'big')" in nb_runner.get_output(4)

    def test_chained_comprehension_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5]",
                "filtered = [x for x in data if x > 2]",
                "doubled = [x * 2 for x in filtered]\nprint(f'doubled={doubled}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "doubled=[6, 8, 10]" in nb_runner.get_output(3)
        # Edit filter threshold
        nb_runner.set_cell_source(2, "filtered = [x for x in data if x > 3]")
        nb_runner.run_all()
        assert "doubled=[8, 10]" in nb_runner.get_output(3)

    def test_nested_dict_comprehension(self, nb_runner):
        nb_runner.create_notebook(
            [
                "keys = ['a', 'b', 'c']\nvals = [1, 2, 3]",
                "mapping = {k: v * 10 for k, v in zip(keys, vals)}\nprint(f'mapping={mapping}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mapping={'a': 10, 'b': 20, 'c': 30}" in nb_runner.get_output(2)


class TestDictComprehensionConditional:
    """dict comprehension with conditional logic."""

    def test_dict_comp_filter(self, nb_runner):
        nb_runner.create_notebook(
            [
                "scores = {'Alice': 95, 'Bob': 67, 'Carol': 82, 'Dave': 45, 'Eve': 91}",
                "passing = {k: v for k, v in scores.items() if v >= 70}\nfailing = {k: v for k, v in scores.items() if v < 70}\nprint(f'passing={sorted(passing.keys())} failing={sorted(failing.keys())}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "passing=['Alice', 'Carol', 'Eve']" in out
        assert "failing=['Bob', 'Dave']" in out

    def test_dict_comp_transform(self, nb_runner):
        nb_runner.create_notebook(
            [
                "words = ['hello', 'world', 'python', 'code']",
                "lengths = {w: len(w) for w in words}\nuppered = {w: w.upper() for w in words}\nprint(f'lengths={lengths}')\nprint(f'uppered={uppered}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'hello': 5" in out
        assert "'HELLO'" in out

    def test_dict_comp_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "nums = [1, 2, 3]",
                "d = {n: n**2 for n in nums}\nprint(f'd={d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d={1: 1, 2: 4, 3: 9}" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "nums = [5, 10, 15]")
        nb_runner.run_all()
        assert "d={5: 25, 10: 100, 15: 225}" in nb_runner.get_output(2)


# Interaction test: dict comprehension with conditional expressions.
# Tests dict comprehension with ternary operators, nested conditions,
# and cross-cell dict transformation pipelines.
class TestDictCompConditionalExpr:
    """Test dict comprehension with conditional expressions across cells."""

    def test_dict_comp_ternary(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: dict comp with ternary
                "scores = {'Alice': 85, 'Bob': 62, 'Charlie': 91, 'Diana': 45, 'Eve': 78}\ngrades = {name: ('pass' if score >= 60 else 'fail') for name, score in scores.items()}\nprint(f'grades={grades}')",
                # Cell 2: filter and transform
                "passing = {k: v for k, v in scores.items() if grades[k] == 'pass'}\navg_pass = sum(passing.values()) / len(passing)\nprint(f'passing_count={len(passing)}')\nprint(f'avg_pass={avg_pass:.1f}')",
                # Cell 3: categorize
                "categories = {name: ('A' if s >= 90 else 'B' if s >= 80 else 'C' if s >= 70 else 'D' if s >= 60 else 'F') for name, s in scores.items()}\nprint(f'categories={categories}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "'Alice': 'pass'" in out1
        assert "'Diana': 'fail'" in out1
        out2 = nb_runner.get_output(2)
        assert "passing_count=4" in out2
        out3 = nb_runner.get_output(3)
        assert "'Charlie': 'A'" in out3
        assert "'Alice': 'B'" in out3
        assert "'Diana': 'F'" in out3

    def test_dict_comp_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "prices = {'apple': 1.5, 'banana': 0.5, 'cherry': 3.0}\ndiscounted = {k: round(v * 0.9, 2) for k, v in prices.items()}\nprint(f'disc={discounted}')",
                "total = sum(discounted.values())\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()

        # Change discount rate
        nb_runner.set_cell_source(
            1,
            "prices = {'apple': 1.5, 'banana': 0.5, 'cherry': 3.0}\ndiscounted = {k: round(v * 0.8, 2) for k, v in prices.items()}\nprint(f'disc={discounted}')",
        )
        nb_runner.run_cells([1, 2])
        assert "'apple': 1.2" in nb_runner.get_output(1)
        assert "total=4.0" in nb_runner.get_output(2)

    def test_dict_comp_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = [('a', 1), ('b', 2), ('c', 3)]\nd = {k: v ** 2 for k, v in data}\nprint(f'd={d}')",
                "total = sum(d.values())\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d={'a': 1, 'b': 4, 'c': 9}" in nb_runner.get_output(1)
        assert "total=14" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "total=14" in nb_runner.get_output(2)


# Interaction test: list comprehension with multiple for-clauses.
# Tests nested list comprehensions with multiple iterables,
# conditions, and cross-cell flattening patterns.
class TestMultiForComprehension:
    """Test multi-for comprehensions across cells."""

    def test_multi_for_comp(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: cartesian product via comprehension
                "colors = ['red', 'blue']\nsizes = ['S', 'M', 'L']\ncombos = [(c, s) for c in colors for s in sizes]\nprint(f'combos={combos}')\nprint(f'count={len(combos)}')",
                # Cell 2: filtered cartesian
                "nums1 = range(1, 5)\nnums2 = range(1, 5)\npairs = [(a, b) for a in nums1 for b in nums2 if a < b]\nprint(f'pairs={pairs}')",
                # Cell 3: aggregate
                "total_combos = len(combos)\ntotal_pairs = len(pairs)\nprint(f'combos={total_combos}')\nprint(f'pairs={total_pairs}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "count=6" in out1
        out2 = nb_runner.get_output(2)
        assert "(1, 2)" in out2
        assert "(1, 3)" in out2
        out3 = nb_runner.get_output(3)
        assert "combos=6" in out3
        assert "pairs=6" in out3

    def test_multi_for_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "rows = [1, 2, 3]\ncols = ['a', 'b']\ngrid = [(r, c) for r in rows for c in cols]\nprint(f'grid_size={len(grid)}')",
                "first = grid[0]\nlast = grid[-1]\nprint(f'first={first}')\nprint(f'last={last}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "grid_size=6" in nb_runner.get_output(1)
        assert "first=(1, 'a')" in nb_runner.get_output(2)

        # Add more rows
        nb_runner.set_cell_source(
            1,
            "rows = [1, 2, 3, 4]\ncols = ['a', 'b']\ngrid = [(r, c) for r in rows for c in cols]\nprint(f'grid_size={len(grid)}')",
        )
        nb_runner.run_cells([1, 2])
        assert "grid_size=8" in nb_runner.get_output(1)
        assert "last=(4, 'b')" in nb_runner.get_output(2)

    def test_multi_for_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "matrix = [[i * j for j in range(1, 4)] for i in range(1, 4)]\nprint(f'matrix={matrix}')",
                "flat = [x for row in matrix for x in row]\nprint(f'flat={flat}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "flat=[1, 2, 3, 2, 4, 6, 3, 6, 9]" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "flat=[1, 2, 3, 2, 4, 6, 3, 6, 9]" in nb_runner.get_output(2)


# Interaction test: set comprehension with complex filtering.
# Tests set comprehension with multi-condition filters, set algebra,
# and cross-cell set-based analysis.
class TestSetComprehensionFilter:
    """Test set comprehension with complex filtering across cells."""

    def test_set_comp_filter(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: set comprehension with conditions
                "nums = range(1, 51)\nevens = {n for n in nums if n % 2 == 0}\ndiv_by_3 = {n for n in nums if n % 3 == 0}\nprint(f'evens_count={len(evens)}')\nprint(f'div3_count={len(div_by_3)}')",
                # Cell 2: set operations
                "both = evens & div_by_3  # div by 6\neither = evens | div_by_3\nonly_even = evens - div_by_3\nprint(f'both={sorted(both)}')\nprint(f'either_count={len(either)}')\nprint(f'only_even_count={len(only_even)}')",
                # Cell 3: symmetric difference
                "sym_diff = evens ^ div_by_3\nprint(f'sym_diff_count={len(sym_diff)}')\nprint(f'verify={len(sym_diff) == len(either) - len(both)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "evens_count=25" in out1
        assert "div3_count=16" in out1
        out2 = nb_runner.get_output(2)
        assert "6" in out2 and "12" in out2
        out3 = nb_runner.get_output(3)
        assert "verify=True" in out3

    def test_set_comp_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "nums = range(1, 21)\nprimes = {n for n in nums if n > 1 and all(n % i != 0 for i in range(2, int(n**0.5)+1))}\nprint(f'primes={sorted(primes)}')",
                "prime_count = len(primes)\nprint(f'count={prime_count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=8" in nb_runner.get_output(2)

        # Extend range
        nb_runner.set_cell_source(
            1,
            "nums = range(1, 31)\nprimes = {n for n in nums if n > 1 and all(n % i != 0 for i in range(2, int(n**0.5)+1))}\nprint(f'primes={sorted(primes)}')",
        )
        nb_runner.run_cells([1, 2])
        assert "count=10" in nb_runner.get_output(2)

    def test_set_comp_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "words = ['hello', 'HELLO', 'World', 'world', 'Python']\nunique_lower = {w.lower() for w in words}\nprint(f'unique={sorted(unique_lower)}')",
                "count = len(unique_lower)\nprint(f'count={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "unique=['hello', 'python', 'world']" in nb_runner.get_output(1)
        assert "count=3" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "count=3" in nb_runner.get_output(2)
