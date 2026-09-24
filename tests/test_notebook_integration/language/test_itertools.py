"""itertools, groupby and accumulate across cells."""

import textwrap

import pytest


# List accumulation and cross-cell aggregation.
#
# Tests patterns where data is built across multiple cells then aggregated.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestAccumulationAggregation:
    """Cross-cell accumulation and aggregation patterns."""

    def test_build_list_across_cells(self, nb_runner):
        """Build list in separate cells, aggregate in final."""
        nb_runner.create_notebook(
            [
                "part1 = [1, 2, 3]",
                "part2 = [4, 5, 6]",
                "part3 = [7, 8, 9]",
                "combined = part1 + part2 + part3\ntotal = sum(combined)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 45" in nb_runner.get_output(4)

        nb_runner.set_cell_source(2, "part2 = [40, 50, 60]")
        nb_runner.run_all()
        # 1+2+3+40+50+60+7+8+9 = 180
        assert "total = 180" in nb_runner.get_output(4)

    def test_dict_merge_across_cells(self, nb_runner):
        """Build dict across cells, query in final."""
        nb_runner.create_notebook(
            [
                "user_data = {'name': 'Alice', 'age': 30}",
                "settings = {'theme': 'dark', 'lang': 'en'}",
                "profile = {**user_data, **settings}\nprint(f'profile = {sorted(profile.items())}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "('name', 'Alice')" in out
        assert "('theme', 'dark')" in out

        nb_runner.set_cell_source(2, "settings = {'theme': 'light', 'lang': 'de'}")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "('theme', 'light')" in out2
        assert "('lang', 'de')" in out2

    def test_reduce_across_cells(self, nb_runner):
        """Multiple reduction steps across cells."""
        nb_runner.create_notebook(
            [
                "data = [10, 20, 30, 40, 50]",
                "filtered = [x for x in data if x >= 20]",
                "squared = [x**2 for x in filtered]",
                "avg = sum(squared) / len(squared)\nprint(f'avg = {avg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # filtered=[20,30,40,50], squared=[400,900,1600,2500], avg=5400/4=1350
        assert "avg = 1350.0" in nb_runner.get_output(4)

        nb_runner.set_cell_source(1, "data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]")
        nb_runner.run_all()
        # filtered=[20..100], squared, avg
        out = nb_runner.get_output(4)
        assert "avg =" in out


# Interaction test: itertools accumulate with custom function.
# Tests itertools.accumulate with operator.mul, custom functions,
# initial value, and cross-cell running computation pipelines.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestAccumulateCustomFunc:
    """Test itertools.accumulate with custom function across cells."""

    def test_accumulate_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: running sum and product
                "from itertools import accumulate\nimport operator\nnums = [1, 2, 3, 4, 5]\nrunning_sum = list(accumulate(nums))\nrunning_prod = list(accumulate(nums, operator.mul))\nprint(f'sum={running_sum}')\nprint(f'prod={running_prod}')",
                # Cell 2: custom max accumulate
                "running_max = list(accumulate(nums, max))\nprint(f'max={running_max}')",
                # Cell 3: with initial value
                "with_init = list(accumulate(nums, operator.add, initial=100))\nprint(f'with_init={with_init}')\nprint(f'final={with_init[-1]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "sum=[1, 3, 6, 10, 15]" in out1
        assert "prod=[1, 2, 6, 24, 120]" in out1
        out2 = nb_runner.get_output(2)
        assert "max=[1, 2, 3, 4, 5]" in out2
        out3 = nb_runner.get_output(3)
        assert "with_init=[100, 101, 103, 106, 110, 115]" in out3
        assert "final=115" in out3

    def test_accumulate_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import accumulate\ndata = [10, 20, 30]\nresult = list(accumulate(data))\nprint(f'result={result}')",
                "last = result[-1]\nprint(f'last={last}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[10, 30, 60]" in nb_runner.get_output(1)
        assert "last=60" in nb_runner.get_output(2)

        # Edit data
        nb_runner.set_cell_source(
            1,
            "from itertools import accumulate\ndata = [10, 20, 30, 40]\nresult = list(accumulate(data))\nprint(f'result={result}')",
        )
        nb_runner.run_cells([1, 2])
        assert "last=100" in nb_runner.get_output(2)

    def test_accumulate_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import accumulate\nimport operator\nfactors = [2, 3, 5, 7]\nrunning = list(accumulate(factors, operator.mul))\nprint(f'running={running}')",
                "final_prod = running[-1]\nprint(f'final={final_prod}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "running=[2, 6, 30, 210]" in nb_runner.get_output(1)
        assert "final=210" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "final=210" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestItertoolsAccumulateTakewhile:
    """itertools accumulate and takewhile."""

    def test_accumulate_running_sum(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import itertools",
                "data = [1, 2, 3, 4, 5]\nrunning = list(itertools.accumulate(data))\nprint(f'running={running}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "running=[1, 3, 6, 10, 15]" in nb_runner.get_output(2)

    def test_takewhile_dropwhile(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import itertools",
                "data = [1, 3, 5, 2, 4, 6]\ntaken = list(itertools.takewhile(lambda x: x < 4, data))\ndropped = list(itertools.dropwhile(lambda x: x < 4, data))\nprint(f'taken={taken} dropped={dropped}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "taken=[1, 3]" in out
        assert "dropped=[5, 2, 4, 6]" in out

    def test_accumulate_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import itertools\nimport operator",
                "vals = [2, 3, 4]\nprods = list(itertools.accumulate(vals, operator.mul))\nprint(f'prods={prods}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "prods=[2, 6, 24]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2, "vals = [1, 2, 3, 4]\nprods = list(itertools.accumulate(vals, operator.mul))\nprint(f'prods={prods}')"
        )
        nb_runner.run_all()
        assert "prods=[1, 2, 6, 24]" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestFunctoolsReduceAccumulate:
    """functools.reduce and accumulate patterns."""

    def test_reduce_sum(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import reduce\nnums = [1, 2, 3, 4, 5]",
                "total = reduce(lambda a, b: a + b, nums)\nproduct = reduce(lambda a, b: a * b, nums)\nprint(f'total={total} product={product}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=15" in nb_runner.get_output(2)
        assert "product=120" in nb_runner.get_output(2)

    def test_reduce_with_initial(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import reduce\nitems = ['a', 'b', 'c']",
                "result = reduce(lambda acc, x: acc + '-' + x, items, 'start')\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=start-a-b-c" in nb_runner.get_output(2)

    def test_itertools_accumulate(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import accumulate\nimport operator\nvals = [1, 2, 3, 4, 5]",
                "running_sum = list(accumulate(vals))\nrunning_product = list(accumulate(vals, operator.mul))\nprint(f'sums={running_sum} products={running_product}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "sums=[1, 3, 6, 10, 15]" in out
        assert "products=[1, 2, 6, 24, 120]" in out


# functools.reduce and accumulate patterns with caching.
# Tests reduce, accumulate, and edit propagation.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestReduceAccumulate:
    """Test functools.reduce and itertools.accumulate caching."""

    def test_reduce_sum(self, nb_runner):
        """functools.reduce for summation with caching."""
        nb_runner.create_notebook(
            [
                "from functools import reduce",
                "nums = [1, 2, 3, 4, 5]",
                "total = reduce(lambda a, b: a + b, nums)",
                "print(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=15" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "total=15" in out2

    def test_reduce_edit(self, nb_runner):
        """Edit input, verify reduce result changes."""
        nb_runner.create_notebook(
            [
                "from functools import reduce",
                "nums = [2, 3, 4]",
                "product = reduce(lambda a, b: a * b, nums)",
                "print(f'product={product}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "product=24" in out

        nb_runner.set_cell_source(2, "nums = [2, 3, 4, 5]")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "product=120" in out2

    def test_accumulate_pattern(self, nb_runner):
        """itertools.accumulate running totals."""
        nb_runner.create_notebook(
            [
                "from itertools import accumulate",
                "payments = [100, 200, 150, 300]",
                "running = list(accumulate(payments))\nlast = running[-1]",
                "print(f'last={last}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "last=750" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "last=750" in out2


# Interaction test: itertools.chain.from_iterable with nested data.
# Tests chain.from_iterable for flattening nested structures,
# combined with map and filter across cells.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestChainFromIterable:
    """Test itertools.chain.from_iterable across cells."""

    def test_chain_from_iterable(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: flatten nested lists
                "import itertools\nnested = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]\nflat = list(itertools.chain.from_iterable(nested))\nprint(f'flat={flat}')\nprint(f'len={len(flat)}')",
                # Cell 2: flatten with transformation
                "words = [['hello', 'world'], ['foo', 'bar', 'baz']]\nall_chars = list(itertools.chain.from_iterable(w.upper() for w in itertools.chain.from_iterable(words)))\nunique_chars = sorted(set(all_chars))\nprint(f'unique_count={len(unique_chars)}')",
                # Cell 3: combine
                "total = sum(flat)\nchar_count = len(all_chars)\nprint(f'sum={total}')\nprint(f'chars={char_count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "flat=[1, 2, 3, 4, 5, 6, 7, 8, 9]" in out1
        assert "len=9" in out1
        out3 = nb_runner.get_output(3)
        assert "sum=45" in out3

    def test_chain_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import itertools\ngroups = [[10, 20], [30, 40], [50]]\nflat = list(itertools.chain.from_iterable(groups))\nprint(f'flat={flat}')",
                "total = sum(flat)\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=150" in nb_runner.get_output(2)

        # Add more groups
        nb_runner.set_cell_source(
            1,
            "import itertools\ngroups = [[10, 20], [30, 40], [50], [60, 70]]\nflat = list(itertools.chain.from_iterable(groups))\nprint(f'flat={flat}')",
        )
        nb_runner.run_cells([1, 2])
        assert "total=280" in nb_runner.get_output(2)

    def test_chain_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import itertools\npairs = [(1, 'a'), (2, 'b'), (3, 'c')]\nflat = list(itertools.chain.from_iterable(pairs))\nprint(f'flat={flat}')",
                "strs = [x for x in flat if isinstance(x, str)]\nprint(f'strs={strs}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "strs=['a', 'b', 'c']" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "strs=['a', 'b', 'c']" in nb_runner.get_output(2)


# Interaction test: itertools chain with generators and lazy evaluation.
# Tests chain with generator expressions, lazy flattening,
# and cross-cell lazy pipeline patterns.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestChainLazyGenerator:
    """Test itertools chain with generators across cells."""

    def test_chain_lazy(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: chain generators
                "from itertools import chain\ndef gen_range(start, end):\n    for i in range(start, end):\n        yield i * i\n\nchained = list(chain(gen_range(1, 4), gen_range(4, 7)))\nprint(f'chained={chained}')",
                # Cell 2: chain with filter
                "evens = [x for x in chained if x % 2 == 0]\nprint(f'evens={evens}')",
                # Cell 3: sum from chained
                "total = sum(chained)\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        # 1, 4, 9, 16, 25, 36
        assert "chained=[1, 4, 9, 16, 25, 36]" in out1
        out2 = nb_runner.get_output(2)
        assert "evens=[4, 16, 36]" in out2
        out3 = nb_runner.get_output(3)
        assert "total=91" in out3

    def test_chain_lazy_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import chain\na = [1, 2, 3]\nb = [4, 5, 6]\nresult = list(chain(a, b))\nprint(f'result={result}')",
                "length = len(result)\nprint(f'length={length}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "length=6" in nb_runner.get_output(2)

        # Edit to add more
        nb_runner.set_cell_source(
            1,
            "from itertools import chain\na = [1, 2, 3]\nb = [4, 5, 6]\nc = [7, 8]\nresult = list(chain(a, b, c))\nprint(f'result={result}')",
        )
        nb_runner.run_cells([1, 2])
        assert "length=8" in nb_runner.get_output(2)

    def test_chain_lazy_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import chain\nwords = list(chain(['a', 'b'], ['c', 'd'], ['e']))\nprint(f'words={words}')",
                "joined = ''.join(words)\nprint(f'joined={joined}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "words=['a', 'b', 'c', 'd', 'e']" in nb_runner.get_output(1)
        assert "joined=abcde" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "joined=abcde" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestItertoolsChainFlat:
    """itertools.chain and chain.from_iterable."""

    def test_chain_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import chain\na = [1, 2]\nb = [3, 4]\nc = [5, 6]",
                "result = list(chain(a, b, c))\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[1, 2, 3, 4, 5, 6]" in nb_runner.get_output(2)

    def test_chain_from_iterable(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import chain\nnested = [[1, 2], [3, 4], [5]]",
                "flat = list(chain.from_iterable(nested))\nprint(f'flat={flat}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "flat=[1, 2, 3, 4, 5]" in nb_runner.get_output(2)

    def test_chain_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import chain\nx = ['a', 'b']\ny = ['c']",
                "combined = list(chain(x, y))\nprint(f'combined={combined}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "combined=['a', 'b', 'c']" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from itertools import chain\nx = ['x', 'y', 'z']\ny = ['w']")
        nb_runner.run_all()
        assert "combined=['x', 'y', 'z', 'w']" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestItertoolsChainFromProduct:
    """itertools chain from iterable and product repeat."""

    def test_chain_from_iterable(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import itertools",
                "nested = [[1, 2], [3, 4], [5]]\nflat = list(itertools.chain.from_iterable(nested))\nprint(f'flat={flat} sum={sum(flat)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "flat=[1, 2, 3, 4, 5]" in out
        assert "sum=15" in out

    def test_product_repeat(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import itertools",
                "bits = list(itertools.product([0, 1], repeat=3))\nprint(f'combos={len(bits)} first={bits[0]} last={bits[-1]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "combos=8" in out
        assert "(0, 0, 0)" in out
        assert "(1, 1, 1)" in out

    def test_chain_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import itertools",
                "data = list(itertools.chain.from_iterable([[1], [2]]))\nprint(f'data={data}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "data=[1, 2]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2, "data = list(itertools.chain.from_iterable([[10, 20], [30]]))\nprint(f'data={data}')"
        )
        nb_runner.run_all()
        assert "data=[10, 20, 30]" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestItertoolsChainProduct:
    """itertools.chain, product, starmap combinations."""

    def test_product_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import product\ncolors = ['red', 'blue']\nsizes = ['S', 'M']",
                "combos = list(product(colors, sizes))\nprint(f'combos={combos}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "('red', 'S')" in nb_runner.get_output(2)
        assert "('blue', 'M')" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "from itertools import product\ncolors = ['green']\nsizes = ['L', 'XL']")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "('green', 'L')" in out
        assert "('green', 'XL')" in out

    def test_starmap(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import starmap\npairs = [(2, 3), (4, 5), (6, 7)]",
                "products = list(starmap(lambda a, b: a * b, pairs))\nprint(f'products={products}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "products=[6, 20, 42]" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestCombinationsPermutations:
    """itertools combinations permutations."""

    def test_combinations(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import combinations",
                "items = ['A', 'B', 'C', 'D']\ncomb2 = list(combinations(items, 2))\ncomb3 = list(combinations(items, 3))\nprint(f'comb2_count={len(comb2)} comb3_count={len(comb3)}')\nprint(f'comb2={comb2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "comb2_count=6" in out
        assert "comb3_count=4" in out

    def test_permutations(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import permutations",
                "items = [1, 2, 3]\nperms = list(permutations(items))\nprint(f'count={len(perms)} first={perms[0]} last={perms[-1]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "count=6" in out
        assert "first=(1, 2, 3)" in out
        assert "last=(3, 2, 1)" in out

    def test_combinations_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import combinations",
                "result = list(combinations([1, 2, 3], 2))\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[(1, 2), (1, 3), (2, 3)]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "result = list(combinations([1, 2, 3, 4], 2))\nprint(f'count={len(result)}')")
        nb_runner.run_all()
        assert "count=6" in nb_runner.get_output(2)


# itertools combinatorial patterns with caching.
# Tests combinations, permutations, product, chain, and edit propagation.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestItertoolsCombinatorial:
    """Test itertools combinatorial operation caching."""

    def test_combinations_basic(self, nb_runner):
        """itertools.combinations with caching."""
        nb_runner.create_notebook(
            [
                "from itertools import combinations",
                "items = ['a', 'b', 'c', 'd']",
                "combos = list(combinations(items, 2))\ncount = len(combos)",
                "print(f'count={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "count=6" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "count=6" in out2

    def test_permutations_edit(self, nb_runner):
        """Edit input, verify permutations count changes."""
        nb_runner.create_notebook(
            [
                "from itertools import permutations",
                "items = [1, 2, 3]",
                "perms = list(permutations(items))\ncount = len(perms)",
                "print(f'count={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "count=6" in out

        nb_runner.set_cell_source(2, "items = [1, 2, 3, 4]")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "count=24" in out2

    def test_chain_flatten(self, nb_runner):
        """itertools.chain to flatten nested lists."""
        nb_runner.create_notebook(
            [
                "from itertools import chain",
                "lists = [[1, 2], [3, 4], [5]]",
                "flat = list(chain.from_iterable(lists))\ntotal = sum(flat)",
                "print(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=15" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "total=15" in out2


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestItertoolsProductCombs:
    """itertools.product and combinations_with_replacement."""

    def test_product(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import product\na = [1, 2]\nb = ['x', 'y']",
                "result = list(product(a, b))\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[(1, 'x'), (1, 'y'), (2, 'x'), (2, 'y')]" in nb_runner.get_output(2)

    def test_combinations_with_replacement(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import combinations_with_replacement\nitems = ['a', 'b', 'c']",
                "result = list(combinations_with_replacement(items, 2))\ncount = len(result)\nprint(f'count={count} first={result[0]} last={result[-1]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "count=6" in out
        assert "first=('a', 'a')" in out
        assert "last=('c', 'c')" in out

    def test_product_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import product\ncolors = ['R', 'G']\nsizes = ['S', 'L']",
                "combos = list(product(colors, sizes))\ncount = len(combos)\nprint(f'count={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=4" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from itertools import product\ncolors = ['R', 'G', 'B']\nsizes = ['S', 'M', 'L']")
        nb_runner.run_all()
        assert "count=9" in nb_runner.get_output(2)


# Interaction test: itertools product and combinations_with_replacement.
# Tests cartesian products, combinations with replacement,
# and cross-cell combinatorial analysis.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestProductCombReplace:
    """Test itertools product and combinations_with_replacement across cells."""

    def test_product_comb_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: product
                "from itertools import product, combinations_with_replacement\ncolors = ['R', 'G', 'B']\nsizes = ['S', 'M', 'L']\ncombos = list(product(colors, sizes))\nprint(f'product_count={len(combos)}')\nprint(f'first={combos[0]}')\nprint(f'last={combos[-1]}')",
                # Cell 2: combinations with replacement
                "coins = [1, 5, 10]\nways = list(combinations_with_replacement(coins, 2))\nprint(f'ways_count={len(ways)}')\nfor w in ways:\n    print(f'pair={w} sum={sum(w)}')",
                # Cell 3: filter products
                "matching = [(c, s) for c, s in combos if c == 'R' or s == 'L']\nprint(f'matching_count={len(matching)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "product_count=9" in out1
        assert "first=('R', 'S')" in out1
        assert "last=('B', 'L')" in out1
        out2 = nb_runner.get_output(2)
        assert "ways_count=6" in out2
        out3 = nb_runner.get_output(3)
        assert "matching_count=5" in out3

    def test_product_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import product\ndice = [1, 2, 3, 4, 5, 6]\nrolls = list(product(dice, repeat=2))\ntotal = len(rolls)\nprint(f'total={total}')",
                "sevens = [r for r in rolls if sum(r) == 7]\nprint(f'sevens={len(sevens)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=36" in nb_runner.get_output(1)
        assert "sevens=6" in nb_runner.get_output(2)

        # Edit to 3 dice
        nb_runner.set_cell_source(
            1,
            "from itertools import product\ndice = [1, 2, 3, 4, 5, 6]\nrolls = list(product(dice, repeat=3))\ntotal = len(rolls)\nprint(f'total={total}')",
        )
        nb_runner.set_cell_source(
            2, "sevens = [r for r in rolls if sum(r) == 7]  # 3-dice\nprint(f'sevens={len(sevens)}')"
        )
        nb_runner.run_cells([1, 2])
        assert "total=216" in nb_runner.get_output(1)
        assert "sevens=15" in nb_runner.get_output(2)

    def test_product_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import combinations_with_replacement\nitems = ['a', 'b', 'c']\npairs = list(combinations_with_replacement(items, 2))\nprint(f'count={len(pairs)}')",
                "as_strings = ['+'.join(p) for p in pairs]\nprint(f'strings={as_strings}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=6" in nb_runner.get_output(1)

        # Re-run - cache
        nb_runner.run_all()
        assert "count=6" in nb_runner.get_output(1)


# Interaction test: itertools.groupby with key function.
# Tests groupby with sorted data, key extraction, group aggregation,
# and cross-cell grouped data processing.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestGroupbyKeyFunction:
    """Test itertools.groupby with key function across cells."""

    def test_groupby_aggregation(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: sort and group
                "from itertools import groupby\ndata = [('A', 10), ('B', 20), ('A', 30), ('B', 40), ('C', 50)]\nsorted_data = sorted(data, key=lambda x: x[0])\ngroups = {k: [v for _, v in g] for k, g in groupby(sorted_data, key=lambda x: x[0])}\nprint(f'groups={groups}')",
                # Cell 2: aggregate per group
                "sums = {k: sum(v) for k, v in groups.items()}\navgs = {k: sum(v)/len(v) for k, v in groups.items()}\nprint(f'sums={sums}')\nprint(f'avgs={avgs}')",
                # Cell 3: find best group
                "best = max(sums, key=sums.get)\nprint(f'best={best}')\nprint(f'best_sum={sums[best]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "'A': [10, 30]" in out1
        assert "'B': [20, 40]" in out1
        assert "'C': [50]" in out1
        out2 = nb_runner.get_output(2)
        assert "'A': 40" in out2
        assert "'B': 60" in out2
        out3 = nb_runner.get_output(3)
        assert "best=B" in out3
        assert "best_sum=60" in out3

    def test_groupby_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import groupby\nwords = ['apple', 'avocado', 'banana', 'blueberry', 'cherry']\nby_letter = {k: list(g) for k, g in groupby(sorted(words), key=lambda w: w[0])}\nprint(f'groups={by_letter}')",
                "counts = {k: len(v) for k, v in by_letter.items()}\nprint(f'counts={counts}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': ['apple', 'avocado']" in nb_runner.get_output(1)
        assert "'a': 2" in nb_runner.get_output(2)

        # Add more words
        nb_runner.set_cell_source(
            1,
            "from itertools import groupby\nwords = ['apple', 'avocado', 'apricot', 'banana', 'blueberry', 'cherry', 'coconut']\nby_letter = {k: list(g) for k, g in groupby(sorted(words), key=lambda w: w[0])}\nprint(f'groups={by_letter}')",
        )
        nb_runner.run_cells([1, 2])
        assert "'a': 3" in nb_runner.get_output(2)
        assert "'c': 2" in nb_runner.get_output(2)

    def test_groupby_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import groupby\nnums = [1, 1, 2, 2, 2, 3, 3]\nruns = [(k, len(list(g))) for k, g in groupby(nums)]\nprint(f'runs={runs}')",
                "longest_run = max(runs, key=lambda x: x[1])\nprint(f'longest={longest_run}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "runs=[(1, 2), (2, 3), (3, 2)]" in nb_runner.get_output(1)
        assert "longest=(2, 3)" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "longest=(2, 3)" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestItertoolsGroupby:
    """itertools.groupby with sorting and key functions."""

    def test_groupby_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import groupby\ndata = [('A', 1), ('A', 2), ('B', 3), ('B', 4), ('C', 5)]",
                "groups = {k: list(v) for k, v in groupby(data, key=lambda x: x[0])}\nprint(f'keys={sorted(groups.keys())}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys=['A', 'B', 'C']" in nb_runner.get_output(2)

    def test_groupby_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import groupby\nnums = [1, 1, 2, 2, 2, 3, 3]",
                "runs = [(k, len(list(g))) for k, g in groupby(nums)]\nprint(f'runs={runs}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "runs=[(1, 2), (2, 3), (3, 2)]" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "from itertools import groupby\nnums = [5, 5, 5, 1, 1]")
        nb_runner.run_all()
        assert "runs=[(5, 3), (1, 2)]" in nb_runner.get_output(2)

    def test_groupby_with_sort(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import groupby\nitems = [('b', 2), ('a', 1), ('b', 3), ('a', 4)]",
                "sorted_items = sorted(items, key=lambda x: x[0])\ngrouped = {k: [v for _, v in g] for k, g in groupby(sorted_items, key=lambda x: x[0])}\nprint(f'grouped={dict(sorted(grouped.items()))}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': [1, 4]" in nb_runner.get_output(2)
        assert "'b': [2, 3]" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestItertoolsGroupbySorted:
    """itertools groupby with sorted data."""

    def test_groupby_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import groupby",
                "data = [('A', 1), ('A', 2), ('B', 3), ('B', 4), ('C', 5)]\ngroups = {k: [v for _, v in g] for k, g in groupby(data, key=lambda x: x[0])}\nprint(f'groups={groups}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'A': [1, 2]" in out
        assert "'B': [3, 4]" in out
        assert "'C': [5]" in out

    def test_groupby_words(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import groupby",
                "words = sorted(['apple', 'ant', 'banana', 'bat', 'cherry'])\ngroups = {k: list(g) for k, g in groupby(words, key=lambda w: w[0])}\ncounts = {k: len(v) for k, v in groups.items()}\nprint(f'counts={counts}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'a': 2" in out
        assert "'b': 2" in out
        assert "'c': 1" in out

    def test_groupby_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import groupby",
                "nums = [1, 1, 2, 2, 2, 3]\ngroups = [(k, len(list(g))) for k, g in groupby(nums)]\nprint(f'groups={groups}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "groups=[(1, 2), (2, 3), (3, 1)]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2,
            "nums = [1, 1, 1, 2, 3, 3]\ngroups = [(k, len(list(g))) for k, g in groupby(nums)]\nprint(f'groups={groups}')",
        )
        nb_runner.run_all()
        assert "groups=[(1, 3), (2, 1), (3, 2)]" in nb_runner.get_output(2)


# Aggregation with groupby interaction tests.
# Tests that editing groupby logic or data properly invalidates
# aggregated results downstream.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestGroupbyAggregationInteraction:
    """Test groupby/aggregation patterns with cache invalidation."""

    def test_manual_groupby_edit(self, nb_runner):
        """Editing data grouped manually should propagate."""
        nb_runner.create_notebook(
            [
                "data = [('a', 1), ('b', 2), ('a', 3), ('b', 4), ('a', 5)]",
                "groups = {}\nfor key, val in data:\n    groups.setdefault(key, []).append(val)",
                "sums = {k: sum(v) for k, v in sorted(groups.items())}",
                "print(f'sums={sums}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "sums={'a': 9, 'b': 6}" in out

        nb_runner.set_cell_source(1, "data = [('x', 10), ('y', 20), ('x', 30)]")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "sums={'x': 40, 'y': 20}" in out

    def test_itertools_groupby_edit(self, nb_runner):
        """Editing sorted data for itertools.groupby should propagate."""
        nb_runner.create_notebook(
            [
                "from itertools import groupby\ndata = sorted([('a', 1), ('b', 2), ('a', 3), ('b', 4)])",
                "grouped = {k: [v for _, v in g] for k, g in groupby(data, key=lambda x: x[0])}",
                "counts = {k: len(v) for k, v in sorted(grouped.items())}",
                "print(f'counts={counts}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "counts={'a': 2, 'b': 2}" in out

        nb_runner.set_cell_source(
            1, "from itertools import groupby\ndata = sorted([('a', 1), ('a', 2), ('a', 3), ('b', 4)])"
        )
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "counts={'a': 3, 'b': 1}" in out


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestItertoolsSliceTakewhile:
    """itertools.islice and takewhile/dropwhile."""

    def test_islice(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import islice, count\nnatural = count(1)",
                "first10 = list(islice(natural, 10))\nprint(f'first10={first10}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first10=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]" in nb_runner.get_output(2)

    def test_takewhile_dropwhile(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import takewhile, dropwhile\nnums = [1, 3, 5, 7, 2, 4, 6]",
                "taken = list(takewhile(lambda x: x < 6, nums))\ndropped = list(dropwhile(lambda x: x < 6, nums))\nprint(f'taken={taken} dropped={dropped}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "taken=[1, 3, 5]" in nb_runner.get_output(2)
        assert "dropped=[7, 2, 4, 6]" in nb_runner.get_output(2)

    def test_islice_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import islice\ndata = list(range(100))",
                "chunk = list(islice(data, 5, 10))\nprint(f'chunk={chunk}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "chunk=[5, 6, 7, 8, 9]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "chunk = list(islice(data, 90, 95))\nprint(f'chunk={chunk}')")
        nb_runner.run_all()
        assert "chunk=[90, 91, 92, 93, 94]" in nb_runner.get_output(2)


# Interaction test: itertools.tee and islice with multiple consumers.
# Tests tee for creating independent iterators, islice for windows,
# and cross-cell iterator consumption patterns.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestItertoolsTeeIslice:
    """Test itertools.tee and islice across cells."""

    def test_tee_islice_operations(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: create and tee an iterator
                "import itertools\noriginal = iter(range(10))\nit1, it2, it3 = itertools.tee(original, 3)\nprint('teed=3')",
                # Cell 2: consume differently using islice
                "first_5 = list(itertools.islice(it1, 5))\nevens = list(itertools.islice(it2, 0, 10, 2))\nlast_3 = list(itertools.islice(it3, 7, 10))\nprint(f'first_5={first_5}')\nprint(f'evens={evens}')\nprint(f'last_3={last_3}')",
                # Cell 3: combine results (union of [0..4] + [0,2,4,6,8] + [7,8,9] = 9 unique, missing 5)
                "all_unique = sorted(set(first_5 + evens + last_3))\nprint(f'unique_count={len(all_unique)}')\nprint(f'has_five={5 in all_unique}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "first_5=[0, 1, 2, 3, 4]" in out2
        assert "evens=[0, 2, 4, 6, 8]" in out2
        assert "last_3=[7, 8, 9]" in out2
        out3 = nb_runner.get_output(3)
        assert "unique_count=9" in out3
        assert "has_five=False" in out3

    def test_tee_edit_range(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import itertools\noriginal = iter(range(10))\nit1, it2 = itertools.tee(original, 2)\nprint('teed=2')",
                "a = list(itertools.islice(it1, 3))\nb = list(itertools.islice(it2, 3, 6))\nprint(f'a={a}')\nprint(f'b={b}')",
                "combined = a + b\ntotal = sum(combined)\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=[0, 1, 2]" in nb_runner.get_output(2)
        assert "b=[3, 4, 5]" in nb_runner.get_output(2)
        assert "total=15" in nb_runner.get_output(3)

        # Change range
        nb_runner.set_cell_source(
            1, "import itertools\noriginal = iter(range(20))\nit1, it2 = itertools.tee(original, 2)\nprint('teed=2')"
        )
        nb_runner.run_cells([1, 2, 3])
        assert "a=[0, 1, 2]" in nb_runner.get_output(2)
        assert "b=[3, 4, 5]" in nb_runner.get_output(2)
        # Same slices so same results
        assert "total=15" in nb_runner.get_output(3)

    def test_tee_islice_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import itertools\ndata = iter([10, 20, 30, 40, 50])\na, b = itertools.tee(data, 2)\nprint('teed')",
                "head = list(itertools.islice(a, 2))\ntail = list(itertools.islice(b, 3, 5))\nprint(f'head={head}')\nprint(f'tail={tail}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "head=[10, 20]" in out
        assert "tail=[40, 50]" in out

        # Re-run - cache
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "head=[10, 20]" in out


# Interaction test: itertools starmap and repeat.
# Tests starmap for unpacking arguments, repeat for infinite iterators,
# islice for limiting, and cross-cell functional patterns.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestStarmapRepeat:
    """Test itertools starmap and repeat across cells."""

    def test_starmap_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: starmap with pairs
                "from itertools import starmap, repeat, islice\npairs = [(2, 3), (4, 5), (6, 7)]\nproducts = list(starmap(lambda x, y: x * y, pairs))\nprint(f'products={products}')",
                # Cell 2: starmap with pow
                "powers = list(starmap(pow, [(2, 10), (3, 5), (10, 3)]))\nprint(f'powers={powers}')",
                # Cell 3: repeat + islice
                "repeated = list(islice(repeat('hello', 5), 5))\nprint(f'repeated={repeated}')\nprint(f'count={len(repeated)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "products=[6, 20, 42]" in out1
        out2 = nb_runner.get_output(2)
        assert "powers=[1024, 243, 1000]" in out2
        out3 = nb_runner.get_output(3)
        assert "count=5" in out3

    def test_starmap_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import starmap\ncoords = [(0, 0), (3, 4), (6, 8)]\nimport math\ndistances = list(starmap(math.hypot, coords))\nprint(f'distances={distances}')",
                "total_dist = sum(distances)\nprint(f'total={total_dist}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "distances=[0.0, 5.0, 10.0]" in nb_runner.get_output(1)
        assert "total=15.0" in nb_runner.get_output(2)

        # Edit coords
        nb_runner.set_cell_source(
            1,
            "from itertools import starmap\ncoords = [(0, 0), (3, 4), (5, 12)]\nimport math\ndistances = list(starmap(math.hypot, coords))\nprint(f'distances={distances}')",
        )
        nb_runner.run_cells([1, 2])
        assert "distances=[0.0, 5.0, 13.0]" in nb_runner.get_output(1)
        assert "total=18.0" in nb_runner.get_output(2)

    def test_starmap_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import starmap\ndata = [(1, 'a'), (2, 'b'), (3, 'c')]\nformatted = list(starmap(lambda n, s: f'{n}:{s}', data))\nprint(f'formatted={formatted}')",
                "joined = ', '.join(formatted)\nprint(f'joined={joined}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "formatted=['1:a', '2:b', '3:c']" in nb_runner.get_output(1)
        assert "joined=1:a, 2:b, 3:c" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "joined=1:a, 2:b, 3:c" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestItertoolsStarmapRepeat:
    """itertools.starmap and repeat patterns."""

    def test_repeat(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import repeat\nvals = list(repeat('x', 5))",
                "print(f'vals={vals}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "vals=['x', 'x', 'x', 'x', 'x']" in nb_runner.get_output(2)

    def test_starmap_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import starmap\ndata = [(1, 10), (2, 20), (3, 30)]",
                "sums = list(starmap(lambda a, b: a + b, data))\nprint(f'sums={sums}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sums=[11, 22, 33]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from itertools import starmap\ndata = [(5, 5), (10, 10)]")
        nb_runner.run_all()
        assert "sums=[10, 20]" in nb_runner.get_output(2)


# Itertools advanced — cash caching with itertools combinatorial patterns.
@pytest.mark.stress
class TestItertoolsInfinite:
    """Test infinite iterator patterns."""

    def test_itertools_propagation(self, nb_runner):
        """Itertools results propagate on change."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                from itertools import combinations
                items = ['A', 'B', 'C']
                pairs = list(combinations(items, 2))
            """),
                textwrap.dedent("""\
                labels = [f"{a}-{b}" for a, b in pairs]
                print(f"labels={labels}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "labels=['A-B', 'A-C', 'B-C']" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            from itertools import combinations
            items = ['X', 'Y', 'Z', 'W']
            pairs = list(combinations(items, 2))
        """),
        )
        nb_runner.run_cells([1, 2])
        out = nb_runner.get_output(2)
        assert "X-Y" in out
        assert "Z-W" in out
