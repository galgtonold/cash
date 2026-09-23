"""Tuple unpacking, starred assignment and multiple return values across cells."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress]


# Multi-assignment & augmented assignment interaction tests.
#
# Tests editing multi-target assignments, augmented assignments (+=, *=),
# and walrus operator patterns.
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestMultiAssignmentEdits:
    """Editing multi-assignment patterns."""

    def test_edit_multi_assign(self, nb_runner):
        """Edit a multi-assignment statement."""
        nb_runner.create_notebook(
            [
                "a = b = c = 10  # multi assign",
                "total = a + b + c\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "a = b = c = 20  # multi assign v2")
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)

    def test_edit_swap_assignment(self, nb_runner):
        """Edit a swap assignment."""
        nb_runner.create_notebook(
            [
                "x, y = 1, 2  # swap source",
                "x, y = y, x\nprint(f'x={x} y={y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=2 y=1" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "x, y = 10, 20  # swap source v2")
        nb_runner.run_all()
        assert "x=20 y=10" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestAugmentedAssignmentEdits:
    """Editing augmented assignment operations."""

    def test_edit_augmented_op(self, nb_runner):
        """Edit the augmented assignment operator."""
        nb_runner.create_notebook(
            [
                "val = 10  # augmented source",
                "val += 5\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 15" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "val *= 5\nprint(f'val = {val}')")
        nb_runner.run_all()
        assert "val = 50" in nb_runner.get_output(2)

    def test_edit_augmented_source(self, nb_runner):
        """Edit the source value for augmented assignment."""
        nb_runner.create_notebook(
            [
                "base = 100  # augmented base",
                "base //= 3\nprint(f'base = {base}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "base = 33" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "base = 200  # augmented base v2")
        nb_runner.run_all()
        assert "base = 66" in nb_runner.get_output(2)

    def test_chain_augmented_assignments(self, nb_runner):
        """Chain of augmented assignments across cells."""
        nb_runner.create_notebook(
            [
                "n = 1  # chain augmented start",
                "n += 9  # step 1",
                "n *= 2  # step 2",
                "print(f'n = {n}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 1+9=10, 10*2=20
        assert "n = 20" in nb_runner.get_output(4)

        # Edit middle step
        nb_runner.set_cell_source(2, "n += 99  # step 1 v2")
        nb_runner.run_all()
        # 1+99=100, 100*2=200
        assert "n = 200" in nb_runner.get_output(4)


# Tuple unpacking and multi-return interaction tests.
#
# Tests editing functions that return tuples, and editing
# unpacking patterns in downstream cells.
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestTupleUnpackingEdits:
    """Editing tuple unpacking patterns."""

    def test_edit_multi_return_function(self, nb_runner):
        """Edit a function that returns a tuple."""
        nb_runner.create_notebook(
            [
                "def stats(data):\n    return min(data), max(data)",
                "lo, hi = stats([3, 1, 4, 1, 5])\nprint(f'lo={lo} hi={hi}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lo=1 hi=5" in nb_runner.get_output(2)

        # Change function to return mean too
        nb_runner.set_cell_source(
            1,
            "def stats(data):\n    return min(data), max(data), sum(data)/len(data)",
        )
        nb_runner.set_cell_source(
            2,
            "lo, hi, avg = stats([3, 1, 4, 1, 5])\nprint(f'lo={lo} hi={hi} avg={avg}')",
        )
        nb_runner.run_all()
        assert "lo=1 hi=5 avg=2.8" in nb_runner.get_output(2)

    def test_edit_unpacking_target(self, nb_runner):
        """Edit the variables that receive unpacked values."""
        nb_runner.create_notebook(
            [
                "pair = (10, 20)  # source tuple",
                "a, b = pair\nprint(f'a={a} b={b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=10 b=20" in nb_runner.get_output(2)

        # Change source
        nb_runner.set_cell_source(1, "pair = (100, 200)  # source tuple bigger")
        nb_runner.run_all()
        assert "a=100 b=200" in nb_runner.get_output(2)

    def test_star_unpacking_edit(self, nb_runner):
        """Edit star unpacking patterns."""
        nb_runner.create_notebook(
            [
                "items = [1, 2, 3, 4, 5]  # items to unpack",
                "first, *rest = items\nprint(f'first={first} rest={rest}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first=1 rest=[2, 3, 4, 5]" in nb_runner.get_output(2)

        # Change to different unpacking
        nb_runner.set_cell_source(2, "*start, last = items\nprint(f'start={start} last={last}')")
        nb_runner.run_all()
        assert "start=[1, 2, 3, 4] last=5" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestDictUnpackingEdits:
    """Dict unpacking patterns."""

    def test_edit_dict_values_method(self, nb_runner):
        """Edit dict and verify unpacking updates."""
        nb_runner.create_notebook(
            [
                "config = {'host': 'localhost', 'port': 8080}",
                "host = config['host']\nport = config['port']\nprint(f'addr={host}:{port}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "addr=localhost:8080" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "config = {'host': '0.0.0.0', 'port': 9090}")
        nb_runner.run_all()
        assert "addr=0.0.0.0:9090" in nb_runner.get_output(2)

    def test_nested_unpacking_edit(self, nb_runner):
        """Edit nested tuple unpacking."""
        nb_runner.create_notebook(
            [
                "data = ((1, 2), (3, 4))  # nested tuples",
                "(a, b), (c, d) = data\ntotal = a + b + c + d\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "data = ((10, 20), (30, 40))  # nested tuples bigger")
        nb_runner.run_all()
        assert "total = 100" in nb_runner.get_output(2)


# Multiple variable assignment and swap patterns.
#
# Tests tuple assignment, swap, augmented assignment with edits.
@pytest.mark.timeout(90)
class TestMultiAssignSwap:
    """Multiple variable assignment edit patterns."""

    def test_tuple_swap_edit(self, nb_runner):
        """Edit initial values, swap and downstream reflect."""
        nb_runner.create_notebook(
            [
                "x, y = 10, 20",
                "x, y = y, x\nprint(f'x={x} y={y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=20 y=10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "x, y = 100, 200")
        nb_runner.run_all()
        assert "x=200 y=100" in nb_runner.get_output(2)

    def test_augmented_assignment_edit(self, nb_runner):
        """Edit initial value, augmented assignments chain correctly."""
        nb_runner.create_notebook(
            [
                "val = 10",
                "val = val + 5\nval = val * 2\nprint(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "val = 100")
        nb_runner.run_all()
        assert "val = 210" in nb_runner.get_output(2)

    def test_chained_assignment_edit(self, nb_runner):
        """Edit one var in chained assignment."""
        nb_runner.create_notebook(
            [
                "a = 1\nb = 2\nc = 3",
                "total = a + b + c\navg = total / 3\nprint(f'total={total} avg={avg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=6 avg=2.0" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "a = 10\nb = 20\nc = 30")
        nb_runner.run_all()
        assert "total=60 avg=20.0" in nb_runner.get_output(2)

    def test_multi_target_from_function(self, nb_runner):
        """Edit multi-target assignment from function return."""
        nb_runner.create_notebook(
            [
                "def get_bounds(data):\n    return min(data), max(data)",
                "nums = [5, 2, 8, 1, 9]",
                "lo, hi = get_bounds(nums)\nspan = hi - lo\nprint(f'lo={lo} hi={hi} span={span}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lo=1 hi=9 span=8" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "nums = [100, 200, 300]")
        nb_runner.run_all()
        assert "lo=100 hi=300 span=200" in nb_runner.get_output(3)


# Multi-return function with unpacking interaction tests.
#
# Tests editing functions that return multiple values and
# various unpacking patterns.
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestMultiReturnEdits:
    """Editing multi-return function patterns."""

    def test_edit_returned_values(self, nb_runner):
        """Edit the values returned by a multi-return function."""
        nb_runner.create_notebook(
            [
                "def compute(x):\n    return x * 2, x ** 2, x + 10",
                "a, b, c = compute(5)\nprint(f'a={a} b={b} c={c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=10 b=25 c=15" in nb_runner.get_output(2)

        # Edit function
        nb_runner.set_cell_source(1, "def compute(x):\n    return x * 3, x ** 3, x + 100")
        nb_runner.run_all()
        assert "a=15 b=125 c=105" in nb_runner.get_output(2)

    def test_edit_unpacking_target(self, nb_runner):
        """Edit which returned values are used."""
        nb_runner.create_notebook(
            [
                "def stats(data):\n    return min(data), max(data), sum(data)",
                "lo, hi, total = stats([3, 1, 4, 1, 5])\nprint(f'lo={lo} hi={hi} total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lo=1 hi=5 total=14" in nb_runner.get_output(2)

        # Change input data
        nb_runner.set_cell_source(2, "lo, hi, total = stats([10, 20, 30])\nprint(f'lo={lo} hi={hi} total={total}')")
        nb_runner.run_all()
        assert "lo=10 hi=30 total=60" in nb_runner.get_output(2)

    def test_edit_star_unpacking(self, nb_runner):
        """Edit star unpacking patterns."""
        nb_runner.create_notebook(
            [
                "def get_items():\n    return 1, 2, 3, 4, 5",
                "first, *rest = get_items()\nprint(f'first={first} rest={rest}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first=1 rest=[2, 3, 4, 5]" in nb_runner.get_output(2)

        # Change function
        nb_runner.set_cell_source(1, "def get_items():\n    return 10, 20, 30")
        nb_runner.run_all()
        assert "first=10 rest=[20, 30]" in nb_runner.get_output(2)

    def test_edit_namedtuple_return(self, nb_runner):
        """Edit function returning namedtuple."""
        nb_runner.create_notebook(
            [
                "from collections import namedtuple\nResult = namedtuple('Result', ['value', 'status'])",
                "def process(x):\n    if x > 0:\n        return Result(x * 2, 'ok')\n    return Result(0, 'error')",
                "r = process(5)\nprint(f'value={r.value} status={r.status}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "value=10 status=ok" in nb_runner.get_output(3)

        # Edit function
        nb_runner.set_cell_source(
            2,
            "def process(x):\n    if x > 0:\n        return Result(x ** 2, 'success')\n    return Result(-1, 'fail')",
        )
        nb_runner.run_all()
        assert "value=25 status=success" in nb_runner.get_output(3)


# multiple assignment / unpacking patterns with caching.
# Tests tuple unpacking, star expressions, swap, and edit propagation.
@pytest.mark.integration
@pytest.mark.timeout(90)
class TestMultipleAssignUnpack:
    """Test multiple assignment and unpacking caching."""

    def test_tuple_unpack(self, nb_runner):
        """Basic tuple unpacking with caching."""
        nb_runner.create_notebook(
            [
                "data = (10, 20, 30)",
                "a, b, c = data",
                "total = a + b + c\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "total=60" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "total=60" in out2

    def test_star_unpack_edit(self, nb_runner):
        """Star unpacking with edit propagation."""
        nb_runner.create_notebook(
            [
                "values = [1, 2, 3, 4, 5]",
                "first, *middle, last = values",
                "print(f'first={first} middle={middle} last={last}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "first=1" in out
        assert "middle=[2, 3, 4]" in out
        assert "last=5" in out

        nb_runner.set_cell_source(1, "values = [10, 20, 30]")
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "first=10" in out2
        assert "middle=[20]" in out2
        assert "last=30" in out2

    def test_swap_pattern(self, nb_runner):
        """Variable swap pattern with caching."""
        nb_runner.create_notebook(
            [
                "x = 'hello'\ny = 'world'",
                "x, y = y, x",
                "print(f'x={x} y={y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "x=world" in out
        assert "y=hello" in out

        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "x=world" in out2


@pytest.mark.timeout(90)
class TestMultiReturnCrossCell:
    """multiple return values across cells with edits."""

    def test_function_multi_return(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def stats(data):\n    return min(data), max(data), sum(data) / len(data)",
                "data = [10, 20, 30, 40, 50]",
                "lo, hi, avg = stats(data)\nprint(f'lo={lo} hi={hi} avg={avg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lo=10 hi=50 avg=30.0" in nb_runner.get_output(3)

    def test_multi_return_edit_data(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def analyze(nums):\n    evens = [n for n in nums if n % 2 == 0]\n    odds = [n for n in nums if n % 2 != 0]\n    return evens, odds",
                "nums = [1, 2, 3, 4, 5, 6]",
                "evens, odds = analyze(nums)\nprint(f'evens={evens} odds={odds}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "evens=[2, 4, 6]" in nb_runner.get_output(3)
        assert "odds=[1, 3, 5]" in nb_runner.get_output(3)
        # Edit data
        nb_runner.set_cell_source(2, "nums = [10, 15, 20, 25]")
        nb_runner.run_all()
        assert "evens=[10, 20]" in nb_runner.get_output(3)
        assert "odds=[15, 25]" in nb_runner.get_output(3)

    def test_multi_return_edit_function(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def transform(x):\n    return x * 2, x + 10",
                "a, b = transform(5)\nprint(f'a={a} b={b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=10 b=15" in nb_runner.get_output(2)
        # Edit function
        nb_runner.set_cell_source(1, "def transform(x):\n    return x * 3, x + 100")
        nb_runner.run_all()
        assert "a=15 b=105" in nb_runner.get_output(2)


@pytest.mark.timeout(90)
class TestMultiReturnTupleUnpack:
    """multiple return values and tuple unpacking."""

    def test_multi_return(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def stats(nums):\n    return min(nums), max(nums), sum(nums) / len(nums)",
                "lo, hi, avg = stats([10, 20, 30, 40, 50])\nprint(f'lo={lo} hi={hi} avg={avg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lo=10" in nb_runner.get_output(2)
        assert "hi=50" in nb_runner.get_output(2)
        assert "avg=30.0" in nb_runner.get_output(2)

    def test_star_unpack(self, nb_runner):
        nb_runner.create_notebook(
            [
                "items = [1, 2, 3, 4, 5]",
                "first, *middle, last = items\nprint(f'first={first} middle={middle} last={last}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first=1" in nb_runner.get_output(2)
        assert "middle=[2, 3, 4]" in nb_runner.get_output(2)
        assert "last=5" in nb_runner.get_output(2)

    def test_multi_return_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def divmod_custom(a, b):\n    return a // b, a % b",
                "q, r = divmod_custom(17, 5)\nprint(f'q={q} r={r}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "q=3 r=2" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "q, r = divmod_custom(100, 7)\nprint(f'q={q} r={r}')")
        nb_runner.run_all()
        assert "q=14 r=2" in nb_runner.get_output(2)


# Interaction test: multiple return unpacking with nested tuples.
# Tests complex unpacking patterns with nested structures, star unpacking
# in function returns, and cross-cell value threading.
@pytest.mark.timeout(90)
class TestNestedUnpackReturn:
    """Test complex unpacking patterns across cells."""

    def test_nested_unpack(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: define function with complex return
                "def analyze_data(data):\n    total = sum(data)\n    avg = total / len(data)\n    extremes = (min(data), max(data))\n    spread = extremes[1] - extremes[0]\n    return total, avg, extremes, spread\nprint('analyze_data defined')",
                # Cell 2: unpack nested results
                "data = [10, 20, 30, 40, 50]\ntotal, avg, (lo, hi), spread = analyze_data(data)\nprint(f'total={total}')\nprint(f'avg={avg}')\nprint(f'lo={lo} hi={hi}')\nprint(f'spread={spread}')",
                # Cell 3: use unpacked values
                "normalized = [(x - lo) / spread * 100 for x in data]\nprint(f'norm={[int(n) for n in normalized]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "total=150" in out2
        assert "avg=30.0" in out2
        assert "lo=10 hi=50" in out2
        assert "spread=40" in out2
        out3 = nb_runner.get_output(3)
        assert "norm=[0, 25, 50, 75, 100]" in out3

    def test_nested_unpack_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def stats(nums):\n    s = sorted(nums)\n    return s[0], s[-1], s[len(s)//2]\nprint('stats defined')",
                "lo, hi, med = stats([5, 3, 8, 1, 9])\nprint(f'lo={lo} hi={hi} med={med}')",
                "rng = hi - lo\nprint(f'range={rng}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lo=1 hi=9 med=5" in nb_runner.get_output(2)
        assert "range=8" in nb_runner.get_output(3)

        # Edit data
        nb_runner.set_cell_source(2, "lo, hi, med = stats([10, 20, 30, 40, 50])\nprint(f'lo={lo} hi={hi} med={med}')")
        nb_runner.run_cells([2, 3])
        assert "lo=10 hi=50 med=30" in nb_runner.get_output(2)
        assert "range=40" in nb_runner.get_output(3)

    def test_nested_unpack_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def split_name(full):\n    parts = full.split()\n    first, *middle, last = parts\n    return first, middle, last\nprint('split_name defined')",
                "first, mid, last = split_name('John Michael Smith Jr')\nprint(f'first={first} mid={mid} last={last}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first=John mid=['Michael', 'Smith'] last=Jr" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "first=John mid=['Michael', 'Smith'] last=Jr" in nb_runner.get_output(2)


# Star unpacking and extended iterable unpacking.
#
# Tests *args, **kwargs, and extended unpacking with edits.
@pytest.mark.timeout(90)
class TestStarUnpacking:
    """Star unpacking edit patterns."""

    def test_star_rest_edit(self, nb_runner):
        """Edit list, star unpack head/*rest changes."""
        nb_runner.create_notebook(
            [
                "data = [10, 20, 30, 40, 50]",
                "head, *rest = data\nprint(f'head = {head}, rest = {rest}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "head = 10" in out
        assert "rest = [20, 30, 40, 50]" in out

        nb_runner.set_cell_source(1, "data = [99, 88]")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "head = 99" in out
        assert "rest = [88]" in out

    def test_dict_merge_unpack_edit(self, nb_runner):
        """Edit dict, merge with ** changes."""
        nb_runner.create_notebook(
            [
                "base = {'a': 1, 'b': 2}",
                "extra = {'c': 3}",
                "merged = {**base, **extra}\nprint(f'merged = {merged}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 1" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "base = {'a': 100, 'b': 200}")
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "'a': 100" in out
        assert "'c': 3" in out

    def test_function_args_kwargs_edit(self, nb_runner):
        """Edit function with *args/**kwargs."""
        nb_runner.create_notebook(
            [
                "def combine(*args, **kwargs):\n    return list(args) + list(kwargs.values())",
                "result = combine(1, 2, x=10, y=20)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [1, 2, 10, 20]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def combine(*args, **kwargs):\n    return [a * 2 for a in args] + [v * 3 for v in kwargs.values()]",
        )
        nb_runner.run_all()
        assert "result = [2, 4, 30, 60]" in nb_runner.get_output(2)


@pytest.mark.timeout(90)
class TestTupleUnpackingStarred:
    """tuple unpacking and starred assignment."""

    def test_basic_unpack(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = (10, 20, 30, 40, 50)",
                "first, second, *rest = data\nprint(f'first={first} second={second} rest={rest}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "first=10" in out
        assert "second=20" in out
        assert "rest=[30, 40, 50]" in out

    def test_nested_unpack(self, nb_runner):
        nb_runner.create_notebook(
            [
                "records = [('Alice', 90), ('Bob', 85), ('Carol', 95)]",
                "names = []\nscores = []\nfor name, score in records:\n    names.append(name)\n    scores.append(score)\navg = sum(scores) / len(scores)\nprint(f'names={names} avg={avg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Alice" in out
        assert "avg=90.0" in out

    def test_unpack_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "coords = (1, 2, 3)",
                "x, y, z = coords\nprint(f'x={x} y={y} z={z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=1 y=2 z=3" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "coords = (100, 200, 300)")
        nb_runner.run_all()
        assert "x=100 y=200 z=300" in nb_runner.get_output(2)


@pytest.mark.timeout(90)
class TestUnpackGeneralization:
    """unpacking generalization (**kwargs, *args) across cells."""

    def test_kwargs_merge(self, nb_runner):
        nb_runner.create_notebook(
            [
                "defaults = {'color': 'red', 'size': 10}\noverrides = {'size': 20, 'weight': 5}",
                "merged = {**defaults, **overrides}\nprint(f'merged={dict(sorted(merged.items()))}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "merged={'color': 'red', 'size': 20, 'weight': 5}" in nb_runner.get_output(2)

    def test_args_spread_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "parts = ([1, 2], [3, 4], [5])",
                "combined = [*parts[0], *parts[1], *parts[2]]\nprint(f'combined={combined}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "combined=[1, 2, 3, 4, 5]" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "parts = ([10], [20, 30])")
        nb_runner.set_cell_source(2, "combined = [*parts[0], *parts[1]]\nprint(f'combined={combined}')")
        nb_runner.run_all()
        assert "combined=[10, 20, 30]" in nb_runner.get_output(2)

    def test_func_kwargs(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def build_url(scheme='http', host='localhost', port=80):\n    return f'{scheme}://{host}:{port}'",
                "params = {'scheme': 'https', 'host': 'example.com', 'port': 443}\nurl = build_url(**params)\nprint(f'url={url}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "url=https://example.com:443" in nb_runner.get_output(2)


# multiple assignment, unpacking, and star expressions.
@pytest.mark.integration
class TestUnpacking:
    """Unpacking and multiple assignment patterns."""

    def test_star_unpacking(self, nb_runner):
        """Star (*) unpacking in assignments."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                data = [1, 2, 3, 4, 5, 6, 7]
                first, *middle, last = data
                a, b, *rest = data
                *init, x, y = data
            """),
                "print(f'first={first} middle={middle} last={last}')\n"
                "print(f'a={a} b={b} rest={rest}')\n"
                "print(f'init={init} x={x} y={y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "first=1" in out
        assert "middle=[2, 3, 4, 5, 6]" in out
        assert "last=7" in out
        assert "rest=[3, 4, 5, 6, 7]" in out
        assert "x=6" in out
        assert "y=7" in out

    def test_swap_and_multi_assign(self, nb_runner):
        """Swap and multiple assignment in one line."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                a, b = 10, 20
                a, b = b, a
                x = y = z = 42
                p, q = divmod(100, 7)
            """),
                "print(f'a={a} b={b} x={x} y={y} z={z} p={p} q={q}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "a=20" in out
        assert "b=10" in out
        assert "x=42" in out
        assert "p=14" in out
        assert "q=2" in out

    def test_unpacking_propagation(self, nb_runner):
        """Unpacking with upstream data change."""
        nb_runner.create_notebook(
            [
                "data = (10, 20, 30)",
                textwrap.dedent("""\
                a, b, c = data
                total = a + b + c
            """),
                "print(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=60" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "data = (100, 200, 300)")
        nb_runner.run_cells([1, 2, 3])
        assert "total=600" in nb_runner.get_output(3)


# Unpacking and star expression edit tests.
#
# Tests editing cells with tuple unpacking, star expressions,
# and chained assignments.
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestUnpackingStarEdits:
    """Editing cells with unpacking and star expressions."""

    def test_edit_star_first_to_last(self, nb_runner):
        """Switch from first/*rest to *init/last pattern."""
        nb_runner.create_notebook(
            [
                "first, *rest = [1, 2, 3, 4, 5]",
                "print(f'first={first} rest={rest}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first=1" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "*init, last = [1, 2, 3, 4, 5]")
        nb_runner.set_cell_source(2, "print(f'init={init} last={last}')")
        nb_runner.run_all()
        assert "last=5" in nb_runner.get_output(2)

    def test_edit_chained_value(self, nb_runner):
        """Edit chained assignment x = y = z = value."""
        nb_runner.create_notebook(
            [
                "x = y = z = 5",
                "total = x + y + z\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 15" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "x = y = z = 10")
        nb_runner.run_all()
        assert "total = 30" in nb_runner.get_output(2)
