"""collections.Counter across cells."""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


@pytest.mark.upstream
class TestCounterPatternEdits:
    """Editing Counter and accumulator patterns."""

    def test_edit_counter_source(self, nb_runner):
        """Edit source data for Counter."""
        nb_runner.create_notebook(
            [
                "from collections import Counter\nwords = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']",
                "counts = Counter(words)\nprint(f'most = {counts.most_common(2)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "('apple', 3)" in nb_runner.get_output(2)

        # Change words
        nb_runner.set_cell_source(
            1, "from collections import Counter\nwords = ['x', 'y', 'x', 'x', 'y', 'z', 'z', 'z', 'z']"
        )
        nb_runner.run_all()
        assert "('z', 4)" in nb_runner.get_output(2)

    def test_edit_defaultdict_factory(self, nb_runner):
        """Edit defaultdict population."""
        nb_runner.create_notebook(
            [
                "from collections import defaultdict\npairs = [('a', 1), ('b', 2), ('a', 3)]",
                "d = defaultdict(list)\nfor k, v in pairs:\n    d[k].append(v)\nprint(f'a = {d[\"a\"]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = [1, 3]" in nb_runner.get_output(2)

        # Change pairs
        nb_runner.set_cell_source(1, "from collections import defaultdict\npairs = [('a', 10), ('a', 20), ('b', 5)]")
        nb_runner.run_all()
        assert "a = [10, 20]" in nb_runner.get_output(2)

    def test_edit_running_total(self, nb_runner):
        """Edit accumulation source."""
        nb_runner.create_notebook(
            [
                "transactions = [100, -50, 200, -75]",
                "balance = 0\nfor t in transactions:\n    balance += t\nprint(f'balance = {balance}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "balance = 175" in nb_runner.get_output(2)

        # Change transactions
        nb_runner.set_cell_source(1, "transactions = [500, -100, -200]")
        nb_runner.run_all()
        assert "balance = 200" in nb_runner.get_output(2)

    def test_edit_histogram(self, nb_runner):
        """Edit data for histogram-style grouping."""
        nb_runner.create_notebook(
            [
                "scores = [85, 92, 78, 95, 88, 72, 91]",
                "bins = {'A': 0, 'B': 0, 'C': 0}\nfor s in scores:\n    if s >= 90:\n        bins['A'] += 1\n    elif s >= 80:\n        bins['B'] += 1\n    else:\n        bins['C'] += 1\nprint(f'bins = {bins}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'A': 3" in nb_runner.get_output(2)

        # Change scores
        nb_runner.set_cell_source(1, "scores = [60, 65, 70, 75]")
        nb_runner.run_all()
        assert "'A': 0" in nb_runner.get_output(2)
        assert "'C': 4" in nb_runner.get_output(2)


class TestCounterArithAdvanced:
    """Test Counter arithmetic operations across cells."""

    def test_counter_arithmetic(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: create counters
                "from collections import Counter\nc1 = Counter(a=3, b=2, c=1)\nc2 = Counter(a=1, b=3, d=2)\nprint(f'c1={dict(c1)}')\nprint(f'c2={dict(c2)}')",
                # Cell 2: arithmetic operations
                "added = c1 + c2\nsubtracted = c1 - c2  # drops zero/negative\nintersected = c1 & c2  # min of each\nunioned = c1 | c2  # max of each\nprint(f'add={dict(added)}')\nprint(f'sub={dict(subtracted)}')\nprint(f'inter={dict(intersected)}')\nprint(f'union={dict(unioned)}')",
                # Cell 3: most_common and total
                "most = added.most_common(2)\ntotal_val = added.total()\nprint(f'most={most}')\nprint(f'total={total_val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "'a': 4" in out2
        assert "'b': 5" in out2
        out3 = nb_runner.get_output(3)
        assert "total=" in out3

    def test_counter_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import Counter\ntext = 'hello world'\nc = Counter(text)\nprint(f'l_count={c[\"l\"]}')",
                "top3 = c.most_common(3)\nprint(f'top3={top3}')",
                "vowels = sum(c[v] for v in 'aeiou')\nprint(f'vowels={vowels}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "l_count=3" in nb_runner.get_output(1)

        # Edit text
        nb_runner.set_cell_source(
            1, "from collections import Counter\ntext = 'banana split'\nc = Counter(text)\nprint(f'a_count={c[\"a\"]}')"
        )
        nb_runner.run_cells([1, 2, 3])
        assert "a_count=3" in nb_runner.get_output(1)

    def test_counter_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import Counter\nwords = ['the', 'cat', 'sat', 'on', 'the', 'mat', 'the']\nwc = Counter(words)\nprint(f'the_count={wc[\"the\"]}')",
                "unique = len(wc)\nprint(f'unique={unique}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "the_count=3" in nb_runner.get_output(1)
        assert "unique=5" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "unique=5" in nb_runner.get_output(2)


class TestCounterArithmetic:
    """collections.Counter most_common and arithmetic."""

    def test_counter_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import Counter\nc1 = Counter('aabbc')\nc2 = Counter('bccdd')",
                "added = c1 + c2\nsubtracted = c1 - c2\ncommon = c1 & c2\nprint(f'added={dict(sorted(added.items()))}')\nprint(f'common={dict(sorted(common.items()))}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'a': 2" in out
        assert "'b': 3" in out

    def test_counter_most_common_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import Counter\nwords = ['the', 'cat', 'sat', 'on', 'the', 'mat', 'the', 'cat']",
                "counts = Counter(words)\ntop2 = counts.most_common(2)\nprint(f'top2={top2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "('the', 3)" in nb_runner.get_output(2)
        assert "('cat', 2)" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "from collections import Counter\nwords = ['a', 'b', 'a', 'c', 'b', 'a']")
        nb_runner.run_all()
        assert "('a', 3)" in nb_runner.get_output(2)

    def test_counter_elements(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import Counter\nc = Counter(a=3, b=1)",
                "elements = sorted(c.elements())\nprint(f'elements={elements}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "elements=['a', 'a', 'a', 'b']" in nb_runner.get_output(2)


class TestCounterElements:
    """Test Counter.elements and arithmetic across cells."""

    def test_counter_elements(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: create counter
                "from collections import Counter\nwords = 'apple banana apple cherry banana apple'.split()\nc = Counter(words)\nmost = c.most_common(2)\nprint(f'most={most}')",
                # Cell 2: elements iteration
                "elems = sorted(c.elements())\nprint(f'total={len(elems)}')\nprint(f'first_3={elems[:3]}')",
                # Cell 3: counter arithmetic
                "c2 = Counter({'apple': 1, 'date': 2})\ncombined = c + c2\nprint(f'apple_count={combined[\"apple\"]}')\nprint(f'date_count={combined[\"date\"]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "('apple', 3)" in out1
        out2 = nb_runner.get_output(2)
        assert "total=6" in out2
        out3 = nb_runner.get_output(3)
        assert "apple_count=4" in out3
        assert "date_count=2" in out3

    def test_counter_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import Counter\nc = Counter('aabbbcccc')\nprint(f'a={c[\"a\"]}')\nprint(f'b={c[\"b\"]}')\nprint(f'c_count={c[\"c\"]}')",
                "top = c.most_common(1)[0]\nprint(f'top_char={top[0]}')\nprint(f'top_count={top[1]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=2" in nb_runner.get_output(1)
        assert "top_char=c" in nb_runner.get_output(2)
        assert "top_count=4" in nb_runner.get_output(2)

        # Edit input string
        nb_runner.set_cell_source(
            1,
            "from collections import Counter\nc = Counter('aaaaaabb')\nprint(f'a={c[\"a\"]}')\nprint(f'b={c[\"b\"]}')",
        )
        nb_runner.run_cells([1, 2])
        assert "a=6" in nb_runner.get_output(1)
        assert "top_char=a" in nb_runner.get_output(2)
        assert "top_count=6" in nb_runner.get_output(2)

    def test_counter_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import Counter\nc = Counter([1, 1, 2, 3, 3, 3])\nunique = len(c)\nprint(f'unique={unique}')",
                "total = sum(c.values())\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "unique=3" in nb_runner.get_output(1)
        assert "total=6" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "total=6" in nb_runner.get_output(2)


class TestCounterMostCommonSubtract:
    """collections Counter most_common subtract."""

    def test_most_common(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import Counter",
                "text = 'abracadabra'\nc = Counter(text)\ntop3 = c.most_common(3)\nprint(f'top3={top3}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "('a', 5)" in out

    def test_counter_subtract(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import Counter",
                'inventory = Counter(apples=10, bananas=5, oranges=8)\nsold = Counter(apples=3, bananas=2)\ninventory.subtract(sold)\nprint(f\'apples={inventory["apples"]} bananas={inventory["bananas"]} oranges={inventory["oranges"]}\')',
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "apples=7" in out
        assert "bananas=3" in out
        assert "oranges=8" in out

    def test_counter_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import Counter",
                "c = Counter([1, 1, 2, 3, 3, 3])\nprint(f'most={c.most_common(1)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "most=[(3, 3)]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "c = Counter([1, 1, 1, 1, 2, 3])\nprint(f'most={c.most_common(1)}')")
        nb_runner.run_all()
        assert "most=[(1, 4)]" in nb_runner.get_output(2)


@pytest.mark.integration
class TestCounterStatistics:
    """Test Counter and statistics operation caching."""

    def test_counter_most_common(self, nb_runner):
        """Counter.most_common with caching."""
        nb_runner.create_notebook(
            [
                "from collections import Counter",
                "words = ['apple', 'banana', 'apple', 'cherry', 'apple', 'banana']",
                "counts = Counter(words)\ntop = counts.most_common(2)",
                "print(f'top={top}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "('apple', 3)" in out
        assert "('banana', 2)" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "('apple', 3)" in out2

    def test_counter_edit_data(self, nb_runner):
        """Edit data, verify counter updates."""
        nb_runner.create_notebook(
            [
                "from collections import Counter",
                "data = [1, 1, 2, 2, 2, 3]",
                "c = Counter(data)\nmost = c.most_common(1)[0]",
                "print(f'most={most}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "most=(2, 3)" in out

        nb_runner.set_cell_source(2, "data = [1, 1, 1, 1, 2, 3]")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "most=(1, 4)" in out2

    def test_statistics_measures(self, nb_runner):
        """statistics.mean/median/stdev with caching."""
        nb_runner.create_notebook(
            [
                "import statistics",
                "data = [10, 20, 30, 40, 50]",
                "m = statistics.mean(data)\nmed = statistics.median(data)",
                "print(f'mean={m} median={med}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "mean=30" in out
        assert "median=30" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "mean=30" in out2


class TestCounterIntersectionUnion:
    """Counter intersection and union operations."""

    def test_counter_intersect(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import Counter\nc1 = Counter(a=3, b=1, c=5)\nc2 = Counter(a=1, b=4, c=2)",
                'inter = c1 & c2\nunion = c1 | c2\nprint(f\'inter_a={inter["a"]} inter_b={inter["b"]} union_c={union["c"]}\')',
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "inter_a=1" in out
        assert "inter_b=1" in out
        assert "union_c=5" in out

    def test_counter_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from collections import Counter\ntext = 'aabbc'",
                "c = Counter(text)\ntotal = c.total()\nprint(f'total={total} a={c[\"a\"]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=5" in nb_runner.get_output(2)
        assert "a=2" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from collections import Counter\ntext = 'aaabbbcccdddd'")
        nb_runner.run_all()
        assert "total=13" in nb_runner.get_output(2)
        assert "a=3" in nb_runner.get_output(2)
