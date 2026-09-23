"""zip, zip_longest and enumerate across cells."""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


# Interaction test: enumerate with start parameter and custom step.
# Tests enumerate with start offset, zip+enumerate patterns,
# and cross-cell indexed iteration pipelines.
class TestEnumerateStartStep:
    """Test enumerate with start parameter across cells."""

    def test_enumerate_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: enumerate with start
                "items = ['apple', 'banana', 'cherry']\nindexed = list(enumerate(items, start=1))\nprint(f'indexed={indexed}')",
                # Cell 2: use indexed in computation
                "formatted = [f'{i}. {name}' for i, name in indexed]\nprint(f'list={formatted}')",
                # Cell 3: reversed enumerate
                "rev = list(enumerate(reversed(items), start=1))\nprint(f'reversed={rev}')\ntotal_idx = sum(i for i, _ in rev)\nprint(f'idx_sum={total_idx}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "indexed=[(1, 'apple'), (2, 'banana'), (3, 'cherry')]" in out1
        out2 = nb_runner.get_output(2)
        assert "1. apple" in out2
        assert "3. cherry" in out2
        out3 = nb_runner.get_output(3)
        assert "idx_sum=6" in out3

    def test_enumerate_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "colors = ['red', 'green', 'blue']\nnumbered = {i: c for i, c in enumerate(colors, 100)}\nprint(f'numbered={numbered}')",
                "keys_sum = sum(numbered.keys())\nprint(f'keys_sum={keys_sum}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys_sum=303" in nb_runner.get_output(2)

        # Edit start value
        nb_runner.set_cell_source(
            1,
            "colors = ['red', 'green', 'blue', 'yellow']\nnumbered = {i: c for i, c in enumerate(colors, 200)}\nprint(f'numbered={numbered}')",
        )
        nb_runner.run_cells([1, 2])
        # 200+201+202+203 = 806
        assert "keys_sum=806" in nb_runner.get_output(2)

    def test_enumerate_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "words = ['hello', 'world']\nresult = [(i, w.upper()) for i, w in enumerate(words)]\nprint(f'result={result}')",
                "count = len(result)\nprint(f'count={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[(0, 'HELLO'), (1, 'WORLD')]" in nb_runner.get_output(1)
        assert "count=2" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "count=2" in nb_runner.get_output(2)


class TestEnumerateZipUnpack:
    """enumerate with start, zip with strict, and unpacking."""

    def test_enumerate_start(self, nb_runner):
        nb_runner.create_notebook(
            [
                "items = ['apple', 'banana', 'cherry']",
                "numbered = list(enumerate(items, start=1))\nprint(f'numbered={numbered}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "numbered=[(1, 'apple'), (2, 'banana'), (3, 'cherry')]" in nb_runner.get_output(2)

    def test_enumerate_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "data = ['x', 'y', 'z']",
                "pairs = {i: v for i, v in enumerate(data)}\nprint(f'pairs={pairs}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "pairs={0: 'x', 1: 'y', 2: 'z'}" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "data = ['a', 'b']")
        nb_runner.run_all()
        assert "pairs={0: 'a', 1: 'b'}" in nb_runner.get_output(2)

    def test_zip_unpack_pattern(self, nb_runner):
        nb_runner.create_notebook(
            [
                "names = ['Alice', 'Bob', 'Charlie']\nages = [30, 25, 35]\ncities = ['NY', 'LA', 'SF']",
                "records = list(zip(names, ages, cities))\noldest = max(records, key=lambda r: r[1])\nprint(f'oldest={oldest}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "oldest=('Charlie', 35, 'SF')" in nb_runner.get_output(2)


# zip-to-dict construction patterns with caching.
# Tests zip pairing, dict construction, key/value extraction, and edit propagation.
@pytest.mark.integration
class TestZipDictConstruct:
    """Test zip-based dict construction caching."""

    def test_zip_dict_basic(self, nb_runner):
        """Zip two lists into a dict, verify caching."""
        nb_runner.create_notebook(
            [
                "keys = ['a', 'b', 'c']",
                "values = [1, 2, 3]",
                "mapping = dict(zip(keys, values))",
                "result = ', '.join(f'{k}={v}' for k, v in sorted(mapping.items()))\nprint(result)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "a=1" in out
        assert "b=2" in out
        assert "c=3" in out

        # Re-run: should be cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "a=1" in out2

    def test_zip_dict_edit_keys(self, nb_runner):
        """Edit keys list, verify dict reconstruction."""
        nb_runner.create_notebook(
            [
                "keys = ['x', 'y']",
                "vals = [10, 20]",
                "d = dict(zip(keys, vals))",
                "total = sum(d.values())\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=30" in out

        nb_runner.set_cell_source(1, "keys = ['x', 'y', 'z']")
        nb_runner.set_cell_source(2, "vals = [10, 20, 30]")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "total=60" in out2

    def test_zip_enumerate_pattern(self, nb_runner):
        """Zip with enumerate for indexed pairs."""
        nb_runner.create_notebook(
            [
                "items = ['apple', 'banana', 'cherry']",
                "indexed = dict(enumerate(items))",
                "lines = [f'{i}: {v}' for i, v in sorted(indexed.items())]\nprint('\\n'.join(lines))",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "0: apple" in out
        assert "2: cherry" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "0: apple" in out2


# Interaction test: zip_longest with fillvalue and dict construction.
# Tests zip_longest for unequal iterables, fillvalue parameter,
# dict construction from zipped pairs, and cross-cell data alignment.
class TestZipLongestDict:
    """Test zip_longest with dict construction across cells."""

    def test_zip_longest_fillvalue(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: zip_longest with fill
                "from itertools import zip_longest\nkeys = ['a', 'b', 'c', 'd', 'e']\nvals = [1, 2, 3]\npairs = list(zip_longest(keys, vals, fillvalue=0))\nprint(f'pairs={pairs}')",
                # Cell 2: build dict
                "d = dict(pairs)\nprint(f'dict={d}')\nprint(f'filled={sum(1 for v in d.values() if v == 0)}')",
                # Cell 3: aggregate
                "total = sum(d.values())\nnon_zero = {k: v for k, v in d.items() if v != 0}\nprint(f'total={total}')\nprint(f'non_zero={non_zero}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "('a', 1)" in out1
        assert "('d', 0)" in out1
        out2 = nb_runner.get_output(2)
        assert "filled=2" in out2
        out3 = nb_runner.get_output(3)
        assert "total=6" in out3
        assert "'a': 1" in out3

    def test_zip_longest_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import zip_longest\nnames = ['Alice', 'Bob', 'Charlie']\nages = [30, 25]\npaired = list(zip_longest(names, ages, fillvalue='N/A'))\nprint(f'count={len(paired)}')",
                "result = {n: a for n, a in paired}\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=3" in nb_runner.get_output(1)
        assert "'Charlie': 'N/A'" in nb_runner.get_output(2)

        # Add more ages
        nb_runner.set_cell_source(
            1,
            "from itertools import zip_longest\nnames = ['Alice', 'Bob', 'Charlie']\nages = [30, 25, 35, 40]\npaired = list(zip_longest(names, ages, fillvalue='Unknown'))\nprint(f'count={len(paired)}')",
        )
        nb_runner.run_cells([1, 2])
        assert "count=4" in nb_runner.get_output(1)
        assert "'Unknown': 40" in nb_runner.get_output(2)

    def test_zip_longest_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import zip_longest\ncols = ['x', 'y', 'z']\nrow1 = [1, 2]\nrow2 = [4, 5, 6]\naligned = [dict(zip_longest(cols, r, fillvalue=0)) for r in [row1, row2]]\nprint(f'rows={aligned}')",
                "z_vals = [r['z'] for r in aligned]\nprint(f'z_vals={z_vals}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "'z': 0" in out1
        assert "'z': 6" in out1
        assert "z_vals=[0, 6]" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "z_vals=[0, 6]" in nb_runner.get_output(2)


# Interaction test: zip_longest with fillvalue and multi-iterator.
# Tests itertools.zip_longest with custom fillvalue,
# multiple iterables of different lengths, and cross-cell processing.
class TestZipLongestFillvalue:
    """Test zip_longest with fillvalue across cells."""

    def test_zip_longest_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: zip_longest with fillvalue
                "from itertools import zip_longest\nnames = ['Alice', 'Bob', 'Charlie']\nscores = [95, 87]\ngrades = ['A', 'B', 'C', 'D']\ncombined = list(zip_longest(names, scores, grades, fillvalue='N/A'))\nprint(f'count={len(combined)}')\nfor name, score, grade in combined:\n    print(f'{name}:{score}:{grade}')",
                # Cell 2: process combined data
                "valid = [(n, s, g) for n, s, g in combined if s != 'N/A' and g != 'N/A']\nprint(f'valid_count={len(valid)}')\nprint(f'first_valid={valid[0]}')",
                # Cell 3: transform
                "result_dict = {n: {'score': s, 'grade': g} for n, s, g in combined if n != 'N/A'}\nprint(f'entries={len(result_dict)}')\nprint(f'alice_score={result_dict[\"Alice\"][\"score\"]}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "count=4" in out1
        assert "Alice:95:A" in out1
        out2 = nb_runner.get_output(2)
        assert "valid_count=2" in out2
        out3 = nb_runner.get_output(3)
        assert "entries=3" in out3
        assert "alice_score=95" in out3

    def test_zip_longest_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import zip_longest\nkeys = ['a', 'b', 'c']\nvals = [1, 2]\npairs = dict(zip_longest(keys, vals, fillvalue=0))\nprint(f'pairs={pairs}')",
                "total = sum(pairs.values())\nprint(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=3" in nb_runner.get_output(2)

        # Edit to add more values
        nb_runner.set_cell_source(
            1,
            "from itertools import zip_longest\nkeys = ['a', 'b', 'c', 'd']\nvals = [1, 2, 3]\npairs = dict(zip_longest(keys, vals, fillvalue=0))\nprint(f'pairs={pairs}')",
        )
        nb_runner.run_cells([1, 2])
        assert "total=6" in nb_runner.get_output(2)

    def test_zip_longest_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import zip_longest\ncols = ['x', 'y']\nrow1 = [1, 2]\nrow2 = [3]\nmatrix = [dict(zip_longest(cols, r, fillvalue=0)) for r in [row1, row2]]\nprint(f'rows={len(matrix)}')",
                "x_sum = sum(row['x'] for row in matrix)\ny_sum = sum(row['y'] for row in matrix)\nprint(f'x_sum={x_sum}')\nprint(f'y_sum={y_sum}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x_sum=4" in nb_runner.get_output(2)
        assert "y_sum=2" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "x_sum=4" in nb_runner.get_output(2)


class TestZipLongestStarmap:
    """zip_longest and starmap from itertools."""

    def test_zip_longest_fill(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import zip_longest",
                "a = [1, 2, 3]\nb = ['x', 'y']\npaired = list(zip_longest(a, b, fillvalue='?'))\nprint(f'paired={paired}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "paired=[(1, 'x'), (2, 'y'), (3, '?')]" in nb_runner.get_output(2)

    def test_starmap_operations(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import starmap",
                "pairs = [(2, 3), (4, 5), (6, 7)]\nprods = list(starmap(lambda a, b: a * b, pairs))\nprint(f'prods={prods} sum={sum(prods)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "prods=[6, 20, 42]" in out
        assert "sum=68" in out

    def test_zip_longest_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import zip_longest",
                "r = list(zip_longest([1], [2, 3], fillvalue=0))\nprint(f'r={r}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r=[(1, 2), (0, 3)]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "r = list(zip_longest([1, 2, 3], [10], fillvalue=-1))\nprint(f'r={r}')")
        nb_runner.run_all()
        assert "r=[(1, 10), (2, -1), (3, -1)]" in nb_runner.get_output(2)


class TestMatrixTransposeZipStar:
    """matrix transpose and zip star pattern."""

    def test_unzip(self, nb_runner):
        nb_runner.create_notebook(
            [
                "pairs = [('a', 1), ('b', 2), ('c', 3)]",
                "keys, vals = zip(*pairs)\nprint(f'keys={list(keys)} vals={list(vals)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys=['a', 'b', 'c']" in nb_runner.get_output(2)
        assert "vals=[1, 2, 3]" in nb_runner.get_output(2)

    def test_transpose_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "m = [[1, 2], [3, 4], [5, 6]]",
                "t = [list(row) for row in zip(*m)]\nrows = len(t)\ncols = len(t[0])\nprint(f'rows={rows} cols={cols}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "rows=2" in nb_runner.get_output(2)
        assert "cols=3" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "m = [[1, 2, 3, 4], [5, 6, 7, 8]]")
        nb_runner.run_all()
        assert "rows=4" in nb_runner.get_output(2)
        assert "cols=2" in nb_runner.get_output(2)


# Zip and enumerate interaction tests.
#
# Tests editing cells with zip, enumerate, and itertools
# patterns and verifying cache invalidation.
@pytest.mark.upstream
class TestZipEnumerateEdits:
    """Editing zip and enumerate patterns."""

    def test_edit_enumerate_start(self, nb_runner):
        """Edit list and re-enumerate."""
        nb_runner.create_notebook(
            [
                "items = ['apple', 'banana', 'cherry']",
                "indexed = list(enumerate(items, start=1))\nfor i, item in indexed:\n    print(f'{i}: {item}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "1: apple" in nb_runner.get_output(2)

        # Edit items
        nb_runner.set_cell_source(1, "items = ['mango', 'kiwi', 'grape', 'plum']")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "1: mango" in out
        assert "4: plum" in out

    def test_edit_multi_zip(self, nb_runner):
        """Edit cells with multiple zip operations."""
        nb_runner.create_notebook(
            [
                "first = [1, 2, 3]\nsecond = [4, 5, 6]",
                "sums = [a + b for a, b in zip(first, second)]\nprint(f'sums = {sums}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sums = [5, 7, 9]" in nb_runner.get_output(2)

        # Double the first list values
        nb_runner.set_cell_source(1, "first = [10, 20, 30]\nsecond = [4, 5, 6]")
        nb_runner.run_all()
        assert "sums = [14, 25, 36]" in nb_runner.get_output(2)


class TestZipLongestPairwise:
    """zip_longest, pairwise, and batched iteration patterns."""

    def test_zip_longest(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import zip_longest\na = [1, 2, 3]\nb = ['x', 'y']",
                "result = list(zip_longest(a, b, fillvalue='?'))\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[(1, 'x'), (2, 'y'), (3, '?')]" in nb_runner.get_output(2)

    def test_batched_manual(self, nb_runner):
        nb_runner.create_notebook(
            [
                "def batched(iterable, n):\n    from itertools import islice\n    it = iter(iterable)\n    while batch := list(islice(it, n)):\n        yield tuple(batch)",
                "data = list(range(10))\nresult = list(batched(data, 3))\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "(0, 1, 2)" in nb_runner.get_output(2)
        assert "(9,)" in nb_runner.get_output(2)


class TestZipLongestPatterns:
    """zip with unequal lengths and zip_longest."""

    def test_zip_strict_truncate(self, nb_runner):
        nb_runner.create_notebook(
            [
                "names = ['Alice', 'Bob', 'Charlie']\nscores = [90, 85]",
                "paired = list(zip(names, scores))\ncount = len(paired)\nprint(f'paired={paired} count={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "count=2" in out
        assert "('Alice', 90)" in out

    def test_zip_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from itertools import zip_longest\nk = ['a', 'b']\nv = [1, 2, 3]",
                "result = dict(zip_longest(k, v, fillvalue='?'))\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 1" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from itertools import zip_longest\nk = ['x', 'y', 'z']\nv = [10, 20]")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'x': 10" in out
        assert "'z': '?'" in out


class TestZipUnzipEnumerate:
    """zip unzip and enumerate patterns."""

    def test_zip_and_unzip(self, nb_runner):
        nb_runner.create_notebook(
            [
                "names = ['Alice', 'Bob', 'Carol']\nages = [30, 25, 35]",
                "paired = list(zip(names, ages))\nun_names, un_ages = zip(*paired)\nprint(f'paired={paired}')\nprint(f'names={list(un_names)} ages={list(un_ages)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "('Alice', 30)" in out
        assert "names=['Alice', 'Bob', 'Carol']" in out

    def test_zip_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "a = [1, 2]\nb = ['x', 'y']",
                "result = list(zip(a, b))\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=[(1, 'x'), (2, 'y')]" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "a = [10, 20, 30]\nb = ['p', 'q', 'r']")
        nb_runner.run_all()
        assert "result=[(10, 'p'), (20, 'q'), (30, 'r')]" in nb_runner.get_output(2)
