"""
Interaction test: itertools.chain.from_iterable with nested data.
Tests chain.from_iterable for flattening nested structures,
combined with map and filter across cells.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestChainFromIterable:
    """Test itertools.chain.from_iterable across cells."""

    def test_chain_from_iterable(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: flatten nested lists
            "import itertools\nnested = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]\nflat = list(itertools.chain.from_iterable(nested))\nprint(f'flat={flat}')\nprint(f'len={len(flat)}')",
            # Cell 2: flatten with transformation
            "words = [['hello', 'world'], ['foo', 'bar', 'baz']]\nall_chars = list(itertools.chain.from_iterable(w.upper() for w in itertools.chain.from_iterable(words)))\nunique_chars = sorted(set(all_chars))\nprint(f'unique_count={len(unique_chars)}')",
            # Cell 3: combine
            "total = sum(flat)\nchar_count = len(all_chars)\nprint(f'sum={total}')\nprint(f'chars={char_count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "flat=[1, 2, 3, 4, 5, 6, 7, 8, 9]" in out1
        assert "len=9" in out1
        out3 = nb_runner.get_output(3)
        assert "sum=45" in out3

    def test_chain_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import itertools\ngroups = [[10, 20], [30, 40], [50]]\nflat = list(itertools.chain.from_iterable(groups))\nprint(f'flat={flat}')",
            "total = sum(flat)\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=150" in nb_runner.get_output(2)

        # Add more groups
        nb_runner.set_cell_source(1, "import itertools\ngroups = [[10, 20], [30, 40], [50], [60, 70]]\nflat = list(itertools.chain.from_iterable(groups))\nprint(f'flat={flat}')")
        nb_runner.run_cells([1, 2])
        assert "total=280" in nb_runner.get_output(2)

    def test_chain_cache(self, nb_runner):
        nb_runner.create_notebook([
            "import itertools\npairs = [(1, 'a'), (2, 'b'), (3, 'c')]\nflat = list(itertools.chain.from_iterable(pairs))\nprint(f'flat={flat}')",
            "strs = [x for x in flat if isinstance(x, str)]\nprint(f'strs={strs}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "strs=['a', 'b', 'c']" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "strs=['a', 'b', 'c']" in nb_runner.get_output(2)
