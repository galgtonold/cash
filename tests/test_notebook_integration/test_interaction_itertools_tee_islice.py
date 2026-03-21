"""
Interaction test: itertools.tee and islice with multiple consumers.
Tests tee for creating independent iterators, islice for windows,
and cross-cell iterator consumption patterns.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestItertoolsTeeIslice:
    """Test itertools.tee and islice across cells."""

    def test_tee_islice_operations(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: create and tee an iterator
            "import itertools\noriginal = iter(range(10))\nit1, it2, it3 = itertools.tee(original, 3)\nprint('teed=3')",
            # Cell 2: consume differently using islice
            "first_5 = list(itertools.islice(it1, 5))\nevens = list(itertools.islice(it2, 0, 10, 2))\nlast_3 = list(itertools.islice(it3, 7, 10))\nprint(f'first_5={first_5}')\nprint(f'evens={evens}')\nprint(f'last_3={last_3}')",
            # Cell 3: combine results (union of [0..4] + [0,2,4,6,8] + [7,8,9] = 9 unique, missing 5)
            "all_unique = sorted(set(first_5 + evens + last_3))\nprint(f'unique_count={len(all_unique)}')\nprint(f'has_five={5 in all_unique}')",
        ])
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
        nb_runner.create_notebook([
            "import itertools\noriginal = iter(range(10))\nit1, it2 = itertools.tee(original, 2)\nprint('teed=2')",
            "a = list(itertools.islice(it1, 3))\nb = list(itertools.islice(it2, 3, 6))\nprint(f'a={a}')\nprint(f'b={b}')",
            "combined = a + b\ntotal = sum(combined)\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=[0, 1, 2]" in nb_runner.get_output(2)
        assert "b=[3, 4, 5]" in nb_runner.get_output(2)
        assert "total=15" in nb_runner.get_output(3)

        # Change range
        nb_runner.set_cell_source(1, "import itertools\noriginal = iter(range(20))\nit1, it2 = itertools.tee(original, 2)\nprint('teed=2')")
        nb_runner.run_cells([1, 2, 3])
        assert "a=[0, 1, 2]" in nb_runner.get_output(2)
        assert "b=[3, 4, 5]" in nb_runner.get_output(2)
        # Same slices so same results
        assert "total=15" in nb_runner.get_output(3)

    def test_tee_islice_cache(self, nb_runner):
        nb_runner.create_notebook([
            "import itertools\ndata = iter([10, 20, 30, 40, 50])\na, b = itertools.tee(data, 2)\nprint('teed')",
            "head = list(itertools.islice(a, 2))\ntail = list(itertools.islice(b, 3, 5))\nprint(f'head={head}')\nprint(f'tail={tail}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "head=[10, 20]" in out
        assert "tail=[40, 50]" in out

        # Re-run - cache
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "head=[10, 20]" in out
