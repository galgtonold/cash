"""
Interaction test: multiple return unpacking with nested tuples.
Tests complex unpacking patterns with nested structures, star unpacking
in function returns, and cross-cell value threading.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestNestedUnpackReturn:
    """Test complex unpacking patterns across cells."""

    def test_nested_unpack(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: define function with complex return
            "def analyze_data(data):\n    total = sum(data)\n    avg = total / len(data)\n    extremes = (min(data), max(data))\n    spread = extremes[1] - extremes[0]\n    return total, avg, extremes, spread\nprint('analyze_data defined')",
            # Cell 2: unpack nested results
            "data = [10, 20, 30, 40, 50]\ntotal, avg, (lo, hi), spread = analyze_data(data)\nprint(f'total={total}')\nprint(f'avg={avg}')\nprint(f'lo={lo} hi={hi}')\nprint(f'spread={spread}')",
            # Cell 3: use unpacked values
            "normalized = [(x - lo) / spread * 100 for x in data]\nprint(f'norm={[int(n) for n in normalized]}')",
        ])
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
        nb_runner.create_notebook([
            "def stats(nums):\n    s = sorted(nums)\n    return s[0], s[-1], s[len(s)//2]\nprint('stats defined')",
            "lo, hi, med = stats([5, 3, 8, 1, 9])\nprint(f'lo={lo} hi={hi} med={med}')",
            "rng = hi - lo\nprint(f'range={rng}')",
        ])
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
        nb_runner.create_notebook([
            "def split_name(full):\n    parts = full.split()\n    first, *middle, last = parts\n    return first, middle, last\nprint('split_name defined')",
            "first, mid, last = split_name('John Michael Smith Jr')\nprint(f'first={first} mid={mid} last={last}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "first=John mid=['Michael', 'Smith'] last=Jr" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "first=John mid=['Michael', 'Smith'] last=Jr" in nb_runner.get_output(2)
