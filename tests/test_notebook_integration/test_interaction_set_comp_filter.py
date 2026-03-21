"""
Interaction test: set comprehension with complex filtering.
Tests set comprehension with multi-condition filters, set algebra,
and cross-cell set-based analysis.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestSetComprehensionFilter:
    """Test set comprehension with complex filtering across cells."""

    def test_set_comp_filter(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: set comprehension with conditions
            "nums = range(1, 51)\nevens = {n for n in nums if n % 2 == 0}\ndiv_by_3 = {n for n in nums if n % 3 == 0}\nprint(f'evens_count={len(evens)}')\nprint(f'div3_count={len(div_by_3)}')",
            # Cell 2: set operations
            "both = evens & div_by_3  # div by 6\neither = evens | div_by_3\nonly_even = evens - div_by_3\nprint(f'both={sorted(both)}')\nprint(f'either_count={len(either)}')\nprint(f'only_even_count={len(only_even)}')",
            # Cell 3: symmetric difference
            "sym_diff = evens ^ div_by_3\nprint(f'sym_diff_count={len(sym_diff)}')\nprint(f'verify={len(sym_diff) == len(either) - len(both)}')",
        ])
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
        nb_runner.create_notebook([
            "nums = range(1, 21)\nprimes = {n for n in nums if n > 1 and all(n % i != 0 for i in range(2, int(n**0.5)+1))}\nprint(f'primes={sorted(primes)}')",
            "prime_count = len(primes)\nprint(f'count={prime_count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=8" in nb_runner.get_output(2)

        # Extend range
        nb_runner.set_cell_source(1, "nums = range(1, 31)\nprimes = {n for n in nums if n > 1 and all(n % i != 0 for i in range(2, int(n**0.5)+1))}\nprint(f'primes={sorted(primes)}')")
        nb_runner.run_cells([1, 2])
        assert "count=10" in nb_runner.get_output(2)

    def test_set_comp_cache(self, nb_runner):
        nb_runner.create_notebook([
            "words = ['hello', 'HELLO', 'World', 'world', 'Python']\nunique_lower = {w.lower() for w in words}\nprint(f'unique={sorted(unique_lower)}')",
            "count = len(unique_lower)\nprint(f'count={count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "unique=['hello', 'python', 'world']" in nb_runner.get_output(1)
        assert "count=3" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "count=3" in nb_runner.get_output(2)
