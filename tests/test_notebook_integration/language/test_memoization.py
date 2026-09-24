"""Hand-written memoization: dict caches, memoize decorators and lookup tables."""

import textwrap

import pytest


@pytest.mark.stress
class TestManualMemoization:
    """Test manual memoization patterns."""

    def test_dict_memoize(self, nb_runner):
        """Manual dict-based memoization."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                _memo = {}
                def memoized_power(base, exp):
                    key = (base, exp)
                    if key not in _memo:
                        _memo[key] = base ** exp
                    return _memo[key]
            """),
                textwrap.dedent("""\
                r1 = memoized_power(2, 10)
                r2 = memoized_power(3, 5)
                r3 = memoized_power(2, 10)  # should be cached
                cache_size = len(_memo)
                print(f"r1={r1} r2={r2} r3={r3} cache_size={cache_size}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=1024 r2=243 r3=1024 cache_size=2" in nb_runner.get_output(2)

    def test_memoize_decorator(self, nb_runner):
        """Custom memoize decorator across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def memoize(fn):
                    cache = {}
                    def wrapper(*args):
                        if args not in cache:
                            cache[args] = fn(*args)
                        return cache[args]
                    wrapper.cache = cache
                    wrapper.__name__ = fn.__name__
                    return wrapper
            """),
                textwrap.dedent("""\
                @memoize
                def expensive(n):
                    return sum(i**2 for i in range(n))

                r1 = expensive(100)
                r2 = expensive(100)  # cached
                r3 = expensive(50)
                cache_size = len(expensive.cache)
                print(f"r1={r1} r3={r3} cache_size={cache_size}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=328350" in nb_runner.get_output(2)
        assert "cache_size=2" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestMemoizationEdits:
    """Editing memoization patterns."""

    def test_edit_memoized_function(self, nb_runner):
        """Edit a function that uses manual memoization."""
        nb_runner.create_notebook(
            [
                "def fib(n, memo={}):\n    if n in memo:\n        return memo[n]\n    if n <= 1:\n        return n\n    memo[n] = fib(n-1, memo) + fib(n-2, memo)\n    return memo[n]",
                "result = fib(10)\nprint(f'fib(10) = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "fib(10) = 55" in nb_runner.get_output(2)

        # Change to call with different arg
        nb_runner.set_cell_source(2, "result = fib(15)\nprint(f'fib(15) = {result}')")
        nb_runner.run_all()
        assert "fib(15) = 610" in nb_runner.get_output(2)

    def test_edit_lru_cache_function(self, nb_runner):
        """Edit function using lru_cache."""
        nb_runner.create_notebook(
            [
                "from functools import lru_cache\n@lru_cache(maxsize=None)\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
                "result = factorial(5)\nprint(f'5! = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "5! = 120" in nb_runner.get_output(2)

        # Change call
        nb_runner.set_cell_source(2, "result = factorial(7)\nprint(f'7! = {result}')")
        nb_runner.run_all()
        assert "7! = 5040" in nb_runner.get_output(2)

    def test_edit_lookup_table(self, nb_runner):
        """Edit a precomputed lookup table."""
        nb_runner.create_notebook(
            [
                "squares = {i: i**2 for i in range(10)}",
                "vals = [squares[x] for x in [1, 3, 5, 7]]\nprint(f'vals = {vals}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "vals = [1, 9, 25, 49]" in nb_runner.get_output(2)

        # Change to cubes
        nb_runner.set_cell_source(1, "squares = {i: i**3 for i in range(10)}")
        nb_runner.run_all()
        assert "vals = [1, 27, 125, 343]" in nb_runner.get_output(2)
