"""Batch 343: functools.lru_cache, partial application, and caching interaction."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestFunctoolsCache:
    def test_lru_cache_basic(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import lru_cache\n@lru_cache(maxsize=None)\ndef fib(n):\n    if n < 2:\n        return n\n    return fib(n-1) + fib(n-2)",
            "result = fib(10)\nprint(f'fib10={result}')",
            "result2 = fib(15)\nprint(f'fib15={result2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "fib10=55" in nb_runner.get_output(2)
        assert "fib15=610" in nb_runner.get_output(3)

    def test_lru_cache_edit_function(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import lru_cache\n@lru_cache(maxsize=128)\ndef compute(x):\n    return x ** 2",
            "vals = [compute(i) for i in range(5)]\nprint(f'vals={vals}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "vals=[0, 1, 4, 9, 16]" in nb_runner.get_output(2)
        # Edit function
        nb_runner.set_cell_source(1, "from functools import lru_cache\n@lru_cache(maxsize=128)\ndef compute(x):\n    return x ** 3")
        nb_runner.run_all()
        assert "vals=[0, 1, 8, 27, 64]" in nb_runner.get_output(2)

    def test_partial_application(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import partial\ndef power(base, exp):\n    return base ** exp\nsquare = partial(power, exp=2)\ncube = partial(power, exp=3)",
            "s = square(5)\nc = cube(3)\nprint(f's={s} c={c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "s=25 c=27" in nb_runner.get_output(2)
