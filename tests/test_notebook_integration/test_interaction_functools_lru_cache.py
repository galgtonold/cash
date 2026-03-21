"""Batch 429: functools.lru_cache and cache usage."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestFunctoolsLruCache:
    def test_lru_cache_basic(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import lru_cache\n@lru_cache(maxsize=32)\ndef fib(n):\n    if n < 2: return n\n    return fib(n-1) + fib(n-2)",
            "r1 = fib(10)\nr2 = fib(20)\ninfo = fib.cache_info()\nprint(f'r1={r1} r2={r2} hits={info.hits}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "r1=55" in out
        assert "r2=6765" in out

    def test_lru_cache_clear(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import lru_cache\ncall_count = 0\n@lru_cache(maxsize=4)\ndef square(n):\n    global call_count\n    call_count += 1\n    return n * n",
            "r1 = square(3)\nr2 = square(3)\nsquare.cache_clear()\nr3 = square(3)\nprint(f'r1={r1} r2={r2} r3={r3} calls={call_count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "r1=9" in out
        assert "calls=2" in out

    def test_lru_cache_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import lru_cache\n@lru_cache\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
            "r = factorial(5)\nprint(f'r={r}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r=120" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "r = factorial(7)\nprint(f'r={r}')")
        nb_runner.run_all()
        assert "r=5040" in nb_runner.get_output(2)
