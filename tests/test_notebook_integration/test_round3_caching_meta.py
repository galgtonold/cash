"""Batch 55: Caching meta-patterns — cash caching code that itself caches (lru_cache, memoize)."""
import textwrap
import pytest


@pytest.mark.stress
class TestLruCachePatterns:
    """Test functools.lru_cache interaction with cash."""

    def test_lru_cache_basic(self, nb_runner):
        """lru_cache decorated function across cells."""
        nb_runner.create_notebook([
            "from functools import lru_cache",
            textwrap.dedent("""\
                @lru_cache(maxsize=128)
                def fibonacci(n):
                    if n < 2:
                        return n
                    return fibonacci(n-1) + fibonacci(n-2)

                fib_30 = fibonacci(30)
                print(f"fib_30={fib_30}")
            """),
            textwrap.dedent("""\
                fib_50 = fibonacci(50)
                info = fibonacci.cache_info()
                print(f"fib_50={fib_50}")
                print(f"hits={info.hits} misses={info.misses}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "fib_30=832040" in nb_runner.get_output(2)
        assert "fib_50=12586269025" in nb_runner.get_output(3)

    def test_lru_cache_change_function(self, nb_runner):
        """Changing lru_cache function propagates."""
        nb_runner.create_notebook([
            "from functools import lru_cache",
            textwrap.dedent("""\
                @lru_cache(maxsize=32)
                def process(x):
                    return x * 2
            """),
            textwrap.dedent("""\
                result = process(5)
                print(f"result={result}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=10" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, textwrap.dedent("""\
            @lru_cache(maxsize=32)
            def process(x):
                return x * 3
        """))
        nb_runner.run_all()
        assert "result=15" in nb_runner.get_output(3)

    def test_cache_decorator(self, nb_runner):
        """functools.cache (Python 3.9+) pattern."""
        nb_runner.create_notebook([
            "from functools import cache",
            textwrap.dedent("""\
                @cache
                def factorial(n):
                    return n * factorial(n-1) if n else 1

                results = [factorial(i) for i in range(8)]
                print(f"results={results}")
            """),
            textwrap.dedent("""\
                big = factorial(20)
                print(f"big={big}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[1, 1, 2, 6, 24, 120, 720, 5040]" in nb_runner.get_output(2)
        assert "big=2432902008176640000" in nb_runner.get_output(3)


@pytest.mark.stress
class TestManualMemoization:
    """Test manual memoization patterns."""

    def test_dict_memoize(self, nb_runner):
        """Manual dict-based memoization."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=1024 r2=243 r3=1024 cache_size=2" in nb_runner.get_output(2)

    def test_memoize_decorator(self, nb_runner):
        """Custom memoize decorator across cells."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=328350" in nb_runner.get_output(2)
        assert "cache_size=2" in nb_runner.get_output(2)


@pytest.mark.stress
class TestCachedPropertyPattern:
    """Test cached_property and similar patterns."""

    def test_cached_property_class(self, nb_runner):
        """Class with cached_property (Python 3.8+)."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from functools import cached_property

                class DataSet:
                    def __init__(self, data):
                        self.data = data

                    @cached_property
                    def mean(self):
                        return sum(self.data) / len(self.data)

                    @cached_property
                    def variance(self):
                        m = self.mean
                        return sum((x - m) ** 2 for x in self.data) / len(self.data)
            """),
            textwrap.dedent("""\
                ds = DataSet([1, 2, 3, 4, 5])
                print(f"mean={ds.mean} var={ds.variance}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mean=3.0 var=2.0" in nb_runner.get_output(2)

    def test_manual_cached_property(self, nb_runner):
        """Manual cached property implementation."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class lazy_property:
                    def __init__(self, fn):
                        self.fn = fn
                        self.attr_name = f'_lazy_{fn.__name__}'
                    def __get__(self, obj, cls):
                        if obj is None:
                            return self
                        if not hasattr(obj, self.attr_name):
                            setattr(obj, self.attr_name, self.fn(obj))
                        return getattr(obj, self.attr_name)

                class Config:
                    def __init__(self, items):
                        self.items = items

                    @lazy_property
                    def processed(self):
                        return sorted(set(self.items))
            """),
            textwrap.dedent("""\
                cfg = Config([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])
                p = cfg.processed
                print(f"processed={p}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "processed=[1, 2, 3, 4, 5, 6, 9]" in nb_runner.get_output(2)
