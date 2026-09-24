"""functools: partial, lru_cache, singledispatch and reduce across cells."""

import textwrap

import pytest


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestPartialApplyEdits:
    """Editing partial application patterns."""

    def test_edit_partial_arg(self, nb_runner):
        """Edit a partial function's fixed argument."""
        nb_runner.create_notebook(
            [
                "from functools import partial\ndef power(base, exp):\n    return base ** exp\nsquare = partial(power, exp=2)",
                "result = square(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 25" in nb_runner.get_output(2)

        # Change to cube
        nb_runner.set_cell_source(
            1,
            "from functools import partial\ndef power(base, exp):\n    return base ** exp\nsquare = partial(power, exp=3)",
        )
        nb_runner.run_all()
        assert "result = 125" in nb_runner.get_output(2)

    def test_edit_closure_value(self, nb_runner):
        """Edit a closure factory's captured value."""
        nb_runner.create_notebook(
            [
                "def make_adder(n):\n    def adder(x):\n        return x + n\n    return adder\nadd5 = make_adder(5)",
                "result = add5(10)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 15" in nb_runner.get_output(2)

        # Change to add10
        nb_runner.set_cell_source(
            1, "def make_adder(n):\n    def adder(x):\n        return x + n\n    return adder\nadd5 = make_adder(10)"
        )
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(2)

    def test_edit_curry_chain(self, nb_runner):
        """Edit curried function application."""
        nb_runner.create_notebook(
            [
                "def mul(a):\n    def inner(b):\n        return a * b\n    return inner\ndouble = mul(2)",
                "result = double(7)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 14" in nb_runner.get_output(2)

        # Change multiplier
        nb_runner.set_cell_source(
            1, "def mul(a):\n    def inner(b):\n        return a * b\n    return inner\ndouble = mul(5)"
        )
        nb_runner.run_all()
        assert "result = 35" in nb_runner.get_output(2)

    def test_edit_partial_with_kwargs(self, nb_runner):
        """Edit partial with keyword arguments."""
        nb_runner.create_notebook(
            [
                "from functools import partial\ndef greet(name, greeting, punct):\n    return f'{greeting} {name}{punct}'\nhello = partial(greet, greeting='Hello', punct='!')",
                "msg = hello('Alice')\nprint(msg)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Hello Alice!" in nb_runner.get_output(2)

        # Change greeting
        nb_runner.set_cell_source(
            1,
            "from functools import partial\ndef greet(name, greeting, punct):\n    return f'{greeting} {name}{punct}'\nhello = partial(greet, greeting='Hi', punct='.')",
        )
        nb_runner.run_all()
        assert "Hi Alice." in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestFunctoolsPartial:
    """functools.partial and partialmethod."""

    def test_partial_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import partial\ndef power(base, exp):\n    return base ** exp",
                "square = partial(power, exp=2)\ncube = partial(power, exp=3)\nprint(f'sq5={square(5)} cu3={cube(3)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sq5=25" in nb_runner.get_output(2)
        assert "cu3=27" in nb_runner.get_output(2)

    def test_partial_chain(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import partial\ndef greet(greeting, name, punct):\n    return f'{greeting}, {name}{punct}'",
                "hello = partial(greet, 'Hello')\nhello_exc = partial(hello, punct='!')\nresult = hello_exc('Alice')\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=Hello, Alice!" in nb_runner.get_output(2)

    def test_partial_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import partial\ndef multiply(a, b):\n    return a * b",
                "double = partial(multiply, 2)\nresult = double(7)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=14" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from functools import partial\ndef multiply(a, b):\n    return a * b")
        nb_runner.set_cell_source(2, "triple = partial(multiply, 3)\nresult = triple(7)\nprint(f'result={result}')")
        nb_runner.run_all()
        assert "result=21" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestFunctoolsPartialMethod:
    """functools partial and partialmethod."""

    def test_partial_function(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import partial",
                "def power(base, exp):\n    return base ** exp\nsquare = partial(power, exp=2)\ncube = partial(power, exp=3)\nprint(f'sq5={square(5)} cube3={cube(3)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "sq5=25" in out
        assert "cube3=27" in out

    def test_partial_chain(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import partial",
                "def greet(greeting, name, punct='.'):\n    return f'{greeting}, {name}{punct}'\nhello = partial(greet, 'Hello')\nhello_excited = partial(hello, punct='!')\nprint(f'r1={hello(\"Alice\")} r2={hello_excited(\"Bob\")}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "r1=Hello, Alice." in out
        assert "r2=Hello, Bob!" in out

    def test_partial_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import partial",
                "def add(a, b): return a + b\nadd5 = partial(add, 5)\nresult = add5(10)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=15" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2, "def add(a, b): return a + b\nadd100 = partial(add, 100)\nresult = add100(50)\nprint(f'result={result}')"
        )
        nb_runner.run_all()
        assert "result=150" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestPartialPartialmethod:
    """Test functools partial and partialmethod across cells."""

    def test_partial_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: partial
                "from functools import partial\ndef power(base, exp):\n    return base ** exp\n\nsquare = partial(power, exp=2)\ncube = partial(power, exp=3)\nprint(f'square_5={square(5)}')\nprint(f'cube_3={cube(3)}')",
                # Cell 2: partial with multiple args
                "def greet(greeting, name, punctuation='!'):\n    return f'{greeting}, {name}{punctuation}'\n\nhello = partial(greet, 'Hello')\nresult = hello('World')\nresult2 = hello('Python', punctuation='.')\nprint(f'result={result}')\nprint(f'result2={result2}')",
                # Cell 3: use partials from cell 1
                "vals = [2, 3, 4, 5]\nsquares = list(map(square, vals))\ncubes = list(map(cube, vals))\nprint(f'squares={squares}')\nprint(f'cubes={cubes}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "square_5=25" in out1
        assert "cube_3=27" in out1
        out2 = nb_runner.get_output(2)
        assert "result=Hello, World!" in out2
        assert "result2=Hello, Python." in out2
        out3 = nb_runner.get_output(3)
        assert "squares=[4, 9, 16, 25]" in out3
        assert "cubes=[8, 27, 64, 125]" in out3

    def test_partial_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import partial\ndef add(a, b):\n    return a + b\nadd10 = partial(add, 10)\nresult = add10(5)\nprint(f'result={result}')",
                "doubled = result * 2\nprint(f'doubled={doubled}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=15" in nb_runner.get_output(1)
        assert "doubled=30" in nb_runner.get_output(2)

        # Edit partial
        nb_runner.set_cell_source(
            1,
            "from functools import partial\ndef add(a, b):\n    return a + b\nadd100 = partial(add, 100)\nresult = add100(5)\nprint(f'result={result}')",
        )
        nb_runner.run_cells([1, 2])
        assert "result=105" in nb_runner.get_output(1)
        assert "doubled=210" in nb_runner.get_output(2)

    def test_partial_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import partial\ndef multiply(a, b):\n    return a * b\ndouble = partial(multiply, 2)\nresult = double(21)\nprint(f'result={result}')",
                "is_42 = result == 42\nprint(f'is_42={is_42}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=42" in nb_runner.get_output(1)
        assert "is_42=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "is_42=True" in nb_runner.get_output(2)


@pytest.mark.stress
class TestPartialPatterns:
    """Test functools.partial patterns across cells."""

    def test_lru_cache_cross_cell(self, nb_runner):
        """lru_cache function used across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                from functools import lru_cache

                @lru_cache(maxsize=128)
                def fib(n):
                    if n < 2:
                        return n
                    return fib(n - 1) + fib(n - 2)

                result = fib(30)
                print(f"fib30={result}")
            """),
                textwrap.dedent("""\
                info = fib.cache_info()
                print(f"hits={info.hits} misses={info.misses}")
                fib50 = fib(50)
                print(f"fib50={fib50}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "fib30=832040" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "fib50=12586269025" in out2


@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestPartialCurryingInteraction:
    """Test partial application and currying with cache invalidation."""

    def test_partial_edit_fixed_arg(self, nb_runner):
        """Editing a partial function's fixed argument should propagate."""
        nb_runner.create_notebook(
            [
                "from functools import partial\ndef multiply(x, y):\n    return x * y",
                "double = partial(multiply, 2)",
                "result = double(5)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=10" in out

        nb_runner.set_cell_source(2, "double = partial(multiply, 10)")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=50" in out

    def test_curry_chain_edit(self, nb_runner):
        """Editing a currying chain should propagate."""
        nb_runner.create_notebook(
            [
                "def add(a):\n    def inner(b):\n        return a + b\n    return inner",
                "add5 = add(5)",
                "result = add5(10)",
                "print(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=15" in out

        nb_runner.set_cell_source(2, "add5 = add(50)")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=60" in out

    def test_partial_with_kwargs_edit(self, nb_runner):
        """Editing partial with keyword arguments should propagate."""
        nb_runner.create_notebook(
            [
                "from functools import partial\ndef greet(name, greeting='Hello', punctuation='!'):\n    return f'{greeting}, {name}{punctuation}'",
                "formal = partial(greet, greeting='Good day', punctuation='.')",
                "msg = formal('Alice')",
                "print(f'msg={msg}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "msg=Good day, Alice." in out

        nb_runner.set_cell_source(2, "formal = partial(greet, greeting='Hey', punctuation='!!')")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "msg=Hey, Alice!!" in out


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestFunctoolsPatterns:
    """functools pattern edit propagation."""

    def test_partial_edit(self, nb_runner):
        """Edit partial application, downstream updates."""
        nb_runner.create_notebook(
            [
                "from functools import partial\ndef power(base, exp):\n    return base ** exp",
                "square = partial(power, exp=2)",
                "results = [square(x) for x in [2, 3, 4, 5]]\nprint(f'results = {results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results = [4, 9, 16, 25]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "square = partial(power, exp=3)")
        nb_runner.run_all()
        assert "results = [8, 27, 64, 125]" in nb_runner.get_output(3)

    def test_partial_base_function_edit(self, nb_runner):
        """Edit the base function used in partial."""
        nb_runner.create_notebook(
            [
                "from functools import partial\ndef combine(a, b, sep):\n    return f'{a}{sep}{b}'",
                "dash_join = partial(combine, sep='-')",
                "result = dash_join('hello', 'world')\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = hello-world" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            1,
            "from functools import partial\ndef combine(a, b, sep):\n    return f'{a.upper()}{sep}{b.upper()}'",
        )
        nb_runner.run_all()
        assert "result = HELLO-WORLD" in nb_runner.get_output(3)

    def test_cached_function_edit(self, nb_runner):
        """Edit function logic, even with lru_cache behavior changes."""
        nb_runner.create_notebook(
            [
                "def expensive(n):\n    return sum(range(n))",
                "results = [expensive(x) for x in [10, 100, 1000]]\nprint(f'results = {results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results = [45, 4950, 499500]" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def expensive(n):\n    return sum(range(n)) * 2",
        )
        nb_runner.run_all()
        assert "results = [90, 9900, 999000]" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestFunctoolsLruCache:
    """functools.lru_cache and cache usage."""

    def test_lru_cache_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import lru_cache\n@lru_cache(maxsize=32)\ndef fib(n):\n    if n < 2: return n\n    return fib(n-1) + fib(n-2)",
                "r1 = fib(10)\nr2 = fib(20)\ninfo = fib.cache_info()\nprint(f'r1={r1} r2={r2} hits={info.hits}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "r1=55" in out
        assert "r2=6765" in out

    def test_lru_cache_clear(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import lru_cache\ncall_count = 0\n@lru_cache(maxsize=4)\ndef square(n):\n    global call_count\n    call_count += 1\n    return n * n",
                "r1 = square(3)\nr2 = square(3)\nsquare.cache_clear()\nr3 = square(3)\nprint(f'r1={r1} r2={r2} r3={r3} calls={call_count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "r1=9" in out
        assert "calls=2" in out

    def test_lru_cache_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import lru_cache\n@lru_cache\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
                "r = factorial(5)\nprint(f'r={r}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r=120" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "r = factorial(7)\nprint(f'r={r}')")
        nb_runner.run_all()
        assert "r=5040" in nb_runner.get_output(2)


@pytest.mark.stress
class TestLruCachePatterns:
    """Test functools.lru_cache interaction with cash."""

    def test_lru_cache_basic(self, nb_runner):
        """lru_cache decorated function across cells."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "fib_30=832040" in nb_runner.get_output(2)
        assert "fib_50=12586269025" in nb_runner.get_output(3)

    def test_lru_cache_change_function(self, nb_runner):
        """Changing lru_cache function propagates."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=10" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2,
            textwrap.dedent("""\
            @lru_cache(maxsize=32)
            def process(x):
                return x * 3
        """),
        )
        nb_runner.run_all()
        assert "result=15" in nb_runner.get_output(3)

    def test_cache_decorator(self, nb_runner):
        """functools.cache (Python 3.9+) pattern."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[1, 1, 2, 6, 24, 120, 720, 5040]" in nb_runner.get_output(2)
        assert "big=2432902008176640000" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestFunctoolsCache:
    """functools.lru_cache, partial application, and caching interaction."""

    def test_lru_cache_basic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import lru_cache\n@lru_cache(maxsize=None)\ndef fib(n):\n    if n < 2:\n        return n\n    return fib(n-1) + fib(n-2)",
                "result = fib(10)\nprint(f'fib10={result}')",
                "result2 = fib(15)\nprint(f'fib15={result2}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "fib10=55" in nb_runner.get_output(2)
        assert "fib15=610" in nb_runner.get_output(3)

    def test_lru_cache_edit_function(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import lru_cache\n@lru_cache(maxsize=128)\ndef compute(x):\n    return x ** 2",
                "vals = [compute(i) for i in range(5)]\nprint(f'vals={vals}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "vals=[0, 1, 4, 9, 16]" in nb_runner.get_output(2)
        # Edit function
        nb_runner.set_cell_source(
            1, "from functools import lru_cache\n@lru_cache(maxsize=128)\ndef compute(x):\n    return x ** 3"
        )
        nb_runner.run_all()
        assert "vals=[0, 1, 8, 27, 64]" in nb_runner.get_output(2)

    def test_partial_application(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import partial\ndef power(base, exp):\n    return base ** exp\nsquare = partial(power, exp=2)\ncube = partial(power, exp=3)",
                "s = square(5)\nc = cube(3)\nprint(f's={s} c={c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "s=25 c=27" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.stress
class TestLruCacheAcrossCells:
    """Test caching with functools utilities."""

    def test_functools_lru_cache(self, nb_runner):
        """@lru_cache decorator on a function."""
        nb_runner.create_notebook(
            [
                "from functools import lru_cache",
                textwrap.dedent("""\
                @lru_cache(maxsize=128)
                def expensive(n):
                    return sum(range(n))
            """),
                textwrap.dedent("""\
                result = expensive(1000)
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "499500" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.integration
class TestFunctionalPatterns:
    """Test functional programming patterns across cells."""

    def test_lru_cache_decorator(self, nb_runner):
        """lru_cache decorated function across cells."""
        nb_runner.create_notebook(
            [
                "from functools import lru_cache",
                textwrap.dedent("""\
                @lru_cache(maxsize=128)
                def fib(n):
                    if n < 2:
                        return n
                    return fib(n-1) + fib(n-2)
            """),
                textwrap.dedent("""\
                result = fib(30)
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "832040" in nb_runner.get_output(3)

    def test_higher_order_function_change(self, nb_runner):
        """Changing a higher-order function's component should invalidate."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def apply_twice(f, x):
                    return f(f(x))
            """),
                textwrap.dedent("""\
                def increment(x):
                    return x + 1
            """),
                textwrap.dedent("""\
                result = apply_twice(increment, 5)
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "7" in nb_runner.get_output(3)

        nb_runner.set_cell_source(
            2,
            textwrap.dedent("""\
            def increment(x):
                return x + 10
        """),
        )
        nb_runner.run_all()
        assert "25" in nb_runner.get_output(3)  # increment(increment(5)) = 5+10+10=25


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestFunctoolsReduceOps:
    """Test functools.reduce with various operators across cells."""

    def test_reduce_ops(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: basic reduce operations
                "from functools import reduce\nimport operator\nnums = [1, 2, 3, 4, 5]\nproduct = reduce(operator.mul, nums)\ntotal = reduce(operator.add, nums)\nprint(f'product={product}')\nprint(f'total={total}')",
                # Cell 2: reduce with initial value
                "concatenated = reduce(lambda acc, x: acc + str(x), nums, 'nums:')\nmax_val = reduce(lambda a, b: a if a > b else b, nums)\nprint(f'concat={concatenated}')\nprint(f'max={max_val}')",
                # Cell 3: nested reduce
                "matrix = [[1, 2], [3, 4], [5, 6]]\nflat = reduce(operator.add, matrix)\nflat_sum = reduce(operator.add, flat)\nprint(f'flat={flat}')\nprint(f'flat_sum={flat_sum}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "product=120" in out1
        assert "total=15" in out1
        out2 = nb_runner.get_output(2)
        assert "concat=nums:12345" in out2
        assert "max=5" in out2
        out3 = nb_runner.get_output(3)
        assert "flat=[1, 2, 3, 4, 5, 6]" in out3
        assert "flat_sum=21" in out3

    def test_reduce_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import reduce\nimport operator\ndata = [2, 3, 4]\nresult = reduce(operator.mul, data)\nprint(f'result={result}')",
                "info = f'product of {len(data)} numbers = {result}'\nprint(f'info={info}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "info=product of 3 numbers = 24" in nb_runner.get_output(2)

        # Edit data
        nb_runner.set_cell_source(
            1,
            "from functools import reduce\nimport operator\ndata = [2, 3, 4, 5]\nresult = reduce(operator.mul, data)\nprint(f'result={result}')",
        )
        nb_runner.run_cells([1, 2])
        assert "info=product of 4 numbers = 120" in nb_runner.get_output(2)

    def test_reduce_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import reduce\nwords = ['hello', 'world', 'foo']\nlongest = reduce(lambda a, b: a if len(a) >= len(b) else b, words)\nprint(f'longest={longest}')",
                "length = len(longest)\nprint(f'length={length}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "longest=hello" in nb_runner.get_output(1)
        assert "length=5" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "length=5" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestSingledispatch:
    """Test functools.singledispatch across cells."""

    def test_singledispatch_types(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: define singledispatch function
                "from functools import singledispatch\n@singledispatch\ndef format_val(val):\n    return f'unknown:{val}'\n@format_val.register(int)\ndef _(val):\n    return f'int:{val:,}'\n@format_val.register(float)\ndef _(val):\n    return f'float:{val:.2f}'\n@format_val.register(str)\ndef _(val):\n    return f'str:\"{val}\"'\n@format_val.register(list)\ndef _(val):\n    return f'list[{len(val)}]'\nprint('format_val defined')",
                # Cell 2: dispatch various types
                "results = [\n    format_val(42000),\n    format_val(3.14159),\n    format_val('hello'),\n    format_val([1, 2, 3]),\n    format_val((1, 2)),\n]\nfor r in results:\n    print(r)",
                # Cell 3: collect
                "types_seen = len(results)\nprint(f'dispatched={types_seen}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "int:42,000" in out2
        assert "float:3.14" in out2
        assert 'str:"hello"' in out2
        assert "list[3]" in out2
        assert "unknown:(1, 2)" in out2
        out3 = nb_runner.get_output(3)
        assert "dispatched=5" in out3

    def test_singledispatch_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import singledispatch\n@singledispatch\ndef stringify(val):\n    return str(val)\n@stringify.register(int)\ndef _(val):\n    return f'N={val}'\n@stringify.register(str)\ndef _(val):\n    return f'S={val}'\nprint('stringify defined')",
                "r1 = stringify(42)\nr2 = stringify('hi')\nprint(f'r1={r1}')\nprint(f'r2={r2}')",
                "combined = r1 + '|' + r2\nprint(f'combined={combined}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=N=42" in nb_runner.get_output(2)
        assert "combined=N=42|S=hi" in nb_runner.get_output(3)

        # Edit the call site to use different values
        nb_runner.set_cell_source(
            2, "r1 = stringify(1000)\nr2 = stringify('world')\nprint(f'r1={r1}')\nprint(f'r2={r2}')"
        )
        nb_runner.run_cells([2, 3])
        assert "r1=N=1000" in nb_runner.get_output(2)
        assert "combined=N=1000|S=world" in nb_runner.get_output(3)

    def test_singledispatch_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import singledispatch\n@singledispatch\ndef double(val):\n    return val\n@double.register(int)\ndef _(val):\n    return val * 2\n@double.register(str)\ndef _(val):\n    return val + val\nprint('double defined')",
                "results = [double(5), double('ab'), double(3.14)]\nprint(f'results={results}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[10, 'abab', 3.14]" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "results=[10, 'abab', 3.14]" in nb_runner.get_output(2)


@pytest.mark.stress
class TestFunctoolsChaining:
    """Test functools chaining patterns."""

    def test_partial_propagation(self, nb_runner):
        """Partial function propagation on change."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                from functools import partial

                def multiply(x, factor):
                    return x * factor

                double = partial(multiply, factor=2)
            """),
                textwrap.dedent("""\
                results = [double(i) for i in range(5)]
                print(f"results={results}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[0, 2, 4, 6, 8]" in nb_runner.get_output(2)

        # Change to triple
        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            from functools import partial

            def multiply(x, factor):
                return x * factor

            double = partial(multiply, factor=3)
        """),
        )
        nb_runner.run_cells([1, 2])
        assert "results=[0, 3, 6, 9, 12]" in nb_runner.get_output(2)
