"""Batch 74: Functools advanced — partial, lru_cache, reduce, singledispatch across cells."""
import textwrap
import pytest


@pytest.mark.stress
class TestPartialPatterns:
    """Test functools.partial patterns across cells."""

    def test_partial_function(self, nb_runner):
        """Partial application across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from functools import partial

                def power(base, exp):
                    return base ** exp

                square = partial(power, exp=2)
                cube = partial(power, exp=3)
                print(f"sq5={square(5)} cb3={cube(3)}")
            """),
            textwrap.dedent("""\
                results = [square(i) for i in range(1, 6)]
                print(f"squares={results}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sq5=25 cb3=27" in nb_runner.get_output(1)
        assert "squares=[1, 4, 9, 16, 25]" in nb_runner.get_output(2)

    def test_lru_cache_cross_cell(self, nb_runner):
        """lru_cache function used across cells."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "fib30=832040" in nb_runner.get_output(1)
        out2 = nb_runner.get_output(2)
        assert "fib50=12586269025" in out2

    def test_singledispatch(self, nb_runner):
        """singledispatch across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from functools import singledispatch

                @singledispatch
                def process(data):
                    return f"unknown: {type(data).__name__}"

                @process.register(int)
                def _(data):
                    return f"int: {data * 2}"

                @process.register(str)
                def _(data):
                    return f"str: {data.upper()}"

                @process.register(list)
                def _(data):
                    return f"list: {len(data)} items"
            """),
            textwrap.dedent("""\
                results = [process(42), process("hello"), process([1,2,3]), process(3.14)]
                for r in results:
                    print(r)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "int: 84" in out
        assert "str: HELLO" in out
        assert "list: 3 items" in out
        assert "unknown: float" in out


@pytest.mark.stress
class TestFunctoolsChaining:
    """Test functools chaining patterns."""

    def test_reduce_accumulator(self, nb_runner):
        """functools.reduce across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from functools import reduce

                data = [1, 2, 3, 4, 5]
                product = reduce(lambda a, b: a * b, data)
                factorial_5 = reduce(lambda a, b: a * b, range(1, 6))
                print(f"product={product} factorial={factorial_5}")
            """),
            textwrap.dedent("""\
                # Running max using reduce
                from functools import reduce
                sequence = [3, 1, 4, 1, 5, 9, 2, 6]
                running_max = reduce(lambda acc, x: acc + [max(acc[-1], x)], sequence[1:], [sequence[0]])
                print(f"running_max={running_max}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "product=120 factorial=120" in nb_runner.get_output(1)
        assert "running_max=[3, 3, 4, 4, 5, 9, 9, 9]" in nb_runner.get_output(2)

    def test_partial_propagation(self, nb_runner):
        """Partial function propagation on change."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[0, 2, 4, 6, 8]" in nb_runner.get_output(2)

        # Change to triple
        nb_runner.set_cell_source(1, textwrap.dedent("""\
            from functools import partial

            def multiply(x, factor):
                return x * factor

            double = partial(multiply, factor=3)
        """))
        nb_runner.run_cells([1, 2])
        assert "results=[0, 3, 6, 9, 12]" in nb_runner.get_output(2)
