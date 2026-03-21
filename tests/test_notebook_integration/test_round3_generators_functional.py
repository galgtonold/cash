"""
Batch 26: Generator, iterator, and functional patterns — yield, send/throw,
itertools, functools, closures, higher-order functions.
"""
import pytest
import textwrap

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestGeneratorPatterns:
    """Test caching with generator functions and itertools."""

    def test_generator_consumed_to_list(self, nb_runner):
        """Generator consumed to list should cache."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def gen_range(n):
                    for i in range(n):
                        yield i * i
            """),
            textwrap.dedent("""\
                result = list(gen_range(5))
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[0, 1, 4, 9, 16]" in nb_runner.get_output(2)

        # Re-run - should use cache
        nb_runner.run_all()
        assert "[0, 1, 4, 9, 16]" in nb_runner.get_output(2)

    def test_generator_expression(self, nb_runner):
        """Generator expression consumed to sum."""
        nb_runner.create_notebook([
            "values = [1, 2, 3, 4, 5]",
            textwrap.dedent("""\
                total = sum(x**2 for x in values)
                print(total)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "55" in nb_runner.get_output(2)

    def test_itertools_chain(self, nb_runner):
        """itertools.chain across cells."""
        nb_runner.create_notebook([
            "import itertools",
            "a = [1, 2, 3]\nb = [4, 5, 6]",
            textwrap.dedent("""\
                combined = list(itertools.chain(a, b))
                print(combined)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[1, 2, 3, 4, 5, 6]" in nb_runner.get_output(3)

    def test_itertools_groupby(self, nb_runner):
        """itertools.groupby pattern."""
        nb_runner.create_notebook([
            "import itertools",
            textwrap.dedent("""\
                data = [('a', 1), ('a', 2), ('b', 3), ('b', 4), ('a', 5)]
                data.sort(key=lambda x: x[0])
                groups = {k: list(g) for k, g in itertools.groupby(data, key=lambda x: x[0])}
                print(sorted(groups.keys()))
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "['a', 'b']" in nb_runner.get_output(2)

    def test_generator_function_change_propagation(self, nb_runner):
        """Change generator function → re-run consumer."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def fib(n):
                    a, b = 0, 1
                    for _ in range(n):
                        yield a
                        a, b = b, a + b
            """),
            textwrap.dedent("""\
                result = list(fib(6))
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[0, 1, 1, 2, 3, 5]" in nb_runner.get_output(2)

        # Change to start from 1, 1
        nb_runner.set_cell_source(1, textwrap.dedent("""\
            def fib(n):
                a, b = 1, 1
                for _ in range(n):
                    yield a
                    a, b = b, a + b
        """))
        nb_runner.run_all()
        assert "[1, 1, 2, 3, 5, 8]" in nb_runner.get_output(2)


class TestClosurePatterns:
    """Test caching with closures and higher-order functions."""

    def test_closure_captures_outer_variable(self, nb_runner):
        """Closure capturing outer variable."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def make_multiplier(factor):
                    def mul(x):
                        return x * factor
                    return mul
            """),
            "double = make_multiplier(2)",
            textwrap.dedent("""\
                result = double(21)
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "42" in nb_runner.get_output(3)

    def test_closure_factory_change(self, nb_runner):
        """Change factory argument → closure output changes."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def make_adder(n):
                    def add(x):
                        return x + n
                    return add
            """),
            "add5 = make_adder(5)",
            textwrap.dedent("""\
                result = add5(10)
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "15" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "add5 = make_adder(100)")
        nb_runner.run_all()
        assert "110" in nb_runner.get_output(3)

    def test_higher_order_map_filter(self, nb_runner):
        """Higher-order functions: map + filter."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
            textwrap.dedent("""\
                evens = list(filter(lambda x: x % 2 == 0, data))
                squared = list(map(lambda x: x**2, evens))
                print(squared)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[4, 16, 36, 64, 100]" in nb_runner.get_output(2)


class TestFunctoolsPatterns:
    """Test caching with functools utilities."""

    def test_functools_partial(self, nb_runner):
        """functools.partial creates a new callable."""
        nb_runner.create_notebook([
            "from functools import partial",
            textwrap.dedent("""\
                def power(base, exp):
                    return base ** exp
                square = partial(power, exp=2)
                cube = partial(power, exp=3)
            """),
            textwrap.dedent("""\
                print(f"sq={square(5)} cu={cube(3)}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sq=25 cu=27" in nb_runner.get_output(3)

    def test_functools_reduce(self, nb_runner):
        """functools.reduce across cells."""
        nb_runner.create_notebook([
            "from functools import reduce",
            "numbers = [1, 2, 3, 4, 5]",
            textwrap.dedent("""\
                product = reduce(lambda a, b: a * b, numbers)
                print(product)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "120" in nb_runner.get_output(3)

    def test_functools_lru_cache(self, nb_runner):
        """@lru_cache decorator on a function."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "499500" in nb_runner.get_output(3)


class TestComprehensionEdgeCases:
    """Test complex comprehension patterns."""

    def test_nested_dict_comprehension(self, nb_runner):
        """Nested dict comprehension depending on cross-cell vars."""
        nb_runner.create_notebook([
            "keys = ['a', 'b', 'c']\nvalues = [1, 2, 3]",
            textwrap.dedent("""\
                mapping = {k: {f'{k}_{i}': v * i for i in range(1, 4)}
                           for k, v in zip(keys, values)}
                print(mapping['b'])
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "'b_1': 2" in output

    def test_walrus_in_comprehension(self, nb_runner):
        """Walrus operator (:=) in list comprehension."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]",
            textwrap.dedent("""\
                result = [y for x in data if (y := x**2) > 25]
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[36, 49, 64, 81, 100]" in nb_runner.get_output(2)

    def test_set_comprehension(self, nb_runner):
        """Set comprehension with cross-cell input."""
        nb_runner.create_notebook([
            "words = ['hello', 'HELLO', 'World', 'world', 'FOO']",
            textwrap.dedent("""\
                unique_lower = {w.lower() for w in words}
                print(sorted(unique_lower))
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "['foo', 'hello', 'world']" in nb_runner.get_output(2)
