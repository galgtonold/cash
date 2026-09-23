"""Functools advanced — partial, lru_cache, reduce, singledispatch across cells."""

import textwrap

import pytest


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
