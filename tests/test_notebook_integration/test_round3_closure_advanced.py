"""Batch 90 – advanced closure, nonlocal, and scope edge cases."""

import textwrap, pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestAdvancedClosures:
    """Advanced closure and nonlocal patterns."""

    def test_closure_memoize(self, nb_runner):
        """Closure-based memoization."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def memoize(fn):
                    cache = {}
                    call_count = [0]
                    def wrapper(*args):
                        call_count[0] += 1
                        if args not in cache:
                            cache[args] = fn(*args)
                        return cache[args]
                    wrapper.calls = call_count
                    wrapper.cache = cache
                    return wrapper

                @memoize
                def fib(n):
                    if n < 2: return n
                    return fib(n - 1) + fib(n - 2)

                result = fib(20)
                cache_size = len(fib.cache)
            """),
            "print(f'fib20={result} cache_size={cache_size}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "fib20=6765" in out
        assert "cache_size=21" in out

    def test_nonlocal_accumulator(self, nb_runner):
        """Accumulator using nonlocal with multiple operations."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def make_accumulator():
                    history = []
                    total = 0
                    def add(val):
                        nonlocal total
                        total += val
                        history.append((val, total))
                        return total
                    def get_history():
                        return history[:]
                    return add, get_history

                add, get_hist = make_accumulator()
                add(10)
                add(20)
                add(-5)
                final = add(15)
                hist = get_hist()
            """),
            "print(f'final={final}')\nprint(f'hist={hist}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "final=40" in out
        assert "(10, 10)" in out
        assert "(20, 30)" in out

    def test_closure_chain(self, nb_runner):
        """Chain of closures composing functions."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def compose(*funcs):
                    def composed(x):
                        result = x
                        for f in reversed(funcs):
                            result = f(result)
                        return result
                    return composed

                add1 = lambda x: x + 1
                double = lambda x: x * 2
                square = lambda x: x ** 2

                pipe = compose(square, double, add1)
                results = [pipe(i) for i in range(5)]
            """),
            "print(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        # pipe(0) = square(double(add1(0))) = square(double(1)) = square(2) = 4
        # pipe(1) = square(double(2)) = square(4) = 16
        assert "4" in out
        assert "16" in out

    def test_scope_global_local_interaction(self, nb_runner):
        """Global vs local scope interactions across cells."""
        nb_runner.create_notebook([
            "MULTIPLIER = 10",
            textwrap.dedent("""\
                def compute(x):
                    local_offset = 5
                    return x * MULTIPLIER + local_offset

                r1 = compute(3)
                r2 = compute(7)
            """),
            "print(f'r1={r1} r2={r2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "r1=35" in out
        assert "r2=75" in out

    def test_closure_propagation_memoize(self, nb_runner):
        """Memoized closure with upstream change propagation."""
        nb_runner.create_notebook([
            "scale = 2",
            textwrap.dedent("""\
                def make_scaler(s):
                    cache = {}
                    def scaled(x):
                        if x not in cache:
                            cache[x] = x * s
                        return cache[x]
                    return scaled

                scaler = make_scaler(scale)
                vals = [scaler(i) for i in range(5)]
            """),
            "print(f'vals={vals}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[0, 2, 4, 6, 8]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "scale = 5")
        nb_runner.run_cells([1, 2, 3])
        assert "[0, 5, 10, 15, 20]" in nb_runner.get_output(3)
