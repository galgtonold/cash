"""Batch 60: Functional programming patterns — compose, pipe, curry, partial application."""
import textwrap
import pytest


@pytest.mark.stress
class TestFunctionalComposition:
    """Test function composition and piping."""

    def test_compose_functions(self, nb_runner):
        """Compose multiple functions."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def compose(*fns):
                    def composed(x):
                        result = x
                        for fn in reversed(fns):
                            result = fn(result)
                        return result
                    return composed

                double = lambda x: x * 2
                add_one = lambda x: x + 1
                square = lambda x: x ** 2
            """),
            textwrap.dedent("""\
                transform = compose(square, add_one, double)
                result = transform(3)  # double(3)=6, add_one(6)=7, square(7)=49
                print(f"result={result}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=49" in nb_runner.get_output(2)

    def test_pipe_functions(self, nb_runner):
        """Pipe data through function chain."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def pipe(value, *fns):
                    for fn in fns:
                        value = fn(value)
                    return value
            """),
            textwrap.dedent("""\
                result = pipe(
                    [3, 1, 4, 1, 5, 9, 2, 6],
                    sorted,
                    lambda xs: [x for x in xs if x > 2],
                    lambda xs: [x * 10 for x in xs],
                    sum
                )
                print(f"result={result}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # sorted: [1,1,2,3,4,5,6,9], filter>2: [3,4,5,6,9], *10: [30,40,50,60,90], sum: 270
        assert "result=270" in nb_runner.get_output(2)

    def test_compose_change_propagates(self, nb_runner):
        """Changing composed function propagates."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def step1(x):
                    return x + 10
                def step2(x):
                    return x * 2
            """),
            textwrap.dedent("""\
                result = step2(step1(5))
                print(f"result={result}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, textwrap.dedent("""\
            def step1(x):
                return x + 100
            def step2(x):
                return x * 2
        """))
        nb_runner.run_all()
        assert "result=210" in nb_runner.get_output(2)


@pytest.mark.stress
class TestCurrying:
    """Test currying and partial application patterns."""

    def test_manual_curry(self, nb_runner):
        """Manual currying implementation."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def curry(fn):
                    import inspect
                    params = inspect.signature(fn).parameters
                    n = len(params)
                    def curried(*args):
                        if len(args) >= n:
                            return fn(*args[:n])
                        return lambda *more: curried(*args, *more)
                    return curried

                @curry
                def add3(a, b, c):
                    return a + b + c
            """),
            textwrap.dedent("""\
                r1 = add3(1, 2, 3)
                r2 = add3(1)(2)(3)
                r3 = add3(1, 2)(3)
                print(f"r1={r1} r2={r2} r3={r3}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=6 r2=6 r3=6" in nb_runner.get_output(2)

    def test_reduce_patterns(self, nb_runner):
        """functools.reduce across cells."""
        nb_runner.create_notebook([
            "from functools import reduce",
            textwrap.dedent("""\
                product = reduce(lambda a, b: a * b, range(1, 6))
                nested = [[1, 2], [3, 4], [5, 6]]
                flat = reduce(lambda a, b: a + b, nested)
                print(f"product={product} flat={flat}")
            """),
            textwrap.dedent("""\
                # Running total using reduce
                data = [10, 20, 30, 40, 50]
                from itertools import accumulate
                running = list(accumulate(data))
                print(f"running={running}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "product=120" in nb_runner.get_output(2)
        assert "flat=[1, 2, 3, 4, 5, 6]" in nb_runner.get_output(2)
        assert "running=[10, 30, 60, 100, 150]" in nb_runner.get_output(3)

    def test_map_filter_chain(self, nb_runner):
        """Chained map/filter operations."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                data = range(1, 11)
                step1 = list(map(lambda x: x ** 2, data))
                step2 = list(filter(lambda x: x > 20, step1))
                step3 = list(map(str, step2))
                print(f"step1={step1}")
            """),
            textwrap.dedent("""\
                result = ', '.join(step3)
                print(f"result={result}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "step1=[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]" in nb_runner.get_output(1)
        assert "25" in nb_runner.get_output(2)
