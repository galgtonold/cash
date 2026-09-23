"""Functional programming patterns — compose, pipe, curry, partial application."""

import textwrap

import pytest


@pytest.mark.stress
class TestFunctionalComposition:
    """Test function composition and piping."""

    def test_compose_change_propagates(self, nb_runner):
        """Changing composed function propagates."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            textwrap.dedent("""\
            def step1(x):
                return x + 100
            def step2(x):
                return x * 2
        """),
        )
        nb_runner.run_all()
        assert "result=210" in nb_runner.get_output(2)


@pytest.mark.stress
class TestCurrying:
    """Test currying and partial application patterns."""

    def test_manual_curry(self, nb_runner):
        """Manual currying implementation."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=6 r2=6 r3=6" in nb_runner.get_output(2)
