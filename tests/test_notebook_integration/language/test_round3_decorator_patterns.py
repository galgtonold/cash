"""
Decorator patterns — function decorators, class decorators,
decorator with arguments, stacked decorators, method decorators.
"""

import textwrap

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestFunctionDecorators:
    """Test function decorators across cells."""

    def test_decorator_with_arguments(self, nb_runner):
        """Decorator factory with arguments across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def repeat(n):
                    def decorator(fn):
                        def wrapper(*args, **kwargs):
                            results = []
                            for _ in range(n):
                                results.append(fn(*args, **kwargs))
                            return results
                        return wrapper
                    return decorator
            """),
                textwrap.dedent("""\
                @repeat(3)
                def greet(name):
                    return f"Hi {name}"
            """),
                textwrap.dedent("""\
                result = greet("World")
                print(result)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "['Hi World', 'Hi World', 'Hi World']" in nb_runner.get_output(3)

    def test_decorator_change_propagation(self, nb_runner):
        """Change decorator → function behavior updates."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def multiply_result(factor):
                    def decorator(fn):
                        def wrapper(*args, **kwargs):
                            return fn(*args, **kwargs) * factor
                        return wrapper
                    return decorator
            """),
                textwrap.dedent("""\
                @multiply_result(2)
                def compute(x):
                    return x + 10
            """),
                textwrap.dedent("""\
                print(compute(5))
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # (5+10)*2 = 30
        assert "30" in nb_runner.get_output(3)

        # Change multiplier
        nb_runner.set_cell_source(
            2,
            textwrap.dedent("""\
            @multiply_result(5)
            def compute(x):
                return x + 10
        """),
        )
        nb_runner.run_all()
        # (5+10)*5 = 75
        assert "75" in nb_runner.get_output(3)


class TestClassDecorators:
    """Test class decorators across cells."""

    def test_class_decorator(self, nb_runner):
        """Class decorator that adds method.

        Note: Cash attaches _cash_hash to objects, so vars(self) includes it.
        We filter it out in the __repr__ to keep the test clean.
        """
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                def add_repr(cls):
                    def __repr__(self):
                        attrs = ', '.join(
                            f'{k}={v!r}' for k, v in vars(self).items()
                            if not k.startswith('_cash')
                        )
                        return f'{cls.__name__}({attrs})'
                    cls.__repr__ = __repr__
                    return cls
            """),
                textwrap.dedent("""\
                @add_repr
                class Point:
                    def __init__(self, x, y):
                        self.x = x
                        self.y = y
            """),
                textwrap.dedent("""\
                p = Point(3, 4)
                print(repr(p))
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Point(x=3, y=4)" in nb_runner.get_output(3)
