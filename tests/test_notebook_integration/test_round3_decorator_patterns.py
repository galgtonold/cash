"""
Batch 38: Decorator patterns — function decorators, class decorators,
decorator with arguments, stacked decorators, method decorators.
"""
import pytest
import textwrap

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestFunctionDecorators:
    """Test function decorators across cells."""

    def test_simple_logging_decorator(self, nb_runner):
        """Simple logging decorator defined and used across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def logged(fn):
                    def wrapper(*args, **kwargs):
                        print(f"Calling {fn.__name__}")
                        result = fn(*args, **kwargs)
                        print(f"Done: {result}")
                        return result
                    return wrapper
            """),
            textwrap.dedent("""\
                @logged
                def add(a, b):
                    return a + b
            """),
            textwrap.dedent("""\
                result = add(3, 7)
                print(f"result={result}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "Calling add" in output
        assert "result=10" in output

    def test_decorator_with_arguments(self, nb_runner):
        """Decorator factory with arguments across cells."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "['Hi World', 'Hi World', 'Hi World']" in nb_runner.get_output(3)

    def test_stacked_decorators(self, nb_runner):
        """Multiple decorators stacked on a function."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def uppercase(fn):
                    def wrapper(*args, **kwargs):
                        result = fn(*args, **kwargs)
                        return result.upper()
                    return wrapper

                def exclaim(fn):
                    def wrapper(*args, **kwargs):
                        result = fn(*args, **kwargs)
                        return result + "!"
                    return wrapper
            """),
            textwrap.dedent("""\
                @uppercase
                @exclaim
                def greet(name):
                    return f"hello {name}"
            """),
            textwrap.dedent("""\
                # exclaim runs first (bottom up), then uppercase
                result = greet("world")
                print(result)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "HELLO WORLD!" in nb_runner.get_output(3)

    def test_decorator_change_propagation(self, nb_runner):
        """Change decorator → function behavior updates."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # (5+10)*2 = 30
        assert "30" in nb_runner.get_output(3)

        # Change multiplier
        nb_runner.set_cell_source(2, textwrap.dedent("""\
            @multiply_result(5)
            def compute(x):
                return x + 10
        """))
        nb_runner.run_all()
        # (5+10)*5 = 75
        assert "75" in nb_runner.get_output(3)

    def test_preserving_metadata_with_wraps(self, nb_runner):
        """functools.wraps preserves function metadata."""
        nb_runner.create_notebook([
            "from functools import wraps",
            textwrap.dedent("""\
                def timer(fn):
                    @wraps(fn)
                    def wrapper(*args, **kwargs):
                        return fn(*args, **kwargs)
                    return wrapper
            """),
            textwrap.dedent("""\
                @timer
                def process(data):
                    \"\"\"Process the data.\"\"\"
                    return sum(data)
            """),
            textwrap.dedent("""\
                print(f"name={process.__name__} doc={process.__doc__}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(4)
        assert "name=process" in output
        assert "doc=Process the data." in output


class TestClassDecorators:
    """Test class decorators across cells."""

    def test_class_decorator(self, nb_runner):
        """Class decorator that adds method.
        
        Note: Cash attaches _cash_hash to objects, so vars(self) includes it.
        We filter it out in the __repr__ to keep the test clean.
        """
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Point(x=3, y=4)" in nb_runner.get_output(3)


class TestMethodDecorators:
    """Test method-level decorators."""

    def test_staticmethod_classmethod(self, nb_runner):
        """@staticmethod and @classmethod across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class MathUtils:
                    factor = 2
                    
                    @staticmethod
                    def add(a, b):
                        return a + b
                    
                    @classmethod
                    def scaled_add(cls, a, b):
                        return cls.add(a, b) * cls.factor
            """),
            textwrap.dedent("""\
                r1 = MathUtils.add(3, 4)
                r2 = MathUtils.scaled_add(3, 4)
                print(f"add={r1} scaled={r2}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "add=7 scaled=14" in nb_runner.get_output(2)

    def test_property_with_setter(self, nb_runner):
        """@property with @setter across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class Temperature:
                    def __init__(self, celsius):
                        self._celsius = celsius
                    
                    @property
                    def fahrenheit(self):
                        return self._celsius * 9/5 + 32
                    
                    @property
                    def celsius(self):
                        return self._celsius
                    
                    @celsius.setter
                    def celsius(self, value):
                        self._celsius = value
            """),
            textwrap.dedent("""\
                t = Temperature(100)
                print(f"F={t.fahrenheit}")
                t.celsius = 0
                print(f"F={t.fahrenheit}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "F=212.0" in output
        assert "F=32.0" in output
