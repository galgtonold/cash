"""Batch 92 – decorator stacking and parameterized decorators."""

import textwrap, pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestDecoratorStacking:
    """Stacking multiple decorators."""

    def test_triple_decorator_stack(self, nb_runner):
        """Three decorators stacked on one function."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def logger(fn):
                    def wrapper(*args, **kwargs):
                        result = fn(*args, **kwargs)
                        return f"[log]{result}"
                    return wrapper

                def timer(fn):
                    def wrapper(*args, **kwargs):
                        result = fn(*args, **kwargs)
                        return f"[time]{result}"
                    return wrapper

                def validator(fn):
                    def wrapper(*args, **kwargs):
                        for a in args:
                            if not isinstance(a, (int, float)):
                                return "[invalid]"
                        return f"[valid]{fn(*args, **kwargs)}"
                    return wrapper

                @logger
                @timer
                @validator
                def compute(a, b):
                    return a + b

                r1 = compute(3, 4)
                r2 = compute("x", 4)
            """),
            "print(f'r1={r1}')\nprint(f'r2={r2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "[log][time][valid]7" in out
        assert "[log][time][invalid]" in out

    def test_parameterized_decorator(self, nb_runner):
        """Decorator that takes arguments."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                def repeat(n):
                    def decorator(fn):
                        def wrapper(*args, **kwargs):
                            results = [fn(*args, **kwargs) for _ in range(n)]
                            return results
                        return wrapper
                    return decorator

                @repeat(3)
                def greet(name):
                    return f"Hello, {name}!"

                output = greet("World")
            """),
            "print(f'output={output}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Hello, World!" in out
        assert out.count("Hello, World!") >= 3

    def test_class_decorator(self, nb_runner):
        """Class used as a decorator."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                class CacheDecorator:
                    def __init__(self, fn):
                        self.fn = fn
                        self.cache = {}
                    def __call__(self, *args):
                        if args not in self.cache:
                            self.cache[args] = self.fn(*args)
                        return self.cache[args]

                @CacheDecorator
                def expensive(n):
                    return sum(range(n))

                r1 = expensive(100)
                r2 = expensive(100)
                r3 = expensive(50)
                cache_size = len(expensive.cache)
            """),
            "print(f'r1={r1} r2={r2} r3={r3} cache={cache_size}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "r1=4950" in out
        assert "r2=4950" in out
        assert "r3=1225" in out
        assert "cache=2" in out

    def test_decorator_propagation(self, nb_runner):
        """Decorator behavior changes when upstream config changes."""
        nb_runner.create_notebook([
            "prefix = 'INFO'",
            textwrap.dedent("""\
                def tag(fn):
                    def wrapper(*args, **kwargs):
                        return f"[{prefix}] {fn(*args, **kwargs)}"
                    return wrapper

                @tag
                def message(text):
                    return text

                result = message("test message")
            """),
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[INFO] test message" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "prefix = 'ERROR'")
        nb_runner.run_cells([1, 2, 3])
        assert "[ERROR] test message" in nb_runner.get_output(3)

    def test_wraps_preservation(self, nb_runner):
        """functools.wraps preserves function metadata."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                from functools import wraps

                def my_decorator(fn):
                    @wraps(fn)
                    def wrapper(*args, **kwargs):
                        return fn(*args, **kwargs)
                    return wrapper

                @my_decorator
                def documented_fn(x: int) -> int:
                    '''Multiply x by 2.'''
                    return x * 2

                name = documented_fn.__name__
                doc = documented_fn.__doc__
                result = documented_fn(5)
            """),
            "print(f'name={name} doc={doc} result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "name=documented_fn" in out
        assert "Multiply x by 2" in out
        assert "result=10" in out
