"""Function decorators, stacked decorators and functools.wraps across cells."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress]


@pytest.mark.core
@pytest.mark.timeout(30)
class TestDecoratorEdits:
    """Basic decorator + edit scenarios."""

    def test_edit_decorated_function(self, nb_runner):
        """Edit a decorated function body."""
        nb_runner.create_notebook(
            [
                "def logged(func):\n    def wrapper(*a, **kw):\n        return func(*a, **kw)\n    return wrapper",
                "@logged\ndef compute(x):\n    return x * 2",
                "result = compute(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "@logged\ndef compute(x):\n    return x ** 2")
        nb_runner.run_all()
        assert "result = 25" in nb_runner.get_output(3)

    def test_edit_decorator(self, nb_runner):
        """Edit the decorator itself."""
        nb_runner.create_notebook(
            [
                "def multiplier(factor):\n    def decorator(func):\n        def wrapper(*a):\n            return func(*a) * factor\n        return wrapper\n    return decorator",
                "@multiplier(2)\ndef value(x):\n    return x",
                "result = value(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(3)

        # Change the multiplier implementation
        nb_runner.set_cell_source(
            1,
            "def multiplier(factor):\n    def decorator(func):\n        def wrapper(*a):\n            return func(*a) + factor\n        return wrapper\n    return decorator",
        )
        nb_runner.run_all()
        assert "result = 7" in nb_runner.get_output(3)

    def test_change_decorator_argument(self, nb_runner):
        """Change the decorator argument (factor)."""
        nb_runner.create_notebook(
            [
                "def scale(factor):\n    def decorator(func):\n        def wrapper(*a):\n            return func(*a) * factor\n        return wrapper\n    return decorator",
                "@scale(3)\ndef base(x):\n    return x",
                "result = base(10)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "@scale(10)\ndef base(x):\n    return x")
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(3)

    def test_remove_decorator(self, nb_runner):
        """Remove a decorator from a function."""
        nb_runner.create_notebook(
            [
                "def double(func):\n    def wrapper(*a):\n        return func(*a) * 2\n    return wrapper",
                "@double\ndef calc(x):\n    return x + 1",
                "result = calc(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # (5 + 1) * 2 = 12
        assert "result = 12" in nb_runner.get_output(3)

        # Remove decorator
        nb_runner.set_cell_source(2, "def calc(x):\n    return x + 1")
        nb_runner.run_all()
        assert "result = 6" in nb_runner.get_output(3)


@pytest.mark.integration
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


@pytest.mark.core
@pytest.mark.timeout(30)
class TestStackedDecorators:
    """Stacked decorators + edits."""

    def test_edit_with_stacked_decorators(self, nb_runner):
        """Edit function with two decorators stacked."""
        nb_runner.create_notebook(
            [
                "def add_one(func):\n    def wrapper(*a):\n        return func(*a) + 1\n    return wrapper",
                "def double(func):\n    def wrapper(*a):\n        return func(*a) * 2\n    return wrapper",
                "@double\n@add_one\ndef base(x):\n    return x",
                "result = base(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # base(5) = 5, add_one → 6, double → 12
        assert "result = 12" in nb_runner.get_output(4)

        # Edit base function
        nb_runner.set_cell_source(3, "@double\n@add_one\ndef base(x):\n    return x * 10")
        nb_runner.run_all()
        # base(5) = 50, add_one → 51, double → 102
        assert "result = 102" in nb_runner.get_output(4)

    def test_swap_decorator_order(self, nb_runner):
        """Swap the order of stacked decorators."""
        nb_runner.create_notebook(
            [
                "def add_one(func):\n    def wrapper(*a):\n        return func(*a) + 1\n    return wrapper",
                "def double(func):\n    def wrapper(*a):\n        return func(*a) * 2\n    return wrapper",
                "@double\n@add_one\ndef base(x):\n    return x",
                "result = base(5)\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # base(5) = 5, add_one → 6, double → 12
        assert "result = 12" in nb_runner.get_output(4)

        # Swap order: now add_one wraps double
        nb_runner.set_cell_source(3, "@add_one\n@double\ndef base(x):\n    return x")
        nb_runner.run_all()
        # base(5) = 5, double → 10, add_one → 11
        assert "result = 11" in nb_runner.get_output(4)


@pytest.mark.integration
class TestDecoratorStacking:
    """Stacking multiple decorators."""

    def test_triple_decorator_stack(self, nb_runner):
        """Three decorators stacked on one function."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "[log][time][valid]7" in out
        assert "[log][time][invalid]" in out

    def test_parameterized_decorator(self, nb_runner):
        """Decorator that takes arguments."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Hello, World!" in out
        assert out.count("Hello, World!") >= 3

    def test_class_decorator(self, nb_runner):
        """Class used as a decorator."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "r1=4950" in out
        assert "r2=4950" in out
        assert "r3=1225" in out
        assert "cache=2" in out

    def test_decorator_propagation(self, nb_runner):
        """Decorator behavior changes when upstream config changes."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[INFO] test message" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "prefix = 'ERROR'")
        nb_runner.run_cells([1, 2, 3])
        assert "[ERROR] test message" in nb_runner.get_output(3)

    def test_wraps_preservation(self, nb_runner):
        """functools.wraps preserves function metadata."""
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "name=documented_fn" in out
        assert "Multiply x by 2" in out
        assert "result=10" in out


@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestDecoratorStackEdits:
    """Editing stacked decorators."""

    def test_edit_decorator_order(self, nb_runner):
        """Edit the order of stacked decorators."""
        nb_runner.create_notebook(
            [
                "def add_prefix(fn):\n    def wrapper(*args, **kwargs):\n        return 'PREFIX_' + fn(*args, **kwargs)\n    return wrapper",
                "def add_suffix(fn):\n    def wrapper(*args, **kwargs):\n        return fn(*args, **kwargs) + '_SUFFIX'\n    return wrapper",
                "@add_prefix\n@add_suffix\ndef greet():\n    return 'hello'",
                "result = greet()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # add_suffix first: 'hello_SUFFIX', then add_prefix: 'PREFIX_hello_SUFFIX'
        assert "result = PREFIX_hello_SUFFIX" in nb_runner.get_output(4)

        # Swap decorator order
        nb_runner.set_cell_source(3, "@add_suffix\n@add_prefix\ndef greet():\n    return 'hello'")
        nb_runner.run_all()
        # add_prefix first: 'PREFIX_hello', then add_suffix: 'PREFIX_hello_SUFFIX'
        assert "result = PREFIX_hello_SUFFIX" in nb_runner.get_output(4)

    def test_add_decorator(self, nb_runner):
        """Add a new decorator to an existing function."""
        nb_runner.create_notebook(
            [
                "def uppercase(fn):\n    def wrapper(*args, **kwargs):\n        return fn(*args, **kwargs).upper()\n    return wrapper",
                "def repeat(fn):\n    def wrapper(*args, **kwargs):\n        r = fn(*args, **kwargs)\n        return r + r\n    return wrapper",
                "@uppercase\ndef say():\n    return 'hi'",
                "result = say()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = HI" in nb_runner.get_output(4)

        # Add repeat decorator
        nb_runner.set_cell_source(3, "@uppercase\n@repeat\ndef say():\n    return 'hi'")
        nb_runner.run_all()
        # repeat first: 'hihi', then uppercase: 'HIHI'
        assert "result = HIHI" in nb_runner.get_output(4)

    def test_edit_decorator_implementation(self, nb_runner):
        """Edit the implementation of a stacked decorator."""
        nb_runner.create_notebook(
            [
                "def wrap_brackets(fn):\n    def w(*a, **k):\n        return '[' + fn(*a, **k) + ']'\n    return w",
                "def wrap_parens(fn):\n    def w(*a, **k):\n        return '(' + fn(*a, **k) + ')'\n    return w",
                "@wrap_brackets\n@wrap_parens\ndef msg():\n    return 'hi'",
                "result = msg()\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [(hi)]" in nb_runner.get_output(4)

        # Change brackets to braces
        nb_runner.set_cell_source(
            1,
            "def wrap_brackets(fn):\n    def w(*a, **k):\n        return '{' + fn(*a, **k) + '}'\n    return w",
        )
        nb_runner.run_all()
        assert "result = {(hi)}" in nb_runner.get_output(4)


@pytest.mark.timeout(90)
class TestDecoratorChainWraps:
    """Test decorator chaining with wraps across cells."""

    def test_decorator_chain(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: define decorators
                "from functools import wraps\ndef logged(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        wrapper.calls = getattr(wrapper, 'calls', 0) + 1\n        return func(*args, **kwargs)\n    wrapper.calls = 0\n    return wrapper\ndef validated(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        for a in args:\n            if not isinstance(a, (int, float)):\n                raise TypeError(f'Expected number, got {type(a).__name__}')\n        return func(*args, **kwargs)\n    return wrapper\nprint('decorators defined')",
                # Cell 2: apply stacked decorators
                "@logged\n@validated\ndef add(a, b):\n    '''Add two numbers.'''\n    return a + b\nresult1 = add(3, 4)\nresult2 = add(10, 20)\nprint(f'r1={result1}')\nprint(f'r2={result2}')\nprint(f'calls={add.calls}')\nprint(f'name={add.__name__}')\nprint(f'doc={add.__doc__}')",
                # Cell 3: check metadata preservation
                "has_wraps = add.__name__ == 'add' and add.__doc__ == 'Add two numbers.'\nprint(f'preserved={has_wraps}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "r1=7" in out2
        assert "r2=30" in out2
        assert "calls=2" in out2
        assert "name=add" in out2
        out3 = nb_runner.get_output(3)
        assert "preserved=True" in out3

    def test_decorator_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import wraps\ndef double_result(func):\n    @wraps(func)\n    def wrapper(*a, **kw):\n        return func(*a, **kw) * 2\n    return wrapper\nprint('double_result defined')",
                "@double_result\ndef compute(x):\n    return x + 1\nresult = compute(5)\nprint(f'result={result}')",
                "is_doubled = result == (5 + 1) * 2\nprint(f'doubled={is_doubled}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=12" in nb_runner.get_output(2)
        assert "doubled=True" in nb_runner.get_output(3)

        # Edit function
        nb_runner.set_cell_source(
            2, "@double_result\ndef compute(x):\n    return x * 3\nresult = compute(5)\nprint(f'result={result}')"
        )
        nb_runner.run_cells([2, 3])
        assert "result=30" in nb_runner.get_output(2)

    def test_decorator_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import wraps\ndef memoize(func):\n    cache = {}\n    @wraps(func)\n    def wrapper(n):\n        if n not in cache:\n            cache[n] = func(n)\n        return cache[n]\n    return wrapper\nprint('memoize defined')",
                "@memoize\ndef fib(n):\n    if n < 2: return n\n    return fib(n-1) + fib(n-2)\nresult = fib(10)\nprint(f'fib10={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "fib10=55" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "fib10=55" in nb_runner.get_output(2)


@pytest.mark.timeout(90)
class TestDecoratorWraps:
    """functools.wraps and decorator metadata preservation."""

    def test_wraps_preserves_name(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import wraps\ndef my_decorator(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        return func(*args, **kwargs)\n    return wrapper\n@my_decorator\ndef greet(name):\n    '''Greet someone.'''\n    return f'Hello, {name}!'",
                "result = greet('World')\nfn_name = greet.__name__\nfn_doc = greet.__doc__\nprint(f'result={result} name={fn_name} doc={fn_doc}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "result=Hello, World!" in out
        assert "name=greet" in out
        assert "doc=Greet someone." in out

    def test_decorator_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import wraps\ndef double_result(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        return func(*args, **kwargs) * 2\n    return wrapper",
                "@double_result\ndef compute(x):\n    return x + 10",
                "result = compute(5)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=30" in nb_runner.get_output(3)
        # Edit decorator
        nb_runner.set_cell_source(
            1,
            "from functools import wraps\ndef double_result(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        return func(*args, **kwargs) * 3\n    return wrapper",
        )
        nb_runner.run_all()
        assert "result=45" in nb_runner.get_output(3)

    def test_stacked_decorators(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import wraps\ndef add_one(func):\n    @wraps(func)\n    def wrapper(*args):\n        return func(*args) + 1\n    return wrapper\ndef times_two(func):\n    @wraps(func)\n    def wrapper(*args):\n        return func(*args) * 2\n    return wrapper",
                "@times_two\n@add_one\ndef base(x):\n    return x",
                "result = base(5)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # base(5) -> add_one(5) = 6 -> times_two(6) = 12
        assert "result=12" in nb_runner.get_output(3)


@pytest.mark.timeout(90)
class TestFunctoolsWrapsChain:
    """functools wraps and decorator chaining."""

    def test_wraps_preserves_metadata(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import wraps",
                "def timer(fn):\n    @wraps(fn)\n    def wrapper(*a, **kw):\n        return fn(*a, **kw)\n    return wrapper\n@timer\ndef add(x, y):\n    '''Add two numbers'''\n    return x + y\nprint(f'name={add.__name__} doc={add.__doc__}')\nprint(f'result={add(3, 4)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "name=add" in out
        assert "doc=Add two numbers" in out
        assert "result=7" in out

    def test_decorator_chain(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import wraps",
                "def bold(fn):\n    @wraps(fn)\n    def w(*a, **k): return f'<b>{fn(*a, **k)}</b>'\n    return w\ndef italic(fn):\n    @wraps(fn)\n    def w(*a, **k): return f'<i>{fn(*a, **k)}</i>'\n    return w\n@bold\n@italic\ndef greet(name): return f'Hello {name}'\nresult = greet('World')\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=<b><i>Hello World</i></b>" in nb_runner.get_output(2)

    def test_wraps_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "from functools import wraps",
                "def double(fn):\n    @wraps(fn)\n    def w(*a, **k): return fn(*a, **k) * 2\n    return w\n@double\ndef val(): return 5\nprint(f'result={val()}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=10" in nb_runner.get_output(2)
        nb_runner.set_cell_source(
            2,
            "def double(fn):\n    @wraps(fn)\n    def w(*a, **k): return fn(*a, **k) * 2\n    return w\n@double\ndef val(): return 50\nprint(f'result={val()}')",
        )
        nb_runner.run_all()
        assert "result=100" in nb_runner.get_output(2)


@pytest.mark.timeout(90)
class TestFunctoolsWrapsUpdate:
    """Test functools.wraps decorator metadata across cells."""

    def test_wraps_metadata(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: create decorated function
                "import functools\ndef my_decorator(func):\n    @functools.wraps(func)\n    def wrapper(*args, **kwargs):\n        return func(*args, **kwargs)\n    return wrapper\n\n@my_decorator\ndef greet(name):\n    '''Say hello'''\n    return f'Hello, {name}!'\n\nprint(f'name={greet.__name__}')\nprint(f'doc={greet.__doc__}')",
                # Cell 2: use the decorated function
                "result = greet('Alice')\nprint(f'result={result}')",
                # Cell 3: check wrapped attribute
                "has_wrapped = hasattr(greet, '__wrapped__')\nprint(f'has_wrapped={has_wrapped}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "name=greet" in out1
        assert "doc=Say hello" in out1
        out2 = nb_runner.get_output(2)
        assert "result=Hello, Alice!" in out2
        out3 = nb_runner.get_output(3)
        assert "has_wrapped=True" in out3

    def test_wraps_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import functools\ndef logged(func):\n    @functools.wraps(func)\n    def wrapper(*args):\n        return func(*args)\n    return wrapper\n\n@logged\ndef add(a, b):\n    return a + b\nresult = add(3, 4)\nprint(f'result={result}')",
                "doubled = result * 2\nprint(f'doubled={doubled}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=7" in nb_runner.get_output(1)
        assert "doubled=14" in nb_runner.get_output(2)

        # Edit function
        nb_runner.set_cell_source(
            1,
            "import functools\ndef logged(func):\n    @functools.wraps(func)\n    def wrapper(*args):\n        return func(*args)\n    return wrapper\n\n@logged\ndef add(a, b):\n    return a + b + 10\nresult = add(3, 4)\nprint(f'result={result}')",
        )
        nb_runner.run_cells([1, 2])
        assert "result=17" in nb_runner.get_output(1)
        assert "doubled=34" in nb_runner.get_output(2)

    def test_wraps_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import functools\ndef timer(func):\n    @functools.wraps(func)\n    def wrapper(*args):\n        return func(*args)\n    return wrapper\n\n@timer\ndef square(n):\n    return n * n\nval = square(5)\nprint(f'val={val}')",
                "name_ok = square.__name__ == 'square'\nprint(f'name_ok={name_ok}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=25" in nb_runner.get_output(1)
        assert "name_ok=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "name_ok=True" in nb_runner.get_output(2)


@pytest.mark.integration
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
