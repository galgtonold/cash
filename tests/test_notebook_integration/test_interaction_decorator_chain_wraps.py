"""
Interaction test: decorator chaining with wraps and metadata.
Tests multiple decorators stacked, functools.wraps preservation,
and cross-cell decorated function behavior.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDecoratorChainWraps:
    """Test decorator chaining with wraps across cells."""

    def test_decorator_chain(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: define decorators
            "from functools import wraps\ndef logged(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        wrapper.calls = getattr(wrapper, 'calls', 0) + 1\n        return func(*args, **kwargs)\n    wrapper.calls = 0\n    return wrapper\ndef validated(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        for a in args:\n            if not isinstance(a, (int, float)):\n                raise TypeError(f'Expected number, got {type(a).__name__}')\n        return func(*args, **kwargs)\n    return wrapper\nprint('decorators defined')",
            # Cell 2: apply stacked decorators
            "@logged\n@validated\ndef add(a, b):\n    '''Add two numbers.'''\n    return a + b\nresult1 = add(3, 4)\nresult2 = add(10, 20)\nprint(f'r1={result1}')\nprint(f'r2={result2}')\nprint(f'calls={add.calls}')\nprint(f'name={add.__name__}')\nprint(f'doc={add.__doc__}')",
            # Cell 3: check metadata preservation
            "has_wraps = add.__name__ == 'add' and add.__doc__ == 'Add two numbers.'\nprint(f'preserved={has_wraps}')",
        ])
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
        nb_runner.create_notebook([
            "from functools import wraps\ndef double_result(func):\n    @wraps(func)\n    def wrapper(*a, **kw):\n        return func(*a, **kw) * 2\n    return wrapper\nprint('double_result defined')",
            "@double_result\ndef compute(x):\n    return x + 1\nresult = compute(5)\nprint(f'result={result}')",
            "is_doubled = result == (5 + 1) * 2\nprint(f'doubled={is_doubled}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=12" in nb_runner.get_output(2)
        assert "doubled=True" in nb_runner.get_output(3)

        # Edit function
        nb_runner.set_cell_source(2, "@double_result\ndef compute(x):\n    return x * 3\nresult = compute(5)\nprint(f'result={result}')")
        nb_runner.run_cells([2, 3])
        assert "result=30" in nb_runner.get_output(2)

    def test_decorator_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import wraps\ndef memoize(func):\n    cache = {}\n    @wraps(func)\n    def wrapper(n):\n        if n not in cache:\n            cache[n] = func(n)\n        return cache[n]\n    return wrapper\nprint('memoize defined')",
            "@memoize\ndef fib(n):\n    if n < 2: return n\n    return fib(n-1) + fib(n-2)\nresult = fib(10)\nprint(f'fib10={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "fib10=55" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "fib10=55" in nb_runner.get_output(2)
