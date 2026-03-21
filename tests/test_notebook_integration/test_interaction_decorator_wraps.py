"""Batch 388: functools.wraps and decorator metadata preservation."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDecoratorWraps:
    def test_wraps_preserves_name(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import wraps\ndef my_decorator(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        return func(*args, **kwargs)\n    return wrapper\n@my_decorator\ndef greet(name):\n    '''Greet someone.'''\n    return f'Hello, {name}!'",
            "result = greet('World')\nfn_name = greet.__name__\nfn_doc = greet.__doc__\nprint(f'result={result} name={fn_name} doc={fn_doc}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "result=Hello, World!" in out
        assert "name=greet" in out
        assert "doc=Greet someone." in out

    def test_decorator_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import wraps\ndef double_result(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        return func(*args, **kwargs) * 2\n    return wrapper",
            "@double_result\ndef compute(x):\n    return x + 10",
            "result = compute(5)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=30" in nb_runner.get_output(3)
        # Edit decorator
        nb_runner.set_cell_source(1, "from functools import wraps\ndef double_result(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        return func(*args, **kwargs) * 3\n    return wrapper")
        nb_runner.run_all()
        assert "result=45" in nb_runner.get_output(3)

    def test_stacked_decorators(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import wraps\ndef add_one(func):\n    @wraps(func)\n    def wrapper(*args):\n        return func(*args) + 1\n    return wrapper\ndef times_two(func):\n    @wraps(func)\n    def wrapper(*args):\n        return func(*args) * 2\n    return wrapper",
            "@times_two\n@add_one\ndef base(x):\n    return x",
            "result = base(5)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # base(5) -> add_one(5) = 6 -> times_two(6) = 12
        assert "result=12" in nb_runner.get_output(3)
