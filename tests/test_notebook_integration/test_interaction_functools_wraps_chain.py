"""Batch 513: functools wraps and decorator chaining."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestFunctoolsWrapsChain:
    def test_wraps_preserves_metadata(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import wraps",
            "def timer(fn):\n    @wraps(fn)\n    def wrapper(*a, **kw):\n        return fn(*a, **kw)\n    return wrapper\n@timer\ndef add(x, y):\n    '''Add two numbers'''\n    return x + y\nprint(f'name={add.__name__} doc={add.__doc__}')\nprint(f'result={add(3, 4)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "name=add" in out
        assert "doc=Add two numbers" in out
        assert "result=7" in out

    def test_decorator_chain(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import wraps",
            "def bold(fn):\n    @wraps(fn)\n    def w(*a, **k): return f'<b>{fn(*a, **k)}</b>'\n    return w\ndef italic(fn):\n    @wraps(fn)\n    def w(*a, **k): return f'<i>{fn(*a, **k)}</i>'\n    return w\n@bold\n@italic\ndef greet(name): return f'Hello {name}'\nresult = greet('World')\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=<b><i>Hello World</i></b>" in nb_runner.get_output(2)

    def test_wraps_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import wraps",
            "def double(fn):\n    @wraps(fn)\n    def w(*a, **k): return fn(*a, **k) * 2\n    return w\n@double\ndef val(): return 5\nprint(f'result={val()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=10" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "def double(fn):\n    @wraps(fn)\n    def w(*a, **k): return fn(*a, **k) * 2\n    return w\n@double\ndef val(): return 50\nprint(f'result={val()}')")
        nb_runner.run_all()
        assert "result=100" in nb_runner.get_output(2)
