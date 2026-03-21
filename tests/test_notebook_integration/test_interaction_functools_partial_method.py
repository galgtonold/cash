"""Batch 499: functools partial and partialmethod."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestFunctoolsPartialMethod:
    def test_partial_function(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import partial",
            "def power(base, exp):\n    return base ** exp\nsquare = partial(power, exp=2)\ncube = partial(power, exp=3)\nprint(f'sq5={square(5)} cube3={cube(3)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "sq5=25" in out
        assert "cube3=27" in out

    def test_partial_chain(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import partial",
            "def greet(greeting, name, punct='.'):\n    return f'{greeting}, {name}{punct}'\nhello = partial(greet, 'Hello')\nhello_excited = partial(hello, punct='!')\nprint(f'r1={hello(\"Alice\")} r2={hello_excited(\"Bob\")}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "r1=Hello, Alice." in out
        assert "r2=Hello, Bob!" in out

    def test_partial_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import partial",
            "def add(a, b): return a + b\nadd5 = partial(add, 5)\nresult = add5(10)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=15" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "def add(a, b): return a + b\nadd100 = partial(add, 100)\nresult = add100(50)\nprint(f'result={result}')")
        nb_runner.run_all()
        assert "result=150" in nb_runner.get_output(2)
