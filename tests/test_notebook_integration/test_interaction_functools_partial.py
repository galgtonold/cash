"""Batch 441: functools.partial and partialmethod."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestFunctoolsPartial:
    def test_partial_basic(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import partial\ndef power(base, exp):\n    return base ** exp",
            "square = partial(power, exp=2)\ncube = partial(power, exp=3)\nprint(f'sq5={square(5)} cu3={cube(3)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sq5=25" in nb_runner.get_output(2)
        assert "cu3=27" in nb_runner.get_output(2)

    def test_partial_chain(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import partial\ndef greet(greeting, name, punct):\n    return f'{greeting}, {name}{punct}'",
            "hello = partial(greet, 'Hello')\nhello_exc = partial(hello, punct='!')\nresult = hello_exc('Alice')\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=Hello, Alice!" in nb_runner.get_output(2)

    def test_partial_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import partial\ndef multiply(a, b):\n    return a * b",
            "double = partial(multiply, 2)\nresult = double(7)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=14" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "from functools import partial\ndef multiply(a, b):\n    return a * b")
        nb_runner.set_cell_source(2, "triple = partial(multiply, 3)\nresult = triple(7)\nprint(f'result={result}')")
        nb_runner.run_all()
        assert "result=21" in nb_runner.get_output(2)
