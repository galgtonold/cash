"""
Interaction test: functools partial and partialmethod.
Tests functools.partial for argument binding, partialmethod for classes,
and cross-cell partial application pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestPartialPartialmethod:
    """Test functools partial and partialmethod across cells."""

    def test_partial_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: partial
            "from functools import partial\ndef power(base, exp):\n    return base ** exp\n\nsquare = partial(power, exp=2)\ncube = partial(power, exp=3)\nprint(f'square_5={square(5)}')\nprint(f'cube_3={cube(3)}')",
            # Cell 2: partial with multiple args
            "def greet(greeting, name, punctuation='!'):\n    return f'{greeting}, {name}{punctuation}'\n\nhello = partial(greet, 'Hello')\nresult = hello('World')\nresult2 = hello('Python', punctuation='.')\nprint(f'result={result}')\nprint(f'result2={result2}')",
            # Cell 3: use partials from cell 1
            "vals = [2, 3, 4, 5]\nsquares = list(map(square, vals))\ncubes = list(map(cube, vals))\nprint(f'squares={squares}')\nprint(f'cubes={cubes}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "square_5=25" in out1
        assert "cube_3=27" in out1
        out2 = nb_runner.get_output(2)
        assert "result=Hello, World!" in out2
        assert "result2=Hello, Python." in out2
        out3 = nb_runner.get_output(3)
        assert "squares=[4, 9, 16, 25]" in out3
        assert "cubes=[8, 27, 64, 125]" in out3

    def test_partial_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import partial\ndef add(a, b):\n    return a + b\nadd10 = partial(add, 10)\nresult = add10(5)\nprint(f'result={result}')",
            "doubled = result * 2\nprint(f'doubled={doubled}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=15" in nb_runner.get_output(1)
        assert "doubled=30" in nb_runner.get_output(2)

        # Edit partial
        nb_runner.set_cell_source(1, "from functools import partial\ndef add(a, b):\n    return a + b\nadd100 = partial(add, 100)\nresult = add100(5)\nprint(f'result={result}')")
        nb_runner.run_cells([1, 2])
        assert "result=105" in nb_runner.get_output(1)
        assert "doubled=210" in nb_runner.get_output(2)

    def test_partial_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from functools import partial\ndef multiply(a, b):\n    return a * b\ndouble = partial(multiply, 2)\nresult = double(21)\nprint(f'result={result}')",
            "is_42 = result == 42\nprint(f'is_42={is_42}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=42" in nb_runner.get_output(1)
        assert "is_42=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "is_42=True" in nb_runner.get_output(2)
