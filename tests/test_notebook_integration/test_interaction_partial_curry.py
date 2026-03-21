"""
Batch 305: Partial application and currying interaction tests.
Tests that editing partial functions and curried arguments properly
invalidates downstream computations.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestPartialCurryingInteraction:
    """Test partial application and currying with cache invalidation."""

    def test_partial_edit_fixed_arg(self, nb_runner):
        """Editing a partial function's fixed argument should propagate."""
        nb_runner.create_notebook([
            "from functools import partial\ndef multiply(x, y):\n    return x * y",
            "double = partial(multiply, 2)",
            "result = double(5)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=10" in out

        nb_runner.set_cell_source(2, "double = partial(multiply, 10)")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=50" in out

    def test_curry_chain_edit(self, nb_runner):
        """Editing a currying chain should propagate."""
        nb_runner.create_notebook([
            "def add(a):\n    def inner(b):\n        return a + b\n    return inner",
            "add5 = add(5)",
            "result = add5(10)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=15" in out

        nb_runner.set_cell_source(2, "add5 = add(50)")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=60" in out

    def test_partial_with_kwargs_edit(self, nb_runner):
        """Editing partial with keyword arguments should propagate."""
        nb_runner.create_notebook([
            "from functools import partial\ndef greet(name, greeting='Hello', punctuation='!'):\n    return f'{greeting}, {name}{punctuation}'",
            "formal = partial(greet, greeting='Good day', punctuation='.')",
            "msg = formal('Alice')",
            "print(f'msg={msg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "msg=Good day, Alice." in out

        nb_runner.set_cell_source(2, "formal = partial(greet, greeting='Hey', punctuation='!!')")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "msg=Hey, Alice!!" in out
