"""
Interaction test: class with __call__ and callable protocol.
Tests classes implementing __call__, callable checks,
and cross-cell callable composition.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestCallableProtocol:
    """Test callable protocol via __call__ across cells."""

    def test_callable_class(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: define callable class
            "class Multiplier:\n    def __init__(self, factor):\n        self.factor = factor\n        self.call_count = 0\n    def __call__(self, value):\n        self.call_count += 1\n        return value * self.factor\n    def __repr__(self):\n        return f'Multiplier(x{self.factor})'\nprint('Multiplier defined')",
            # Cell 2: use as callable
            "double = Multiplier(2)\ntriple = Multiplier(3)\nresults = [double(5), triple(5), double(10), triple(10)]\nprint(f'results={results}')\nprint(f'double_calls={double.call_count}')\nprint(f'triple_calls={triple.call_count}')",
            # Cell 3: compose
            "composed = double(triple(7))\nprint(f'composed={composed}')\nprint(f'all_callable={all(callable(f) for f in [double, triple])}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "results=[10, 15, 20, 30]" in out2
        assert "double_calls=2" in out2
        assert "triple_calls=2" in out2
        out3 = nb_runner.get_output(3)
        assert "composed=42" in out3
        assert "all_callable=True" in out3

    def test_callable_edit(self, nb_runner):
        nb_runner.create_notebook([
            "class Adder:\n    def __init__(self, n):\n        self.n = n\n    def __call__(self, x):\n        return x + self.n\nprint('Adder defined')",
            "add5 = Adder(5)\nresult = add5(10)\nprint(f'result={result}')",
            "doubled = add5(result)\nprint(f'doubled={doubled}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=15" in nb_runner.get_output(2)
        assert "doubled=20" in nb_runner.get_output(3)

        # Edit adder value
        nb_runner.set_cell_source(2, "add5 = Adder(10)\nresult = add5(10)\nprint(f'result={result}')")
        nb_runner.run_cells([2, 3])
        assert "result=20" in nb_runner.get_output(2)
        assert "doubled=30" in nb_runner.get_output(3)

    def test_callable_cache(self, nb_runner):
        nb_runner.create_notebook([
            "class Squarer:\n    def __call__(self, x):\n        return x ** 2\nsq = Squarer()\nprint(f'callable={callable(sq)}')",
            "vals = [sq(i) for i in range(5)]\nprint(f'vals={vals}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "callable=True" in nb_runner.get_output(1)
        assert "vals=[0, 1, 4, 9, 16]" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "vals=[0, 1, 4, 9, 16]" in nb_runner.get_output(2)
