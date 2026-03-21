"""Batch 410: closure scope and nonlocal variable capture."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestClosureScopeCapture:
    def test_closure_nonlocal(self, nb_runner):
        nb_runner.create_notebook([
            "def make_counter():\n    count = 0\n    def increment():\n        nonlocal count\n        count += 1\n        return count\n    return increment",
            "counter = make_counter()\nr1 = counter()\nr2 = counter()\nr3 = counter()\nprint(f'r1={r1} r2={r2} r3={r3}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=1 r2=2 r3=3" in nb_runner.get_output(2)

    def test_closure_captures(self, nb_runner):
        nb_runner.create_notebook([
            "multiplier = 10",
            "def scale(x):\n    return x * multiplier\nresult = scale(5)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=50" in nb_runner.get_output(2)

    def test_closure_edit(self, nb_runner):
        nb_runner.create_notebook([
            "def make_adder(n):\n    def add(x):\n        return x + n\n    return add",
            "add5 = make_adder(5)\nresult = add5(10)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=15" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "add100 = make_adder(100)\nresult = add100(10)\nprint(f'result={result}')")
        nb_runner.run_all()
        assert "result=110" in nb_runner.get_output(2)
