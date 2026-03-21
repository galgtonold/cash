"""Batch 351: global/nonlocal keyword scoping with edits."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestGlobalNonlocalScope:
    def test_global_in_function(self, nb_runner):
        nb_runner.create_notebook([
            "counter = 0\ndef increment(n):\n    global counter\n    counter += n\n    return counter",
            "r1 = increment(5)\nr2 = increment(3)\nprint(f'r1={r1} r2={r2} counter={counter}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r1=5 r2=8 counter=8" in nb_runner.get_output(2)

    def test_nonlocal_in_nested(self, nb_runner):
        nb_runner.create_notebook([
            "def outer():\n    total = 0\n    def inner(x):\n        nonlocal total\n        total += x\n        return total\n    return inner",
            "acc = outer()\nresults = [acc(i) for i in [10, 20, 30]]\nprint(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results=[10, 30, 60]" in nb_runner.get_output(2)

    def test_scope_edit_function(self, nb_runner):
        nb_runner.create_notebook([
            "def make_adder(base):\n    def add(x):\n        return base + x\n    return add",
            "add5 = make_adder(5)\nresult = add5(10)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=15" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "def make_adder(base):\n    def add(x):\n        return base * x\n    return add")
        nb_runner.run_all()
        assert "result=50" in nb_runner.get_output(2)
