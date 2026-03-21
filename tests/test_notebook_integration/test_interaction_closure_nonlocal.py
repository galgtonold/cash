"""
Batch 307: Closure and nonlocal interaction tests.
Tests that editing closures with nonlocal variables properly
invalidates downstream computations.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestClosureNonlocalInteraction:
    """Test closure/nonlocal patterns with cache invalidation."""

    def test_closure_capture_edit(self, nb_runner):
        """Editing closure-captured variable should propagate."""
        nb_runner.create_notebook([
            "factor = 3",
            "def make_multiplier():\n    f = factor\n    def mul(x):\n        return x * f\n    return mul",
            "m = make_multiplier()\nresult = m(7)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=21" in out

        nb_runner.set_cell_source(1, "factor = 10")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=70" in out

    def test_counter_closure_edit(self, nb_runner):
        """Editing a counter closure's starting value should propagate."""
        nb_runner.create_notebook([
            "start = 0",
            (
                "def make_counter():\n"
                "    count = start\n"
                "    def increment():\n"
                "        nonlocal count\n"
                "        count += 1\n"
                "        return count\n"
                "    return increment"
            ),
            "counter = make_counter()\nvals = [counter() for _ in range(3)]",
            "print(f'vals={vals}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "vals=[1, 2, 3]" in out

        nb_runner.set_cell_source(1, "start = 100")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "vals=[101, 102, 103]" in out

    def test_accumulator_closure_edit(self, nb_runner):
        """Editing accumulator initial value should propagate."""
        nb_runner.create_notebook([
            "initial = 10",
            (
                "def make_accumulator():\n"
                "    total = initial\n"
                "    def add(n):\n"
                "        nonlocal total\n"
                "        total += n\n"
                "        return total\n"
                "    return add"
            ),
            "acc = make_accumulator()\nresults = [acc(1), acc(2), acc(3)]",
            "print(f'results={results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "results=[11, 13, 16]" in out

        nb_runner.set_cell_source(1, "initial = 0")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "results=[1, 3, 6]" in out
