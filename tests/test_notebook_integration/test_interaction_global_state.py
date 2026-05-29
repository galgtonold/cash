"""Batch 125 – Global state & side-effect interaction tests.

Tests that exercise global variables, print statements, accumulation
patterns, and other side-effecting code with cell edits.
"""

import pytest

pytestmark = [pytest.mark.core, pytest.mark.stress, pytest.mark.timeout(30)]


class TestGlobalStateEdits:
    """Global state manipulation + cell edits."""

    def test_global_list_accumulation(self, nb_runner):
        """Accumulate into a global list, edit accumulation steps."""
        nb_runner.create_notebook([
            "results = []",
            "results.append(1)  # step 1",
            "results.append(2)  # step 2",
            "results.append(3)  # step 3",
            "print(f'results = {results}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "results = [1, 2, 3]" in nb_runner.get_output(5)

    def test_global_dict_update(self, nb_runner):
        """Update a global dict across cells, edit one update."""
        nb_runner.create_notebook([
            "config = {}",
            "config['a'] = 1",
            "config['b'] = 2",
            "total = config['a'] + config['b']\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 3" in nb_runner.get_output(4)

        nb_runner.set_cell_source(2, "config['a'] = 100")
        nb_runner.run_all()
        assert "total = 102" in nb_runner.get_output(4)


class TestPrintOutputEdits:
    """Print output + cell edits."""

    def test_edit_print_format(self, nb_runner):
        """Edit the format of a print statement."""
        nb_runner.create_notebook([
            "x = 42",
            "print(f'The answer is {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "The answer is 42" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "print(f'x = {x}')")
        nb_runner.run_all()
        assert "x = 42" in nb_runner.get_output(2)

    def test_multiple_prints_edit(self, nb_runner):
        """Cell with multiple prints, edit one value."""
        nb_runner.create_notebook([
            "a = 1\nb = 2\nc = 3",
            "print(f'a={a}')\nprint(f'b={b}')\nprint(f'c={c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(2)
        assert "a=1" in output
        assert "b=2" in output
        assert "c=3" in output

        nb_runner.set_cell_source(1, "a = 10\nb = 20\nc = 30")
        nb_runner.run_all()
        output2 = nb_runner.get_output(2)
        assert "a=10" in output2
        assert "b=20" in output2
        assert "c=30" in output2


class TestCounterPatterns:
    """Counter / running total patterns + edits."""

    def test_running_total(self, nb_runner):
        """Running total across cells, edit one addition."""
        nb_runner.create_notebook([
            "total = 0",
            "total = total + 10  # first add",
            "total = total + 20  # second add",
            "total = total + 30  # third add",
            "print(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(5)

        nb_runner.set_cell_source(3, "total = total + 200  # second add (boosted)")
        nb_runner.run_all()
        assert "total = 240" in nb_runner.get_output(5)

    def test_flag_based_flow(self, nb_runner):
        """Flag variable controls flow, edit the flag."""
        nb_runner.create_notebook([
            "debug = True",
            "label = str(debug)\nprint(f'label = {label}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label = True" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "debug = False")
        nb_runner.run_all()
        assert "label = False" in nb_runner.get_output(2)


class TestClosureEdits:
    """Closure/factory patterns + cell edits."""


    def test_edit_closure_implementation(self, nb_runner):
        """Edit the closure implementation itself."""
        nb_runner.create_notebook([
            "def make_op(n):\n    def op(x):\n        return x + n\n    return op",
            "op = make_op(3)",
            "result = op(10)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 13" in nb_runner.get_output(3)

        # Change closure to multiply
        nb_runner.set_cell_source(
            1, "def make_op(n):\n    def op(x):\n        return x * n\n    return op"
        )
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)
