"""Batch 109 – Control structure + cell edit interaction tests.

Tests that exercise loops and conditionals combined with cell edits,
out-of-order execution, and kernel restarts.
"""

import pytest

pytestmark = [pytest.mark.control, pytest.mark.stress, pytest.mark.timeout(30)]


class TestLoopCellEdits:
    """Loops with cell edits."""

    def test_edit_loop_body(self, nb_runner):
        """Edit the body of a for-loop cell."""
        nb_runner.create_notebook([
            "total = 0",
            "for i in range(5):\n    total += i",
            "print(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 10" in nb_runner.get_output(3)

        # Edit loop body: multiply instead of add
        nb_runner.set_cell_source(2, "for i in range(5):\n    total += i * 2")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 20" in nb_runner.get_output(3)

    def test_edit_loop_range(self, nb_runner):
        """Change the range of a loop."""
        nb_runner.create_notebook([
            "total = 0",
            "for i in range(3):\n    total += i",
            "print(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 3" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "for i in range(10):\n    total += i")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 45" in nb_runner.get_output(3)

    def test_edit_loop_init_value(self, nb_runner):
        """Change the initial value before a loop."""
        nb_runner.create_notebook([
            "total = 100",
            "for i in range(3):\n    total += i",
            "print(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 103" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "total = 0")
        nb_runner.run_all()
        assert "total = 3" in nb_runner.get_output(3)


    def test_nested_loop_edit(self, nb_runner):
        """Edit a nested loop."""
        nb_runner.create_notebook([
            "pairs = []\nfor i in range(3):\n    for j in range(2):\n        pairs.append((i, j))",
            "print(f'count = {len(pairs)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count = 6" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "pairs = []\nfor i in range(4):\n    for j in range(3):\n        pairs.append((i, j))",
        )
        nb_runner.run_all()
        assert "count = 12" in nb_runner.get_output(2)


class TestConditionalCellEdits:
    """Conditionals with cell edits."""


    def test_edit_branch_bodies(self, nb_runner):
        """Edit what the branches produce."""
        nb_runner.create_notebook([
            "flag = True",
            "if flag:\n    val = 'yes'\nelse:\n    val = 'no'",
            "print(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = yes" in nb_runner.get_output(3)

        # Edit branch bodies
        nb_runner.set_cell_source(
            2, "if flag:\n    val = 'TRUE'\nelse:\n    val = 'FALSE'"
        )
        nb_runner.run_all()
        assert "val = TRUE" in nb_runner.get_output(3)

    def test_flip_flag_multiple_times(self, nb_runner):
        """Flip a boolean flag back and forth."""
        nb_runner.create_notebook([
            "flag = True",
            "result = 'A' if flag else 'B'\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = A" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "flag = False")
        nb_runner.run_all()
        assert "result = B" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "flag = True")
        nb_runner.run_all()
        assert "result = A" in nb_runner.get_output(2)

    def test_conditional_with_function_call(self, nb_runner):
        """Conditional that calls a function, both get edited."""
        nb_runner.create_notebook([
            "def classify(n):\n    return 'even' if n % 2 == 0 else 'odd'",
            "x = 4",
            "label = classify(x)\nprint(f'label = {label}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label = even" in nb_runner.get_output(3)

        # Edit both: change function and value
        nb_runner.set_cell_source(
            1, "def classify(n):\n    return 'pos' if n > 0 else 'neg'"
        )
        nb_runner.set_cell_source(2, "x = -3")
        nb_runner.run_all()
        assert "label = neg" in nb_runner.get_output(3)


class TestLoopRerunConsistency:
    """Loops must not accumulate on rerun."""

    def test_for_loop_no_double_count(self, nb_runner):
        """Re-running a loop cell should give same result."""
        nb_runner.create_notebook([
            "items = []",
            "for i in range(3):\n    items.append(i)",
            "print(f'items = {items}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "items = [0, 1, 2]" in nb_runner.get_output(3)

        # Re-run
        nb_runner.run_all()
        assert "items = [0, 1, 2]" in nb_runner.get_output(3)

    def test_while_loop_edit_condition(self, nb_runner):
        """Edit while loop condition."""
        nb_runner.create_notebook([
            "i = 0\nresult = []",
            "while i < 3:\n    result.append(i)\n    i += 1",
            "print(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [0, 1, 2]" in nb_runner.get_output(3)

        # Change condition to < 5
        nb_runner.set_cell_source(
            2, "while i < 5:\n    result.append(i)\n    i += 1"
        )
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = [0, 1, 2, 3, 4]" in nb_runner.get_output(3)
