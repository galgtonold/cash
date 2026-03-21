"""Batch 172 – Multi-cell conditional branching with shared state.

Tests where conditional logic spans multiple cells, with shared
state that changes based on which branch was taken, and edits
to the condition or branch bodies.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.control, pytest.mark.timeout(90)]


class TestConditionalStateEdits:
    """Conditional branching affecting shared state."""

    def test_edit_condition_value(self, nb_runner):
        """Edit the condition variable, verify different branch taken."""
        nb_runner.create_notebook([
            "mode = 'fast'  # processing mode",
            "if mode == 'fast':\n    factor = 10\nelse:\n    factor = 1",
            "result = 42 * factor\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 420" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "mode = 'slow'  # processing mode changed")
        nb_runner.run_all()
        assert "result = 42" in nb_runner.get_output(3)

    def test_edit_branch_body(self, nb_runner):
        """Edit a branch body."""
        nb_runner.create_notebook([
            "flag = True  # branch flag",
            "if flag:\n    val = 'YES'\nelse:\n    val = 'NO'",
            "print(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = YES" in nb_runner.get_output(3)

        # Edit the true branch
        nb_runner.set_cell_source(
            2, "if flag:\n    val = 'AFFIRMATIVE'\nelse:\n    val = 'NEGATIVE'"
        )
        nb_runner.run_all()
        assert "val = AFFIRMATIVE" in nb_runner.get_output(3)

    def test_add_elif_branch(self, nb_runner):
        """Add an elif branch to existing if/else."""
        nb_runner.create_notebook([
            "level = 5  # level value",
            "if level > 10:\n    label = 'high'\nelse:\n    label = 'low'",
            "print(f'label = {label}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label = low" in nb_runner.get_output(3)

        # Add elif
        nb_runner.set_cell_source(
            2,
            "if level > 10:\n    label = 'high'\nelif level > 3:\n    label = 'medium'\nelse:\n    label = 'low'",
        )
        nb_runner.run_all()
        assert "label = medium" in nb_runner.get_output(3)


class TestMultiCellBranching:
    """Branching logic split across multiple cells."""

    def test_config_driven_pipeline(self, nb_runner):
        """Config cell drives processing in multiple downstream cells."""
        nb_runner.create_notebook([
            "config = {'scale': 2, 'offset': 10}  # config dict",
            "scaled = 5 * config['scale']",
            "result = scaled + config['offset']\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(3)

        # Edit config
        nb_runner.set_cell_source(
            1, "config = {'scale': 5, 'offset': 100}  # config dict updated"
        )
        nb_runner.run_all()
        assert "result = 125" in nb_runner.get_output(3)

    def test_flag_toggle_multiple_cells(self, nb_runner):
        """Toggle a flag that affects multiple downstream cells."""
        nb_runner.create_notebook([
            "use_prefix = True  # flag for prefix",
            "if use_prefix:\n    label = 'ENABLED'\nelse:\n    label = 'DISABLED'\nprint(f'label = {label}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "label = ENABLED" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "use_prefix = False  # flag for prefix off")
        nb_runner.run_all()
        assert "label = DISABLED" in nb_runner.get_output(2)

    def test_switch_case_pattern(self, nb_runner):
        """Dictionary-based switch/case pattern with edits."""
        nb_runner.create_notebook([
            "action = 'add'  # action selector",
            "ops = {'add': lambda a, b: a + b, 'mul': lambda a, b: a * b}",
            "result = ops[action](3, 4)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 7" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "action = 'mul'  # action selector changed")
        nb_runner.run_all()
        assert "result = 12" in nb_runner.get_output(3)
