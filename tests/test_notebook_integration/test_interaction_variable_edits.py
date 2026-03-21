"""Batch 110 – Variable shadowing, deletion, and scope interaction tests.

Tests that exercise variable shadowing, overwriting, deletion,
and scope changes combined with cell edits and reruns.
"""

import pytest

pytestmark = [pytest.mark.upstream, pytest.mark.stress, pytest.mark.timeout(30)]


class TestVariableShadowing:
    """Variable defined in one cell, redefined in another."""

    def test_shadow_variable(self, nb_runner):
        """Two cells define same variable — last one wins."""
        nb_runner.create_notebook([
            "x = 10",
            "x = 20",
            "print(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 20" in nb_runner.get_output(3)

    def test_shadow_then_edit_first(self, nb_runner):
        """Shadow variable, then edit the first definition."""
        nb_runner.create_notebook([
            "x = 10",
            "x = 20",
            "print(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 20" in nb_runner.get_output(3)

        # Edit first cell — should not change result
        nb_runner.set_cell_source(1, "x = 999")
        nb_runner.run_all()
        assert "x = 20" in nb_runner.get_output(3)

    def test_shadow_then_edit_second(self, nb_runner):
        """Shadow variable, then edit the shadowing cell."""
        nb_runner.create_notebook([
            "x = 10",
            "x = 20",
            "print(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 20" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "x = 50")
        nb_runner.run_all()
        assert "x = 50" in nb_runner.get_output(3)

    def test_remove_shadowing_cell(self, nb_runner):
        """Remove the shadowing cell, original value should take effect.
        Replacing with pass effectively removes the override."""
        nb_runner.create_notebook([
            "x = 10",
            "x = 20",
            "print(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 20" in nb_runner.get_output(3)

        # Replace shadow cell with pass
        nb_runner.set_cell_source(2, "pass")
        nb_runner.run_all()
        assert "x = 10" in nb_runner.get_output(3)

    def test_variable_type_change(self, nb_runner):
        """Variable changes type between edits."""
        nb_runner.create_notebook([
            "x = 42",
            "print(f'type = {type(x).__name__}, val = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "type = int, val = 42" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "x = 'hello'")
        nb_runner.run_all()
        assert "type = str, val = hello" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "x = [1, 2, 3]")
        nb_runner.run_all()
        assert "type = list, val = [1, 2, 3]" in nb_runner.get_output(2)


class TestVariableOverwriting:
    """Variable computed from itself (self-assignment)."""

    def test_self_assignment_basic(self, nb_runner):
        """x = x + 1 pattern."""
        nb_runner.create_notebook([
            "x = 10",
            "x = x + 5\nprint(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 15" in nb_runner.get_output(2)

    def test_self_assignment_edit_init(self, nb_runner):
        """Edit init, self-assignment should use new value."""
        nb_runner.create_notebook([
            "x = 10",
            "x = x + 5\nprint(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 15" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_all()
        assert "x = 105" in nb_runner.get_output(2)

    def test_self_assignment_chain(self, nb_runner):
        """Multiple self-assignments across cells."""
        nb_runner.create_notebook([
            "x = 1",
            "x = x * 2",
            "x = x + 10",
            "print(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 12" in nb_runner.get_output(4)

        nb_runner.set_cell_source(1, "x = 5")
        nb_runner.run_all()
        assert "x = 20" in nb_runner.get_output(4)

    def test_self_assignment_rerun_no_double(self, nb_runner):
        """Self-assignment must not double on rerun."""
        nb_runner.create_notebook([
            "x = 10",
            "x = x + 5\nprint(f'x = {x}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 15" in nb_runner.get_output(2)

        # Rerun — should still be 15, not 20
        nb_runner.run_all()
        assert "x = 15" in nb_runner.get_output(2)


class TestMultipleVariables:
    """Multiple variables with interleaved edits."""

    def test_edit_one_of_two_independent_vars(self, nb_runner):
        """Two independent variables, edit one."""
        nb_runner.create_notebook([
            "a = 10\nb = 20",
            "print(f'a = {a}, b = {b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 10, b = 20" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "a = 99\nb = 20")
        nb_runner.run_all()
        assert "a = 99, b = 20" in nb_runner.get_output(2)

    def test_swap_variables(self, nb_runner):
        """Swap variable values between cells."""
        nb_runner.create_notebook([
            "a = 1\nb = 2",
            "result = a + b * 10\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 21" in nb_runner.get_output(2)

        # Swap values
        nb_runner.set_cell_source(1, "a = 2\nb = 1")
        nb_runner.run_all()
        assert "result = 12" in nb_runner.get_output(2)

    def test_add_new_variable_midstream(self, nb_runner):
        """Edit a cell to produce an additional variable."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x * 2\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)

        # Cell 1 now produces x and z
        nb_runner.set_cell_source(1, "x = 10\nz = 5")
        nb_runner.set_cell_source(2, "y = x * 2 + z\nprint(f'y = {y}')")
        nb_runner.run_all()
        assert "y = 25" in nb_runner.get_output(2)

    def test_remove_variable_from_cell(self, nb_runner):
        """Remove a variable from a cell, downstream breaks gracefully."""
        nb_runner.create_notebook([
            "x = 10\ny = 20",
            "result = x + y\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(2)

        # Remove y, now use only x
        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.set_cell_source(2, "result = x * 3\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(2)
