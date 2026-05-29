"""Batch 150 – Variable shadowing and scope interaction tests.

Tests where variables are overwritten in later cells,
edits change which version of a variable is used, and
scoping rules interact with caching.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(45)]


class TestVariableOverwriting:
    """Variable overwritten in subsequent cells."""

    def test_overwrite_then_edit_first_def(self, nb_runner):
        """Variable defined twice, edit first definition."""
        nb_runner.create_notebook([
            "x = 10",
            "x = x + 5",  # now x = 15
            "result = x * 2\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 30" in nb_runner.get_output(3)

        # Edit first definition
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_all()
        assert "result = 210" in nb_runner.get_output(3)


    def test_three_overwrites_edit_middle(self, nb_runner):
        """Variable overwritten 3 times, edit middle one."""
        nb_runner.create_notebook([
            "val = 1",
            "val = val + 10",
            "val = val * 2",
            "print(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # val = (1+10)*2 = 22
        assert "val = 22" in nb_runner.get_output(4)

        # Edit middle
        nb_runner.set_cell_source(2, "val = val + 100")
        nb_runner.run_all()
        # val = (1+100)*2 = 202
        assert "val = 202" in nb_runner.get_output(4)


class TestMultipleVariables:
    """Multiple variables with interleaved definitions."""



    def test_introduce_new_variable(self, nb_runner):
        """Introduce a new variable mid-notebook."""
        nb_runner.create_notebook([
            "x = 5",
            "result = x * 2\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        # Add a multiplier variable
        nb_runner.set_cell_source(1, "x = 5\nmultiplier = 10")
        nb_runner.set_cell_source(
            2, "result = x * multiplier\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = 50" in nb_runner.get_output(2)
