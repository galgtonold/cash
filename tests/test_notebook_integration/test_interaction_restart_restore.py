"""Batch 173 – Kernel restart with dirty state interaction tests.

Tests that establish cached state, restart the kernel, and verify
that cache restoration works correctly after restart.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.restore, pytest.mark.timeout(90)]


class TestRestartRestore:
    """Restart kernel and verify cache restoration."""

    def test_basic_restart_restore(self, nb_runner):
        """After restart, run_all should restore/recompute."""
        nb_runner.create_notebook([
            "x = 42  # basic value",
            "y = x * 2\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 84" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 84" in nb_runner.get_output(2)

    def test_restart_after_edit(self, nb_runner):
        """Edit, restart, verify new values are computed."""
        nb_runner.create_notebook([
            "a = 10  # param a",
            "b = a + 5\nprint(f'b = {b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "b = 15" in nb_runner.get_output(2)

        # Edit then restart
        nb_runner.set_cell_source(1, "a = 100  # param a big")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "b = 105" in nb_runner.get_output(2)

    def test_restart_chain_restore(self, nb_runner):
        """3-cell chain, restart, verify chain recomputes."""
        nb_runner.create_notebook([
            "base = 5  # chain base",
            "mid = base * 3",
            "final = mid + 7\nprint(f'final = {final}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # base=5, mid=15, final=22
        assert "final = 22" in nb_runner.get_output(3)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "final = 22" in nb_runner.get_output(3)


class TestRestartWithFunction:
    """Restart with function definitions."""

    def test_restart_function_def(self, nb_runner):
        """Function definition survives restart via re-execution."""
        nb_runner.create_notebook([
            "def double(x):\n    return x * 2",
            "result = double(7)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 14" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 14" in nb_runner.get_output(2)

    def test_restart_edit_function_then_run(self, nb_runner):
        """Edit function, restart, verify new function is used."""
        nb_runner.create_notebook([
            "def process(x):\n    return x + 1",
            "out = process(10)\nprint(f'out = {out}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "out = 11" in nb_runner.get_output(2)

        # Edit function
        nb_runner.set_cell_source(1, "def process(x):\n    return x * 10")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "out = 100" in nb_runner.get_output(2)
