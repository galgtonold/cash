"""Batch 141 – Kernel restart + cell edit combined interaction tests.

Tests where users restart the kernel (via shutdown + start_kernel)
combined with cell edits before/after restart, verifying disk
restore and re-computation work correctly together.
"""

import pytest

pytestmark = [pytest.mark.restore, pytest.mark.stress, pytest.mark.timeout(60)]


class TestRestartThenEditCells:
    """Restart kernel, then edit cells."""

    def test_restart_then_edit_root(self, nb_runner):
        """Restart kernel, then edit root cell."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x * 2",
            "z = y + 5\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 25" in nb_runner.get_output(3)

        nb_runner.shutdown()
        nb_runner.start_kernel()

        # Edit root AFTER restart
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_all()
        assert "z = 205" in nb_runner.get_output(3)

    def test_restart_then_edit_leaf(self, nb_runner):
        """Restart kernel, then edit leaf cell."""
        nb_runner.create_notebook([
            "a = 5",
            "b = a + 10",
            "result = b * 3\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 45" in nb_runner.get_output(3)

        nb_runner.shutdown()
        nb_runner.start_kernel()

        nb_runner.set_cell_source(3, "result = b * 100\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 1500" in nb_runner.get_output(3)


class TestEditThenRestartCells:
    """Edit cells, then restart."""

    def test_edit_then_restart_run(self, nb_runner):
        """Edit, restart, run uses edited code."""
        nb_runner.create_notebook([
            "val = 1",
            "out = val + 10\nprint(f'out = {out}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "out = 11" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "val = 50")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "out = 60" in nb_runner.get_output(2)

    def test_edit_run_restart_run_should_restore(self, nb_runner):
        """Edit, run, restart, run restores from cache."""
        nb_runner.create_notebook([
            "data = [1, 2, 3]",
            "total = sum(data)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 6" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "data = [10, 20, 30]")
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)


class TestMultipleRestartsWithEdits:
    """Multiple restarts with edits between."""

    def test_edit_restart_edit_restart(self, nb_runner):
        """Edit, restart, edit again, restart again."""
        nb_runner.create_notebook([
            "n = 1",
            "result = n * 100\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(2)

        # Round 1
        nb_runner.set_cell_source(1, "n = 2")
        nb_runner.run_all()
        assert "result = 200" in nb_runner.get_output(2)
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 200" in nb_runner.get_output(2)

        # Round 2
        nb_runner.set_cell_source(1, "n = 5")
        nb_runner.run_all()
        assert "result = 500" in nb_runner.get_output(2)
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 500" in nb_runner.get_output(2)
