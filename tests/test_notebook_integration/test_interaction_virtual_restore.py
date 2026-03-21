"""Batch 114 – Virtual restore + cell edit interaction tests.

Tests that exercise virtual restore from disk cache after kernel restart,
combined with cell edits and dependency changes.
"""

import pytest

pytestmark = [pytest.mark.restore, pytest.mark.stress, pytest.mark.timeout(30)]


class TestVirtualRestoreBasic:
    """Virtual restore from disk after restart."""

    def test_restore_simple_chain(self, nb_runner):
        """After restart, run last cell — upstream should be restored."""
        nb_runner.create_notebook([
            "x = 42",
            "y = x * 2",
            "z = y + 1\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 85" in nb_runner.get_output(3)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_cell(3)
        assert "z = 85" in nb_runner.get_output(3)

    def test_restore_then_edit_upstream(self, nb_runner):
        """Restore from cache, then edit upstream cell and re-run."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x + 5\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 15" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        # First restore
        nb_runner.run_cell(2)
        assert "y = 15" in nb_runner.get_output(2)

        # Now edit upstream and re-run all
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_all()
        assert "y = 105" in nb_runner.get_output(2)

    def test_restore_multi_variable(self, nb_runner):
        """Restore multiple variables from a single cell."""
        nb_runner.create_notebook([
            "a = 10\nb = 20",
            "c = a + b\nprint(f'c = {c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 30" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_cell(2)
        assert "c = 30" in nb_runner.get_output(2)


class TestVirtualRestoreWithEdits:
    """Edit cells after restart, verify correct behavior."""

    def test_edit_before_restore(self, nb_runner):
        """Edit a cell before restarting, then run all."""
        nb_runner.create_notebook([
            "x = 5",
            "y = x * 3\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 15" in nb_runner.get_output(2)

        # Edit then restart
        nb_runner.set_cell_source(1, "x = 50")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 150" in nb_runner.get_output(2)

    def test_edit_leaf_after_restore(self, nb_runner):
        """Restore, then edit only the leaf cell."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x + 1\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 11" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        # Restore
        nb_runner.run_cell(2)
        assert "y = 11" in nb_runner.get_output(2)

        # Edit leaf
        nb_runner.set_cell_source(2, "y = x * 10\nprint(f'y = {y}')")
        nb_runner.run_cell(2)
        assert "y = 100" in nb_runner.get_output(2)

    def test_two_restarts_with_edits(self, nb_runner):
        """Restart twice with edits in between."""
        nb_runner.create_notebook([
            "x = 1",
            "y = x + 1\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 2" in nb_runner.get_output(2)

        # First restart + edit
        nb_runner.shutdown()
        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 11" in nb_runner.get_output(2)

        # Second restart + edit
        nb_runner.shutdown()
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 101" in nb_runner.get_output(2)


class TestVirtualRestoreWithFunctions:
    """Virtual restore of function definitions."""

    def test_restore_function_def(self, nb_runner):
        """Function defined in cell 1, used in cell 2 — restore cell 2."""
        nb_runner.create_notebook([
            "def double(x):\n    return x * 2",
            "result = double(7)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 14" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_cell(2)
        assert "result = 14" in nb_runner.get_output(2)

    def test_edit_function_after_restore(self, nb_runner):
        """Restore, then edit the function and re-run."""
        nb_runner.create_notebook([
            "def square(x):\n    return x ** 2",
            "val = square(5)\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 25" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        # Edit function before running
        nb_runner.set_cell_source(1, "def square(x):\n    return x ** 3")
        nb_runner.run_all()
        assert "val = 125" in nb_runner.get_output(2)


class TestVirtualRestoreWithFiles:
    """Virtual restore with file dependencies."""

    def test_restore_with_unchanged_file(self, nb_runner, tmp_path):
        """File unchanged after restart — restore should work."""
        data_file = tmp_path / "data.txt"
        data_file.write_text("42")
        path_str = str(data_file).replace("\\", "/")

        nb_runner.create_notebook([
            f"with open('{path_str}') as f:\n    val = int(f.read().strip())",
            "result = val * 2\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 84" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 84" in nb_runner.get_output(2)

    def test_restore_with_changed_file(self, nb_runner, tmp_path):
        """File changed after restart — should recompute."""
        data_file = tmp_path / "data.txt"
        data_file.write_text("10")
        path_str = str(data_file).replace("\\", "/")

        nb_runner.create_notebook([
            f"with open('{path_str}') as f:\n    val = int(f.read().strip())",
            "result = val * 2\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(2)

        # Change file
        import time
        time.sleep(0.1)
        data_file.write_text("99")

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 198" in nb_runner.get_output(2)
