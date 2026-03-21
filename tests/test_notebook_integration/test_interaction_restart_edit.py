"""
Batch 104 — Kernel restart + cell edit interactions.

Tests that exercise the most fragile path: editing cells after a kernel restart.
After restart, cash must:
- Rebuild lineage state from disk cache
- Detect that cell code has changed since the cached state
- Re-execute changed statements and propagate properly
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.restore]


class TestEditAfterRestart:
    """Edit cells after a kernel restart — the trickiest path."""

    def test_edit_upstream_after_restart(self, nb_runner):
        """Run all, restart, edit cell 1, run cell 3 → should see new value."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x + 5",
            "print(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 15" in nb_runner.get_output(3)

        # Restart kernel
        nb_runner.shutdown()
        nb_runner.start_kernel()

        # Edit upstream before running
        nb_runner.set_cell_source(1, "x = 99")
        nb_runner.run_cell(3)
        assert "y = 104" in nb_runner.get_output(3)

    def test_no_edit_after_restart_restores_from_disk(self, nb_runner):
        """Run all, restart, run cell 3 without edits → should restore from disk."""
        nb_runner.create_notebook([
            "a = 42",
            "b = a * 2",
            "print(f'b = {b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "b = 84" in nb_runner.get_output(3)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_cell(3)
        assert "b = 84" in nb_runner.get_output(3)

    def test_edit_then_restart_then_run(self, nb_runner):
        """Edit cell 1, restart BEFORE running, then run all."""
        nb_runner.create_notebook([
            "x = 5",
            "y = x * 3",
            "print(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 15" in nb_runner.get_output(3)

        # Edit but DON'T run — then restart
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 300" in nb_runner.get_output(3)

    def test_edit_middle_cell_after_restart(self, nb_runner):
        """Edit middle cell (formula change) after restart."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x + 5",
            "z = y * 2\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 30" in nb_runner.get_output(3)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.set_cell_source(2, "y = x * 100")
        nb_runner.run_cell(3)
        assert "z = 2000" in nb_runner.get_output(3)


class TestMultipleRestartsWithEdits:
    """Multiple restart-edit cycles."""

    def test_restart_edit_restart_edit(self, nb_runner):
        """Two restart-edit cycles in sequence."""
        nb_runner.create_notebook([
            "x = 1",
            "y = x + 10",
            "print(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 11" in nb_runner.get_output(3)

        # First restart + edit
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_all()
        assert "y = 110" in nb_runner.get_output(3)

        # Second restart + edit
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.set_cell_source(1, "x = 500")
        nb_runner.run_all()
        assert "y = 510" in nb_runner.get_output(3)

    def test_restart_without_edit_then_edit(self, nb_runner):
        """Restart without edit, run, then edit and run again."""
        nb_runner.create_notebook([
            "a = 7",
            "b = a * 3",
            "print(f'b = {b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "b = 21" in nb_runner.get_output(3)

        # Restart without edit
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "b = 21" in nb_runner.get_output(3)

        # Now edit
        nb_runner.set_cell_source(1, "a = 10")
        nb_runner.run_cell(3)
        assert "b = 30" in nb_runner.get_output(3)


class TestRestartWithFunctions:
    """Restart interactions with function definitions."""

    def test_function_edit_after_restart(self, nb_runner):
        """Edit a function definition cell after kernel restart."""
        nb_runner.create_notebook([
            "def compute(x):\n    return x * 2",
            "result = compute(5)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 10" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.set_cell_source(1, "def compute(x):\n    return x * 3")
        nb_runner.run_cell(2)
        assert "result = 15" in nb_runner.get_output(2)

    def test_function_unchanged_after_restart_uses_cache(self, nb_runner):
        """Unchanged function after restart should restore from cache."""
        nb_runner.create_notebook([
            "def add(a, b):\n    return a + b",
            "val = add(3, 4)\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 7" in nb_runner.get_output(2)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_cell(2)
        assert "val = 7" in nb_runner.get_output(2)


class TestRestartWithLongChains:
    """Restart with long dependency chains."""

    def test_five_cell_chain_restart_edit_root(self, nb_runner):
        """5-cell chain, restart, edit root, run last."""
        nb_runner.create_notebook([
            "a = 1",
            "b = a + 1",
            "c = b + 1",
            "d = c + 1",
            "e = d + 1\nprint(f'e = {e}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "e = 5" in nb_runner.get_output(5)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.set_cell_source(1, "a = 100")
        nb_runner.run_cell(5)
        assert "e = 104" in nb_runner.get_output(5)

    def test_four_cell_chain_restart_edit_middle(self, nb_runner):
        """4-cell chain, restart, edit middle, run last."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x * 2",
            "z = y + 5",
            "print(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 25" in nb_runner.get_output(4)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.set_cell_source(2, "y = x * 100")
        nb_runner.run_cell(4)
        assert "z = 1005" in nb_runner.get_output(4)

    def test_chain_restart_revert_to_original(self, nb_runner):
        """Edit root, restart, revert to original, run last."""
        nb_runner.create_notebook([
            "x = 5",
            "y = x + 10",
            "print(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 15" in nb_runner.get_output(3)

        # Edit and run
        nb_runner.set_cell_source(1, "x = 50")
        nb_runner.run_cell(3)
        assert "y = 60" in nb_runner.get_output(3)

        # Restart and revert
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.set_cell_source(1, "x = 5")
        nb_runner.run_cell(3)
        assert "y = 15" in nb_runner.get_output(3)
