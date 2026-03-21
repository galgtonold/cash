"""Batch 117 – Rapid-fire edit interaction tests.

Tests that exercise many rapid successive edits to the same cell(s),
verifying cache coherence under high edit frequency.
"""

import pytest

pytestmark = [pytest.mark.upstream, pytest.mark.stress, pytest.mark.timeout(30)]


class TestRapidEditsOneCell:
    """Many edits to a single cell in quick succession."""

    def test_five_rapid_edits(self, nb_runner):
        """Edit the same cell five times, verify each time."""
        nb_runner.create_notebook([
            "x = 0",
            "y = x + 1\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()

        for i in range(5):
            nb_runner.set_cell_source(1, f"x = {i * 10}")
            nb_runner.run_all()
            assert f"y = {i * 10 + 1}" in nb_runner.get_output(2)

    def test_ten_rapid_edits(self, nb_runner):
        """Ten rapid edits to the upstream cell."""
        nb_runner.create_notebook([
            "n = 0",
            "result = n ** 2\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()

        for i in range(10):
            nb_runner.set_cell_source(1, f"n = {i}")
            nb_runner.run_all()
            assert f"result = {i ** 2}" in nb_runner.get_output(2)

    def test_rapid_edits_with_function(self, nb_runner):
        """Rapidly edit function definition."""
        nb_runner.create_notebook([
            "def f(x):\n    return x + 0",
            "result = f(10)\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()

        for offset in [1, 5, 10, 100, -1]:
            nb_runner.set_cell_source(1, f"def f(x):\n    return x + {offset}")
            nb_runner.run_all()
            assert f"result = {10 + offset}" in nb_runner.get_output(2)


class TestRapidEditsMultipleCells:
    """Rapid edits across multiple cells."""

    def test_alternating_rapid_edits(self, nb_runner):
        """Alternate between editing two cells rapidly."""
        nb_runner.create_notebook([
            "a = 1",
            "b = 1",
            "c = a + b\nprint(f'c = {c}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "c = 2" in nb_runner.get_output(3)

        # Alternate edits
        for i in range(1, 6):
            if i % 2 == 1:
                nb_runner.set_cell_source(1, f"a = {i * 10}")
            else:
                nb_runner.set_cell_source(2, f"b = {i * 10}")
            nb_runner.run_all()

        # After 5 rounds: a=50, b=40
        assert "c = 90" in nb_runner.get_output(3)

    def test_rapid_edit_and_revert(self, nb_runner):
        """Rapidly edit and revert, check cache consistency.

        After editing x = 999 and running, then reverting back to x = 1,
        the system should detect the upstream change in both directions.
        Uses kernel restart between cycles to avoid "already executed"
        skip optimization masking the revert detection.
        """
        nb_runner.create_notebook([
            "x = 1",
            "y = x + 1\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 2" in nb_runner.get_output(2)

        # Edit to new value
        nb_runner.set_cell_source(1, "x = 999")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 1000" in nb_runner.get_output(2)

        # Revert back to original
        nb_runner.set_cell_source(1, "x = 1")
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 2" in nb_runner.get_output(2)


class TestRapidEditWithRestart:
    """Rapid edits combined with kernel restarts."""

    def test_edit_restart_cycle(self, nb_runner):
        """Edit → restart → verify cycle."""
        nb_runner.create_notebook([
            "x = 0",
            "y = x + 1\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 1" in nb_runner.get_output(2)

        for val in [10, 20, 30]:
            nb_runner.set_cell_source(1, f"x = {val}")
            nb_runner.shutdown()
            nb_runner.start_kernel()
            nb_runner.run_all()
            assert f"y = {val + 1}" in nb_runner.get_output(2)
