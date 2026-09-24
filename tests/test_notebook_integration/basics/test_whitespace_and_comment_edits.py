"""Edits that change only whitespace or comments."""

import pytest


# Edge case interaction tests.
#
# Tests empty cells, whitespace-only changes, very large output,
# cell reordering scenarios, and other boundary conditions.
@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestWhitespaceEdits:
    """Whitespace-only cell edits."""

    def test_add_trailing_newline(self, nb_runner):
        """Adding trailing newline should not invalidate cache."""
        nb_runner.create_notebook(
            [
                "x = 42",
                "print(f'x = {x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 42" in nb_runner.get_output(2)

        # Add trailing newlines — should still work
        nb_runner.set_cell_source(1, "x = 42\n\n")
        nb_runner.run_all()
        assert "x = 42" in nb_runner.get_output(2)

    def test_add_comment_only(self, nb_runner):
        """Adding a comment changes the code hash → recomputes."""
        nb_runner.create_notebook(
            [
                "val = 10",
                "print(f'val = {val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 10" in nb_runner.get_output(2)

        # Add a comment — semantically identical but different hash
        nb_runner.set_cell_source(1, "# Important value\nval = 10")
        nb_runner.run_all()
        assert "val = 10" in nb_runner.get_output(2)
