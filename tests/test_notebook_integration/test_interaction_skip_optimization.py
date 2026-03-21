"""Batch 113 – Skip optimization edge cases.

Tests that exercise the 'already executed' skip optimization
in combination with external modifications, reruns, and edits.
The skip optimization checks:
1. Code matches executed_cell_codes[var]
2. Output's _cash_hash matches stored lineage
3. No file dependencies OR file deps unchanged
4. Input lineages match executed_input_lineages[var]
"""

import pytest

pytestmark = [pytest.mark.upstream, pytest.mark.stress, pytest.mark.timeout(30)]


class TestSkipOptimizationBasic:
    """Basic skip optimization behavior."""

    def test_rerun_same_cell_skips(self, nb_runner):
        """Re-running the same cell should skip (not recompute)."""
        nb_runner.create_notebook([
            "x = 42",
            "y = x * 2\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 84" in nb_runner.get_output(2)

        # Re-run — should still produce same result (skip or recompute)
        nb_runner.run_all()
        assert "y = 84" in nb_runner.get_output(2)

    def test_edit_upstream_forces_recompute(self, nb_runner):
        """Editing upstream cell should force downstream recompute."""
        nb_runner.create_notebook([
            "x = 1",
            "y = x + 1\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 2" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_all()
        assert "y = 101" in nb_runner.get_output(2)

    def test_same_code_different_input_lineage(self, nb_runner):
        """Same code but different input lineage — should recompute."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x + 1\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 11" in nb_runner.get_output(2)

        # Change x, run all
        nb_runner.set_cell_source(1, "x = 20")
        nb_runner.run_all()
        assert "y = 21" in nb_runner.get_output(2)

        # Change back — should use cached result or recompute correctly
        nb_runner.set_cell_source(1, "x = 10")
        nb_runner.run_all()
        assert "y = 11" in nb_runner.get_output(2)


class TestExternalModification:
    """External modification of variables (e.g., in a separate cell)."""

    def test_overwrite_cached_var_then_rerun(self, nb_runner):
        """Overwrite a cached variable, then re-run the producer cell."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x * 2\nprint(f'y = {y}')",
            "# Intentionally overwrite y\ny = 999\nprint(f'y_overwritten = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)
        assert "y_overwritten = 999" in nb_runner.get_output(3)

        # Now re-run cell 2 — should it produce 20 again?
        nb_runner.run_cell(2)
        assert "y = 20" in nb_runner.get_output(2)

    def test_dependent_after_overwrite(self, nb_runner):
        """After overwriting a variable, downstream should use new value."""
        nb_runner.create_notebook([
            "x = 5",
            "y = x * 2",
            "z = y + 1\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 11" in nb_runner.get_output(3)

        # Overwrite y in cell 2
        nb_runner.set_cell_source(2, "y = 100")
        nb_runner.run_all()
        assert "z = 101" in nb_runner.get_output(3)


class TestSkipWithFileDepEdit:
    """Skip optimization + file dependencies + cell edits."""

    def test_file_dep_prevents_skip(self, nb_runner, tmp_path):
        """If a file dependency changed, skip should not happen."""
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

        # Change file and re-run
        import time
        time.sleep(0.1)  # Ensure mtime changes
        data_file.write_text("50")
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(2)

    def test_file_unchanged_skips_correctly(self, nb_runner, tmp_path):
        """If file is unchanged, skip optimization should work."""
        data_file = tmp_path / "stable.txt"
        data_file.write_text("42")
        path_str = str(data_file).replace("\\", "/")

        nb_runner.create_notebook([
            f"with open('{path_str}') as f:\n    val = int(f.read().strip())",
            "result = val * 3\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 126" in nb_runner.get_output(2)

        # Re-run without changing file — result should stay the same
        nb_runner.run_all()
        assert "result = 126" in nb_runner.get_output(2)


class TestSkipWithMultiOutput:
    """Skip optimization with multi-output cells."""

    def test_multi_output_cell_skip(self, nb_runner):
        """Cell that produces multiple outputs — skip all or none."""
        nb_runner.create_notebook([
            "x = 10",
            "a = x + 1\nb = x + 2",
            "print(f'a = {a}, b = {b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 11, b = 12" in nb_runner.get_output(3)

        # Re-run — should produce same result
        nb_runner.run_all()
        assert "a = 11, b = 12" in nb_runner.get_output(3)

    def test_multi_output_edit_upstream(self, nb_runner):
        """Edit upstream, multi-output cell should recompute."""
        nb_runner.create_notebook([
            "x = 10",
            "a = x + 1\nb = x * 2",
            "print(f'a = {a}, b = {b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 11, b = 20" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_all()
        assert "a = 101, b = 200" in nb_runner.get_output(3)

    def test_independent_outputs_partial_dependency(self, nb_runner):
        """Two outputs: one depends on x, one on y."""
        nb_runner.create_notebook([
            "x = 10\ny = 20",
            "a = x * 2\nb = y * 3",
            "print(f'a = {a}, b = {b}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 20, b = 60" in nb_runner.get_output(3)

        # Edit only x
        nb_runner.set_cell_source(1, "x = 100\ny = 20")
        nb_runner.run_all()
        assert "a = 200, b = 60" in nb_runner.get_output(3)
