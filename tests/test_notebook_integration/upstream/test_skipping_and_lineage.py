"""Statements skipped as already executed, and the lineage behind them."""

import time

import pytest

pytestmark = [pytest.mark.stress]


# Stress tests: Skip Logic & Lineage Edge Cases (Scenarios 1-30)
#
# Tests the skip optimization, cache key computation, lineage tracking,
# and various edge cases around when statements should/shouldn't be skipped.
@pytest.mark.skip_optimization
class TestSkipLogic:
    """Tests for the already-executed skip optimization."""

    def test_01_scalar_chain_skip(self, nb_runner):
        """Scenario 1: Scalar chain x→y→z, re-run middle — z should still skip."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x + 1",
                "z = y + 1\nprint(f'z={z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z=3" in nb_runner.get_output(3)
        # Re-run cell 2 (same code, same input) — should skip
        nb_runner.run_cell(2)
        # Re-run cell 3 — should skip since y hasn't changed
        nb_runner.run_cell(3)
        assert "z=3" in nb_runner.get_output(3)

    def test_02_self_assignment_skip(self, nb_runner):
        """Scenario 2: df = df.sort_values(...) — re-run should skip."""
        nb_runner.create_notebook(
            [
                "import pandas as pd\ndf = pd.DataFrame({'a': [3,1,2], 'b': [6,4,5]})",
                "df = df.sort_values('a')\nprint(df['a'].tolist())",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[1, 2, 3]" in nb_runner.get_output(2)
        # Re-run cell 2 — should skip (self-assignment, same df)
        nb_runner.run_cell(2)
        assert "[1, 2, 3]" in nb_runner.get_output(2)

    def test_03_self_assignment_changed_input(self, nb_runner):
        """Scenario 3: df = df.sort_values(...) where df changed upstream — should NOT skip."""
        nb_runner.create_notebook(
            [
                "import pandas as pd\ndf = pd.DataFrame({'a': [3,1,2]})",
                "df = df.sort_values('a')\nprint(df['a'].tolist())",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "[1, 2, 3]" in nb_runner.get_output(2)
        # Change cell 1
        nb_runner.set_cell_source(1, "import pandas as pd\ndf = pd.DataFrame({'a': [9,7,8]})")
        nb_runner.run_cell(1)
        nb_runner.run_cell(2)
        assert "[7, 8, 9]" in nb_runner.get_output(2)

    def test_04_skip_with_file_dep_unchanged(self, nb_runner, tmp_path):
        """Scenario 7: Statement reads file, file unchanged — should skip on re-run."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        csv_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_str}')\nprint(len(df))",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        assert "2" in nb_runner.get_output(1)
        # Re-run same cell — file unchanged, should skip
        nb_runner.run_cell(1)
        out2 = nb_runner.get_output(1)
        assert "2" in out2

    def test_05_skip_with_file_dep_changed(self, nb_runner, tmp_path):
        """Scenario 8: Statement reads file, file CHANGED — should NOT skip."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        csv_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"import pandas as pd\ndf = pd.read_csv('{csv_str}')\nprint(len(df))",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_cell(1)
        assert "2" in nb_runner.get_output(1)
        # Modify the file
        time.sleep(0.1)  # ensure mtime changes
        csv_path.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
        # Re-run — should detect file change and re-execute
        nb_runner.run_cell(1)
        assert "3" in nb_runner.get_output(1)

    def test_06_augmented_assignment_different_results(self, nb_runner):
        """Scenario 10: x += 1 on an isolated re-run is idempotent (run-from-start).

        ``x`` is a no-lineage primitive the cell self-modifies. Re-running cell 2
        alone == running the notebook from the start (``x = 0`` then ``x += 1``),
        so ``x`` is restored to its cell-entry base before re-execution and the
        result stays ``1`` — it does NOT accumulate to ``2``. (An earlier snapshot
        of the doubling bug asserted ``x=2``; idempotent re-run is the intended
        semantic — see test_rerunning_a_self_modifying_cell_alone.)
        """
        nb_runner.create_notebook(
            [
                "x = 0",
                "x += 1\nprint(f'x={x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=1" in nb_runner.get_output(2)
        # Re-run cell 2 — base x=0 is restored first, so x += 1 stays 1.
        nb_runner.run_cell(2)
        assert "x=1" in nb_runner.get_output(2)

    def test_09_cell_with_only_print(self, nb_runner):
        """Scenario 13: print() only — no outputs to cache."""
        nb_runner.create_notebook(
            [
                "x = 42",
                "print(f'Value: {x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Value: 42" in nb_runner.get_output(2)
        # Re-run — should skip or at minimum be correct
        nb_runner.run_cell(2)
        assert "Value: 42" in nb_runner.get_output(2)

    def test_10_none_assignment_lineage(self, nb_runner):
        """Scenario 14: x = None then y = x — None should have valid lineage."""
        nb_runner.create_notebook(
            [
                "x = None",
                "y = x\nprint(f'y is None: {y is None}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y is None: True" in nb_runner.get_output(2)
        # Re-run — should skip
        nb_runner.run_cell(2)
        assert "y is None: True" in nb_runner.get_output(2)

    def test_11_boolean_caching(self, nb_runner):
        """Scenario 15: Boolean result caching."""
        nb_runner.create_notebook(
            [
                "data = list(range(200))",
                "flag = len(data) > 100\nprint(f'flag={flag}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "flag=True" in nb_runner.get_output(2)
        # Change data to smaller
        nb_runner.set_cell_source(1, "data = list(range(50))")
        nb_runner.run_cell(1)
        nb_runner.run_cell(2)
        assert "flag=False" in nb_runner.get_output(2)

    def test_12_multiple_outputs_one_cell(self, nb_runner):
        """Scenario 9: Statement produces multiple outputs."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "a = x + 1\nb = x + 2\nprint(f'a={a}, b={b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=11, b=12" in nb_runner.get_output(2)
        # Re-run — both should skip
        nb_runner.run_cell(2)
        assert "a=11, b=12" in nb_runner.get_output(2)

    def test_15_cell_rerun_identical_output(self, nb_runner):
        """Re-running a cell that produces same value should still skip."""
        nb_runner.create_notebook(
            [
                "x = 42",
                "y = x\nprint(f'y={y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y=42" in nb_runner.get_output(2)
        # Re-run cell 1 (same code) then cell 2
        nb_runner.run_cell(1)
        nb_runner.run_cell(2)
        assert "y=42" in nb_runner.get_output(2)


@pytest.mark.skip_optimization
class TestLineageIntegrity:
    """Tests for lineage computation and cache key correctness."""

    def test_16_lineage_through_function(self, nb_runner):
        """Scenario 16: Function source change should invalidate cache."""
        nb_runner.create_notebook(
            [
                "def double(x):\n    return x * 2",
                "result = double(5)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=10" in nb_runner.get_output(2)
        # Change function
        nb_runner.set_cell_source(1, "def double(x):\n    return x * 3")
        nb_runner.run_cell(1)
        nb_runner.run_cell(2)
        assert "result=15" in nb_runner.get_output(2)

    def test_17_lineage_through_lambda(self, nb_runner):
        """Scenario 17: Lambda source change should invalidate cache."""
        nb_runner.create_notebook(
            [
                "f = lambda x: x * 2",
                "result = f(5)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=10" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "f = lambda x: x * 3")
        nb_runner.run_cell(1)
        nb_runner.run_cell(2)
        assert "result=15" in nb_runner.get_output(2)

    def test_18_transitive_lineage_invalidation(self, nb_runner):
        """Scenario 19: A→B→C→D chain, change A → all downstream invalid."""
        nb_runner.create_notebook(
            [
                "a = 1",
                "b = a + 1",
                "c = b + 1",
                "d = c + 1\nprint(f'd={d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d=4" in nb_runner.get_output(4)
        # Change A
        nb_runner.set_cell_source(1, "a = 10")
        nb_runner.run_all()
        assert "d=13" in nb_runner.get_output(4)

    def test_19_diamond_dependency(self, nb_runner):
        """Scenario 20: A→B, A→C, B+C→D — change A, D invalidates."""
        nb_runner.create_notebook(
            [
                "a = 10",
                "b = a * 2",
                "c = a * 3",
                "d = b + c\nprint(f'd={d}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "d=50" in nb_runner.get_output(4)
        nb_runner.set_cell_source(1, "a = 100")
        nb_runner.run_all()
        assert "d=500" in nb_runner.get_output(4)

    def test_20_self_referential_increment(self, nb_runner):
        """Scenario 21: x = x + 1 on an isolated re-run is idempotent (run-from-start).

        Same family as test_06 with a pure self-reassignment rather than an
        augmented assignment. Re-running cell 2 restores the cell-entry base
        ``x = 0`` first, so ``x = x + 1`` stays ``1`` rather than accumulating to
        ``2`` — see test_rerunning_a_self_modifying_cell_alone for the catalogue.
        """
        nb_runner.create_notebook(
            [
                "x = 0",
                "x = x + 1\nprint(f'x={x}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=1" in nb_runner.get_output(2)
        # Run cell 2 again — base x=0 is restored first, so x = x + 1 stays 1.
        nb_runner.run_cell(2)
        assert "x=1" in nb_runner.get_output(2)

    def test_27_chain_modification_propagates(self, nb_runner):
        """Modifying early cell propagates through chain on re-run."""
        nb_runner.create_notebook(
            [
                "base = 5",
                "step1 = base * 2",
                "step2 = step1 + 3",
                "step3 = step2 ** 2\nprint(f'step3={step3}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        # (5*2 + 3)^2 = 13^2 = 169
        assert "step3=169" in nb_runner.get_output(4)
        # Change base
        nb_runner.set_cell_source(1, "base = 10")
        nb_runner.run_all()
        # (10*2 + 3)^2 = 23^2 = 529
        assert "step3=529" in nb_runner.get_output(4)

    def test_29_tuple_unpacking(self, nb_runner):
        """Tuple unpacking produces multiple outputs."""
        nb_runner.create_notebook(
            [
                "data = (1, 2, 3)",
                "a, b, c = data\nprint(f'a={a}, b={b}, c={c}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=1, b=2, c=3" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "data = (10, 20, 30)")
        nb_runner.run_cell(1)
        nb_runner.run_cell(2)
        assert "a=10, b=20, c=30" in nb_runner.get_output(2)


# Skip optimization edge cases.
#
# Tests that exercise the 'already executed' skip optimization
# in combination with external modifications, reruns, and edits.
# The skip optimization checks:
# 1. Code matches executed_cell_codes[var]
# 2. Output's _cash_hash matches stored lineage
# 3. No file dependencies OR file deps unchanged
# 4. Input lineages match executed_input_lineages[var]
@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestSkipOptimizationBasic:
    """Basic skip optimization behavior."""

    def test_rerun_same_cell_skips(self, nb_runner):
        """Re-running the same cell should skip (not recompute)."""
        nb_runner.create_notebook(
            [
                "x = 42",
                "y = x * 2\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 84" in nb_runner.get_output(2)

        # Re-run — should still produce same result (skip or recompute)
        nb_runner.run_all()
        assert "y = 84" in nb_runner.get_output(2)

    def test_edit_upstream_forces_recompute(self, nb_runner):
        """Editing upstream cell should force downstream recompute."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "y = x + 1\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 2" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_all()
        assert "y = 101" in nb_runner.get_output(2)

    def test_same_code_different_input_lineage(self, nb_runner):
        """Same code but different input lineage — should recompute."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x + 1\nprint(f'y = {y}')",
            ]
        )
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


@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestExternalModification:
    """External modification of variables (e.g., in a separate cell)."""

    def test_overwrite_cached_var_then_rerun(self, nb_runner):
        """Overwrite a cached variable, then re-run the producer cell."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x * 2\nprint(f'y = {y}')",
                "# Intentionally overwrite y\ny = 999\nprint(f'y_overwritten = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)
        assert "y_overwritten = 999" in nb_runner.get_output(3)

        # Now re-run cell 2 — should it produce 20 again?
        nb_runner.run_cell(2)
        assert "y = 20" in nb_runner.get_output(2)

    def test_dependent_after_overwrite(self, nb_runner):
        """After overwriting a variable, downstream should use new value."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "y = x * 2",
                "z = y + 1\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 11" in nb_runner.get_output(3)

        # Overwrite y in cell 2
        nb_runner.set_cell_source(2, "y = 100")
        nb_runner.run_all()
        assert "z = 101" in nb_runner.get_output(3)


@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestSkipWithFileDepEdit:
    """Skip optimization + file dependencies + cell edits."""

    def test_file_dep_prevents_skip(self, nb_runner, tmp_path):
        """If a file dependency changed, skip should not happen."""
        data_file = tmp_path / "data.txt"
        data_file.write_text("10", encoding="utf-8")
        path_str = str(data_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"with open('{path_str}') as f:\n    val = int(f.read().strip())",
                "result = val * 2\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 20" in nb_runner.get_output(2)

        # Change file and re-run
        import time

        time.sleep(0.1)  # Ensure mtime changes
        data_file.write_text("50", encoding="utf-8")
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(2)

    def test_file_unchanged_skips_correctly(self, nb_runner, tmp_path):
        """If file is unchanged, skip optimization should work."""
        data_file = tmp_path / "stable.txt"
        data_file.write_text("42", encoding="utf-8")
        path_str = str(data_file).replace("\\", "/")

        nb_runner.create_notebook(
            [
                f"with open('{path_str}') as f:\n    val = int(f.read().strip())",
                "result = val * 3\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 126" in nb_runner.get_output(2)

        # Re-run without changing file — result should stay the same
        nb_runner.run_all()
        assert "result = 126" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(30)
class TestSkipWithMultiOutput:
    """Skip optimization with multi-output cells."""

    def test_multi_output_cell_skip(self, nb_runner):
        """Cell that produces multiple outputs — skip all or none."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "a = x + 1\nb = x + 2",
                "print(f'a = {a}, b = {b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 11, b = 12" in nb_runner.get_output(3)

        # Re-run — should produce same result
        nb_runner.run_all()
        assert "a = 11, b = 12" in nb_runner.get_output(3)

    def test_multi_output_edit_upstream(self, nb_runner):
        """Edit upstream, multi-output cell should recompute."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "a = x + 1\nb = x * 2",
                "print(f'a = {a}, b = {b}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a = 11, b = 20" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 100")
        nb_runner.run_all()
        assert "a = 101, b = 200" in nb_runner.get_output(3)
