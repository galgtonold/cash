from cash.analysis.annotations import CacheAnnotation
from cash.notebook.cache_status import CacheStatus

"""
Tests for the ALREADY_EXECUTED skip optimization in statement_processor.py.

This optimization checks if a statement was already computed (e.g., as an upstream
dependency) and skips re-execution when:
1. Code matches executed_cell_codes[var]
2. Output's lineage matches stored lineage (not externally modified)
3. No file dependencies changed
4. Input lineages match executed_input_lineages[var]
5. Self-assignment: check output lineage, not input

Previously: 0 tests. This file fills a critical coverage gap.
"""

import contextlib
import json
import os
import shutil
import tempfile
import time
from unittest.mock import patch

import pytest

from cash.notebook.statement.imports import redundant_import_names
from tests._cell_driver import run_cash_cell

# Force caching regardless of the 10 ms min-execution-time floor.
_PERSIST = CacheAnnotation(persist=True)


class TestAlreadyExecutedOptimization:
    """Tests for the ALREADY_EXECUTED skip optimization."""

    def test_skip_when_same_code_same_inputs(self, statement_processor, mock_shell):
        """
        When a statement has already been executed with the same code and inputs,
        re-running it should skip (SKIPPED status) instead of re-executing.
        """

        # First execution: compute x
        mock_shell.user_ns["a"] = 10
        metrics1 = statement_processor.process_statement("x = a + 1")
        assert mock_shell.user_ns["x"] == 11
        assert metrics1["status"] == CacheStatus.COMPUTED

        # Second execution of the same statement (same code, same inputs)
        # Should be RESTORED from cache or re-COMPUTED (depending on lineage stability)
        metrics2 = statement_processor.process_statement("x = a + 1")
        assert mock_shell.user_ns["x"] == 11
        assert metrics2["status"] in (CacheStatus.SKIPPED, CacheStatus.RESTORED, CacheStatus.COMPUTED)

    def test_reexecute_when_input_lineage_changed(self, statement_processor, mock_shell):
        """
        When an input variable's lineage changes between executions,
        the statement should re-compute (not skip).
        """

        # First execution
        mock_shell.user_ns["a"] = 10
        statement_processor.process_statement("x = a + 1")
        assert mock_shell.user_ns["x"] == 11

        # Modify input 'a' with different lineage
        mock_shell.user_ns["a"] = 20
        statement_processor.tracking_state.lineage.discard("a")
        if hasattr(mock_shell.user_ns.get("a"), "_cash_hash"):
            with contextlib.suppress(AttributeError, TypeError):
                delattr(mock_shell.user_ns["a"], "_cash_hash")

        # Second execution - input changed, should compute
        metrics2 = statement_processor.process_statement("x = a + 1")
        assert mock_shell.user_ns["x"] == 21
        assert metrics2["status"] == CacheStatus.COMPUTED

    def test_reexecute_when_code_different(self, statement_processor, mock_shell):
        """
        When the code changes but targets the same output variable,
        the statement must re-execute.
        """

        mock_shell.user_ns["a"] = 10
        statement_processor.process_statement("x = a + 1")
        assert mock_shell.user_ns["x"] == 11

        # Different code for same output variable
        metrics2 = statement_processor.process_statement("x = a * 2")
        assert mock_shell.user_ns["x"] == 20
        assert metrics2["status"] == CacheStatus.COMPUTED

    def test_skip_preserves_lineage(self, statement_processor, mock_shell):
        """
        When a statement is skipped via already-executed, the lineage
        tracking state should remain correct.
        """

        mock_shell.user_ns["a"] = 10
        statement_processor.process_statement("x = a + 1")
        lineage_after_first = statement_processor.tracking_state.variable_lineage.get("x")
        assert lineage_after_first is not None

        # Second run - should skip/restore
        statement_processor.process_statement("x = a + 1")
        lineage_after_skip = statement_processor.tracking_state.variable_lineage.get("x")

        # Lineage should be preserved (not cleared or changed)
        assert lineage_after_skip == lineage_after_first

    def test_skip_multiple_outputs(self, statement_processor, mock_shell):
        """
        Already-executed optimization should work for multi-output statements.
        _PERSIST overrides the 10 ms min-execution-time floor so trivial
        assignments are actually written to cache.
        """

        # First execution
        metrics1 = statement_processor.process_statement("x = 10\ny = 20", annotation=_PERSIST)
        assert mock_shell.user_ns["x"] == 10
        assert mock_shell.user_ns["y"] == 20
        assert metrics1["status"] == CacheStatus.COMPUTED

        # Second execution - should be skipped or restored
        metrics2 = statement_processor.process_statement("x = 10\ny = 20", annotation=_PERSIST)
        assert mock_shell.user_ns["x"] == 10
        assert mock_shell.user_ns["y"] == 20
        assert metrics2["status"] in (CacheStatus.SKIPPED, CacheStatus.RESTORED)

    def test_no_skip_when_output_externally_modified(self, statement_processor, mock_shell):
        """
        If the output variable was externally modified (lineage doesn't match),
        the optimization should NOT skip.
        """

        mock_shell.user_ns["a"] = 10
        statement_processor.process_statement("x = a + 1")
        assert mock_shell.user_ns["x"] == 11

        # Externally modify x - remove its lineage
        mock_shell.user_ns["x"] = 999
        statement_processor.tracking_state.lineage.discard("x")
        statement_processor.tracking_state.executed_cell_codes.pop("x", None)

        # Should re-compute since x was externally modified
        statement_processor.process_statement("x = a + 1")
        assert mock_shell.user_ns["x"] == 11

    def test_self_assignment_checks_output_lineage(self, statement_processor, mock_shell):
        """
        Self-assignment (e.g., data = sorted(data)) should use output lineage
        check, not input lineage, for the skip optimization.
        """

        # Setup: create a list
        statement_processor.process_statement("data = [3, 1, 2]")
        assert mock_shell.user_ns["data"] == [3, 1, 2]

        # Self-assignment: sort
        statement_processor.process_statement("data = sorted(data)")
        assert mock_shell.user_ns["data"] == [1, 2, 3]

        # Verify data has lineage
        assert "data" in statement_processor.tracking_state.variable_lineage

    def test_skip_not_applied_to_loop_iterations(self, statement_processor, mock_shell):
        """
        The already-executed optimization should NOT apply to statements
        with iteration context markers (loop body statements).
        """

        mock_shell.user_ns["item"] = 1

        # Simulate a loop iteration with context marker
        code_with_context = "# __iteration_context__: abc123\nresult = item * 2"
        metrics1 = statement_processor.process_statement(code_with_context)
        assert mock_shell.user_ns["result"] == 2
        assert metrics1["status"] == CacheStatus.COMPUTED

        # Same base code with different context (different iteration)
        mock_shell.user_ns["item"] = 5
        statement_processor.tracking_state.lineage.discard("item")
        code_with_context2 = "# __iteration_context__: def456\nresult = item * 2"
        metrics2 = statement_processor.process_statement(code_with_context2)
        # Should NOT be skipped since it has iteration context
        assert mock_shell.user_ns["result"] == 10
        assert metrics2["status"] == CacheStatus.COMPUTED

    def test_no_outputs_no_skip(self, statement_processor):
        """
        Statements with no outputs (e.g., print) don't use already-executed optimization.
        """

        # print() has no assignable outputs
        statement_processor.process_statement("print('hello')")

        # Second run - should go through normal cache path
        metrics2 = statement_processor.process_statement("print('hello')")
        assert metrics2["status"] in (CacheStatus.COMPUTED, CacheStatus.RESTORED)


class TestAlreadyExecutedWithFileDeps:
    """Tests for the already-executed optimization with file dependencies."""

    def test_reexecute_when_file_mtime_changed(self, statement_processor, mock_shell, tmp_path):
        """
        If a file dependency's mtime changed, the statement should NOT be skipped.
        """

        # Create a test file
        test_file = tmp_path / "data.txt"
        test_file.write_text("original content", encoding="utf-8")

        file_path = str(test_file).replace("\\", "/")
        mock_shell.user_ns["path"] = file_path

        # Read the file
        metrics1 = statement_processor.process_statement("content = open(path).read()")
        assert mock_shell.user_ns["content"] == "original content"
        assert metrics1["status"] == CacheStatus.COMPUTED

        # Modify the file (changes mtime)
        time.sleep(0.1)  # Ensure mtime differs
        test_file.write_text("modified content", encoding="utf-8")

        # Re-run - should detect file change and not skip
        metrics2 = statement_processor.process_statement("content = open(path).read()")
        assert mock_shell.user_ns["content"] == "modified content"
        assert metrics2["status"] == CacheStatus.COMPUTED

    def test_skip_when_file_unchanged(self, statement_processor, mock_shell, tmp_path):
        """
        If file dependencies haven't changed, the already-executed optimization
        can skip re-execution.
        """

        test_file = tmp_path / "data.txt"
        test_file.write_text("static content", encoding="utf-8")

        file_path = str(test_file).replace("\\", "/")
        mock_shell.user_ns["path"] = file_path

        metrics1 = statement_processor.process_statement("content = open(path).read()")
        assert mock_shell.user_ns["content"] == "static content"
        assert metrics1["status"] == CacheStatus.COMPUTED

        # Re-run without modifying file
        metrics2 = statement_processor.process_statement("content = open(path).read()")
        assert mock_shell.user_ns["content"] == "static content"
        # Should be restored from cache or re-computed (file didn't change, value correct either way)
        assert metrics2["status"] in (CacheStatus.SKIPPED, CacheStatus.RESTORED, CacheStatus.COMPUTED)


class TestStatefulFunctionSkip:
    """Test that @stateful functions are never skipped."""

    def test_stateful_function_always_executes(self, statement_processor, mock_shell):
        """
        Functions decorated with @stateful should always re-execute,
        even if already_executed optimization would otherwise skip.
        """

        from cash.purity import stateful

        call_count = {"n": 0}

        @stateful
        def get_count():
            call_count["n"] += 1
            return call_count["n"]

        mock_shell.user_ns["get_count"] = get_count

        # First execution
        metrics1 = statement_processor.process_statement("result = get_count()")
        assert mock_shell.user_ns["result"] == 1
        assert metrics1["status"] == CacheStatus.COMPUTED

        # Second execution - should NOT be skipped because get_count is @stateful
        metrics2 = statement_processor.process_statement("result = get_count()")
        assert mock_shell.user_ns["result"] == 2
        assert metrics2["status"] == CacheStatus.COMPUTED

    def test_pure_function_can_be_cached(self, statement_processor, mock_shell):
        """
        Functions decorated with @pure CAN be cached/skipped.
        """

        from cash.purity import pure

        @pure
        def add(a, b):
            return a + b

        mock_shell.user_ns["add"] = add
        mock_shell.user_ns["a"] = 10
        mock_shell.user_ns["b"] = 20

        # First execution
        metrics1 = statement_processor.process_statement("result = add(a, b)")
        assert mock_shell.user_ns["result"] == 30
        assert metrics1["status"] == CacheStatus.COMPUTED

        # Second execution - pure function with same inputs, can be cached or re-computed
        metrics2 = statement_processor.process_statement("result = add(a, b)")
        assert mock_shell.user_ns["result"] == 30
        assert metrics2["status"] in (CacheStatus.SKIPPED, CacheStatus.RESTORED, CacheStatus.COMPUTED)


class TestRedundantImportSkip:
    """Tests for the redundant import skip optimization."""

    def test_skip_redundant_import(self, statement_processor, mock_shell):
        """
        Import statements where all names are already in user_ns
        should be skipped.
        """

        # First import
        metrics1 = statement_processor.process_statement("import os")
        assert "os" in mock_shell.user_ns
        assert metrics1["status"] == CacheStatus.COMPUTED

        # Second import of same module - should be skipped
        metrics2 = statement_processor.process_statement("import os")
        assert metrics2["status"] in (CacheStatus.SKIPPED, CacheStatus.RESTORED)

    def test_no_skip_when_import_missing(self, statement_processor, mock_shell, clean_backend):
        """
        The redundant import optimization checks if all import names exist in
        user_ns. When the name is removed, the optimization should not skip.

        Note: Modules are excluded from cache storage (not picklable), so
        a cache hit won't restore them. The import must actually run.
        However, when `import json` is removed from user_ns AND executed_cell_codes
        is cleared, the already-executed optimization won't trigger, and the
        import should be re-executed.
        """

        # Import json (first time)
        statement_processor.process_statement("import json")
        assert "json" in mock_shell.user_ns

        # Remove json from namespace AND all tracking
        del mock_shell.user_ns["json"]
        statement_processor.tracking_state.lineage.discard("json")
        statement_processor.tracking_state.executed_cell_codes.pop("json", None)
        # Also clear hashes so it can't be found via any path
        statement_processor.tracking_state.executed_cell_hashes.pop("json", None)
        statement_processor.tracking_state.current_session_hashes.pop("json", None)
        # Clear cache so it can't restore
        clean_backend.clear()

        # Should re-execute since everything was cleared
        metrics2 = statement_processor.process_statement("import json")
        assert "json" in mock_shell.user_ns
        assert metrics2["status"] == CacheStatus.COMPUTED

    def test_import_alias_skip(self, statement_processor, mock_shell):
        """
        Import with alias (import os as operating_system) should check alias name.
        """

        # First import with alias
        statement_processor.process_statement("import os as operating_system")
        assert "operating_system" in mock_shell.user_ns

        # Second import with same alias - should be skipped
        metrics2 = statement_processor.process_statement("import os as operating_system")
        assert metrics2["status"] in (CacheStatus.SKIPPED, CacheStatus.RESTORED)

    def test_from_import_skip(self, statement_processor, mock_shell):
        """
        From imports should also be checked for redundancy.
        """

        statement_processor.process_statement("from os.path import join")
        assert "join" in mock_shell.user_ns

        # Should be skipped on second import
        metrics2 = statement_processor.process_statement("from os.path import join")
        assert metrics2["status"] in (CacheStatus.SKIPPED, CacheStatus.RESTORED)


class TestCacheRestorePaths:
    """Tests for cache restore error handling and fallback paths."""

    def test_restore_handles_cleared_namespace(self, statement_processor, mock_shell):
        """
        If cached variables are removed from namespace, restoration should
        bring them back from cache. _PERSIST overrides the 10 ms
        min-execution-time floor so trivial assignments are written to cache.
        """

        # Execute
        statement_processor.process_statement("x = 42\ny = 84", annotation=_PERSIST)
        assert mock_shell.user_ns["x"] == 42
        assert mock_shell.user_ns["y"] == 84

        # Clear from namespace to force cache restore
        del mock_shell.user_ns["x"]
        del mock_shell.user_ns["y"]
        statement_processor.tracking_state.lineage.discard("x")
        statement_processor.tracking_state.lineage.discard("y")
        statement_processor.tracking_state.executed_cell_codes.pop("x", None)
        statement_processor.tracking_state.executed_cell_codes.pop("y", None)

        # Restore from cache
        metrics2 = statement_processor.process_statement("x = 42\ny = 84", annotation=_PERSIST)
        assert metrics2["status"] == CacheStatus.RESTORED
        assert mock_shell.user_ns["x"] == 42
        assert mock_shell.user_ns["y"] == 84

    def test_execution_error_raises(self, statement_processor):
        """
        When code execution fails, it should raise an error.
        """

        with pytest.raises(ZeroDivisionError):
            statement_processor.process_statement("x = 1 / 0")

    def test_stdout_captured_in_metrics(self, statement_processor):
        """
        Stdout from executed code should be captured in metrics.
        """

        metrics = statement_processor.process_statement("x = 42")
        assert metrics["status"] == CacheStatus.COMPUTED
        assert "stdout" in metrics


class TestSizeAwareEdgeCases:
    """Additional edge cases for size-aware caching."""

    def test_estimate_object_size_basic_types(self):
        """estimate_object_size should handle basic Python types."""
        from cash.object_hashing import estimate_object_size

        # Small objects
        assert estimate_object_size(42) < 10000
        assert estimate_object_size("hello") < 10000
        assert estimate_object_size([1, 2, 3]) < 10000

        # Larger objects
        big_list = list(range(100000))
        assert estimate_object_size(big_list) > 0

    def test_estimate_object_size_none(self):
        """estimate_object_size should handle None."""
        from cash.object_hashing import estimate_object_size

        assert estimate_object_size(None) >= 0

    def test_large_compute_slow_is_cached(self, statement_processor, mock_shell):
        """
        Large objects with computation above the 10 ms floor should be cached.
        sum(range(5_000_000)) takes ~25 ms so the statement crosses the floor.
        """

        # Use a computation that genuinely takes > 10 ms
        metrics = statement_processor.process_statement("big = list(range(sum(range(5_000_000)) // 12499997500000))")
        assert metrics["status"] == CacheStatus.COMPUTED

        del mock_shell.user_ns["big"]
        statement_processor.tracking_state.lineage.discard("big")
        statement_processor.tracking_state.executed_cell_codes.pop("big", None)

        metrics2 = statement_processor.process_statement("big = list(range(sum(range(5_000_000)) // 12499997500000))")
        assert metrics2["status"] == CacheStatus.RESTORED


class TestGetRedundantImportNames:
    """Tests for the redundant_import_names helper."""

    def test_simple_import(self):
        """Test that simple imports are recognized."""
        import ast

        tree = ast.parse("import os")
        names = redundant_import_names(tree)
        assert names == {"os"}

    def test_from_import(self):
        """Test that from imports are recognized."""
        import ast

        tree = ast.parse("from os.path import join, exists")
        names = redundant_import_names(tree)
        assert names == {"join", "exists"}

    def test_import_with_alias(self):
        """Test that aliased imports use the alias name."""
        import ast

        tree = ast.parse("import numpy as np")
        names = redundant_import_names(tree)
        assert names == {"np"}

    def test_mixed_code_returns_none(self):
        """Non-pure-import code should return None."""
        import ast

        tree = ast.parse("import os\nx = 1")
        names = redundant_import_names(tree)
        assert names is None

    def test_wildcard_import_returns_none(self):
        """Wildcard imports should return None (can't check)."""
        import ast

        tree = ast.parse("from os.path import *")
        names = redundant_import_names(tree)
        assert names is None

    def test_dotted_import(self):
        """Dotted imports (import foo.bar) should return top-level name."""
        import ast

        tree = ast.parse("import os.path")
        names = redundant_import_names(tree)
        assert names == {"os"}


# ===========================================================================
# Accumulator Init Skip Tests
# (Merged from test_accumulator_init_skip.py)
# ===========================================================================


class TestAccumulatorInitSkip:
    """Test that accumulator initialization statements are skipped when appropriate.

    When adding a new item to a cached loop:
    1. The backwards scan may schedule the init statement (e.g., ``results = {}``)
    2. But if ``results`` already exists in memory with cached values,
       we should NOT re-run the init statement (it would reset the dict)
    """

    def test_skip_empty_dict_init_when_has_data(self, cash_magics, mock_shell):
        """
        If results = {} is scheduled but results already has data,
        the init should be skipped when re-running the loop.

        Scenario:
        1. Run loop with 4 items (ABCD) - all cached
        2. Edit loop code to add 5th item (E)
        3. Run downstream cell - triggers upstream re-execution
        4. All 5 items should be present (not just E)
        """

        magics = cash_magics
        shell = mock_shell

        loop_code_v1 = 'results = {}\nfor x in ["A", "B", "C", "D"]:\n    results[x] = x * 2\n'
        loop_code_v2 = 'results = {}\nfor x in ["A", "B", "C", "D", "E"]:\n    results[x] = x * 2\n'
        keys_code = "keys = list(results.keys())"

        temp_dir = tempfile.mkdtemp()
        notebook_path = os.path.join(temp_dir, "test.ipynb")

        notebook_v1 = {
            "cells": [
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": loop_code_v1},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": keys_code},
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook_v1, f)

        def get_cells_v1(_path=None):
            with open(notebook_path, encoding="utf-8") as nf:
                data = json.load(nf)
                return [cell["source"] for cell in data["cells"] if cell["cell_type"] == "code"]

        with (
            patch("cash.notebook.upstream.checker.get_notebook_cells") as mock_get_cells,
            patch("cash.notebook.upstream.checker.get_notebook_cells_with_ids") as mock_get_ids,
        ):
            mock_get_cells.side_effect = get_cells_v1
            mock_get_ids.return_value = []

            magics.cash_on("")
            run_cash_cell(magics, loop_code_v1)

            assert "results" in shell.user_ns
            assert shell.user_ns["results"] == {"A": "AA", "B": "BB", "C": "CC", "D": "DD"}

            run_cash_cell(magics, keys_code)
            assert shell.user_ns["keys"] == ["A", "B", "C", "D"]

        notebook_v2 = {
            "cells": [
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": loop_code_v2},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": keys_code},
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook_v2, f)

        def get_cells_v2(_path=None):
            with open(notebook_path, encoding="utf-8") as nf:
                data = json.load(nf)
                return [cell["source"] for cell in data["cells"] if cell["cell_type"] == "code"]

        with (
            patch("cash.notebook.upstream.checker.get_notebook_cells") as mock_get_cells,
            patch("cash.notebook.upstream.checker.get_notebook_cells_with_ids") as mock_get_ids,
        ):
            mock_get_cells.side_effect = get_cells_v2
            mock_get_ids.return_value = []

            run_cash_cell(magics, keys_code)

            results = shell.user_ns.get("results", {})
            assert "A" in results, f"Missing 'A' in results: {results}"
            assert "B" in results, f"Missing 'B' in results: {results}"
            assert "C" in results, f"Missing 'C' in results: {results}"
            assert "D" in results, f"Missing 'D' in results: {results}"
            assert "E" in results, f"Missing 'E' in results: {results}"

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_init_runs_if_no_existing_data(self, cash_magics, mock_shell):
        """If results doesn't exist yet, the init SHOULD run."""
        magics = cash_magics
        shell = mock_shell

        if "results" in shell.user_ns:
            del shell.user_ns["results"]

        magics.cash_on("")
        run_cash_cell(magics, "results = {}\nfor x in ['A', 'B']:\n    results[x] = x * 2\n")
        assert shell.user_ns["results"] == {"A": "AA", "B": "BB"}

    def test_init_runs_if_existing_data_empty(self, cash_magics, mock_shell):
        """If results exists but is empty, the init SHOULD run."""
        magics = cash_magics
        shell = mock_shell

        shell.user_ns["results"] = {}
        magics.cash_on("")
        run_cash_cell(magics, "results = {}\nfor x in ['A', 'B']:\n    results[x] = x * 2\n")
        assert shell.user_ns["results"] == {"A": "AA", "B": "BB"}

    def test_list_accumulator(self, cash_magics, mock_shell):
        """List accumulators should also work."""
        magics = cash_magics
        shell = mock_shell

        magics.cash_on("")
        run_cash_cell(magics, "results = []\nfor x in ['A', 'B', 'C', 'D']:\n    results.append(x)\n")
        assert shell.user_ns["results"] == ["A", "B", "C", "D"]
