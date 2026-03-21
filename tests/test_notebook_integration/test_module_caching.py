"""
Tests for module-dependent statement caching.

When a cell contains `import metrics` followed by statements that use the
module (e.g., `print(metrics.increment(5))`), the module-dependent statements
should be fully cached after the second execution of the cell.

Bug scenario (fixed by auto-tracking local modules before _capture_variables):

  Call 1 (fresh session): All statements execute normally.
    - `import metrics` executes.  At this point, `metrics` is NOT in
      `function_tracker._tracked_modules` because auto_track_local_imports
      ran pre-execution when the module wasn't yet in sys.modules.
    - `_capture_variables` runs for `import metrics` and computes a lineage
      WITHOUT the module source hash component (module_lineage_component = "").
    - `print(metrics.increment(5))` executes and is cached with a key derived
      from the input lineage of `metrics` (which lacks the source hash).

  Call 2: `import metrics` is SKIPPED (redundant import optimisation).
    - `_capture_variables` runs via the skip path.  Now `metrics` IS in
      `_tracked_modules` (from post-execution tracking at the end of call 1).
    - The lineage now INCLUDES `module_lineage_component` → different hash.
    - `print(metrics.increment(5))` uses the new `metrics` lineage as input
      → different cache key → cache miss → EXECUTED again (should be RESTORED).

  Call 3: Everything matches → all cached correctly.

Fix: call `auto_track_local_imports(code)` right after execution and BEFORE
`_capture_variables` so the module is tracked from the very first execution.
"""
import pytest
import textwrap

pytestmark = pytest.mark.modules


def _extract_statuses(raw_output: str) -> list:
    """Extract caching statuses from raw debug output.
    
    Looks for patterns like:
    - [ALREADY_EXECUTED] Skipping re-execution  → 'SKIPPED'
    - [CACHE_HIT_DEBUG] Cache hit              → 'RESTORED'
    - SKIPPING redundant import                → 'SKIPPED_IMPORT'
    - [CACHE DEBUG] Executing (cache miss)     → 'EXECUTED'
    """
    statuses = []
    for line in raw_output.split('\n'):
        if '[ALREADY_EXECUTED] Skipping re-execution' in line:
            statuses.append('SKIPPED')
        elif '[CACHE_HIT_DEBUG] Cache hit' in line:
            statuses.append('RESTORED')
        elif 'SKIPPING redundant import' in line:
            statuses.append('SKIPPED_IMPORT')
        elif '[CACHE DEBUG] Executing (cache miss)' in line:
            statuses.append('EXECUTED')
    return statuses


class TestModuleCachingConsistency:
    """Module-dependent statements should be cached after just 2 executions."""

    def test_module_dep_cached_on_second_call(self, nb_runner, tmp_path):
        """
        After the first execution (all COMPUTED), a second execution should
        SKIP the import and RESTORE/SKIP the module-dependent statements.
        No third execution should be necessary.
        """
        # Create a simple local module in the work directory
        module_path = nb_runner.work_dir / "metrics.py"
        module_path.write_text(textwrap.dedent("""\
            _counter = 0
            def increment(n):
                global _counter
                _counter += n
                return _counter
        """))

        # Create notebook that imports and uses the module
        nb_runner.create_notebook([
            "import metrics",
            "result = metrics.increment(5)",
            "print(f'Result: {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()

        # --- Call 1: fresh session, everything should execute ---
        nb_runner.run_all()
        output1 = nb_runner.get_output(3)
        assert "Result:" in output1

        # --- Call 2: import should be SKIPPED, other statements cached ---
        nb_runner.run_all()
        out2_cell2_raw = nb_runner.get_raw_output(2)

        # Cell 2 (result = metrics.increment(5)) should NOT have a cache miss
        assert 'Executing (cache miss)' not in out2_cell2_raw, (
            f"Cell 2 had a cache miss on call 2 (should have been cached):\n{out2_cell2_raw}"
        )

        # --- Call 3: still cached ---
        nb_runner.run_all()
        out3_cell2_raw = nb_runner.get_raw_output(2)

        # Both call 2 and call 3 should show the same caching behavior
        statuses_2 = _extract_statuses(out2_cell2_raw)
        statuses_3 = _extract_statuses(out3_cell2_raw)
        assert statuses_2 == statuses_3, (
            f"Caching statuses differ between call 2 and call 3.\n"
            f"Call 2: {statuses_2}\n"
            f"Call 3: {statuses_3}"
        )

    def test_multi_statement_cell_with_module(self, nb_runner, tmp_path):
        """
        A single cell with import + module usage should cache after 2 calls.
        This is the exact scenario from the user's bug report.
        """
        # Create local module
        module_path = nb_runner.work_dir / "metrics.py"
        module_path.write_text(textwrap.dedent("""\
            _counter = 0
            def increment(n):
                global _counter
                _counter += n
                return _counter
        """))

        # Single cell with import + usage (multi-statement)
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import metrics
                print("Testing metrics module...")
                print(metrics.increment(5))"""),
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()

        # --- Call 1: everything executes ---
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "Testing metrics module..." in out1

        # --- Call 2: should be fully cached (no cache misses) ---
        nb_runner.run_all()
        out2_raw = nb_runner.get_raw_output(1)

        # No statement in the cell should have a cache miss
        assert 'Executing (cache miss)' not in out2_raw, (
            f"Cell had a cache miss on call 2 (should have been fully cached):\n{out2_raw}"
        )

        # --- Call 3: still cached ---
        nb_runner.run_all()
        out3_raw = nb_runner.get_raw_output(1)

        # Same caching behavior on call 2 and call 3
        statuses_2 = _extract_statuses(out2_raw)
        statuses_3 = _extract_statuses(out3_raw)
        assert statuses_2 == statuses_3, (
            f"Caching statuses differ between call 2 and call 3.\n"
            f"Call 2: {statuses_2}\n"
            f"Call 3: {statuses_3}"
        )

    def test_from_import_cached_on_second_call(self, nb_runner, tmp_path):
        """
        `from module import func` should also cache correctly on second call.
        """
        # Create local module
        module_path = nb_runner.work_dir / "helpers.py"
        module_path.write_text(textwrap.dedent("""\
            def double(x):
                return x * 2
        """))

        nb_runner.create_notebook([
            "from helpers import double",
            "result = double(21)",
            "print(f'Answer: {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()

        # Call 1: all execute
        nb_runner.run_all()
        assert "Answer: 42" in nb_runner.get_output(3)

        # Call 2: should be fully cached
        nb_runner.run_all()
        out2_cell3_raw = nb_runner.get_raw_output(3)

        # Cell 3 should not have a cache miss
        assert 'Executing (cache miss)' not in out2_cell3_raw, (
            f"Cell 3 had a cache miss on call 2 (should have been cached):\n{out2_cell3_raw}"
        )

        # Call 3: still cached
        nb_runner.run_all()
        out3_cell3_raw = nb_runner.get_raw_output(3)

        statuses_2 = _extract_statuses(out2_cell3_raw)
        statuses_3 = _extract_statuses(out3_cell3_raw)
        assert statuses_2 == statuses_3, (
            f"Cell 3 statuses differ between call 2 and call 3.\n"
            f"Call 2: {statuses_2}\n"
            f"Call 3: {statuses_3}"
        )
